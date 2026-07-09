"""Multigroup k-eigenvalue solvers: power iteration over the fission source.

Outer loop: standard power iteration on  M phi = (1/k) F phi, where M is the
block (groups) transport-approximation operator and F the fission production
operator. Within one outer iteration the groups are swept Gauss-Seidel style
(downscatter uses the freshest fluxes), and each within-group system is solved
by matrix-free Jacobi-preconditioned CG with a warm start from the previous
outer iteration.

Two angular approximations share this machinery:

- DiffusionEigenSolver: classic diffusion; group state = scalar flux.
- SP3EigenSolver: simplified P3; group state = (Phi1, phi2) moment pair,
  solved as one SPD block system per group (see SP3GroupOperator).

The inner CG tolerance is adapted to the outer residual, so early outers are
cheap and the final ones are tight; combined with warm starts, late outer
iterations cost only a handful of stencil applications.

Everything — fluxes, coupling coefficients, sources — lives on the device for
the entire solve; only per-iteration convergence scalars cross the PCIe bus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import asnumpy, device_name, get_backend, synchronize
from .grid import Grid
from .linalg import neumann_preconditioner, pcg
from .materials import Material
from .operator import BC_VACUUM, BC_ZERO_FLUX, GroupOperator, SP3GroupOperator


@dataclass
class Result:
    k_eff: float
    flux: object  # scalar flux, (G, nx, ny, nz) array on the solve device
    converged: bool
    outer_iterations: int
    inner_iterations: int
    solve_seconds: float
    device: str
    k_history: list = field(default_factory=list)
    source_error_history: list = field(default_factory=list)

    @property
    def flux_numpy(self) -> np.ndarray:
        return asnumpy(self.flux)

    def __repr__(self):
        status = "converged" if self.converged else "NOT CONVERGED"
        return (
            f"Result(k_eff={self.k_eff:.6f}, {status}, "
            f"{self.outer_iterations} outers / {self.inner_iterations} inners, "
            f"{self.solve_seconds:.2f} s on {self.device})"
        )


class Fields:
    """Per-cell, per-group cross-section fields on the solve device.

    Attributes (lists of (nx, ny, nz) device arrays, length G, except
    sigma_s which is a G x G nested list with None for empty couplings):
    nu_sigma_f, chi, removal, diffusion, sigma_t, sigma_s.
    """

    def __init__(self, xp, grid, materials, material_map, dtype):
        mats = [materials] if isinstance(materials, Material) else list(materials)
        G = mats[0].n_groups
        if any(m.n_groups != G for m in mats):
            raise ValueError("all materials must have the same number of groups")
        self.n_groups = G
        self.fissile = any(m.is_fissile for m in mats)

        if material_map is None:
            if len(mats) > 1:
                raise ValueError("material_map is required with multiple materials")
            lookup = lambda table: xp.full(grid.shape, float(table[0]), dtype=dtype)
        else:
            mmap = xp.asarray(np.asarray(material_map))
            if mmap.shape != grid.shape:
                raise ValueError(f"material_map shape {mmap.shape} != grid shape {grid.shape}")
            if int(mmap.min()) < 0 or int(mmap.max()) >= len(mats):
                raise ValueError("material_map indexes outside the materials list")
            lookup = lambda table: xp.asarray(table, dtype=dtype)[mmap]

        def per_group(attr):
            table = np.array([getattr(m, attr) for m in mats])  # (M, G)
            return [lookup(table[:, g]) for g in range(G)]

        self.nu_sigma_f = per_group("nu_sigma_f")
        self.chi = per_group("chi")
        self.removal = per_group("removal")
        self.diffusion = per_group("diffusion")
        self.sigma_t = per_group("sigma_t")

        sig_s = np.array([m.sigma_s for m in mats])  # (M, G, G)
        self.sigma_s = [[lookup(sig_s[:, gf, gt]) if np.any(sig_s[:, gf, gt]) else None
                         for gt in range(G)] for gf in range(G)]

    def fission_source(self, phi_by_group):
        """Sum_g nuSigma_f,g * phi_g over an iterable of per-group fluxes."""
        src = self.nu_sigma_f[0] * phi_by_group[0]
        for g in range(1, self.n_groups):
            src += self.nu_sigma_f[g] * phi_by_group[g]
        return src


class _PowerIterationSolver:
    """Shared field construction and power-iteration outer loop.

    Subclasses set self.ops (one operator per group, exposing .apply and
    .inv_diag) in _build_operators() and define the group state layout via
    _initial_state / _rhs / _phi.

    Parameters
    ----------
    grid         : Grid
    materials    : a single Material, or a list of Materials indexed by
                   material_map.
    material_map : optional int array of shape grid.shape assigning a material
                   index to every cell (omit for a homogeneous reactor).
    bc           : "zero-flux" (default) or "reflective"; a single string
                   applies to all six faces, or pass 3 per-axis entries /
                   6 per-face values, e.g. bc=(("reflective", "zero-flux"),
                   ("reflective", "zero-flux"), "reflective") for a quarter
                   core with reflective symmetry planes, solved in 2D.
    device       : "auto" | "gpu" | "cpu"
    dtype        : floating dtype for the solve (float64 default; float32
                   roughly doubles GPU throughput at ~1e-6 k accuracy).
    precond_degree : degree of the Neumann-polynomial preconditioner for the
                   inner CG (0 = plain Jacobi, the default). Each degree adds
                   one stencil apply per CG iteration but cuts the iteration
                   count -- and with it the global reductions that are the
                   GPU's only synchronization points. Degree 2-3 is a good
                   setting (cf. E et al., NED 320 (2017), where degree-3
                   Neumann-PCG was the fastest GPU solver for 2e4-3e6 cells).
    """

    def __init__(self, grid: Grid, materials, material_map=None,
                 bc: str = BC_ZERO_FLUX, device: str = "auto", dtype=np.float64,
                 active=None, mask_bc=BC_VACUUM, precond_degree: int = 0):
        self.grid = grid
        self.xp = xp = get_backend(device)
        self.device = device_name(xp)
        self.dtype = np.dtype(dtype)
        self.active = active
        self.mask_bc = mask_bc

        f = Fields(xp, grid, materials, material_map, self.dtype)
        if not f.fissile:
            raise ValueError("no fissile material: k-eigenvalue problem is undefined")
        self.fields = f
        self.n_groups = f.n_groups
        self.nu_sigma_f = f.nu_sigma_f  # list of (nx,ny,nz), len G
        self.chi = f.chi
        self.sigma_s = f.sigma_s

        self._build_operators(grid, f.diffusion, f.sigma_t, f.removal, bc)
        self.preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                                int(precond_degree))
                         for op in self.ops]

    # ---- hooks implemented by the angular approximation -------------------
    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        raise NotImplementedError

    def _initial_state(self):
        """Per-group solution state, as a list of device arrays."""
        raise NotImplementedError

    def _rhs(self, g, q0):
        """Within-group right-hand side from the isotropic source q0."""
        raise NotImplementedError

    def _phi(self, state_g):
        """Scalar flux phi0 of one group's state (view or expression)."""
        raise NotImplementedError

    # -----------------------------------------------------------------------
    def _fission_source(self, state):
        src = self.nu_sigma_f[0] * self._phi(state[0])
        for g in range(1, self.n_groups):
            src += self.nu_sigma_f[g] * self._phi(state[g])
        return src

    def solve(self, tol_k: float = 1e-7, tol_source: float = 1e-6,
              max_outer: int = 2000, inner_rtol_floor: float = 1e-10,
              k_guess: float = 1.0, verbose: bool = False) -> Result:
        """Run power iteration until |dk| < tol_k and the relative L2 change of
        the normalized fission source < tol_source."""
        xp, G = self.xp, self.n_groups
        synchronize(xp)
        t0 = time.perf_counter()

        state = self._initial_state()
        fsrc = self._fission_source(state)
        total = xp.sum(fsrc)
        if float(total) <= 0:
            raise RuntimeError("initial fission source is zero")
        k = float(k_guess)

        k_hist, err_hist = [], []
        inner_total = 0
        src_err = 1.0
        converged = False

        for outer in range(1, max_outer + 1):
            # Inner tolerance tracks the outer residual: cheap early, tight late.
            rtol = min(1e-3, max(0.1 * src_err, inner_rtol_floor, 0.01 * tol_source))

            for g in range(G):
                q0 = (self.chi[g] / k) * fsrc
                for gf in range(G):
                    s = self.sigma_s[gf][g]
                    if gf != g and s is not None:
                        q0 += s * self._phi(state[gf])
                state[g], n_it = pcg(self.ops[g].apply, self._rhs(g, q0), state[g],
                                     self.ops[g].inv_diag, xp, rtol=rtol,
                                     precond=self.preconds[g])
                inner_total += n_it

            fsrc_new = self._fission_source(state)
            total_new = xp.sum(fsrc_new)
            k_new = k * float(total_new / total)

            diff = fsrc_new / total_new - fsrc / total
            src_err = float(xp.sqrt(xp.sum(diff * diff) / xp.sum((fsrc_new / total_new) ** 2)))
            dk = abs(k_new - k)

            # Normalize so the mean fission source stays at 1 (avoids drift).
            scale = self.grid.n_cells / total_new
            for g in range(G):
                state[g] *= scale
            fsrc = fsrc_new * scale
            total = xp.sum(fsrc)
            k = k_new

            k_hist.append(k)
            err_hist.append(src_err)
            if verbose:
                print(f"  outer {outer:4d}  k = {k:.8f}  dk = {dk:.2e}  src_err = {src_err:.2e}")

            if dk < tol_k and src_err < tol_source:
                converged = True
                break

        synchronize(xp)
        return Result(
            k_eff=k,
            flux=xp.stack([self._phi(state[g]) for g in range(G)]),
            converged=converged,
            outer_iterations=outer,
            inner_iterations=inner_total,
            solve_seconds=time.perf_counter() - t0,
            device=self.device,
            k_history=k_hist,
            source_error_history=err_hist,
        )


class DiffusionEigenSolver(_PowerIterationSolver):
    """Multigroup neutron diffusion k-eigenvalue solver (see base for args)."""

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        self.ops = [GroupOperator(self.xp, grid, diffusion[g], removal[g], bc=bc,
                                  active=self.active, mask_bc=self.mask_bc)
                    for g in range(self.n_groups)]

    def _initial_state(self):
        return [self.xp.ones(self.grid.shape, dtype=self.dtype)
                for _ in range(self.n_groups)]

    def _rhs(self, g, q0):
        return q0

    def _phi(self, state_g):
        return state_g


class SP3EigenSolver(_PowerIterationSolver):
    """Multigroup simplified-P3 k-eigenvalue solver (see base for args).

    Same interface and cost structure as DiffusionEigenSolver with ~2x the
    work per group; captures leading transport effects that diffusion misses
    (steep flux gradients, strong absorbers, small cores).
    """

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        self.ops = [SP3GroupOperator(self.xp, grid, diffusion[g], sigma_t[g],
                                     removal[g], bc=bc,
                                     active=self.active, mask_bc=self.mask_bc)
                    for g in range(self.n_groups)]

    def _initial_state(self):
        state = []
        for _ in range(self.n_groups):
            u = self.xp.zeros((2,) + self.grid.shape, dtype=self.dtype)
            u[0] = 1.0
            state.append(u)
        return state

    def _rhs(self, g, q0):
        # Symmetrized block RHS: (q0, 5 * (-2/5) q0).
        rhs = self.xp.empty((2,) + self.grid.shape, dtype=self.dtype)
        rhs[0] = q0
        rhs[1] = -2.0 * q0
        return rhs

    def _phi(self, state_g):
        return state_g[0] - 2.0 * state_g[1]
