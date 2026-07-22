"""Discrete-ordinates (S_N) transport on the body-fitted triangular mesh.

The triangular counterpart of ``ndgpu.sn`` (2D Cartesian S_N), so discrete-
ordinates transport -- and the hybrid S_N/diffusion drum treatment -- run on the
actual HP-MR hex/triangular core, not a Cartesian stand-in.

The mesh is the same structured equilateral-triangle lattice the diffusion/SP3
solvers use (:class:`ndgpu.tri.TriGrid`): cells stored on (nrows, ncols, 2) with
the last index the down/up triangle of each rhombus, every interior cell coupled
to three neighbours at fixed offsets. That structure is what makes S_N tractable
here: rather than order the transport sweep by hand (and break cycles), we
assemble, for each ordinate, the sparse streaming+collision operator
L_Omega = Omega.grad + Sigma_t with upwind (step) differencing and factorize it
once. A "sweep" is then a triangular solve; the within-group scattering fixed
point is GMRES'd exactly as in the Cartesian solver, inside a fission power
iteration.

Upwind (step) differencing keeps every angular flux non-negative -- robust
through the near-black B4C drums where diamond differencing would ring -- at the
cost of being first-order in space (more numerically diffusive than the diamond
scheme of ``ndgpu.sn``). The equilateral triangle has edge length h, area
sqrt(3)/4 h^2, and the six (type, edge) outward normals below.

Boundaries: ``vacuum`` (the HP-MR core surface; incoming flux zero on edges
facing an excised/void cell or off-mesh) and ``periodic`` (wrap the lattice to a
torus -- an infinite medium whose flat-flux k is exactly k_inf, used to validate
the operator). CPU/numpy reference, not the GPU path.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, factorized, gmres

from .materials import Material
from .sn import SNResult, _anderson, quadrature_2d
from .tri import TriGrid

_SQRT3_2 = np.sqrt(3.0) / 2.0

# (source type, neighbour rhombus offset di, dj, neighbour type, outward normal).
# Down triangle (t=0): hypotenuse->up(i,j), bottom->up(i-1,j), left->up(i,j-1).
# Up   triangle (t=1): hypotenuse->down(i,j), top->down(i+1,j), right->down(i,j+1).
_EDGES = [
    (0, 0, 0, 1, (_SQRT3_2, 0.5)),
    (0, -1, 0, 1, (0.0, -1.0)),
    (0, 0, -1, 1, (-_SQRT3_2, 0.5)),
    (1, 0, 0, 0, (-_SQRT3_2, -0.5)),
    (1, 1, 0, 0, (0.0, 1.0)),
    (1, 0, 1, 0, (_SQRT3_2, -0.5)),
]


class TriSNTransportSolver:
    """S_N transport k-eigenvalue solver on a triangular mesh.

    Parameters
    ----------
    grid          : TriGrid (2D; shape (nrows, ncols, 2)).
    materials     : a Material or list indexed by material_map.
    material_map  : int array of shape grid.shape; omit for a homogeneous medium.
    active        : optional bool mask of in-core cells (shape grid.shape). Cells
                    outside it are excised; active cells facing them see the
                    ``bc`` boundary law. Defaults to all-active.
    n_polar, n_azi: product-quadrature sizes (n_azi a multiple of 4).
    bc            : "vacuum" (default) or "periodic" (torus / infinite medium).
    """

    def __init__(self, grid: TriGrid, materials, material_map=None, active=None,
                 n_polar: int = 3, n_azi: int = 12, bc: str = "vacuum"):
        if len(grid.shape) != 3 or grid.shape[2] != 2:
            raise ValueError("TriSNTransportSolver is 2D: grid shape (nr, nc, 2)")
        if bc not in ("vacuum", "periodic"):
            raise ValueError("bc must be 'vacuum' or 'periodic'")
        self.grid = grid
        self.nr, self.nc = grid.shape[0], grid.shape[1]
        self.bc = bc
        self.h = grid.side
        self.area = (np.sqrt(3.0) / 4.0) * self.h ** 2
        self.N = self.nr * self.nc * 2

        mats = [materials] if isinstance(materials, Material) else list(materials)
        self.G = mats[0].n_groups
        mmap = (np.zeros(grid.shape, int) if material_map is None
                else np.asarray(material_map).reshape(grid.shape))
        self.active = (np.ones(grid.shape, bool) if active is None
                       else np.asarray(active).reshape(grid.shape))

        def per_group(fn):
            table = np.array([fn(m) for m in mats])
            return np.stack([table[mmap, g] for g in range(self.G)])  # (G, nr, nc, 2)

        self.st = per_group(lambda m: np.asarray(m.sigma_t, float))
        removal = per_group(lambda m: np.asarray(m.removal, float))
        self.ss_self = np.maximum(self.st - removal, 0.0)
        self.nsf = per_group(lambda m: np.asarray(m.nu_sigma_f, float))
        self.chi = per_group(lambda m: np.asarray(m.chi, float))
        sig_s = np.array([m.sigma_s for m in mats])
        self.scatter = [[None] * self.G for _ in range(self.G)]
        for gf in range(self.G):
            for gt in range(self.G):
                if gf != gt and np.any(sig_s[:, gf, gt]):
                    self.scatter[gf][gt] = sig_s[mmap, gf, gt]
        if not np.any(self.nsf):
            raise ValueError("no fissile material: k-eigenvalue is undefined")

        self.mu, self.eta, self.w = quadrature_2d(n_polar, n_azi)
        self.M = self.mu.size
        self._cell = np.arange(self.N).reshape(self.nr, self.nc, 2)
        self._act_flat = self.active.reshape(-1)
        self._prefactor()

    def _prefactor(self):
        """Assemble and LU-factorize L_Omega = Omega.grad + Sigma_t (upwind) for
        every ordinate and group, once."""
        h, area = self.h, self.area
        nr, nc = self.nr, self.nc
        cell = self._cell
        act = self.active
        # base diagonal per group = Sigma_t * area (collision); streaming adds to it.
        self._solvers = [[None] * self.M for _ in range(self.G)]
        for g in range(self.G):
            st_area = (self.st[g] * area).reshape(-1)
            for m in range(self.M):
                mu, eta = self.mu[m], self.eta[m]
                diag = st_area.copy()
                rows, cols, vals = [], [], []
                for (t, di, dj, tn, (nx, ny)) in _EDGES:
                    On = (mu * nx + eta * ny) * h            # (Omega.n) * edge length
                    src = cell[:, :, t]                      # (nr, nc) source cell ids
                    src_act = act[:, :, t]
                    ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
                    ni, nj = ii + di, jj + dj
                    if self.bc == "periodic":
                        ni %= nr; nj %= nc
                        inb = np.ones_like(ni, bool)
                    else:
                        inb = (ni >= 0) & (ni < nr) & (nj >= 0) & (nj < nc)
                    nbr = np.full((nr, nc), -1)
                    nbr[inb] = cell[ni[inb], nj[inb], tn]
                    nbr_act = np.zeros((nr, nc), bool)
                    nbr_act[inb] = act[ni[inb], nj[inb], tn]
                    valid = src_act & nbr_act                # interior coupled edge
                    if On > 0:                               # outflow: psi_face = psi_c
                        # leakage out across every edge of an active cell (to an
                        # active neighbour or across the vacuum/void boundary).
                        d = np.where(src_act, On, 0.0)
                        np.add.at(diag, src.reshape(-1), d.reshape(-1))
                    else:                                    # inflow: psi_face = psi_nbr
                        s = src[valid]; n = nbr[valid]
                        rows.append(s); cols.append(n)
                        vals.append(np.full(s.size, On))
                        # inflow across a vacuum/void boundary contributes 0.
                # inactive cells: identity rows (psi = 0)
                inact = ~self._act_flat
                diag = np.where(inact, 1.0, diag)
                rows.append(np.arange(self.N)); cols.append(np.arange(self.N))
                vals.append(diag)
                A = sp.csr_matrix((np.concatenate(vals),
                                   (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(self.N, self.N))
                # zero any stray couplings out of inactive rows
                self._solvers[g][m] = factorized(A.tocsc())

    def _sweep(self, g, src_flat):
        """phi = Sum_m w_m L_Omega^-1 (src * area) for an isotropic source."""
        rhs = src_flat * self.area
        rhs = np.where(self._act_flat, rhs, 0.0)
        phi = np.zeros(self.N)
        for m in range(self.M):
            phi += self.w[m] * self._solvers[g][m](rhs)
        return phi

    def _solve_group(self, g, qext_flat, phi0, tol):
        """Within-group GMRES on (I - T) phi = b, T = one sweep of the scatter
        source; the boundary is vacuum/periodic (folded into L_Omega), so no
        boundary fixed point is needed."""
        ss = self.ss_self[g].reshape(-1)
        b = self._sweep(g, qext_flat)                        # source-only response

        def op(x):                                           # (I - T) x, T = scatter sweep
            return x - self._sweep(g, ss * x)

        A = LinearOperator((self.N, self.N), matvec=op, dtype=float)
        phi, _ = gmres(A, b, x0=phi0, rtol=min(tol, 1e-4), atol=0.0, maxiter=400)
        return phi

    def solve(self, tol_k: float = 1e-7, tol_source: float = 1e-6,
              max_outer: int = 500, verbose: bool = False) -> SNResult:
        t0 = time.perf_counter()
        G, N = self.G, self.N
        phi = [np.where(self._act_flat, 1.0, 0.0) for _ in range(G)]
        nsf = [self.nsf[g].reshape(-1) for g in range(G)]
        chi = [self.chi[g].reshape(-1) for g in range(G)]
        scat = [[None if self.scatter[gf][g] is None
                 else self.scatter[gf][g].reshape(-1) for g in range(G)]
                for gf in range(G)]
        fiss = sum(nsf[g] * phi[g] for g in range(G))
        n_act = float(self._act_flat.sum())
        total = fiss.sum()
        k = 1.0
        prev_rel = prev_err = 1.0
        k_hist = []
        converged = False
        outer = 0
        hist = []                                            # Anderson (fsrc_in, raw)
        for outer in range(1, max_outer + 1):
            fs = fiss / k
            tol = min(1e-3, max(0.05 * prev_rel, 0.01 * tol_k, 1e-10))
            fiss_in = fiss
            phi_new = [None] * G
            for g in range(G):
                q = chi[g] * fs
                for gf in range(G):
                    if gf != g and scat[gf][g] is not None:
                        src = phi_new[gf] if gf < g else phi[gf]
                        q = q + scat[gf][g] * src
                phi_new[g] = self._solve_group(g, q, phi[g], tol)
            fiss_new = sum(nsf[g] * phi_new[g] for g in range(G))
            total_new = fiss_new.sum()
            k_new = k * total_new / total
            dk = abs(k_new - k)
            rel = max(np.max(np.abs(phi_new[g] - phi[g])) /
                      max(np.max(np.abs(phi_new[g])), 1e-30) for g in range(G))
            phi, k = phi_new, k_new
            k_hist.append(k)
            if verbose:
                print(f"  outer {outer:3d}  k = {k:.7f}  dk = {dk:.2e}  rel = {rel:.2e}")
            if dk < tol_k and rel < tol_source and tol < max(1e-8, tol_k):
                converged = True
                break
            # Anderson-accelerate the fission source (the loosely-coupled core has
            # a dominance ratio near 1, so plain power iteration crawls).
            raw = fiss_new * (n_act / total_new)             # mean active source = 1
            if rel > 1.1 * prev_err:
                hist = []
            hist.append((fiss_in, raw))
            if len(hist) > 6:
                hist.pop(0)
            fiss = _anderson(hist)
            fiss *= n_act / fiss.sum()
            total = fiss.sum()
            prev_rel = max(rel, dk)
            prev_err = rel
        flux = np.stack([phi[g].reshape(self.grid.shape) for g in range(G)])
        return SNResult(k_eff=k, flux=flux, converged=converged,
                        outer_iterations=outer,
                        solve_seconds=time.perf_counter() - t0,
                        n_ordinates=self.M, k_history=k_hist)
