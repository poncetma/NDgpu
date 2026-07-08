"""Time-dependent multigroup neutron diffusion with delayed neutron precursors.

Solves, for each group g and precursor family i:

    (1/v_g) dphi_g/dt = div(D_g grad phi_g) - Sigma_r,g phi_g
                        + sum_{g'!=g} Sigma_s,g'->g phi_g'
                        + chi_g (1 - beta) S(t) + chi_d,g sum_i lambda_i C_i
    dC_i/dt           = beta_i S(t) - lambda_i C_i

with the fission source S = (1/k0) sum_g nuSigma_f,g phi_g. Dividing by the
initial eigenvalue k0 (the standard "critical adjustment") makes the t=0
steady state an exact equilibrium of the transient equations, so power
evolution is driven purely by the applied perturbation.

Time discretization is backward Euler (first order, unconditionally stable —
the right default for stiff reactor kinetics). The precursor update is solved
analytically per step and substituted into the flux equation, so each step is
a fixed-source multigroup problem:

    [A_g + 1/(v_g dt)] phi^{n+1} = phi^n/(v_g dt) + inscatter^{n+1}
        + chi_g [(1-beta) + omega] S^{n+1} + chi_d,g sum_i lambda_i C_i^n/(1+lambda_i dt)

with omega = sum_i lambda_i dt beta_i/(1+lambda_i dt). The bracketed operator
is the steady diffusion operator plus a positive diagonal shift — still SPD
and *better* conditioned, so the same matrix-free CG machinery applies; the
fission/scattering coupling converges in a few Gauss-Seidel sweeps per step
thanks to warm starts.

Time-dependent problems (control rod movement, cross-section ramps) are
described by a callable  problem_at(t) -> (materials, material_map)  whose
results should be cached by the caller: fields and operators are rebuilt only
when the returned objects change identity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import asnumpy, device_name, get_backend, synchronize
from .grid import Grid
from .linalg import pcg
from .materials import Kinetics
from .operator import BC_VACUUM, BC_ZERO_FLUX, GroupOperator
from .solver import DiffusionEigenSolver, Fields, Result


@dataclass
class TransientResult:
    times: np.ndarray        # (n_steps + 1,)
    power: np.ndarray        # (n_steps + 1,), relative to the initial power
    k0: float                # initial eigenvalue used for critical adjustment
    steady: Result           # the initial steady-state solve
    flux: object             # final scalar flux (G, nx, ny, nz), device array
    precursors: object       # final precursor fields (I, nx, ny, nz)
    total_inner_iterations: int
    solve_seconds: float
    device: str

    @property
    def flux_numpy(self) -> np.ndarray:
        return asnumpy(self.flux)

    def __repr__(self):
        return (
            f"TransientResult(t = 0..{self.times[-1]:g} s in {len(self.times) - 1} steps, "
            f"k0={self.k0:.6f}, P(end)={self.power[-1]:.4f} P0, "
            f"{self.total_inner_iterations} inners, {self.solve_seconds:.2f} s on {self.device})"
        )


class TransientSolver:
    """Time-dependent multigroup diffusion solver.

    Parameters
    ----------
    grid       : Grid
    problem_at : callable t -> (materials, material_map). material_map may be
                 None for a homogeneous reactor. Return cached objects while
                 nothing changes — operators are rebuilt only on identity
                 change of either element.
    kinetics   : Kinetics (velocities, delayed families).
    bc, device, dtype : as for DiffusionEigenSolver.
    """

    def __init__(self, grid: Grid, problem_at, kinetics: Kinetics,
                 bc=BC_ZERO_FLUX, device: str = "auto", dtype=np.float64,
                 active=None, mask_bc=BC_VACUUM,
                 group_operator=GroupOperator, eig_solver=DiffusionEigenSolver):
        self.grid = grid
        self.problem_at = problem_at
        self.kinetics = kinetics
        self.bc = bc
        self.active = active
        self.mask_bc = mask_bc
        # Geometry is pluggable: the Cartesian (GroupOperator/DiffusionEigenSolver)
        # or hex (HexGroupOperator/HexDiffusionEigenSolver) pair share signatures.
        self.group_operator = group_operator
        self.eig_solver = eig_solver
        self.xp = get_backend(device)
        self.device = device_name(self.xp)
        self.dtype = np.dtype(dtype)

    def solve(self, t_end: float, dt: float, tol_step: float = 1e-6,
              max_sweeps: int = 200, anderson_depth: int = 5,
              steady_kwargs: dict | None = None,
              verbose: bool = False) -> TransientResult:
        """March from the steady state at t=0 to t_end with fixed step dt.

        anderson_depth : number of past fission-source iterates retained by the
            Anderson acceleration of the within-step fixed point (window m+1 for
            m residual differences). Depth 1 disables it (plain Picard).
        """
        xp, kin = self.xp, self.kinetics
        beta, lam = kin.beta, kin.decay
        n_steps = int(round(t_end / dt))
        synchronize(xp)
        t0 = time.perf_counter()

        # --- initial condition: steady state, critically adjusted ------------
        mats, mmap = self.problem_at(0.0)
        eig = self.eig_solver(self.grid, mats, mmap, bc=self.bc,
                              device="cpu" if xp is np else "gpu",
                              dtype=self.dtype,
                              active=self.active, mask_bc=self.mask_bc)
        G = eig.n_groups
        if len(kin.velocities) != G:
            raise ValueError("kinetics.velocities must have one entry per group")
        steady = eig.solve(**(steady_kwargs or dict(tol_k=1e-8, tol_source=1e-7)))
        if not steady.converged:
            raise RuntimeError(f"initial steady state did not converge: {steady}")
        k0 = steady.k_eff

        fields = eig.fields
        phi = [steady.flux[g].copy() for g in range(G)]
        S = fields.fission_source(phi) / k0
        scale = 1.0 / float(xp.sum(S))          # P(0) = 1
        for g in range(G):
            phi[g] *= scale
        S = S * scale
        C = [(beta[i] / lam[i]) * S for i in range(kin.n_families)]  # equilibrium

        # Delayed-source weight of the end-of-step fission source.
        omega = float(np.sum(lam * dt * beta / (1.0 + lam * dt)))
        fis_w = (1.0 - kin.beta_total) + omega
        inv_vdt = [1.0 / (kin.velocities[g] * dt) for g in range(G)]

        def chi_d(g):
            return (fields.chi[g] if kin.chi_delayed is None
                    else float(kin.chi_delayed[g]))

        ops = [self.group_operator(xp, self.grid, fields.diffusion[g],
                             fields.removal[g] + inv_vdt[g], bc=self.bc,
                             active=self.active, mask_bc=self.mask_bc)
               for g in range(G)]
        last = (mats, mmap)

        times = [0.0]
        power = [1.0]
        inner_total = 0

        for n in range(1, n_steps + 1):
            t = n * dt
            mats, mmap = self.problem_at(t)
            if mats is not last[0] or mmap is not last[1]:
                fields = Fields(xp, self.grid, mats, mmap, self.dtype)
                ops = [self.group_operator(xp, self.grid, fields.diffusion[g],
                                     fields.removal[g] + inv_vdt[g], bc=self.bc,
                                     active=self.active, mask_bc=self.mask_bc)
                       for g in range(G)]
                last = (mats, mmap)

            phi_old = [p.copy() for p in phi]
            # Decayed precursor source, constant within the step.
            dsrc = (lam[0] / (1.0 + lam[0] * dt)) * C[0]
            for i in range(1, kin.n_families):
                dsrc += (lam[i] / (1.0 + lam[i] * dt)) * C[i]

            # Fixed point on the end-of-step fission source (Gauss-Seidel over
            # groups, warm-started from the previous step). Near criticality
            # the plain iteration contracts like the prompt multiplication
            # factor (arbitrarily close to 1), so it is Anderson-accelerated:
            # the next iterate is the residual-minimizing affine combination
            # of the last few sweeps, which collapses the handful of slow
            # error modes (one per perturbed region) in a few sweeps.
            change = 1.0
            hist: list = []  # (S_j, G(S_j)) pairs, oldest first
            for sweep in range(1, max_sweeps + 1):
                # Solve well below both the current sweep change and the step
                # tolerance, so CG noise never becomes the fixed point's floor.
                rtol = min(1e-6, max(1e-3 * change, 1e-3 * tol_step, 1e-12))
                for g in range(G):
                    q = inv_vdt[g] * phi_old[g] + (fis_w * fields.chi[g]) * S \
                        + chi_d(g) * dsrc
                    for gf in range(G):
                        s = fields.sigma_s[gf][g]
                        if gf != g and s is not None:
                            q += s * phi[gf]
                    phi[g], n_it = pcg(ops[g].apply, q, phi[g],
                                       ops[g].inv_diag, xp, rtol=rtol)
                    inner_total += n_it
                G_S = fields.fission_source(phi) / k0
                delta = G_S - S
                change = float(xp.sqrt(xp.sum(delta * delta) / xp.sum(G_S**2)))
                if change < tol_step:
                    S = G_S
                    break
                hist.append((S, G_S))
                hist = hist[-anderson_depth:]
                S = G_S
                if len(hist) >= 2:
                    # min || F_last + sum_j gamma_j (F_j - F_last) ||_2 over the
                    # residuals F_j = G_j - S_j (small dense normal equations).
                    F = [Gj - Sj for Sj, Gj in hist]
                    dF = [Fj - F[-1] for Fj in F[:-1]]
                    m = len(dF)
                    A = np.array([[float(xp.sum(dF[i] * dF[j])) for j in range(m)]
                                  for i in range(m)])
                    b = np.array([-float(xp.sum(dF[i] * F[-1])) for i in range(m)])
                    A[np.diag_indices(m)] += 1e-12 * (np.trace(A) + 1e-300)
                    try:
                        gamma = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        gamma = None
                    if gamma is not None and np.all(np.abs(gamma) < 1e4):
                        for j in range(m):
                            S = S + float(gamma[j]) * (hist[j][1] - hist[-1][1])
            else:
                raise RuntimeError(
                    f"time step at t={t:g} s did not converge "
                    f"({max_sweeps} sweeps, source change {change:.2e})")

            for i in range(kin.n_families):
                C[i] = (C[i] + (dt * beta[i]) * S) / (1.0 + lam[i] * dt)

            times.append(t)
            power.append(float(xp.sum(S)))
            if verbose and (n % max(1, n_steps // 20) == 0 or n == n_steps):
                print(f"  t = {t:8.4f} s   P/P0 = {power[-1]:.5f}   ({sweep} sweeps)")

        synchronize(xp)
        return TransientResult(
            times=np.array(times),
            power=np.array(power),
            k0=k0,
            steady=steady,
            flux=xp.stack(phi),
            precursors=xp.stack(C),
            total_inner_iterations=inner_total,
            solve_seconds=time.perf_counter() - t0,
            device=self.device,
        )
