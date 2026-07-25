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
from .linalg import get_linear_solver, neumann_preconditioner
from .materials import Kinetics
from .sn import SNTransportSolver
from .sp3 import SP3GroupOperator
from .spn import SDPNGroupOperator, _SPN_C
from .stencil import BC_VACUUM, BC_ZERO_FLUX, GroupOperator
from .solver import (DiffusionEigenSolver, Fields, Result, SDP1EigenSolver,
                     SDPNEigenSolver)


# Anderson safeguard: a sweep residual growing past this factor means the
# stored history no longer describes the fixed-point map (see the restart
# comment in TransientSolver.solve); the history is then dropped. 1.5 tolerates
# the mild non-monotonicity of healthy Anderson steps (1.0 restarts so often it
# degenerates to Picard and stalls on C5G7's upscatter; >= 5 lets divergent
# oscillations run).
_ANDERSON_RESTART_GROWTH = 1.5


def _require_global_kinetics(kin: Kinetics, who: str):
    """Per-material kinetics (2D velocities/beta) and per-family chi_delayed
    are currently implemented only by the diffusion TransientSolver."""
    if kin.per_material or (kin.chi_delayed is not None
                            and kin.chi_delayed.ndim == 2):
        raise NotImplementedError(
            f"{who} supports only global kinetics data; per-material "
            f"velocities/beta and per-family chi_delayed need TransientSolver "
            f"(diffusion)")


def _anderson_mix(hist, S, xp):
    """One Anderson update of the fixed-point iterate S from the history of
    (S_j, G(S_j)) pairs (oldest first; the current sweep is hist[-1]): the
    residual-minimizing affine combination of the stored sweeps. Returns S
    unchanged when the small dense system is singular or the coefficients
    blow up (falls back to plain Picard)."""
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
        return S
    if not np.all(np.abs(gamma) < 1e4):
        return S
    for j in range(m):
        S = S + float(gamma[j]) * (hist[j][1] - hist[-1][1])
    return S


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
    # Fixed-point (fission source) iterations per time step -- the transient's
    # outer convergence telemetry, per the benchmark protocol.
    step_iterations: list = field(default_factory=list)

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
    kinetics   : Kinetics (velocities, delayed families). Per-material tables
                 (2D velocities/beta, rows indexing the problem_at materials
                 list) and per-family chi_delayed (I, G) are supported: the
                 solver maps them onto the grid through the material map
                 (mixed cells blend linearly) and the arithmetic proceeds
                 elementwise.
    mix_material, mix_weight : optional per-cell two-material blend on top of
                 the integer material map, as for DiffusionEigenSolver. Static
                 in time -- time dependence belongs in the *values* of the
                 materials returned by problem_at, not in the maps.
    bc, device, dtype, linear_solver, symmetric_operator : as for
                 DiffusionEigenSolver (symmetric_operator=False needs a
                 non-symmetric linear_solver and the default group_operator).
    """

    def __init__(self, grid: Grid, problem_at, kinetics: Kinetics,
                 bc=BC_ZERO_FLUX, device: str = "auto", dtype=np.float64,
                 active=None, mask_bc=BC_VACUUM,
                 group_operator=GroupOperator, eig_solver=DiffusionEigenSolver,
                 precond_degree: int = 0, linear_solver="cg",
                 symmetric_operator: bool = True,
                 mix_material=None, mix_weight=None):
        self.grid = grid
        self.problem_at = problem_at
        self.kinetics = kinetics
        self.bc = bc
        self.active = active
        self.mask_bc = mask_bc
        self.mix_material = mix_material
        self.mix_weight = mix_weight
        # Geometry is pluggable: the Cartesian (GroupOperator/DiffusionEigenSolver)
        # or hex (HexGroupOperator/HexDiffusionEigenSolver) pair share signatures.
        self.group_operator = group_operator
        self.eig_solver = eig_solver
        self.precond_degree = int(precond_degree)
        self.linear_solver = linear_solver
        self.symmetric_operator = bool(symmetric_operator)
        # Only the structured GroupOperator knows the kwarg; pluggable
        # geometries (hex, ...) never see it in the default (True) case.
        self._op_kwargs = {} if self.symmetric_operator else {"symmetric": False}
        self._linsolve = get_linear_solver(linear_solver)
        self.xp = get_backend(device)
        self.device = device_name(self.xp)
        self.dtype = np.dtype(dtype)

    def solve(self, t_end: float, dt: float, tol_step: float = 1e-6,
              max_sweeps: int = 200, anderson_depth: int = 5,
              scatter_subsweeps: int | None = None,
              steady_kwargs: dict | None = None,
              verbose: bool = False) -> TransientResult:
        """March from the steady state at t=0 to t_end with fixed step dt.

        anderson_depth : number of past fission-source iterates retained by the
            Anderson acceleration of the within-step fixed point (window m+1 for
            m residual differences). Depth 1 disables it (plain Picard).
        scatter_subsweeps : Gauss-Seidel passes over the groups per fixed-point
            evaluation (fission source held fixed). With downscatter only, one
            ordered pass is exact, so extra passes are pointless; with
            *upscatter* a single pass leaves the evaluation dependent on the
            incoming flux iterate, and Anderson -- which assumes the sweeps it
            mixes sample a fixed map G(S) -- loses most of its acceleration
            (C5G7's 7-group water: ~450 sweeps instead of ~10). A few passes
            restore an (almost) pure G(S). None (default) auto-selects: 3 when
            any upscatter coupling exists, else 1.
        """
        xp, kin = self.xp, self.kinetics
        beta, lam = kin.beta, kin.decay
        n_steps = int(round(t_end / dt))
        synchronize(xp)
        t0 = time.perf_counter()

        # --- initial condition: steady state, critically adjusted ------------
        mats, mmap = self.problem_at(0.0)
        # Mix arrays only reach solvers that were given them: pluggable
        # geometries (hex, ...) never see the kwargs in the default case.
        mix_kwargs = ({} if self.mix_material is None else
                      dict(mix_material=self.mix_material,
                           mix_weight=self.mix_weight))
        eig = self.eig_solver(self.grid, mats, mmap, bc=self.bc,
                              device="cpu" if xp is np else "gpu",
                              dtype=self.dtype,
                              active=self.active, mask_bc=self.mask_bc,
                              linear_solver=self.linear_solver,
                              symmetric_operator=self.symmetric_operator,
                              **mix_kwargs)
        G = eig.n_groups
        if kin.velocities.shape[-1] != G:
            raise ValueError("kinetics.velocities must have one entry per group")
        n_mats = len(mats) if isinstance(mats, (list, tuple)) else 1
        for name, table in (("velocities", kin.velocities), ("beta", kin.beta)):
            if table.ndim == 2 and len(table) != n_mats:
                raise ValueError(f"per-material kinetics.{name} must have one "
                                 f"row per entry of the materials list")
        steady = eig.solve(**(steady_kwargs or dict(tol_k=1e-8, tol_source=1e-7)))
        if not steady.converged:
            raise RuntimeError(f"initial steady state did not converge: {steady}")
        k0 = steady.k_eff

        # Cylindrical grids: the power integral is always the metric-weighted
        # sum (physics, independent of the operator form), while the source
        # weight fed to the operators follows the operator itself -- the
        # volume-weighted SPD form wants a weighted source (.rhs_weight),
        # the divergence form and Cartesian grids take the source as-is.
        met = getattr(self.grid, "cylindrical_metrics", lambda: None)()
        vol_w = None if met is None else xp.asarray(met[0], dtype=self.dtype)

        def total_power(src):
            return float(xp.sum(src if vol_w is None else src * vol_w))

        fields = eig.fields

        # Kinetics tables become scalars (global data) or per-cell device
        # fields (per-material rows looked up through the material map, mixed
        # cells blended linearly); either shape flows through the same
        # elementwise arithmetic below. The maps are static in time, so the
        # fields are built once, from the t=0 Fields instance.
        if kin.beta.ndim == 2:
            # beta rides on the fission source: mixed cells weight each
            # component by its fission-production share (a fuel/moderator rim
            # cell keeps the fuel's beta), exactly like the chi blend.
            beta = [fields.map_table_fission_weighted(kin.beta[:, i])
                    for i in range(kin.n_families)]
        else:
            beta = [float(b) for b in beta]
        if kin.velocities.ndim == 2:
            inv_vdt = [fields.map_table(1.0 / (kin.velocities[:, g] * dt))
                       for g in range(G)]
        else:
            inv_vdt = [1.0 / (kin.velocities[g] * dt) for g in range(G)]

        phi = [steady.flux[g].copy() for g in range(G)]
        S = fields.fission_source(phi) / k0
        scale = 1.0 / total_power(S)            # P(0) = 1
        for g in range(G):
            phi[g] *= scale
        S = S * scale
        C = [(beta[i] / lam[i]) * S for i in range(kin.n_families)]  # equilibrium

        chi_d2 = kin.chi_delayed if (kin.chi_delayed is not None
                                     and kin.chi_delayed.ndim == 2) else None

        # End-of-step fission-source spectrum weight. The material chi is the
        # *total* (steady) spectrum -- what C5G7-TD calls the cumulative
        # spectrum -- so the prompt spectrum never appears explicitly:
        # (1 - beta) chi_p,g = chi_g - sum_i beta_i chi_d,ig, and substituting
        # the analytic backward-Euler precursor update into the flux equation
        # weights the end-of-step fission source by
        #     w_fis,g = chi_g - sum_i chi_d,ig beta_i / (1 + lam_i dt),
        # which reduces to the familiar chi_g [(1 - beta) + omega] when
        # chi_d = chi. In this form the unperturbed t=0 state is an exact
        # equilibrium of the transient equations for *any* delayed spectrum.
        bcoef = [beta[i] / (1.0 + lam[i] * dt) for i in range(kin.n_families)]
        bcoef_sum = sum(bcoef)

        def fission_weights():
            if chi_d2 is not None:               # one spectrum per family
                ws = []
                for g in range(G):
                    acc = 0.0
                    for i in range(kin.n_families):
                        if chi_d2[i, g] != 0.0:
                            acc = acc + chi_d2[i, g] * bcoef[i]
                    ws.append(fields.chi[g] - acc)
                return ws
            if kin.chi_delayed is not None:      # single global spectrum
                return [fields.chi[g] - float(kin.chi_delayed[g]) * bcoef_sum
                        for g in range(G)]
            return [fields.chi[g] * (1.0 - bcoef_sum) for g in range(G)]

        w_fis = fission_weights()

        def delayed_source_by_group(C):
            """Per-group delayed emission Sum_i chi_d,i,g lam_i C_i/(1+lam_i dt),
            precomputed once per step (C is the start-of-step field)."""
            decayed = [(lam[i] / (1.0 + lam[i] * dt)) * C[i]
                       for i in range(kin.n_families)]
            if chi_d2 is not None:               # one spectrum per family
                out = []
                for g in range(G):
                    dg = 0.0
                    for i in range(kin.n_families):
                        if chi_d2[i, g] != 0.0:
                            dg = dg + chi_d2[i, g] * decayed[i]
                    out.append(dg)
                return out
            dsrc = decayed[0]
            for d in decayed[1:]:
                dsrc = dsrc + d
            if kin.chi_delayed is not None:      # single global spectrum
                return [float(kin.chi_delayed[g]) * dsrc for g in range(G)]
            return [fields.chi[g] * dsrc for g in range(G)]  # material spectrum

        ops = [self.group_operator(xp, self.grid, fields.diffusion[g],
                             fields.removal[g] + inv_vdt[g], bc=self.bc,
                             active=self.active, mask_bc=self.mask_bc,
                             **self._op_kwargs)
               for g in range(G)]
        preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                           self.precond_degree) for op in ops]
        src_w = getattr(ops[0], "rhs_weight", None)
        last = (mats, mmap)

        # sigma_s is indexed [g_from][g_to]: any coupling above the diagonal
        # is upscatter, which one ordered Gauss-Seidel pass cannot resolve.
        has_upscatter = any(fields.sigma_s[gf][gt] is not None
                            for gt in range(G) for gf in range(gt + 1, G))
        n_sub = (int(scatter_subsweeps) if scatter_subsweeps
                 else (3 if has_upscatter else 1))

        times = [0.0]
        power = [1.0]
        inner_total = 0

        for n in range(1, n_steps + 1):
            t = n * dt
            mats, mmap = self.problem_at(t)
            if mats is not last[0] or mmap is not last[1]:
                fields = Fields(xp, self.grid, mats, mmap, self.dtype,
                                mix_material=self.mix_material,
                                mix_weight=self.mix_weight)
                ops = [self.group_operator(xp, self.grid, fields.diffusion[g],
                                     fields.removal[g] + inv_vdt[g], bc=self.bc,
                                     active=self.active, mask_bc=self.mask_bc,
                                     **self._op_kwargs)
                       for g in range(G)]
                preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                                   self.precond_degree)
                            for op in ops]
                src_w = getattr(ops[0], "rhs_weight", None)
                w_fis = fission_weights()
                last = (mats, mmap)

            phi_old = [p.copy() for p in phi]
            # Delayed emission from decayed precursors, constant within the step.
            dsrc_g = delayed_source_by_group(C)

            # Fixed point on the end-of-step fission source (Gauss-Seidel over
            # groups, warm-started from the previous step). Near criticality
            # the plain iteration contracts like the prompt multiplication
            # factor (arbitrarily close to 1), so it is Anderson-accelerated:
            # the next iterate is the residual-minimizing affine combination
            # of the last few sweeps, which collapses the handful of slow
            # error modes (one per perturbed region) in a few sweeps.
            change = 1.0
            change_prev = np.inf
            hist: list = []  # (S_j, G(S_j)) pairs, oldest first
            for sweep in range(1, max_sweeps + 1):
                # Solve well below both the current sweep change and the step
                # tolerance, so CG noise never becomes the fixed point's floor.
                rtol = min(1e-6, max(1e-3 * change, 1e-3 * tol_step, 1e-12))
                for _ in range(n_sub):
                    for g in range(G):
                        q = inv_vdt[g] * phi_old[g] + w_fis[g] * S + dsrc_g[g]
                        for gf in range(G):
                            s = fields.sigma_s[gf][g]
                            if gf != g and s is not None:
                                q += s * phi[gf]
                        if src_w is not None:
                            q = q * src_w
                        phi[g], n_it = self._linsolve(ops[g].apply, q, phi[g],
                                                      ops[g].inv_diag, xp,
                                                      rtol=rtol,
                                                      precond=preconds[g])
                        inner_total += n_it
                G_S = fields.fission_source(phi) / k0
                delta = G_S - S
                change = float(xp.sqrt(xp.sum(delta * delta) / xp.sum(G_S**2)))
                if change < tol_step:
                    S = G_S
                    break
                if change > _ANDERSON_RESTART_GROWTH * change_prev:
                    # A growing residual means the stored sweeps no longer
                    # describe the current fixed-point map: with upscatter
                    # (or a large Anderson jump) one Gauss-Seidel sweep is a
                    # *stateful* function of the flux iterate, not of S alone,
                    # and mixing across inconsistent pairs diverges (seen on
                    # C5G7's 7-group water). Restart from plain Picard.
                    hist = []
                change_prev = change
                hist.append((S, G_S))
                hist = hist[-anderson_depth:]
                S = G_S
                if len(hist) >= 2:
                    S = _anderson_mix(hist, S, xp)
            else:
                raise RuntimeError(
                    f"time step at t={t:g} s did not converge "
                    f"({max_sweeps} sweeps, source change {change:.2e})")

            for i in range(kin.n_families):
                C[i] = (C[i] + (dt * beta[i]) * S) / (1.0 + lam[i] * dt)

            times.append(t)
            power.append(total_power(S))
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


class TransientSDP1Solver:
    """Time-dependent multigroup SDP1 (simplified double-P1) solver.

    The transient counterpart of :class:`~ndgpu.SDP1EigenSolver`: the same
    two-moment (Phi1 = phi0 + 2 phi2, phi2) block as the steady SDP1, marched in
    time with backward Euler and the same delayed-precursor / Anderson machinery
    as :class:`TransientSolver`. The even-moment time derivatives couple through
    the exact (symmetric) time matrix theta * sum_m c^(m) -- see the theta note
    on :class:`~ndgpu.operator.SP3GroupOperator` -- so the transient block stays
    SPD and the within-step solve uses CG, exactly like the steady SDP1.
    Odd-moment time derivatives are neglected (the standard quasi-static
    closure of time-SP3 kinetics). Captures the transport correction to reactor
    kinetics (steeper
    gradients, stronger absorbers) that diffusion kinetics misses, at ~2x the
    per-step cost. Same ``problem_at`` / ``Kinetics`` interface and
    :class:`TransientResult` as the diffusion solver.
    """

    def __init__(self, grid: Grid, problem_at, kinetics: Kinetics,
                 bc=BC_ZERO_FLUX, device: str = "auto", dtype=np.float64,
                 active=None, mask_bc=BC_VACUUM, precond_degree: int = 0):
        _require_global_kinetics(kinetics, "TransientSDP1Solver")
        self.grid = grid
        self.problem_at = problem_at
        self.kinetics = kinetics
        self.bc = bc
        self.active = active
        self.mask_bc = mask_bc
        self.precond_degree = int(precond_degree)
        self._linsolve = get_linear_solver("cg")
        self.xp = get_backend(device)
        self.device = device_name(self.xp)
        self.dtype = np.dtype(dtype)

    @staticmethod
    def _phi0(state_g):
        return state_g[0] - 2.0 * state_g[1]

    def _build_ops(self, xp, fields, inv_vdt, G):
        ops = [SP3GroupOperator(xp, self.grid, fields.diffusion[g],
                                fields.sigma_t[g], fields.removal[g], bc=self.bc,
                                active=self.active, mask_bc=self.mask_bc,
                                variant="sdp1", theta=inv_vdt[g])
               for g in range(G)]
        preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                           self.precond_degree) for op in ops]
        return ops, preconds

    def solve(self, t_end: float, dt: float, tol_step: float = 1e-6,
              max_sweeps: int = 200, anderson_depth: int = 5,
              steady_kwargs: dict | None = None,
              verbose: bool = False) -> TransientResult:
        """March from the steady SDP1 state at t=0 to t_end with fixed step dt
        (see :meth:`TransientSolver.solve` for the shared parameters)."""
        xp, kin = self.xp, self.kinetics
        beta, lam = kin.beta, kin.decay
        n_steps = int(round(t_end / dt))
        synchronize(xp)
        t0 = time.perf_counter()

        mats, mmap = self.problem_at(0.0)
        eig = SDP1EigenSolver(self.grid, mats, mmap, bc=self.bc,
                              device="cpu" if xp is np else "gpu",
                              dtype=self.dtype, active=self.active,
                              mask_bc=self.mask_bc)
        G = eig.n_groups
        if len(kin.velocities) != G:
            raise ValueError("kinetics.velocities must have one entry per group")
        steady = eig.solve(**(steady_kwargs or dict(tol_k=1e-8, tol_source=1e-7)))
        if not steady.converged:
            raise RuntimeError(f"initial steady state did not converge: {steady}")
        k0 = steady.k_eff

        met = getattr(self.grid, "cylindrical_metrics", lambda: None)()
        vol_w = None if met is None else xp.asarray(met[0], dtype=self.dtype)

        def total_power(src):
            return float(xp.sum(src if vol_w is None else src * vol_w))

        fields = eig.fields
        state = [s.copy() for s in eig.state]          # (2, *grid) per group
        phi0 = [self._phi0(state[g]) for g in range(G)]
        S = fields.fission_source(phi0) / k0
        scale = 1.0 / total_power(S)                    # P(0) = 1
        for g in range(G):
            state[g] *= scale
        S = S * scale
        C = [(beta[i] / lam[i]) * S for i in range(kin.n_families)]

        omega = float(np.sum(lam * dt * beta / (1.0 + lam * dt)))
        fis_w = (1.0 - kin.beta_total) + omega
        inv_vdt = [1.0 / (kin.velocities[g] * dt) for g in range(G)]

        def chi_d(g):
            return (fields.chi[g] if kin.chi_delayed is None
                    else float(kin.chi_delayed[g]))

        ops, preconds = self._build_ops(xp, fields, inv_vdt, G)
        src_w = getattr(ops[0], "rhs_weight", None)
        last = (mats, mmap)

        times = [0.0]
        power = [1.0]
        inner_total = 0

        for n in range(1, n_steps + 1):
            t = n * dt
            mats, mmap = self.problem_at(t)
            if mats is not last[0] or mmap is not last[1]:
                fields = Fields(xp, self.grid, mats, mmap, self.dtype)
                ops, preconds = self._build_ops(xp, fields, inv_vdt, G)
                src_w = getattr(ops[0], "rhs_weight", None)
                last = (mats, mmap)

            state_old = [s.copy() for s in state]
            phi0_old = [self._phi0(s) for s in state_old]
            dsrc = (lam[0] / (1.0 + lam[0] * dt)) * C[0]
            for i in range(1, kin.n_families):
                dsrc += (lam[i] / (1.0 + lam[i] * dt)) * C[i]

            change = 1.0
            hist: list = []
            for sweep in range(1, max_sweeps + 1):
                rtol = min(1e-6, max(1e-3 * change, 1e-3 * tol_step, 1e-12))
                phi0 = [self._phi0(state[g]) for g in range(G)]
                for g in range(G):
                    # Isotropic external source into phi0 (fission + delayed +
                    # in-scatter); the time term is carried by the block below.
                    q = (fis_w * fields.chi[g]) * S + chi_d(g) * dsrc
                    for gf in range(G):
                        s = fields.sigma_s[gf][g]
                        if gf != g and s is not None:
                            q += s * phi0[gf]
                    if src_w is not None:
                        q = q * src_w
                    theta = inv_vdt[g]
                    rhs = xp.empty_like(state[g])
                    # Backward-Euler time sources mirror the operator's exact
                    # time terms: theta*phi0 on row 0 and, on the 5x-scaled
                    # row 1, theta*(9 phi2 - 2 Phi1) = theta*(5 phi2 - 2 phi0).
                    rhs[0] = q + theta * phi0_old[g]
                    rhs[1] = -2.0 * q + theta * (5.0 * state_old[g][1]
                                                 - 2.0 * phi0_old[g])
                    state[g], n_it = self._linsolve(
                        ops[g].apply, rhs, state[g], ops[g].inv_diag, xp,
                        rtol=rtol, precond=preconds[g])
                    inner_total += n_it
                    phi0[g] = self._phi0(state[g])
                G_S = fields.fission_source(phi0) / k0
                delta = G_S - S
                change = float(xp.sqrt(xp.sum(delta * delta) / xp.sum(G_S**2)))
                if change < tol_step:
                    S = G_S
                    break
                hist.append((S, G_S))
                hist = hist[-anderson_depth:]
                S = G_S
                if len(hist) >= 2:
                    S = _anderson_mix(hist, S, xp)
            else:
                raise RuntimeError(
                    f"time step at t={t:g} s did not converge "
                    f"({max_sweeps} sweeps, source change {change:.2e})")

            for i in range(kin.n_families):
                C[i] = (C[i] + (dt * beta[i]) * S) / (1.0 + lam[i] * dt)

            times.append(t)
            power.append(total_power(S))
            if verbose and (n % max(1, n_steps // 20) == 0 or n == n_steps):
                print(f"  t = {t:8.4f} s   P/P0 = {power[-1]:.5f}   ({sweep} sweeps)")

        synchronize(xp)
        return TransientResult(
            times=np.array(times), power=np.array(power), k0=k0, steady=steady,
            flux=xp.stack([self._phi0(state[g]) for g in range(G)]),
            precursors=xp.stack(C), total_inner_iterations=inner_total,
            solve_seconds=time.perf_counter() - t0, device=self.device,
        )


class TransientSDPNSolver:
    """Time-dependent multigroup SDPN solver in the diffusive U-form, order N
    in {1, 2, 3} (M = N+1 moments) -- the transient counterpart of
    :class:`~ndgpu.SDPNEigenSolver` and its SDP2/SDP3 subclasses.

    Backward Euler with the exact even-moment time coupling: in the U-form a
    time derivative transforms exactly like a cross section that is equal in
    every even moment, so the time matrix is theta * sum_m c^(m) (theta =
    1/(v dt); ``SDPNGroupOperator.time_weights``), and the step source adds
    the matching theta * time_weights . U_old per moment row. Odd-moment time
    derivatives are neglected -- the standard quasi-static closure of
    time-SPN kinetics, the same approximation as :class:`TransientSDP1Solver`
    (which order 1 reproduces through the dedicated two-moment block). Same
    ``problem_at`` / ``Kinetics`` interface, delayed-precursor treatment and
    :class:`TransientResult` as :class:`TransientSolver`; the within-step
    block is non-symmetric (double-PN closure), so it is solved with CG in a
    symmetrizing basis when the tables and boundary conditions admit one and
    with BiCGStab otherwise (same auto-selection as the steady solver).
    """

    _coeffs = None                 # None -> SDPN tables (operator default)

    def __init__(self, grid: Grid, problem_at, kinetics: Kinetics,
                 order: int = 3, bc=BC_ZERO_FLUX, device: str = "auto",
                 dtype=np.float64, active=None, mask_bc=BC_VACUUM,
                 precond_degree: int = 0, symmetrize: bool | None = None):
        _require_global_kinetics(kinetics, "TransientSDPNSolver")
        self.grid = grid
        self.problem_at = problem_at
        self.kinetics = kinetics
        self.order = int(order)
        self.bc = bc
        self.active = active
        self.mask_bc = mask_bc
        self.precond_degree = int(precond_degree)
        # Symmetrize like the steady solver (the exact time matrix
        # theta * sum_m c^(m) transforms along with A, so the transient block
        # is symmetric exactly when the steady one is) and run CG; otherwise
        # BiCGStab. Auto = the diagonal similarity when the tables admit one
        # (SDP1/SDP2; SPN tables are symmetric as-is); on order 3 (no
        # similarity) auto falls back to the congruence basis whenever the
        # boundary conditions allow it (reflective/zero-flux faces only, see
        # CongruentSDPNOperator).
        from .spn import _SDPN_C, _congruence_available, _diag_similarity
        tabs = (self._coeffs if self._coeffs is not None else _SDPN_C)[self.order]
        r = _diag_similarity(tabs)
        identity = r is not None and all(abs(x - 1.0) < 1e-12 for x in r)
        if symmetrize is None:
            if r is not None:
                symmetrize = not identity
            else:
                symmetrize = _congruence_available(self.order, grid, bc,
                                                   active, mask_bc,
                                                   coeffs=self._coeffs)
        self._symmetrize = bool(symmetrize)
        self._diag_sym = r is not None
        symmetric = identity or self._symmetrize
        self._linsolve = get_linear_solver("cg" if symmetric else "bicgstab")
        self.xp = get_backend(device)
        self.device = device_name(self.xp)
        self.dtype = np.dtype(dtype)

    def _build_ops(self, xp, fields, inv_vdt, G):
        if self._symmetrize and not self._diag_sym:
            from .spn import CongruentSDPNOperator
            ops = [CongruentSDPNOperator(xp, self.grid, fields.diffusion[g],
                                         fields.sigma_t[g], fields.removal[g],
                                         order=self.order, bc=self.bc,
                                         active=self.active,
                                         mask_bc=self.mask_bc,
                                         coeffs=self._coeffs,
                                         theta=inv_vdt[g])
                   for g in range(G)]
        else:
            ops = [SDPNGroupOperator(xp, self.grid, fields.diffusion[g],
                                     fields.sigma_t[g], fields.removal[g],
                                     order=self.order, bc=self.bc,
                                     active=self.active, mask_bc=self.mask_bc,
                                     coeffs=self._coeffs, theta=inv_vdt[g],
                                     symmetrize=self._symmetrize)
               for g in range(G)]
        preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                           self.precond_degree) for op in ops]
        return ops, preconds

    def solve(self, t_end: float, dt: float, tol_step: float = 1e-6,
              max_sweeps: int = 200, anderson_depth: int = 5,
              steady_kwargs: dict | None = None,
              verbose: bool = False) -> TransientResult:
        """March from the steady SDPN state at t=0 to t_end with fixed step dt
        (see :meth:`TransientSolver.solve` for the shared parameters)."""
        xp, kin = self.xp, self.kinetics
        beta, lam = kin.beta, kin.decay
        n_steps = int(round(t_end / dt))
        synchronize(xp)
        t0 = time.perf_counter()

        mats, mmap = self.problem_at(0.0)
        eig_cls = type(f"_SDPN{self.order}Eig", (SDPNEigenSolver,),
                       {"_order": self.order, "_coeffs": self._coeffs})
        # The steady state seeds the marching state, so it must live in the
        # same (possibly symmetrized) moment basis as the transient operators.
        eig = eig_cls(self.grid, mats, mmap, bc=self.bc,
                      device="cpu" if xp is np else "gpu", dtype=self.dtype,
                      active=self.active, mask_bc=self.mask_bc,
                      symmetrize=self._symmetrize)
        G = eig.n_groups
        M = self.order + 1
        if len(kin.velocities) != G:
            raise ValueError("kinetics.velocities must have one entry per group")
        steady = eig.solve(**(steady_kwargs or dict(tol_k=1e-8, tol_source=1e-7)))
        if not steady.converged:
            raise RuntimeError(f"initial steady state did not converge: {steady}")
        k0 = steady.k_eff

        met = getattr(self.grid, "cylindrical_metrics", lambda: None)()
        vol_w = None if met is None else xp.asarray(met[0], dtype=self.dtype)

        def total_power(src):
            return float(xp.sum(src if vol_w is None else src * vol_w))

        fields = eig.fields
        w_phi = eig.ops[0].phi0_weights          # phi0 = w_phi . U
        w_src = eig.ops[0].src_weights           # isotropic source row weights
        tw = eig.ops[0].time_weights             # exact time matrix / theta

        def phi0_of(state_g):
            p = w_phi[0] * state_g[0]
            for j in range(1, M):
                if w_phi[j] != 0.0:
                    p = p + w_phi[j] * state_g[j]
            return p

        state = [s.copy() for s in eig.state]    # (M, *grid) per group
        phi0 = [phi0_of(state[g]) for g in range(G)]
        S = fields.fission_source(phi0) / k0
        scale = 1.0 / total_power(S)             # P(0) = 1
        for g in range(G):
            state[g] *= scale
        S = S * scale
        C = [(beta[i] / lam[i]) * S for i in range(kin.n_families)]

        omega = float(np.sum(lam * dt * beta / (1.0 + lam * dt)))
        fis_w = (1.0 - kin.beta_total) + omega
        inv_vdt = [1.0 / (kin.velocities[g] * dt) for g in range(G)]

        def chi_d(g):
            return (fields.chi[g] if kin.chi_delayed is None
                    else float(kin.chi_delayed[g]))

        ops, preconds = self._build_ops(xp, fields, inv_vdt, G)
        src_w = getattr(ops[0], "rhs_weight", None)
        last = (mats, mmap)

        times = [0.0]
        power = [1.0]
        inner_total = 0

        for n in range(1, n_steps + 1):
            t = n * dt
            mats, mmap = self.problem_at(t)
            if mats is not last[0] or mmap is not last[1]:
                fields = Fields(xp, self.grid, mats, mmap, self.dtype)
                ops, preconds = self._build_ops(xp, fields, inv_vdt, G)
                src_w = getattr(ops[0], "rhs_weight", None)
                last = (mats, mmap)

            state_old = [s.copy() for s in state]
            dsrc = (lam[0] / (1.0 + lam[0] * dt)) * C[0]
            for i in range(1, kin.n_families):
                dsrc += (lam[i] / (1.0 + lam[i] * dt)) * C[i]

            change = 1.0
            hist: list = []
            for sweep in range(1, max_sweeps + 1):
                rtol = min(1e-6, max(1e-3 * change, 1e-3 * tol_step, 1e-12))
                phi0 = [phi0_of(state[g]) for g in range(G)]
                for g in range(G):
                    # Isotropic external source into phi0 (fission + delayed +
                    # in-scatter), distributed over the moment rows by the
                    # first column of c^(1); each row also carries its exact
                    # backward-Euler time source theta * time_weights . U_old.
                    q = (fis_w * fields.chi[g]) * S + chi_d(g) * dsrc
                    for gf in range(G):
                        s = fields.sigma_s[gf][g]
                        if gf != g and s is not None:
                            q += s * phi0[gf]
                    if src_w is not None:
                        q = q * src_w
                    theta = inv_vdt[g]
                    rhs = xp.empty_like(state[g])
                    for i in range(M):
                        rhs[i] = w_src[i] * q
                        for j in range(M):
                            if tw[i][j] != 0.0:
                                rhs[i] += (theta * tw[i][j]) * state_old[g][j]
                    state[g], n_it = self._linsolve(
                        ops[g].apply, rhs, state[g], ops[g].inv_diag, xp,
                        rtol=rtol, precond=preconds[g])
                    inner_total += n_it
                    phi0[g] = phi0_of(state[g])
                G_S = fields.fission_source(phi0) / k0
                delta = G_S - S
                change = float(xp.sqrt(xp.sum(delta * delta) / xp.sum(G_S**2)))
                if change < tol_step:
                    S = G_S
                    break
                hist.append((S, G_S))
                hist = hist[-anderson_depth:]
                S = G_S
                if len(hist) >= 2:
                    S = _anderson_mix(hist, S, xp)
            else:
                raise RuntimeError(
                    f"time step at t={t:g} s did not converge "
                    f"({max_sweeps} sweeps, source change {change:.2e})")

            for i in range(kin.n_families):
                C[i] = (C[i] + (dt * beta[i]) * S) / (1.0 + lam[i] * dt)

            times.append(t)
            power.append(total_power(S))
            if verbose and (n % max(1, n_steps // 20) == 0 or n == n_steps):
                print(f"  t = {t:8.4f} s   P/P0 = {power[-1]:.5f}   ({sweep} sweeps)")

        synchronize(xp)
        return TransientResult(
            times=np.array(times), power=np.array(power), k0=k0, steady=steady,
            flux=xp.stack([phi0_of(state[g]) for g in range(G)]),
            precursors=xp.stack(C), total_inner_iterations=inner_total,
            solve_seconds=time.perf_counter() - t0, device=self.device,
        )


class TransientSDP3Solver(TransientSDPNSolver):
    """Time-dependent multigroup SDP3 solver (4-moment U-form block) -- the
    transient counterpart of :class:`~ndgpu.SDP3EigenSolver`. See
    :class:`TransientSDPNSolver`."""

    def __init__(self, grid: Grid, problem_at, kinetics: Kinetics, **kwargs):
        kwargs.pop("order", None)
        super().__init__(grid, problem_at, kinetics, order=3, **kwargs)


class TransientSPNSolver(TransientSDPNSolver):
    """Time-dependent multigroup standard-SPN solver: order 0/1/2/3 =
    SP1/SP3/SP5/SP7 kinetics. Identical machinery to
    :class:`TransientSDPNSolver` with the standard SPN coefficient tables
    (both A and the time matrix theta * sum_m c^(m) are then symmetric).
    Order 0 is P1/diffusion kinetics run through the block machinery --
    equivalent to :class:`TransientSolver` (tested)."""

    _coeffs = _SPN_C


class TransientSNSolver:
    """Time-dependent multigroup discrete-ordinates (S_N) transport.

    The transport counterpart of :class:`TransientSolver`, driving any S_N engine
    -- Cartesian, triangles, extruded prisms -- through one shared time loop.
    Backward Euler on the angular flux: the transport time term (1/v) dpsi/dt adds
    theta = 1/(v_g dt) to the total cross section (the sweep's collision diagonal)
    and contributes a *per-ordinate* source theta * psi_m^old. So -- unlike
    diffusion, which only shifts the removal term and carries a scalar
    (1/(v dt)) phi_old -- the full *angular* flux is retained between steps (see
    the ``t_*`` engine adapter on :class:`~ndgpu.sn.SNTransportSolver` and
    :class:`~ndgpu.tri_sn.TriSNTransportSolver`). Everything else is shared with
    :class:`TransientSolver`: the analytic backward-Euler precursor update, the
    critical adjustment by k0, the Anderson fixed point on the fission source and
    the :class:`TransientResult` contract. Only the per-group solve changes -- a
    fixed-source transport sweep instead of a diffusion CG solve.

    Parameters
    ----------
    grid, problem_at, kinetics : as for :class:`TransientSolver`. Only global
        kinetics data is supported (per-material velocities/beta and per-family
        chi_delayed remain diffusion-only).
    engine_cls : the steady S_N solver class used as the transport engine --
        :class:`~ndgpu.sn.SNTransportSolver` (2D Cartesian) or
        :class:`~ndgpu.tri_sn.TriSNTransportSolver` (triangles and extruded
        prisms, ``engine="lu"``, step or SCB). ``engine_kwargs`` are forwarded to
        it (n_polar, n_azi, scheme, acceleration, sweep, ...). The driver talks to
        an engine only through its ``t_*`` adapter methods, so it never interprets
        the angular flux -- which differs per geometry and scheme (per cell for
        Cartesian and tri step, per corner sub-volume for tri SCB).
    bc : the engine's boundary law -- "vacuum"/"reflective" for Cartesian,
        "vacuum"/"periodic" for tri. The reflective boundary fixed point is
        Anderson-accelerated and warm-started across sweeps and time steps.
    device : "cpu" (default) or "gpu"/"cuda" -- the transient runs on the
        vectorized wavefront sweep, so the whole per-step transport solve is a
        fixed sequence of batched numpy/cupy kernels.
    step_acceleration : "auto" (default; "cmfd" whenever the engine accumulates
        face currents, else "none"), "cmfd" or "none" -- drift-corrected
        diffusion acceleration of the within-step fission fixed point. The
        theta = 1/(v dt) shift already damps that fixed point (the smaller the
        step, the stronger), so CMFD helps most at *large* steps; measured on a
        40x40 bare core (fixed-point iterations per step, wall time):
        dt = 5e-2 s 12.6 -> 3.2 its, 3.1x faster; 1e-2 s 13.7 -> 3.6, 2.4x;
        1e-4 s 6.0 -> 4.0, 1.3x. Power traces agree to ~1e-6 (the fixed point is
        the same; only its convergence rate changes).
    """

    STEP_ACCELERATIONS = ("auto", "none", "cmfd")

    def __init__(self, grid: Grid, problem_at, kinetics: Kinetics,
                 engine_cls=SNTransportSolver, bc: str = "vacuum",
                 device: str = "cpu", dtype=np.float64,
                 step_acceleration: str = "auto", **engine_kwargs):
        _require_global_kinetics(kinetics, "TransientSNSolver")
        if step_acceleration not in self.STEP_ACCELERATIONS:
            raise ValueError(f"step_acceleration must be one of "
                             f"{self.STEP_ACCELERATIONS}")
        self.step_acceleration = step_acceleration
        self.grid = grid
        self.problem_at = problem_at
        self.kinetics = kinetics
        self.engine_cls = engine_cls
        self.bc = bc
        self.device_arg = device
        self.engine_kwargs = engine_kwargs
        self.xp = get_backend(device)
        self.device = device_name(self.xp)
        self.dtype = np.dtype(dtype)

    def _engine(self, mats, mmap, **extra):
        return self.engine_cls(self.grid, mats, mmap, bc=self.bc,
                               device=self.device_arg, **self.engine_kwargs,
                               **extra)

    def _cmfd_step(self, eng, S, phi, ctx):
        """One CMFD acceleration of the within-step fission fixed point.

        Integrating the time-shifted transport equation over angle gives an
        *exact* scalar balance -- the per-ordinate time source theta*psi_old
        integrates to theta*phi_old -- so the coarse problem is precisely the
        drift-corrected *diffusion* backward-Euler step:

            [div(-D grad) + Sigma_t + theta - Sigma_s,gg] phi
                = w_fis,g S + delayed_g + inscatter_g + theta phi_old,g

        with D and the removal carrying the same theta shift and the face
        currents drift-fitted to the transport currents. At the transport fixed
        point the coarse operator reproduces the transport balance, so CMFD
        changes the convergence rate only, never the converged step. One
        current-accumulating transport sweep per group builds it; the coarse
        fixed point on S is then solved with cheap sparse back-solves.

        Returns the accelerated fission source, or None to fall back."""
        xp, G = self.xp, ctx["G"]
        facs = []
        for g in range(G):
            qext = ctx["w_fis"][g] * S + ctx["dsrc"][g]
            for gf in range(G):
                s = ctx["scatter"][gf][g]
                if gf != g and s is not None:
                    qext = qext + s * phi[gf]
            facs.append(eng.t_cmfd_factor(g, ctx["ss"][g] * phi[g] + qext,
                                          ctx["psi_old"][g], ctx["state"][g]))
        # Coarse fixed point on the fission source: identical structure to the
        # transport fixed point but with sparse back-solves instead of sweeps.
        nsf = ctx["nsf_h"]
        ph = [asnumpy(p).ravel().copy() for p in phi]
        Sv = asnumpy(S).ravel().copy()
        hist = []
        for _ in range(200):
            for g in range(G):
                q = (ctx["w_fis_h"][g] * Sv + ctx["dsrc_h"][g]
                     + ctx["theta"][g] * ctx["phi_old_h"][g])
                for gf in range(G):
                    s = ctx["scatter_h"][gf][g]
                    if gf != g and s is not None:
                        q = q + s * ph[gf]
                ph[g] = facs[g](q)
            S_new = sum(nsf[g] * ph[g] for g in range(G)) / ctx["k0"]
            if not np.all(np.isfinite(S_new)) or np.any(S_new < 0.0):
                return None
            d = np.linalg.norm(S_new - Sv) / max(np.linalg.norm(S_new), 1e-300)
            hist.append((Sv, S_new))
            hist = hist[-5:]
            Sv = S_new if len(hist) < 2 else _anderson_mix(hist, S_new, np)
            if d < 1e-10:
                break
        else:
            return None
        return xp.asarray(Sv.reshape(asnumpy(S).shape))

    def solve(self, t_end: float, dt: float, tol_step: float = 1e-6,
              max_sweeps: int = 200, anderson_depth: int = 5,
              steady_kwargs: dict | None = None,
              verbose: bool = False) -> TransientResult:
        """March from the steady S_N state at t=0 to t_end with fixed step dt
        (see :meth:`TransientSolver.solve` for the shared parameters)."""
        xp, kin = self.xp, self.kinetics
        beta = [float(b) for b in kin.beta]
        lam = kin.decay
        n_steps = int(round(t_end / dt))
        synchronize(xp)
        t0 = time.perf_counter()

        # --- initial condition: steady S_N state, critically adjusted --------
        mats, mmap = self.problem_at(0.0)
        eng = self._engine(mats, mmap)
        G = eng.G
        if kin.velocities.shape[-1] != G:
            raise ValueError("kinetics.velocities must have one entry per group")
        # CMFD needs face currents, which only the wavefront sweep accumulates.
        # Resolved before the (expensive) steady solve so misuse fails fast.
        can_cmfd = (getattr(eng, "T_CMFD_STEP", False)
                    and getattr(eng, "sweep_mode", "wavefront") == "wavefront")
        accel = self.step_acceleration
        if accel == "auto":
            accel = "cmfd" if can_cmfd else "none"
        elif accel == "cmfd" and not can_cmfd:
            raise ValueError(
                "step_acceleration='cmfd' needs an engine that accumulates face "
                "currents in its transient sweep (the Cartesian wavefront sweep; "
                "transient CMFD on the tri/prism mesh is Phase 3b)")
        steady = eng.solve(**(steady_kwargs or dict(tol_k=1e-8, tol_source=1e-7)))
        if not steady.converged:
            raise RuntimeError(f"initial steady state did not converge: {steady}")
        k0 = steady.k_eff

        def fld(a):
            return xp.asarray(a)

        def load_fields(eng):
            ss = [fld(eng.ss_self[g]) for g in range(G)]
            nsf = [fld(eng.nsf[g]) for g in range(G)]
            chi = [fld(eng.chi[g]) for g in range(G)]
            scatter = [[None if s is None else fld(s) for s in row]
                       for row in eng.scatter]
            return ss, nsf, chi, scatter

        ss, nsf, chi, scatter = load_fields(eng)

        def fission_source(phi):
            return sum(nsf[g] * phi[g] for g in range(G))

        def total_power(S):
            return float(xp.sum(S))

        # theta_g = 1/(v_g dt) is the backward-Euler collision shift.
        theta = [1.0 / (float(kin.velocities[g]) * dt) for g in range(G)]

        # Steady scalar flux, and its angular flux from one clean sweep of the
        # converged steady source. This psi seeds the very first time source
        # theta*psi_old; the engine owns its layout (per-cell on a Cartesian or
        # step-differenced mesh, per corner sub-volume for tri SCB) -- the driver
        # only carries and scales it.
        phi = [fld(steady.flux[g]) for g in range(G)]
        S = fission_source(phi) / k0
        state = eng.t_state0()
        psi = []
        for g in range(G):
            qext = chi[g] * S
            for gf in range(G):
                s = scatter[gf][g]
                if gf != g and s is not None:
                    qext = qext + s * phi[gf]
            psi_g, state[g] = eng.t_seed_psi(g, ss[g] * phi[g] + qext, state[g])
            psi.append(psi_g)

        # Marching engine. Geometries that prefactor their transport operator
        # (tri/prism) need Sigma_t + theta baked in at construction; the Cartesian
        # sweep takes the shifted cross section per call and reuses this instance.
        if getattr(self.engine_cls, "T_SHIFT_AT_CONSTRUCTION", False):
            eng = self._engine(mats, mmap, sigma_t_shift=theta)
            ss, nsf, chi, scatter = load_fields(eng)
        eng.t_setup(theta)

        scale = 1.0 / total_power(S)                     # P(0) = 1
        for g in range(G):
            phi[g] = phi[g] * scale
            psi[g] = psi[g] * scale
        S = S * scale
        C = [(beta[i] / lam[i]) * S for i in range(kin.n_families)]  # equilibrium

        # End-of-step fission weight and delayed emission -- the same analytic
        # backward-Euler precursor substitution as TransientSolver, global data.
        bcoef = [beta[i] / (1.0 + lam[i] * dt) for i in range(kin.n_families)]
        bcoef_sum = sum(bcoef)

        def fission_weights():
            if kin.chi_delayed is not None:
                return [chi[g] - float(kin.chi_delayed[g]) * bcoef_sum
                        for g in range(G)]
            return [chi[g] * (1.0 - bcoef_sum) for g in range(G)]

        def delayed_source_by_group(C):
            decayed = [(lam[i] / (1.0 + lam[i] * dt)) * C[i]
                       for i in range(kin.n_families)]
            dsum = decayed[0]
            for d in decayed[1:]:
                dsum = dsum + d
            if kin.chi_delayed is not None:
                return [float(kin.chi_delayed[g]) * dsum for g in range(G)]
            return [chi[g] * dsum for g in range(G)]

        w_fis = fission_weights()
        last = (mats, mmap)

        times = [0.0]
        power = [1.0]
        step_its = []
        inner_total = 0
        sweeps0 = eng._sweep_count

        for n in range(1, n_steps + 1):
            t = n * dt
            mats, mmap = self.problem_at(t)
            if mats is not last[0] or mmap is not last[1]:
                inner_total += eng._sweep_count - sweeps0
                shift = (dict(sigma_t_shift=theta)
                         if getattr(self.engine_cls,
                                    "T_SHIFT_AT_CONSTRUCTION", False) else {})
                eng = self._engine(mats, mmap, **shift)
                ss, nsf, chi, scatter = load_fields(eng)
                eng.t_setup(theta)
                w_fis = fission_weights()
                sweeps0 = eng._sweep_count
                last = (mats, mmap)

            # psi_old is fixed within the step; the time source is theta*psi_old
            # (its angular integral, theta*phi_old, is the CMFD-level source).
            psi_old = list(psi)
            phi_old = [p.copy() for p in phi]
            dsrc_g = delayed_source_by_group(C)
            if accel == "cmfd":
                ctx = dict(G=G, k0=k0, theta=theta, ss=ss, scatter=scatter,
                           w_fis=w_fis, dsrc=dsrc_g, psi_old=psi_old,
                           state=state,
                           nsf_h=[asnumpy(nsf[g]).ravel() for g in range(G)],
                           w_fis_h=[asnumpy(w_fis[g]).ravel() for g in range(G)],
                           dsrc_h=[asnumpy(dsrc_g[g]).ravel() for g in range(G)],
                           phi_old_h=[asnumpy(p).ravel() for p in phi_old],
                           scatter_h=[[None if s is None else asnumpy(s).ravel()
                                       for s in row] for row in scatter])

            change = 1.0
            change_prev = np.inf
            hist: list = []
            for sweep in range(1, max_sweeps + 1):
                rtol = min(1e-6, max(1e-3 * change, 1e-3 * tol_step, 1e-12))
                phi_new = [None] * G
                psi_new = [None] * G
                for g in range(G):
                    qext = w_fis[g] * S + dsrc_g[g]
                    for gf in range(G):
                        s = scatter[gf][g]
                        if gf != g and s is not None:
                            src = phi_new[gf] if gf < g else phi[gf]  # Gauss-Seidel
                            qext = qext + s * src
                    p, ps, state[g] = eng.t_solve_group(
                        g, qext, psi_old[g], rtol, phi[g], state[g])
                    phi_new[g] = fld(p)
                    psi_new[g] = ps
                G_S = fission_source(phi_new) / k0
                delta = G_S - S
                change = float(xp.sqrt(xp.sum(delta * delta) / xp.sum(G_S**2)))
                phi, psi = phi_new, psi_new
                if change < tol_step:
                    S = G_S
                    break
                if accel == "cmfd":
                    # The coarse solve collapses the slow fission modes; on
                    # fallback (non-finite/negative flux or a stalled coarse
                    # fixed point) the plain Anderson iterate is used instead.
                    S_c = self._cmfd_step(eng, G_S, phi, ctx)
                    if S_c is not None:
                        S = S_c
                        hist = []      # stale pairs: the map just changed
                        change_prev = change
                        continue
                if change > _ANDERSON_RESTART_GROWTH * change_prev:
                    hist = []
                change_prev = change
                hist.append((S, G_S))
                hist = hist[-anderson_depth:]
                S = G_S
                if len(hist) >= 2:
                    S = _anderson_mix(hist, S, xp)
            else:
                raise RuntimeError(
                    f"time step at t={t:g} s did not converge "
                    f"({max_sweeps} sweeps, source change {change:.2e})")

            for i in range(kin.n_families):
                C[i] = (C[i] + (dt * beta[i]) * S) / (1.0 + lam[i] * dt)

            times.append(t)
            power.append(total_power(S))
            step_its.append(sweep)
            if verbose and (n % max(1, n_steps // 20) == 0 or n == n_steps):
                print(f"  t = {t:8.4f} s   P/P0 = {power[-1]:.5f}   ({sweep} sweeps)")

        inner_total += eng._sweep_count - sweeps0
        synchronize(xp)
        return TransientResult(
            times=np.array(times), power=np.array(power), k0=k0, steady=steady,
            flux=xp.stack(phi), precursors=xp.stack(C),
            total_inner_iterations=inner_total,
            solve_seconds=time.perf_counter() - t0, device=self.device,
            step_iterations=step_its,
        )
