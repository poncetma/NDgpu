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

import copy
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np

from .backend import asnumpy, device_name, get_backend, synchronize
from .grid import Grid
from .linalg import get_linear_solver, neumann_preconditioner
from .materials import Kinetics
from .sn import SNTransportSolver
from .sp3 import SP3GroupOperator
from .spn import SDPNGroupOperator, _SPN_C
from .timescheme import make_time_scheme
from .stencil import BC_VACUUM, BC_ZERO_FLUX, GroupOperator
from .solver import (DiffusionEigenSolver, Fields, Result, SDP1EigenSolver,
                     SDPNEigenSolver, scatter_stack)
from . import kernels


# Anderson safeguard: a sweep residual growing past this factor means the
# stored history no longer describes the fixed-point map (see the restart
# comment in TransientSolver.solve); the history is then dropped. 1.5 tolerates
# the mild non-monotonicity of healthy Anderson steps (1.0 restarts so often it
# degenerates to Picard and stalls on C5G7's upscatter; >= 5 lets divergent
# oscillations run).
_ANDERSON_RESTART_GROWTH = 1.5

# Gauss-Seidel passes over the energy groups per fixed-point evaluation, when
# the library has upscatter. Raised from 3 to 6 on 2026-08-06 after tolerance
# ladders on BOTH upscattering benchmarks showed 3 to be simultaneously slower
# and less accurate:
#
#   11-group HP-MR, one step, refine 2/3: 3 -> 6 subsweeps takes 248/247 outer
#     sweeps to 27/26 and ~7x fewer CG iterations. Subsweeps 3, 6 and 12 all
#     converge to the SAME power (1.3990072 / 1.3990070 / 1.3990067 at
#     tol_step 1e-9), and against that the production-tolerance errors are
#     2.5e-4 / 7.1e-5 / 1.8e-4 -- so 6 is both the cheapest (11,449 CG to
#     converge, against 43,371 at 3 and 26,387 at 12) and the most accurate.
#     The plateau has a top edge as well as a bottom one.
#   7-group C5G7-TD1-1: 3 -> 8 takes 399 sweeps to 162, 1.9x fewer CG, error
#     1.24e-4 -> 6.4e-5. Both counts converge to the SAME power (0.7814288 and
#     0.7814287 at tol_step 1e-9), which is what licenses the change: they are
#     the same fixed point, reached at different cost.
#
# The mechanism is NOT that the scattering iteration is slow on its own. With
# ``rebalance=False`` the subsweep count barely matters (1347 -> 1080 sweeps
# from 3 to 6). What subsweeps buy is a spectrally CONSISTENT flux, which is a
# precondition for the rebalance: its correction assumes the swept flux
# satisfies a neutron balance with the source it was given, and under
# Gauss-Seidel that holds only once the groups stop moving relative to one
# another. The two together are worth ~50x; neither alone is worth 1.5x.
_UPSCATTER_SUBSWEEPS = 6


def _n_steps(t_end: float, dt: float) -> int:
    """Return the exact number of constant-width steps requested.

    Rounding a non-integral horizon silently changes the requested physical
    problem. Refuse it rather than report a result at a different final time.
    """
    if not np.isfinite(t_end) or not np.isfinite(dt) or t_end < 0.0 or dt <= 0.0:
        raise ValueError("t_end must be finite and non-negative, and dt must be finite and positive")
    n_steps = int(round(t_end / dt))
    if not np.isclose(n_steps * dt, t_end, rtol=1e-12,
                      atol=1e-14 * max(1.0, t_end)):
        raise ValueError("t_end must be an integer multiple of dt for a constant-step transient")
    return n_steps


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
    # True when the caller supplied the compatible time-zero eigenpair instead
    # of asking this solve to repeat the initial critical calculation.
    initial_state_reused: bool = False

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
    problem_at : callable t -> (materials, material_map), or
                 (materials, material_map, mix_material, mix_weight) to move
                 the volume blend in time as well -- which is what a rotating
                 control drum is, since its absorber arc sweeps a different
                 area fraction of each cell at every angle. material_map may be
                 None for a homogeneous reactor. Return cached objects while
                 nothing changes — operators are rebuilt only on identity
                 change of any returned element, so a driver that hands back
                 the same arrays costs nothing. Per-material kinetics are
                 remapped whenever the material or blend geometry changes.
    xs_update_at : optional callable t -> (fields -> None) | None, the seam for
                 STATE-dependent cross sections: a temperature feedback closes
                 over the current temperature field and returns the per-cell
                 update (see :mod:`ndgpu.feedback`). Supplying it forces an
                 operator rebuild every step, because the data really does
                 change every step.
    on_step    : optional callable (t, flux, power) -> None, invoked after each
                 converged step. This is what lets an external solver march
                 alongside: it receives the end-of-step flux and P(t)/P(0), and
                 whatever it computes there can be fed back through
                 ``xs_update_at`` on the next step (operator splitting).
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
                 mix_material=None, mix_weight=None,
                 xs_update_at=None, on_step=None, phase_context=None):
        self.grid = grid
        self.problem_at = problem_at
        self.xs_update_at = xs_update_at
        self.on_step = on_step
        # Optional driver instrumentation: callable(name) -> context manager.
        # Keeping it outside the numerical result avoids timing overhead unless
        # a coupled/benchmark driver explicitly opts in.
        self.phase_context = phase_context
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

    def _group_batch(self, fields, phi, G):
        r"""Stacked group data for the batched source assembly, or None.

        The same trade ``_PowerIterationSolver._make_group_batch`` makes for the
        steady outer loop, applied to the *step* fixed point -- which needs it
        more, because it assembles a right-hand side per group per Gauss-Seidel
        subsweep per sweep, and a step near criticality takes many sweeps. The
        per-(g, g') in-scatter loop is two kernels and a full-size temporary per
        pair, i.e. O(G^2) launches per subsweep; stacking the scattering matrix
        into (G, G, \*grid) and the fluxes into (G, \*grid) lets one kernel walk
        a whole row, taking that to O(G).

        Returned only on GPU and only for G >= 3 with room for the stack, for
        the reasons given there: on CPU the sparse loop is strictly better
        because it skips absent couplings instead of multiplying materialized
        zeros. The returned ``phi`` is the flux array the caller must then work
        through -- group entries are *views* into it, so the batched kernel sees
        each group solve's result without a restack.
        """
        xp = self.xp
        if not kernels.use_fused(xp, "groups") or G < 3:
            return None
        ref = phi[0]
        nbytes = G * G * ref.size * ref.dtype.itemsize
        try:
            free = xp.cuda.Device().mem_info[0]
        except Exception:
            free = 0
        if free and nbytes > free // 8:
            return None                       # not worth the footprint
        return {"phi": xp.stack(phi), "fsrc": xp.zeros_like(ref),
                **self._batch_fields(fields, G, ref)}

    def _batch_fields(self, fields, G, ref):
        """The cross-section half of the batch -- rebuilt whenever Fields is."""
        xp = self.xp
        return {"S": scatter_stack(xp, fields.sigma_s, G, False,
                                   ref.shape, ref.dtype),
                "W": xp.stack([fields.nu_sigma_f[g] for g in range(G)])}

    def _unpack(self, spec):
        """Normalize problem_at's return to (materials, material_map,
        mix_material, mix_weight), defaulting the blend to the static one."""
        if len(spec) == 2:
            mats, mmap = spec
            return mats, mmap, self.mix_material, self.mix_weight
        if len(spec) == 4:
            return tuple(spec)
        raise ValueError("problem_at must return (materials, material_map) or "
                         "(materials, material_map, mix_material, mix_weight), "
                         f"got {len(spec)} elements")

    def solve(self, t_end: float, dt: float, tol_step: float = 1e-6,
              max_sweeps: int = 200, anderson_depth: int = 5,
              scatter_subsweeps: int | None = None,
              steady_kwargs: dict | None = None,
              initial_steady: Result | None = None,
              initial_precursors=None,
              rebalance: bool = False,
              linsolve_kwargs: dict | None = None,
              verbose: bool = False) -> TransientResult:
        """March from the steady state at t=0 to t_end with fixed step dt.

        tol_step : stopping criterion on the RELATIVE CHANGE between successive
            fission-source iterates. **It is not an error bound, and on a
            near-critical core the two differ by a large factor.** Measured on
            the 11-group HP-MR (2D refine 2, +0.5 $ step, dt = 0.02 s), P(end)
            at tol_step 1e-5 / 1e-6 / 1e-7 / 1e-8 is 1.625123 / 1.626469 /
            1.626604 / 1.626618 — clean first-order convergence to ~1.6266196,
            i.e. an error of roughly **150 x tol_step**. So the default 1e-6
            delivers a power good to ~1.5e-4 relative, about four significant
            figures, not six. Quote transient powers accordingly, and tighten
            tol_step (at ~250 sweeps a step, expensively) if more is needed.

            Note also that loosening tol_step can cost *both* accuracy and time,
            because ``rebalance`` switches off below ``10 * tol_step``: the
            1e-5 run above took 316/405 sweeps against 248/245 at 1e-6, having
            handed the last decade to unaccelerated Picard.

        anderson_depth : number of past fission-source iterates retained by the
            Anderson acceleration of the within-step fixed point (window m+1 for
            m residual differences). Depth 1 disables it (plain Picard).

            On a stiff, stateful group sweep its iterate change can be smaller
            than the true fixed-point residual. Before accepting convergence,
            the solver therefore performs one unaccelerated confirmation sweep;
            a failed confirmation restarts the Anderson history.

        rebalance : enforce whole-core neutron balance on each iterate by
            rescaling the fission source (see the derivation at the call site).
            Removes the fundamental AMPLITUDE error, which is the slow mode
            near criticality: 4.5x fewer iterations than plain Picard at an
            answer that agrees to 1.2e-6. Does NOT combine with Anderson -- the
            correction is a rational function of S, so the map stops being
            affine and Anderson's premise fails.
        linsolve_kwargs : extra keyword arguments forwarded to every inner
            linear solve. The one that matters on GPU is ``check_every``: PCG's
            convergence test is a device->host reduction, i.e. a full pipeline
            stall, and at the default of 1 there is one per CG iteration. The
            group solves here run at a *loose*, sweep-adaptive rtol and so take
            few iterations, which makes that stall a large fraction of each
            solve -- see ``examples/hpmr_transient_bench.py``. Costs at most
            ``check_every - 1`` extra iterations per solve, so keep it well
            below the typical iteration count.
        scatter_subsweeps : Gauss-Seidel passes over the groups per fixed-point
            evaluation (fission source held fixed). With downscatter only, one
            ordered pass is exact, so extra passes are pointless; with
            *upscatter* a single pass leaves the evaluation dependent on the
            incoming flux iterate, and Anderson -- which assumes the sweeps it
            mixes sample a fixed map G(S) -- loses most of its acceleration
            (C5G7's 7-group water: ~450 sweeps instead of ~10). A few passes
            restore an (almost) pure G(S). None (default) auto-selects: 3 when
            any upscatter coupling exists, else 1.

            **On a strongly-upscattering multigroup core the auto-selected 3 is
            far too few, and this is the dominant cost knob in the whole solver.**
            Measured on the 11-group HP-MR (+0.5 $ step, dt = 0.02 s,
            rebalance=True), sweeps and total CG for one step:

                subsweeps   3      4      5      6      8
                refine 2  248/35689 144/23421 43/7661 27/5285 25/5536
                refine 3  247/52864 144/34609 30/8068 26/7583 25/8254

            Going from 3 to 6 is **~7x fewer CG iterations and ~3x less wall
            time, at roughly half the error** -- the optimum sits at 6 with a
            5-8 plateau, identically at both mesh sizes. Note the 4->5
            transition is a cliff, not a gradient: below a threshold the group
            cascade cannot propagate within one fixed-point evaluation and the
            outer iteration is left to carry it, which is what produces the
            hundreds of sweeps the auto default has been paying.

            The auto rule is deliberately left at 3 because it is shared with
            the C5G7-TD and TWIGL validations, which have not been re-measured;
            the HP-MR benchmark harness passes 6 explicitly. Raising the default
            is the right change once those are checked.
        initial_steady : optional compatible time-zero diffusion eigenpair.
            Supplying it skips the repeated eigenvalue solve and starts from
            its flux and k_eff. The caller owns physical compatibility: it must
            have been solved on this grid with ``problem_at(0)`` and the same
            state-dependent cross sections. Shape, group count, convergence,
            and finiteness are validated here. This is primarily the hand-off
            from a just-converged coupled hot equilibrium.
        initial_precursors : optional normalized spatial precursor fields,
            shape ``(n_families, *grid.shape)``. These must use the same
            normalization as ``initial_steady.flux`` after its fission source
            is normalized to ``P(0)=1``. Supplying them carries delayed-source
            history across an IQS macro interval; omitting them initializes the
            critical equilibrium fields.
        """
        xp, kin = self.xp, self.kinetics
        beta, lam = kin.beta, kin.decay
        lin_kw = dict(linsolve_kwargs or {})
        n_steps = _n_steps(t_end, dt)
        synchronize(xp)
        t0 = time.perf_counter()

        # --- initial condition: steady state, critically adjusted ------------
        mats, mmap, mix_m0, mix_w0 = self._unpack(self.problem_at(0.0))
        # Mix arrays only reach solvers that were given them: pluggable
        # geometries (hex, ...) never see the kwargs in the default case.
        mix_kwargs = ({} if mix_m0 is None else
                      dict(mix_material=mix_m0, mix_weight=mix_w0))
        # The t=0 steady state must be built with the SAME state-dependent
        # cross sections the first step will see, or the transient starts from
        # an equilibrium of a different problem and shows a spurious jump.
        upd0 = self.xs_update_at(0.0) if self.xs_update_at is not None else None
        initial_phase = (nullcontext() if self.phase_context is None else
                         self.phase_context("initial_eigen_solve"))
        with initial_phase:
            eig = self.eig_solver(self.grid, mats, mmap, bc=self.bc,
                                  device="cpu" if xp is np else "gpu",
                                  dtype=self.dtype,
                                  active=self.active, mask_bc=self.mask_bc,
                                  linear_solver=self.linear_solver,
                                  symmetric_operator=self.symmetric_operator,
                                  xs_update=upd0,
                                  **mix_kwargs)
            G = eig.n_groups
            if kin.velocities.shape[-1] != G:
                raise ValueError("kinetics.velocities must have one entry per group")
            n_mats = len(mats) if isinstance(mats, (list, tuple)) else 1
            for name, table in (("velocities", kin.velocities), ("beta", kin.beta)):
                if table.ndim == 2 and len(table) != n_mats:
                    raise ValueError(f"per-material kinetics.{name} must have one "
                                     f"row per entry of the materials list")
            if initial_steady is None:
                steady = eig.solve(
                    **(steady_kwargs or dict(tol_k=1e-8, tol_source=1e-7)))
            else:
                if not isinstance(initial_steady, Result):
                    raise TypeError("initial_steady must be a solver Result")
                if not initial_steady.converged:
                    raise ValueError("initial_steady must be converged")
                if (not np.isfinite(initial_steady.k_eff)
                        or initial_steady.k_eff <= 0.0):
                    raise ValueError("initial_steady.k_eff must be finite and positive")
                initial_flux = xp.asarray(initial_steady.flux, dtype=self.dtype)
                expected = (G,) + tuple(self.grid.shape)
                if initial_flux.shape != expected:
                    raise ValueError(
                        f"initial_steady flux shape {initial_flux.shape} != {expected}")
                if not bool(xp.all(xp.isfinite(initial_flux))):
                    raise ValueError("initial_steady flux must be finite")
                steady = copy.copy(initial_steady)
                steady.flux = initial_flux.copy()
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

        def kinetic_fields(current_fields):
            """Map kinetics through the current material/blend geometry."""
            if kin.beta.ndim == 2:
                # beta rides on the fission source, like chi: a mixed
                # fuel/moderator cell must keep the fuel's delayed fraction.
                mapped_beta = [current_fields.map_table_fission_weighted(kin.beta[:, i])
                               for i in range(kin.n_families)]
            else:
                mapped_beta = [float(b) for b in kin.beta]
            if kin.velocities.ndim == 2:
                mapped_inv_vdt = [current_fields.map_table(
                    1.0 / (kin.velocities[:, g] * dt)) for g in range(G)]
            else:
                mapped_inv_vdt = [1.0 / (kin.velocities[g] * dt) for g in range(G)]
            return mapped_beta, mapped_inv_vdt

        beta, inv_vdt = kinetic_fields(fields)

        phi = [steady.flux[g].copy() for g in range(G)]
        # On GPU the group loops below collapse into one kernel per row; `phi`
        # then aliases the rows of a single (G, *grid) array so the batched
        # kernel always reads the current Gauss-Seidel iterate.
        batch = self._group_batch(fields, phi, G)
        if batch is not None:
            phi = [batch["phi"][g] for g in range(G)]

        def fission_source(phi_by_group):
            """Sum_g nuSigma_f,g phi_g, batched into one kernel when available."""
            if batch is None:
                return fields.fission_source(phi_by_group)
            # The batched kernel reads the STACK, not the argument -- they are
            # the same storage only for the solver's own `phi`, whose entries
            # are views into it. Called with any other flux list it would
            # silently return the wrong field, so refuse instead.
            if phi_by_group is not phi:
                raise AssertionError("batched fission_source takes the aliased "
                                     "flux list; got a different one")
            # Accumulated into a persistent buffer; every caller divides by k0
            # straight away, which allocates, so the buffer is never aliased.
            out = batch["fsrc"]
            out.fill(0)
            return kernels.group_accumulate(xp, out, batch["W"], batch["phi"])

        S = fission_source(phi) / k0
        scale = 1.0 / total_power(S)            # P(0) = 1
        for g in range(G):
            phi[g] *= scale
        S = S * scale
        if initial_precursors is None:
            C = [(beta[i] / lam[i]) * S
                 for i in range(kin.n_families)]  # equilibrium
        else:
            supplied = xp.asarray(initial_precursors, dtype=self.dtype)
            expected = (kin.n_families,) + tuple(self.grid.shape)
            if supplied.shape != expected:
                raise ValueError(
                    f"initial_precursors shape {supplied.shape} != {expected}")
            if not bool(xp.all(xp.isfinite(supplied))):
                raise ValueError("initial_precursors must be finite")
            if bool(xp.any(supplied < 0.0)):
                raise ValueError("initial_precursors must be non-negative")
            C = [supplied[i].copy() for i in range(kin.n_families)]

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
        w_fis_sum = w_fis[0]
        for g in range(1, G):
            w_fis_sum = w_fis_sum + w_fis[g]
        # Total out-scatter per group; summed over groups this equals
        # the total in-scatter, which is what the rebalance needs.
        out_scatter = []
        for g in range(G):
            acc = None
            for gt in range(G):
                if gt == g:
                    continue
                sc = fields.sigma_s[g][gt]
                if sc is None:
                    continue
                acc = sc if acc is None else acc + sc
            out_scatter.append(xp.zeros(self.grid.shape, dtype=self.dtype)
                       if acc is None else acc)


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
        last = (mats, mmap, mix_m0, mix_w0)
        last_upd = upd0

        # sigma_s is indexed [g_from][g_to]: any coupling above the diagonal
        # is upscatter, which one ordered Gauss-Seidel pass cannot resolve.
        has_upscatter = any(fields.sigma_s[gf][gt] is not None
                            for gt in range(G) for gf in range(gt + 1, G))
        n_sub = (int(scatter_subsweeps) if scatter_subsweeps
                 else (_UPSCATTER_SUBSWEEPS if has_upscatter else 1))

        times = [0.0]
        power = [1.0]
        inner_total = 0
        step_its: list[int] = []

        for n in range(1, n_steps + 1):
            t = n * dt
            spec = self._unpack(self.problem_at(t))
            mats, mmap, mix_m, mix_w = spec
            upd = self.xs_update_at(t) if self.xs_update_at is not None else None
            # Rebuild on identity change of anything the fields depend on --
            # including the update itself, so a driver whose state has not moved
            # can hand back the SAME closure and skip the rebuild. That matters:
            # with a thermal step coarser than the neutronics step the
            # temperature (and hence the cross sections) is unchanged for most
            # steps, and rebuilding G operators for identical data is pure cost.
            if upd is not last_upd or any(a is not b for a, b in zip(spec, last)):
                phase = (nullcontext() if self.phase_context is None else
                         self.phase_context("operator_rebuild"))
                with phase:
                    fields = Fields(xp, self.grid, mats, mmap, self.dtype,
                                    mix_material=mix_m, mix_weight=mix_w,
                                    xs_update=upd)
                    if kin.per_material:
                        beta, inv_vdt = kinetic_fields(fields)
                        bcoef = [beta[i] / (1.0 + lam[i] * dt)
                                 for i in range(kin.n_families)]
                        bcoef_sum = sum(bcoef)
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
                    # Summed emission weight, for the rebalance's W(.) reduction.
                    w_fis_sum = w_fis[0]
                    for g in range(1, G):
                        w_fis_sum = w_fis_sum + w_fis[g]
                    # Total out-scatter per group; summed over groups this equals
                    # the total in-scatter, which is what the rebalance needs.
                    out_scatter = []
                    for g in range(G):
                        acc = None
                        for gt in range(G):
                            if gt == g:
                                continue
                            sc = fields.sigma_s[g][gt]
                            if sc is None:
                                continue
                            acc = sc if acc is None else acc + sc
                        out_scatter.append(xp.zeros(self.grid.shape, dtype=self.dtype)
                                           if acc is None else acc)
                    if batch is not None:
                        # New Fields means new scattering and production data; the
                        # flux stack is state and must survive the rebuild.
                        batch.update(self._batch_fields(fields, G, batch["fsrc"]))
                    last, last_upd = spec, upd

            phi_old = [p.copy() for p in phi]
            # Delayed emission from decayed precursors, constant within the step.
            dsrc_g = delayed_source_by_group(C)

            # Whole-core rebalance needs the step's FIXED source -- everything
            # on the right-hand side that does not scale with the flux. That is
            # exactly what makes the correction determinate: loss and fission
            # both scale with a trial amplitude f, the delayed and time sources
            # do not, so balance has a unique solution for f.
            if rebalance:
                fixed_total = 0.0
                fixed_cell = None
                for g in range(G):
                    term = dsrc_g[g] + inv_vdt[g] * phi_old[g]
                    if src_w is not None:
                        term = term * src_w
                    fixed_total += float(xp.sum(term))
                    # The coarse scheme needs this per CELL, not just its total;
                    # it is constant through the step, so it is built once here.
                    fixed_cell = term if fixed_cell is None else fixed_cell + term

            # Fixed point on the end-of-step fission source (Gauss-Seidel over
            # groups, warm-started from the previous step). Near criticality
            # the plain iteration contracts like the prompt multiplication
            # factor (arbitrarily close to 1), so it is Anderson-accelerated:
            # the next iterate is the residual-minimizing affine combination
            # of the last few sweeps, which collapses the handful of slow
            # error modes (one per perturbed region) in a few sweeps.
            change = 1.0
            change_prev = np.inf
            confirming = False
            anderson_used = False
            hist: list = []  # (S_j, G(S_j)) pairs, oldest first
            for sweep in range(1, max_sweeps + 1):
                # Solve well below both the current sweep change and the step
                # tolerance, so CG noise never becomes the fixed point's floor.
                rtol = min(1e-6, max(1e-3 * change, 1e-3 * tol_step, 1e-12))
                for _ in range(n_sub):
                    for g in range(G):
                        q = inv_vdt[g] * phi_old[g] + w_fis[g] * S + dsrc_g[g]
                        if batch is not None:
                            # One kernel for the whole in-scatter row. The flux
                            # stack is updated in place group by group below, so
                            # this reads the new flux for g' < g and the old one
                            # for g' > g -- the same Gauss-Seidel sweep.
                            kernels.group_accumulate(xp, q, batch["S"][g],
                                                     batch["phi"])
                        else:
                            for gf in range(G):
                                s = fields.sigma_s[gf][g]
                                if gf != g and s is not None:
                                    q += s * phi[gf]
                        if src_w is not None:
                            q = q * src_w
                        sol, n_it = self._linsolve(ops[g].apply, q, phi[g],
                                                   ops[g].inv_diag, xp,
                                                   rtol=rtol,
                                                   precond=preconds[g],
                                                   **lin_kw)
                        if batch is None:
                            phi[g] = sol
                        else:
                            phi[g][...] = sol   # write back into the stack
                        inner_total += n_it
                G_S = fission_source(phi) / k0

                delta = G_S - S
                change = float(xp.sqrt(xp.sum(delta * delta) / xp.sum(G_S**2)))

                # ---- whole-core rebalance -------------------------------
                # The swept flux satisfies the balance with the source it was
                # GIVEN (S), not with the one it implies (G_S). Rescaling the
                # whole solution by f and demanding balance against the implied
                # source pins the fundamental AMPLITUDE in one shot:
                #
                #     f = Fx / (Fx + W(S) - W(G_S)) = Fx / (Fx - W(delta))
                #
                # with W(.) the total weighted fission emission and Fx the
                # step's fixed source (delayed + time). Loss and fission both
                # scale with f while Fx does not, which is what makes f
                # determinate; and f -> 1 as delta -> 0, so the fixed point is
                # untouched and only the path changes.
                #
                # W(S) - W(G_S) is evaluated as a SINGLE reduction over delta,
                # never as two reductions subtracted. Near criticality Fx is
                # ~0.1% of either, so forming them separately loses the answer
                # to cancellation: that version diverged outright (source change
                # stuck at 5e-1 after 6000 sweeps), and an earlier variant that
                # also mis-stated the sweep's balance gave a negative power.
                if rebalance and change > 10.0 * tol_step:
                    w_delta = float(xp.sum(w_fis_sum * delta
                                           * (src_w if src_w is not None else 1.0)))
                    denom = fixed_total - w_delta
                    if denom > 0.0:
                        # Far from 1 means the balance is being asked to
                        # extrapolate through the critical pole; clamp and let
                        # the sweeps carry the rest.
                        f = min(max(fixed_total / denom, 0.5), 2.0)
                        G_S = G_S * f
                        for g in range(G):
                            phi[g] *= f
                        delta = G_S - S

                if change < tol_step:
                    if anderson_used and not confirming:
                        # Anderson can make the change between mixed iterates
                        # look converged while the stateful Gauss-Seidel map is
                        # not. Confirm using a clean Picard sweep; if it fails,
                        # the normal path below restarts from an empty history.
                        confirming = True
                        hist = []
                        S = G_S
                        continue
                    S = G_S
                    break
                confirming = False
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
                    anderson_used = True
            else:
                raise RuntimeError(
                    f"time step at t={t:g} s did not converge "
                    f"({max_sweeps} sweeps, source change {change:.2e})")

            for i in range(kin.n_families):
                C[i] = (C[i] + (dt * beta[i]) * S) / (1.0 + lam[i] * dt)

            step_its.append(sweep)
            times.append(t)
            power.append(total_power(S))
            # The seam an external physics marches through: it sees the
            # converged end-of-step flux and P(t)/P(0), advances its own state,
            # and what it computes reaches the neutronics on the NEXT step via
            # xs_update_at -- one-way per step, i.e. operator splitting.
            if self.on_step is not None:
                # The GPU multigroup path already owns one contiguous
                # (G, *grid) flux stack. Hand its view to a coupled power edit
                # instead of making that edit restack/copy every group. A
                # leading-axis array remains iterable exactly like the legacy
                # list, so callbacks written as ``for p in flux`` keep working.
                callback_phi = batch["phi"] if batch is not None else phi
                self.on_step(t, callback_phi, power[-1])
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
            step_iterations=step_its,
            initial_state_reused=initial_steady is not None,
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
        n_steps = _n_steps(t_end, dt)
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
        n_steps = _n_steps(t_end, dt)
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
              time_scheme: str = "backward-euler", time_weight: float | None = None,
              verbose: bool = False) -> TransientResult:
        """March from the steady S_N state at t=0 to t_end with fixed step dt
        (see :meth:`TransientSolver.solve` for the shared parameters).

        time_weight : the theta-method weight w on the end-of-step operator --
            w = 1 (default) is backward Euler, w = 1/2 Crank-Nicolson. Written
            (1/v) du/dt = F(u), the scheme is

                theta (u^{n+1} - u^n) = w F^{n+1} + (1 - w) F^n,
                theta = 1/(v_g dt)

            which rearranges to a *backward-Euler-shaped* solve -- collision
            shift theta/w and a per-ordinate source (theta/w) Psi^n -- with the
            carried angular field

                Psi^{n+1} = u^{n+1} + ((1 - w)/w) (u^{n+1} - Psi^n),  Psi^0 = u^0.

            The recursion is what keeps F^n out of the picture: evaluating it
            directly would cost an extra streaming apply (a whole sweep) per
            step. At w = 1 it degenerates to Psi = u^{n+1}, i.e. bit-for-bit the
            backward-Euler path, and the shift theta/w >= theta means the
            within-step fixed point is *better* damped for w < 1, not worse.

            Note w < 1 is opt-in on purpose: Crank-Nicolson is A-stable but not
            L-stable, so a prompt jump can ring rather than decay. Backward
            Euler stays the default for stiff reactor kinetics.
        """
        xp, kin = self.xp, self.kinetics
        beta = [float(b) for b in kin.beta]
        lam = kin.decay
        scheme = make_time_scheme(time_scheme, time_weight)
        if not scheme.is_bdf and any(b != 0.0 for b in beta):
            # The precursor substitution is of BDF form (see timescheme.py);
            # the theta-method carries F^n instead of past states and has no
            # matching treatment, so refuse rather than silently run a
            # mismatched-order scheme.
            raise NotImplementedError(
                f"{scheme.name} is prompt-only: its precursor treatment has "
                f"not been derived. Use time_scheme='backward-euler' or 'bdf2' "
                f"with delayed neutrons (beta != 0).")
        n_steps = _n_steps(t_end, dt)
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

        # theta_g = 1/(v_g dt); every scheme solves with the shift a0*theta and
        # a carried field Psi supplied by the scheme (see ndgpu/timescheme.py).
        # a0 may change once, when a multistep formula leaves its bootstrap.
        theta_base = [1.0 / (float(kin.velocities[g]) * dt) for g in range(G)]
        a0 = scheme.a0(1)
        theta = [a0 * tb for tb in theta_base]

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

        # Analytic BDF precursor substitution (see ndgpu/timescheme.py): with
        # the scheme's history combination H,
        #     C^{n+1} = (dt beta S^{n+1} + H) / (a0 + lam dt),
        # so the end-of-step fission weight carries beta a0/(a0 + lam dt) and
        # the delayed source lam H/(a0 + lam dt). At a0 = 1 both reduce to the
        # familiar backward-Euler beta/(1 + lam dt). Consistency (sum a_j = 0)
        # makes H = a0 C at equilibrium, so an unperturbed state is preserved
        # exactly for any a0.
        # The scheme owns the precursor substitution -- BDF form or theta form,
        # each derived in ndgpu/timescheme.py and each preserving the t=0
        # equilibrium exactly. The driver only weights the result by the
        # delayed spectrum.
        lam_l = [float(l) for l in lam]

        def bcoef_sum_at(step):
            return sum(scheme.precursor_bcoef(step, beta, lam_l, dt))

        bcoef_sum = bcoef_sum_at(1)

        def fission_weights(bsum):
            if kin.chi_delayed is not None:
                return [chi[g] - float(kin.chi_delayed[g]) * bsum
                        for g in range(G)]
            return [chi[g] * (1.0 - bsum) for g in range(G)]

        def delayed_source_by_group(decayed):
            dsum = decayed[0]
            for d in decayed[1:]:
                dsum = dsum + d
            if kin.chi_delayed is not None:
                return [float(kin.chi_delayed[g]) * dsum for g in range(G)]
            return [chi[g] * dsum for g in range(G)]

        w_fis = fission_weights(bcoef_sum)
        last = (mats, mmap)

        # Carried fields Psi (angular and its scalar integral), seeded from the
        # steady state; the scheme owns how they advance. One scheme instance
        # per field so each keeps its own history.
        sch_psi, sch_phi = scheme, copy.deepcopy(scheme)
        psi_c = sch_psi.start(psi)
        phi_c = sch_phi.start([p.copy() for p in phi])
        C_hist = [C, None]        # [C^n, C^{n-1}]; the theta form also needs
        S_prev = S                # the previous step's fission source

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
                w_fis = fission_weights(bcoef_sum)
                sweeps0 = eng._sweep_count
                last = (mats, mmap)

            # A multistep scheme raises a0 once it leaves its bootstrap step;
            # the shift, the DSA/CMFD factors and the precursor coefficients
            # all move with it, so refresh them on that transition only.
            a0_n = scheme.a0(n)
            if a0_n != a0:
                a0 = a0_n
                theta = [a0 * tb for tb in theta_base]
                bcoef_sum = bcoef_sum_at(n)
                if getattr(self.engine_cls, "T_SHIFT_AT_CONSTRUCTION", False):
                    inner_total += eng._sweep_count - sweeps0
                    eng = self._engine(mats, mmap, sigma_t_shift=theta)
                    ss, nsf, chi, scatter = load_fields(eng)
                    sweeps0 = eng._sweep_count
                eng.t_setup(theta)
                w_fis = fission_weights(bcoef_sum)

            # The carried Psi is fixed within the step; the time source is
            # a0*theta*Psi (its angular integral is the CMFD-level source).
            # For backward Euler these are just the previous step's psi/phi.
            psi_old = sch_psi.carried(n)
            phi_old = sch_phi.carried(n)
            dsrc_g = delayed_source_by_group(
                scheme.precursor_decayed(n, C_hist, S_prev, beta, lam_l, dt))
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

            # Precursors first: they must see the same step index (hence the
            # same a0) the flux solve just used. Then advance the carried
            # fields.
            C_prev = C
            C = scheme.precursor_update(n, C_hist, S, S_prev, beta, lam_l, dt)
            C_hist, S_prev = [C, C_prev], S

            sch_psi.push(psi)
            sch_phi.push(phi)

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
