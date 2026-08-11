"""Coupled neutronics/thermal steady state.

The two physics are solved separately and reconciled by a fixed-point
iteration on the temperature field:

    T  ->  cross sections (feedback)  ->  k-eigenvalue solve  ->  phi
       ->  q'''  (normalized to rated power)  ->  conduction  ->  T'

**What is being computed, and what is not.** A k-eigenvalue solve renormalizes
its flux, so the power *level* is imposed by the operator -- rated thermal
power -- and not computed. Feedback therefore cannot change how much heat the
core makes; it changes where the heat is made, and it changes `k`. That is the
standard steady-state coupled formulation (the same one Griffin/BISON and
MPACT/CTF use), and it is what makes the iteration strongly contractive: with
the level pinned, Doppler can only redistribute power by a fraction of a
percent, so the map is a near-constant and Picard converges in a handful of
steps without needing acceleration to be viable.

The reportable result is `k` at hot full power versus `k` on the cold
unfed core -- the **temperature defect** -- and, through
:func:`criticality_search`, the control-drum angle that holds the core critical
once that defect is paid for.

**Why the map lives in free functions.** :func:`neutronics_step` and
:func:`thermal_step` are called both by :class:`CoupledSolver` here and by the
preCICE participants in ``examples/precice/``. Cross-verifying an external
coupling against an internal one only means something if the two are running
identical physics; the way to guarantee that is to have one implementation of
each half and let the couplers differ.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import numpy as np

from . import kernels
from .backend import asnumpy, synchronize
from .linalg import AndersonAccelerator
from .power import power_density
from .thermal import ConductionSolver
from .tri import TriDiffusionEigenSolver


class _CoupledPhaseProfiler:
    """Low-intrusion coupled phase timer.

    CPU regions use ``perf_counter``. GPU regions use CUDA events and are
    resolved after one final synchronization, so profiling does not turn every
    phase boundary into the device/host stall it is intended to diagnose.
    """

    def __init__(self, xp, enabled=False):
        self.xp = xp
        self.enabled = bool(enabled)
        self.gpu = self.enabled and xp is not np
        self._pairs = defaultdict(list)
        self._neutronics_start = None
        self.counters = defaultdict(int)

    def _stamp(self):
        if self.gpu:
            event = self.xp.cuda.Event()
            event.record()
            return event
        return time.perf_counter()

    def _add(self, name, start, stop):
        if self.enabled:
            self._pairs[name].append((start, stop))

    @contextmanager
    def region(self, name):
        if not self.enabled:
            yield
            return
        start = self._stamp()
        if self.gpu:
            from .profiling import nvtx_range
            marker = nvtx_range(f"ndgpu.coupled.{name}")
        else:
            marker = nullcontext()
        with marker:
            yield
        self._add(name, start, self._stamp())
        if name == "operator_rebuild":
            self.counters["operator_rebuilds"] += 1

    def resume_neutronics(self):
        if self.enabled:
            self._neutronics_start = self._stamp()

    def pause_neutronics(self):
        if self.enabled and self._neutronics_start is not None:
            self._add("neutronics_total", self._neutronics_start, self._stamp())
            self._neutronics_start = None

    def seconds(self):
        if not self.enabled:
            return {}
        synchronize(self.xp)
        out = {}
        for name, pairs in self._pairs.items():
            if self.gpu:
                out[name] = sum(
                    self.xp.cuda.get_elapsed_time(a, b) for a, b in pairs) / 1e3
            else:
                out[name] = sum(b - a for a, b in pairs)
        # Initial criticality and rebuild work are nested within the interval
        # from transient entry to coupling callbacks. Report an exclusive
        # time-march neutron solve so the table adds up rather than attributing
        # startup to time-step work or double-counting rebuilds.
        neutron_total = out.pop("neutronics_total", 0.0)
        out["neutronics_solve"] = max(
            0.0, neutron_total - out.get("initial_eigen_solve", 0.0)
            - out.get("operator_rebuild", 0.0))
        return out


@dataclass
class CouplingContext:
    """Everything both halves need, assembled once.

    Built from an ``HpmrProblem`` by :func:`context_from_problem`, but nothing
    here is HP-MR specific -- any geometry the eigen solvers accept works.
    """

    grid: object
    materials: list
    material_map: np.ndarray
    thermal_materials: list
    feedback: object                       # ThermalFeedback
    total_power: float                     # watts
    active: np.ndarray | None = None
    mask_bc: object = "vacuum"
    bc: object = "reflective"
    mix_material: np.ndarray | None = None
    mix_weight: np.ndarray | None = None
    thermal_bc: object = "adiabatic"
    thermal_mask_bc: object = 1e-3         # W/(cm^2 K) to the vessel
    ambient_temperature: float = 400.0
    solver_cls: object = TriDiffusionEigenSolver
    #: Within-group operator matching solver_cls; only the transient needs it
    #: explicitly (the eigen solvers pick their own). None => infer from grid.
    group_operator: object = None
    #: Delayed-neutron data. Only the transient reads it.
    kinetics: object = None
    device: str = "cpu"
    dtype: object = np.float64
    eigen_kwargs: dict = field(default_factory=lambda: {"tol_k": 1e-10,
                                                        "tol_source": 1e-9})
    #: Reuse the previous flux and k as the next solve's starting point. A
    #: 3-4x speed-up in production -- but it makes the coupled map depend on
    #: its own history, so a lockstep comparison against another coupling tool
    #: must switch it OFF to make the map a pure function of T.
    warm_start: bool = True

    # -- mutable warm-start cache ------------------------------------------
    _state: object = None
    _k: float = 1.0
    _thermal: object = None

    def __post_init__(self):
        if self.group_operator is None:
            from .stencil import GroupOperator
            from .tri import TriGrid, TriGroupOperator
            self.group_operator = (TriGroupOperator
                                   if isinstance(self.grid, TriGrid)
                                   else GroupOperator)

    def reset_state(self):
        """Forget the warm start, making the coupled map deterministic."""
        self._state = None
        self._k = 1.0

    @property
    def shape(self):
        return tuple(self.grid.shape)

    def thermal_solver(self) -> ConductionSolver:
        """The conduction solver -- built once and reused: unlike the
        neutronics, its coefficients do not depend on the coupled state."""
        if self._thermal is None:
            self._thermal = ConductionSolver(
                self.grid, self.thermal_materials, self.material_map,
                bc=self.thermal_bc, active=self.active,
                mask_bc=self.thermal_mask_bc,
                ambient_temperature=self.ambient_temperature,
                mix_material=self.mix_material, mix_weight=self.mix_weight,
                device=self.device, dtype=self.dtype)
        return self._thermal

    def initial_temperature(self) -> np.ndarray:
        """A physically-scaled starting guess: the temperature the fuel would
        reach if all the power left through the heat pipes with no conduction
        at all, T_sink + q/h. Both participants must start from this same
        field for a lockstep comparison to be meaningful."""
        th = self.thermal_solver()
        xp = th.xp
        h, t_sink = th.h, th.t_sink
        sinked_volume = float(xp.sum(th.cell_volume * (h > 0)))
        if sinked_volume <= 0.0:
            return np.full(self.shape, self.ambient_temperature)
        q_mean = self.total_power / sinked_volume
        safe_h = xp.where(h > 0, h, 1.0)
        return asnumpy(xp.where(h > 0, t_sink + q_mean / safe_h,
                                self.ambient_temperature))


@dataclass
class CoupledResult:
    k_eff: float
    temperature: np.ndarray
    power_density: np.ndarray
    flux: object
    iterations: int
    converged: bool
    seconds: float
    device: str
    k_history: list
    residual_history: list
    thermal: object                 # the last ThermalResult
    neutronics: object              # the last eigen Result

    @property
    def peak_temperature(self) -> float:
        return float(np.max(self.temperature))

    def __repr__(self):
        status = "converged" if self.converged else "NOT CONVERGED"
        return (f"CoupledResult(k_eff = {self.k_eff:.6f}, "
                f"T = {np.min(self.temperature):.1f} .. "
                f"{self.peak_temperature:.1f} K, "
                f"{status} in {self.iterations} coupling iterations, "
                f"{self.seconds:.1f} s)")


def neutronics_step(temperature, ctx: CouplingContext):
    """T (K) -> volumetric fission power q''' (W/cm^3), plus k and the flux.

    Rebuilds the eigen solver, because the operators copy their coefficients at
    construction and so cannot see a mutated ``Fields``. Measured, that rebuild
    is 12 ms against a 530 ms solve on the 2-group HP-MR at refine 6 (2.3%),
    and a smaller share again on the 11-group core where the solve is ~15x
    longer. Not worth a second, incremental code path to keep bit-consistent
    with this one.
    """
    solver = ctx.solver_cls(
        ctx.grid, ctx.materials, ctx.material_map, bc=ctx.bc,
        active=ctx.active, mask_bc=ctx.mask_bc,
        mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
        device=ctx.device, dtype=ctx.dtype,
        xs_update=ctx.feedback.hook(temperature))

    kw = dict(ctx.eigen_kwargs)
    if ctx.warm_start and ctx._state is not None:
        # The flux only -- deliberately NOT k_guess. The power iteration stops
        # on the CHANGE between successive outers, so seeding it with the
        # previous k as well makes the first |dk| vanishingly small and the
        # solve exits before the flux has responded to the new cross sections:
        # measured, that froze the coupling after one iteration and reported a
        # k 0.9 pcm off the converged value, with a residual of exactly zero
        # that looked like convergence. Warm-starting the shape alone leaves
        # k_guess at its default, so the first outer must do real work.
        kw["state0"] = ctx._state
    res = solver.solve(**kw)
    if not res.converged:
        raise RuntimeError(f"eigenvalue solve did not converge: {res}")
    if ctx.warm_start:
        ctx._state = solver.state
        ctx._k = res.k_eff

    # The thermal solver owns the volume convention -- on an r-z grid the true
    # cell volume is the annulus, not the operator's radius weight -- and the
    # rated power must be divided by the SAME volume the sink term integrates
    # over, or the two halves disagree about how big the core is.
    q = power_density(res.flux, ctx.materials, ctx.material_map,
                      total_power=ctx.total_power,
                      cell_volume=ctx.thermal_solver().cell_volume,
                      mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
                      active=ctx.active)
    return q, res


def thermal_step(power, ctx: CouplingContext, t0=None):
    """q''' (W/cm^3) -> T (K)."""
    res = ctx.thermal_solver().solve(power, t0=t0)
    return asnumpy(res.temperature), res


def coupled_map(temperature, ctx: CouplingContext):
    """One Gauss-Seidel application G(T), neutronics first.

    Neutronics-first matches a preCICE ``serial-implicit`` scheme declared with
    ``first="Neutronics"``; running them the other way round is a different
    fixed-point iteration with the same fixed point, and the two would not
    agree iterate by iterate.
    """
    q, nres = neutronics_step(temperature, ctx)
    t_new, tres = thermal_step(q, ctx)
    return t_new, {"power": q, "neutronics": nres, "thermal": tres}


class CoupledSolver:
    """Picard / Anderson fixed point on the temperature field."""

    def __init__(self, ctx: CouplingContext):
        self.ctx = ctx

    def solve(self, t0=None, tol=1e-8, max_iter=50, relaxation=1.0,
              anderson_depth=0, verbose=False) -> CoupledResult:
        """Iterate to a self-consistent (T, phi, k).

        tol        : convergence on max|T_{n+1} - T_n| relative to the
                     temperature span, so it means the same thing whatever the
                     absolute level.

                     There is a FLOOR, and asking below it hangs rather than
                     errors. The inner eigen solve converges only to
                     ``tol_source``, and that noise reappears as jitter in the
                     temperature: on the 11-group HP-MR at tol_source = 1e-9
                     the coupled residual settles around 1e-6 K on a ~430 K
                     span, i.e. ~2e-9 relative. Measured: tol = 1e-8 converges
                     in 4 iterations and 23 s, while tol = 1e-10 chases noise
                     for 31 iterations and 164 s and lands on the same k and
                     the same peak temperature to 7 digits. Keep tol at least
                     ~50x above tol_source / span.
        relaxation : under-relaxation factor for plain Picard. Applied as
                     ``omega*G(T) + (1-omega)*T`` -- the literal form an
                     external coupling tool's constant-relaxation scheme uses,
                     so the two produce identical iterates rather than merely
                     equivalent ones.
        anderson_depth : > 1 switches to Anderson acceleration over that many
                     iterates. The converged answer is unchanged.
        """
        ctx = self.ctx
        T = np.asarray(ctx.initial_temperature() if t0 is None else t0, dtype=float)
        acc = AndersonAccelerator(depth=anderson_depth, beta=relaxation)

        k_hist, res_hist = [], []
        info = None
        converged = False
        t_start = time.perf_counter()
        for n in range(1, max_iter + 1):
            T_raw, info = coupled_map(T, ctx)
            T_new = acc.step(T, T_raw)
            change = float(np.max(np.abs(T_new - T)))
            span = max(float(np.ptp(T_new)), 1.0)
            k_hist.append(info["neutronics"].k_eff)
            res_hist.append(change)
            if verbose:
                print(f"  coupling {n:3d}  k = {k_hist[-1]:.7f}  "
                      f"max|dT| = {change:.3e} K  T_peak = {T_new.max():.1f} K",
                      flush=True)
            T = T_new
            if change / span < tol:
                converged = True
                break
        seconds = time.perf_counter() - t_start

        return CoupledResult(
            k_eff=k_hist[-1], temperature=T, power_density=info["power"],
            flux=info["neutronics"].flux, iterations=len(k_hist),
            converged=converged, seconds=seconds, device=ctx.device,
            k_history=k_hist, residual_history=res_hist,
            thermal=info["thermal"], neutronics=info["neutronics"])


@dataclass
class CoupledTransientResult:
    times: np.ndarray
    power: np.ndarray                # P(t)/P(0), neutronics
    peak_temperature: np.ndarray     # max fuel T at each step, K
    mean_temperature: np.ndarray
    k0: float
    temperature: np.ndarray          # final field
    flux: object
    steady: object                   # the coupled steady state it started from
    seconds: float
    device: str
    steps: int
    phase_seconds: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"CoupledTransientResult(P/P0 {self.power.min():.4f} .. "
                f"{self.power.max():.4f}, peak T {self.peak_temperature.max():.1f} K, "
                f"{self.steps} steps in {self.seconds:.1f} s)")


def coupled_transient(ctx: CouplingContext, t_end, dt, *, problem_at=None,
                      dt_thermal=None, verbose=False, transient_kwargs=None,
                      solver_cls=None, precond_degree=1, check_every=4,
                      thermal_rtol=1e-8, thermal_maxiter=20000,
                      thermal_check_every=4, thermal_precond_degree=0,
                      thermal_diagnostics_every=0, profile=False):
    """March the neutronics and the thermal solution together.

    Operator splitting, one exchange per step: the neutronics step is taken
    with the cross sections evaluated at the temperature the thermal solver
    left behind, then the resulting power advances the temperature. That is
    what production coupled codes do, and it is the only sane choice here --
    a fully implicit coupling would need the eigenvalue-free flux solve and
    the conduction solve inside one Newton iteration, at several times the
    cost, to chase a splitting error that is already far below the modelling
    error in an analytic Doppler law.

    The neutronics step is set by prompt kinetics (milliseconds); the thermal
    response is seconds. Backward Euler on the conduction side is
    unconditionally stable, so ``dt_thermal`` can be a multiple of ``dt`` --
    the temperature is then advanced once every few neutronics steps with the
    power accumulated over them, which is where most of the saving is.

    problem_at : as ``TransientSolver``'s, for the driving perturbation (a
                 drum rotation, a rod). Defaults to a stationary core, i.e. the
                 transient is driven by the feedback alone.
    precond_degree, check_every : GPU-oriented defaults for the neutron
                 Neumann-PCG solves. Explicit values in
                 ``transient_kwargs['linsolve_kwargs']`` take precedence over
                 ``check_every``.
    thermal_rtol, thermal_maxiter, thermal_check_every,
    thermal_precond_degree : controls for the backward-Euler conduction solve.
                 The coupled tolerance is deliberately looser than the
                 conduction solver's verification-oriented standalone default.
    thermal_diagnostics_every : compute the exact thermal energy balance every
                 N thermal advances. Zero disables its four global reductions.
    profile : collect phase times without intermediate GPU synchronization,
                 plus step/iteration/rebuild/transfer counters. GPU phases use
                 CUDA events and emit NVTX ranges.
    """
    from .transient import TransientSolver

    xp = ctx.thermal_solver().xp
    profiler = _CoupledPhaseProfiler(xp, enabled=profile)
    dt = float(dt)
    dt_thermal = float(dt if dt_thermal is None else dt_thermal)
    if not np.isfinite(dt) or not np.isfinite(dt_thermal) or dt <= 0.0 or dt_thermal <= 0.0:
        raise ValueError("dt and dt_thermal must be finite and positive")
    every = int(round(dt_thermal / dt))
    if every < 1 or not np.isclose(every * dt, dt_thermal, rtol=1e-12,
                                   atol=1e-14 * max(1.0, dt_thermal)):
        raise ValueError("dt_thermal must be an integer multiple of dt")
    thermal_diagnostics_every = int(thermal_diagnostics_every)
    if thermal_diagnostics_every < 0:
        raise ValueError("thermal_diagnostics_every must be non-negative")
    precond_degree = int(precond_degree)
    check_every = int(check_every)
    thermal_precond_degree = int(thermal_precond_degree)
    thermal_check_every = int(thermal_check_every)
    thermal_maxiter = int(thermal_maxiter)
    if precond_degree < 0 or thermal_precond_degree < 0:
        raise ValueError("preconditioner degrees must be non-negative")
    if check_every < 1 or thermal_check_every < 1:
        raise ValueError("residual check cadences must be positive")
    if not np.isfinite(thermal_rtol) or thermal_rtol <= 0.0:
        raise ValueError("thermal_rtol must be finite and positive")
    if thermal_maxiter < 1:
        raise ValueError("thermal_maxiter must be positive")

    # Start from the converged coupled steady state, so t=0 is an equilibrium
    # of BOTH physics -- otherwise the run opens with a transient that is
    # nothing but the initial condition relaxing.
    steady = CoupledSolver(ctx).solve(tol=1e-8, anderson_depth=5)
    if not steady.converged:
        raise RuntimeError(f"coupled steady state did not converge: {steady}")

    def thermal_solver_for(step):
        return ConductionSolver(
            ctx.grid, ctx.thermal_materials, ctx.material_map,
            bc=ctx.thermal_bc, active=ctx.active, mask_bc=ctx.thermal_mask_bc,
            ambient_temperature=ctx.ambient_temperature,
            mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
            device=ctx.device, dtype=ctx.dtype, time_step=step,
            precond_degree=thermal_precond_degree)

    thermal = thermal_solver_for(dt_thermal)

    # Everything the loop touches lives on the solve device from here on.
    steady_T = np.asarray(steady.temperature)
    state = {"T": xp.asarray(steady_T, dtype=thermal.dtype),
             "peak": [], "mean": [], "accum": None, "n": 0, "hook": None,
             "min_integral": None}
    state["hook"] = ctx.feedback.hook(state["T"])
    fuel = _fuel_mask(ctx)
    fuel_dev = xp.asarray(fuel)

    def xs_update_at(_t):
        # The SAME object until the temperature actually moves. The transient
        # rebuilds its operators on identity change, so with a thermal step
        # coarser than the neutronics step this skips the rebuild on every
        # sub-step -- the cross sections genuinely have not changed.
        return state["hook"]

    # Per-group kappa*Sigma_f on the DEVICE, blended by the same rules the
    # cross sections use. Keeping this on device is what makes the GPU path
    # worth having: the host-side power_density would pull the whole flux back
    # every step (11 groups x 12k cells is ~1 MB a step, thousands of steps),
    # and that transfer would sit in series with every solve.
    kappa_xs = xp.stack(_device_fission_energy(ctx, xp, thermal.dtype))
    act_dev = None if ctx.active is None else xp.asarray(ctx.active).astype(bool)
    if act_dev is not None:
        kappa_xs = xp.where(act_dev[None, ...], kappa_xs, 0.0)
    vol = thermal.cell_volume
    raw_power = xp.zeros(ctx.shape, dtype=thermal.dtype)
    initial_peak = float(steady_T[fuel].max())
    initial_mean = float(steady_T[fuel].mean())
    state["cached_peak"], state["cached_mean"] = initial_peak, initial_mean

    def on_step(t, phi, power_ratio):
        profiler.pause_neutronics()
        try:
            with profiler.region("power_edit"):
                raw_power.fill(0)
                # One contraction kernel when the multigroup transient hands
                # us its existing contiguous flux stack. If batching was not
                # available, stack only on the fused GPU path; the CPU fallback
                # consumes the legacy group list directly.
                phi_power = (phi if not kernels.use_fused(xp, "groups") or
                             hasattr(phi, "shape") else xp.stack(phi))
                kernels.group_accumulate(xp, raw_power, kappa_xs, phi_power)
                integral = xp.sum(raw_power * vol)       # device scalar
                scale = ctx.total_power * power_ratio / integral
                q = raw_power * scale
                state["min_integral"] = (
                    integral if state["min_integral"] is None else
                    xp.minimum(state["min_integral"], integral))
                # Accumulate power over the sub-steps so a coarser thermal step
                # sees the mean source over its own interval, not a snapshot.
                state["accum"] = (q if state["accum"] is None
                                  else state["accum"] + q)
                state["n"] += 1
            profiler.counters["neutronics_steps"] += 1

            # Flush a trailing partial interval too. A constant-width thermal
            # solver is reused for full windows; the final short window gets a
            # correctly shifted backward-Euler operator.
            final_window = np.isclose(t, t_end, rtol=1e-12,
                                      atol=1e-14 * max(1.0, t_end))
            if state["n"] == every or final_window:
                width = state["n"] * dt
                stepper = (thermal if state["n"] == every
                           else thermal_solver_for(width))
                next_thermal = profiler.counters["thermal_steps"] + 1
                diagnostics = (thermal_diagnostics_every > 0 and
                               next_thermal % thermal_diagnostics_every == 0)
                with profiler.region("thermal_solve"):
                    res = stepper.step(
                        state["accum"] / state["n"], state["T"],
                        rtol=thermal_rtol, maxiter=thermal_maxiter,
                        check_every=thermal_check_every,
                        diagnostics=diagnostics, synchronize_timing=False)
                state["T"] = res.temperature
                profiler.counters["thermal_steps"] += 1
                profiler.counters["thermal_iterations"] += res.iterations
                profiler.counters["thermal_diagnostics"] += int(diagnostics)

                with profiler.region("feedback_update"):
                    state["hook"] = ctx.feedback.hook(state["T"])
                profiler.counters["feedback_updates"] += 1

                # One small transfer per thermal window supplies both
                # telemetry values and validates every device-side power
                # normalization since the previous exchange.
                with profiler.region("telemetry_transfer"):
                    packet = asnumpy(xp.stack((
                        state["min_integral"],
                        xp.max(state["T"][fuel_dev]),
                        xp.mean(state["T"][fuel_dev]))))
                profiler.counters["telemetry_transfers"] += 1
                if not np.isfinite(packet[0]) or packet[0] <= 0.0:
                    raise RuntimeError("no finite positive fission power in "
                                       "the transient flux")
                state["cached_peak"] = float(packet[1])
                state["cached_mean"] = float(packet[2])
                state["accum"], state["n"] = None, 0
                state["min_integral"] = None

            # Temperature is constant between thermal exchanges; reuse the
            # last two host scalars instead of synchronizing twice per neutron
            # step merely to rediscover the same values.
            state["peak"].append(state["cached_peak"])
            state["mean"].append(state["cached_mean"])
        finally:
            profiler.resume_neutronics()

    mats = ctx.materials
    stationary = (lambda _t: (mats, ctx.material_map,
                              ctx.mix_material, ctx.mix_weight))
    tr = TransientSolver(
        ctx.grid, problem_at or stationary, ctx.kinetics, bc=ctx.bc,
        active=ctx.active, mask_bc=ctx.mask_bc,
        mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
        group_operator=ctx.group_operator, eig_solver=ctx.solver_cls,
        device=ctx.device, dtype=ctx.dtype,
        precond_degree=precond_degree,
        xs_update_at=xs_update_at, on_step=on_step,
        phase_context=(profiler.region if profile else None))

    # Whole-core rebalance instead of Anderson, measured on the 11-group HP-MR
    # (refine 3, one 50 ms step, against a tol=1e-10 rebalanced reference):
    #
    #   depth=1 rebalance=on   tol 1e-6   -0.034%    16.7 s   112k inner
    #   depth=1 rebalance=off  tol 1e-6   -0.034%    75.3 s   535k inner
    #   depth=5 rebalance=off  tol 1e-6   -0.693%    38.5 s   273k inner
    #   depth=5 rebalance=off  tol 1e-8   -0.002%   102.8 s   652k inner
    #
    # Two things there. Rebalance cuts plain Picard 4.5x at the SAME answer
    # (the two agree to 1.2e-6), which is the property an accelerator must
    # have. And Anderson's convergence measure -- the change between successive
    # MIXED iterates -- understates the true residual on this stiff map, so at
    # the same nominal tolerance it stops 20x short; tightening to 1e-8 fixes
    # the answer but costs 6x the rebalanced run. The two also do not combine:
    # rebalance makes the map mildly nonlinear in S, which breaks Anderson's
    # affine premise, and depth 5 with rebalance fails to converge outright.
    # ...and the winner depends on the problem, so pick rather than hard-code.
    # On the two-group placeholder core the map is well conditioned and
    # Anderson is 7.5x FASTER than rebalanced Picard (2.0 s vs 15.0 s, agreeing
    # to 1.5e-4). The stiff regime is the multigroup one with upscatter, where
    # the numbers above invert. Group count is the honest proxy for "is this
    # the stiff regime"; override through transient_kwargs when it guesses
    # wrong.
    n_groups = ctx.materials[0].n_groups
    if n_groups > 2:
        # scatter_subsweeps=6 rather than the auto-selected 3: on this exact
        # problem (the 11-group HP-MR) the within-step fixed point is limited by
        # the SPECTRAL coupling, and 3 passes leave the outer iteration to carry
        # the group cascade -- ~7x the CG iterations and ~3x the wall time, at
        # twice the error. See ndgpu.benchmarks.hpmr_transient_bench.SUBSWEEPS
        # for the measured table. The solver's auto rule is left at 3 because it
        # is shared with benchmarks that have not been re-measured; this path is
        # the one the measurement was made on.
        step_kwargs = dict(rebalance=True, anderson_depth=1, max_sweeps=4000,
                           scatter_subsweeps=6)
    else:
        step_kwargs = dict(rebalance=False)
    step_kwargs.update(transient_kwargs or {})
    # The coupled fixed point has already paid for a converged hot eigenpair.
    # Hand it to the transient initializer instead of immediately solving the
    # same time-zero problem again. Callers can still override this explicitly
    # through transient_kwargs for a compatibility/convergence experiment.
    step_kwargs.setdefault("initial_steady", steady.neutronics)
    lin_kw = dict(step_kwargs.get("linsolve_kwargs") or {})
    lin_kw.setdefault("check_every", check_every)
    step_kwargs["linsolve_kwargs"] = lin_kw
    t0 = time.perf_counter()
    profiler.resume_neutronics()
    with (profiler.region("transient_total") if profile else nullcontext()):
        res = tr.solve(t_end=t_end, dt=dt, verbose=verbose, **step_kwargs)
    profiler.pause_neutronics()
    seconds = time.perf_counter() - t0

    with profiler.region("result_transfer"):
        final_temperature = asnumpy(state["T"])
    profiler.counters["result_transfers"] += 1
    profiler.counters["initial_eigen_outer_iterations"] = \
        res.steady.outer_iterations
    profiler.counters["initial_eigen_inner_iterations"] = \
        res.steady.inner_iterations
    profiler.counters["initial_state_reuses"] = int(res.initial_state_reused)
    profiler.counters["neutron_inner_iterations"] = res.total_inner_iterations
    profiler.counters["neutron_fixed_point_sweeps"] = sum(res.step_iterations)
    phase_seconds = profiler.seconds()

    return CoupledTransientResult(
        times=res.times, power=res.power,
        peak_temperature=np.array([steady.peak_temperature] + state["peak"]),
        mean_temperature=np.array(
            [float(np.asarray(steady.temperature)[fuel].mean())] + state["mean"]),
        k0=res.k0, temperature=final_temperature, flux=res.flux, steady=steady,
        seconds=seconds, device=ctx.device, steps=len(res.times) - 1,
        phase_seconds=phase_seconds, counters=dict(profiler.counters))


def _device_fission_energy(ctx, xp, dtype):
    """Per-group kappa*Sigma_f as device fields, blended like the cross
    sections so a drum cell is the same fraction of absorber to the power edit
    as it is to the neutronics."""
    from .blend import MaterialBlend
    from .power import fission_energy_xs

    table, _ = fission_energy_xs(ctx.materials)
    blend = MaterialBlend(xp, ctx.grid.shape, ctx.material_map,
                          len(ctx.materials), dtype=dtype,
                          mix_material=ctx.mix_material,
                          mix_weight=ctx.mix_weight)
    return [blend.linear(table[:, g]) for g in range(table.shape[1])]


def _fuel_mask(ctx):
    """Cells that make power -- where 'peak temperature' actually means
    something. Falls back to the active mask if no material is fissile."""
    fissile = [i for i, m in enumerate(ctx.materials) if m.is_fissile]
    mm = np.asarray(ctx.material_map)
    if not fissile:
        return (np.ones(mm.shape, bool) if ctx.active is None
                else np.asarray(ctx.active).astype(bool))
    mask = np.zeros(mm.shape, bool)
    for i in fissile:
        mask |= mm == i
    return mask


def cell_centroids(grid, raster=None) -> np.ndarray:
    """(n_cells, dim) cell centroids, C-ordered to match ``grid.shape.ravel()``.

    The mesh's own physical frame -- for the triangular lattice that means the
    rasterizer's frame, which is why ``raster`` is required there: a generic
    tri-lattice convention differs by a rotation and an offset, and would place
    every vertex on the wrong cell without erroring.
    """
    from .grid import Grid
    from .tri import TriGrid

    if isinstance(grid, TriGrid):
        if raster is None:
            raise ValueError(
                "triangular centroids need the problem's raster (its lattice "
                "origin and side); pass HpmrProblem.raster")
        from .pin_power import tri_cell_centroids
        xy = tri_cell_centroids(raster)                       # (nr, nc, 2, 2)
        if len(grid.shape) == 3:
            return np.ascontiguousarray(xy.reshape(-1, 2))
        nz = grid.shape[3]
        z = (np.arange(nz) + 0.5) * grid.dz
        out = np.empty((*grid.shape, 3))
        out[..., :2] = xy[:, :, :, None, :]
        out[..., 2] = z
        return np.ascontiguousarray(out.reshape(-1, 3))

    if isinstance(grid, Grid):
        cx, cy, cz = (grid.cell_centers(a) for a in range(3))
        if grid.geometry == "cylindrical":
            r, zz = np.meshgrid(cx, cz, indexing="ij")        # ny == 1
            return np.ascontiguousarray(np.stack([r.reshape(-1),
                                                  zz.reshape(-1)], axis=1))
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        return np.ascontiguousarray(np.stack([gx.reshape(-1), gy.reshape(-1),
                                              gz.reshape(-1)], axis=1))

    raise TypeError(f"no centroid rule for {type(grid).__name__}")


def coupling_vertices(problem):
    """(coords, flat_indices) for the ACTIVE cells, in one place.

    Both preCICE participants must call this on identically-built problems.
    The nearest-neighbour mapping is exact -- the identity permutation,
    contributing precisely zero interpolation error -- only because the two
    coordinate arrays are bit-identical. If the two sides drift (a different
    ``refine``, a different drum angle), preCICE will not complain: it will
    quietly map each vertex to a nearby wrong cell, and the coupling becomes a
    smoother. That is what the echo test in the preCICE test-suite is for.
    """
    coords = cell_centroids(problem.grid, problem.raster)
    active = np.asarray(problem.active).ravel()
    idx = np.flatnonzero(active)
    return np.ascontiguousarray(coords[idx]), idx


def uncoupled_k(ctx: CouplingContext, temperature=None) -> float:
    """k with the feedback evaluated at a frozen temperature -- the reference
    the temperature defect is measured against. ``None`` means the reference
    temperature itself, i.e. the cross sections exactly as tabulated."""
    if temperature is None:
        temperature = np.full(ctx.shape,
                              float(np.max(ctx.feedback.t_ref)))
    solver = ctx.solver_cls(
        ctx.grid, ctx.materials, ctx.material_map, bc=ctx.bc,
        active=ctx.active, mask_bc=ctx.mask_bc,
        mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
        device=ctx.device, dtype=ctx.dtype,
        xs_update=ctx.feedback.hook(temperature))
    return solver.solve(**ctx.eigen_kwargs).k_eff


def temperature_defect_pcm(k_cold: float, k_hot: float) -> float:
    """Reactivity lost between the cold, unfed core and the hot one, in pcm.
    Negative: heating a reactor costs reactivity."""
    return 1e5 * (1.0 / k_cold - 1.0 / k_hot)


def criticality_search(rebuild, ctx_kwargs, bracket=(0.0, 180.0), target_k=1.0,
                       tol=1e-6, max_iter=20, solve_kwargs=None, verbose=False):
    """Find the control-drum angle that holds the core critical at rated power.

    This is the question the coupling exists to answer. `k(T)` on its own is a
    number; the operationally meaningful statement is how far the drums have to
    rotate to pay for the temperature defect, which needs the thermal fixed
    point re-converged at every trial angle.

    rebuild    : callable ``angle_deg -> CouplingContext``, rebuilding the
                 problem (the drum rotation changes the material map and the
                 absorber volume fractions, so it is a new problem, not a new
                 parameter).
    bracket    : (angle_lo, angle_hi) in degrees, 0 = arcs inserted.

    Returns (angle_deg, CoupledResult). Secant rather than bisection: k is
    smooth and monotone in the angle once the absorber is volume-mixed
    ("polar"), so secant lands in ~4 evaluations instead of ~20 -- and each
    evaluation is a full coupled solve.
    """
    solve_kwargs = dict(solve_kwargs or {})
    cache = {}

    def k_at(angle):
        if angle not in cache:
            ctx = rebuild(angle, **ctx_kwargs)
            res = CoupledSolver(ctx).solve(**solve_kwargs)
            if not res.converged:
                raise RuntimeError(f"coupled solve did not converge at "
                                   f"{angle:.3f} deg")
            cache[angle] = res
            if verbose:
                print(f"  drum {angle:7.3f} deg -> k = {res.k_eff:.7f}  "
                      f"T_peak = {res.peak_temperature:.1f} K", flush=True)
        return cache[angle]

    a, b = float(bracket[0]), float(bracket[1])
    fa = k_at(a).k_eff - target_k
    fb = k_at(b).k_eff - target_k
    if fa * fb > 0:
        raise ValueError(
            f"k - {target_k} does not change sign across the bracket "
            f"({fa + target_k:.6f} at {a} deg, {fb + target_k:.6f} at {b} deg): "
            f"the core cannot be made critical by drum rotation alone here")

    for _ in range(max_iter):
        if abs(fb - fa) < 1e-14:
            break
        c = b - fb * (b - a) / (fb - fa)
        c = min(max(c, min(bracket)), max(bracket))
        fc = k_at(c).k_eff - target_k
        a, fa, b, fb = b, fb, c, fc
        if abs(fc) < tol:
            break
    return b, cache[b]
