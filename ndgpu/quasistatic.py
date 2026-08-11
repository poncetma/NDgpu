"""Adjoint projection and point-kinetics foundation for quasi-static solves.

The spatial flux is factored as ``phi(r, g, t) = A(t) psi(r, g, t)``.  This
module implements the inexpensive half of that split: project a diffusion
operator onto a forward/adjoint shape and march the resulting amplitude and
effective delayed-neutron precursors.  Adaptive spatial shape correction is
built on top of these primitives; keeping the primitives standalone makes
their normalization and kinetics testable against exact homogeneous limits.
"""

from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .backend import asnumpy, synchronize
from .materials import Kinetics
from .solver import Result


@dataclass(frozen=True)
class EffectiveKinetics:
    """Adjoint-projected point-kinetics parameters at one spatial state.

    ``rho`` is reactivity relative to the transient's time-zero critical
    adjustment ``k0``. ``generation_time`` is Lambda in seconds. ``beta`` and
    ``decay`` contain one value per delayed-neutron family.
    """

    rho: float
    generation_time: float
    beta: np.ndarray
    decay: np.ndarray
    time_importance: float
    fission_importance: float

    def __post_init__(self):
        beta = np.asarray(self.beta, dtype=float)
        decay = np.asarray(self.decay, dtype=float)
        if beta.ndim != 1 or decay.ndim != 1 or beta.shape != decay.shape:
            raise ValueError("effective beta and decay must be same-length vectors")
        if (not np.isfinite(self.rho)
                or not np.isfinite(self.generation_time)
                or self.generation_time <= 0.0):
            raise ValueError("rho must be finite and generation_time positive")
        if (np.any(~np.isfinite(beta)) or np.any(beta < 0.0)
                or np.any(~np.isfinite(decay)) or np.any(decay <= 0.0)):
            raise ValueError("effective delayed-neutron data are invalid")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "decay", decay)

    @property
    def beta_total(self) -> float:
        return float(self.beta.sum())


@dataclass
class PointKineticsResult:
    """Amplitude and effective precursor history from a point-kinetics march."""

    times: np.ndarray
    amplitude: np.ndarray
    precursors: np.ndarray
    rho: np.ndarray

    @property
    def power(self) -> np.ndarray:
        return self.amplitude


@dataclass
class QuasiStaticResult:
    """Coupled fixed-shape, adiabatic, or improved quasi-static result.

    ``rho`` records the reactivity projected from each current control and
    temperature operator, while ``power`` is the independently marched
    amplitude ``P(t)/P(0)``. IQS results additionally retain the accepted
    spatial precursor field and predictor-disagreement counters.
    """

    times: np.ndarray
    power: np.ndarray
    rho: np.ndarray
    peak_temperature: np.ndarray
    mean_temperature: np.ndarray
    temperature: np.ndarray
    flux: object
    adjoint_flux: object
    k0: float
    steady: object
    seconds: float
    initialization_seconds: float
    device: str
    steps: int
    effective_precursors: np.ndarray | None = None
    spatial_precursors: object = None
    shape_update_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    shape_update_reasons: list = field(default_factory=list)
    shape_k_eff: np.ndarray = field(default_factory=lambda: np.empty(0))
    residual_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    shape_residual: np.ndarray = field(default_factory=lambda: np.empty(0))
    fallback_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    phase_seconds: dict = field(default_factory=dict)
    counters: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"QuasiStaticResult(P/P0 {self.power.min():.4f} .. "
                f"{self.power.max():.4f}, peak T "
                f"{self.peak_temperature.max():.1f} K, {self.steps} steps in "
                f"{self.seconds:.1f} s)")


# Compatibility name used by the Phase-1 public API.
FixedShapeQuasiStaticResult = QuasiStaticResult


def _flux_array(value, name: str, groups: int, shape: tuple[int, ...], xp,
                dtype):
    value = value.flux if isinstance(value, Result) else value
    out = xp.asarray(value, dtype=dtype)
    expected = (groups,) + shape
    if out.shape != expected:
        raise ValueError(f"{name} shape {out.shape} != {expected}")
    if not bool(xp.all(xp.isfinite(out))):
        raise ValueError(f"{name} must be finite")
    return out


def _source_weight(solver):
    """Weight matching the discrete operator's right-hand-side convention."""
    weight = getattr(solver, "_src_weight", None)
    return 1.0 if weight is None else weight


def _loss_apply(solver, flux):
    """Apply M = leakage + removal - in-scatter to a scalar flux shape."""
    weight = _source_weight(solver)
    out = []
    for g in range(solver.n_groups):
        value = solver.ops[g].apply(flux[g])
        for gf in range(solver.n_groups):
            scatter = solver.sigma_s[gf][g]
            if gf != g and scatter is not None:
                value = value - weight * scatter * flux[gf]
        out.append(value)
    return out


def _mapped_kinetics(solver, kinetics: Kinetics):
    """Map material-dependent speeds/beta through the solver's volume blend."""
    fields = solver.fields
    groups = solver.n_groups
    if kinetics.velocities.shape[-1] != groups:
        raise ValueError(f"kinetics needs {groups} group velocities")
    if kinetics.velocities.ndim == 2:
        inv_velocity = [fields.map_table(1.0 / kinetics.velocities[:, g])
                        for g in range(groups)]
    else:
        inv_velocity = [1.0 / float(kinetics.velocities[g])
                        for g in range(groups)]
    if kinetics.beta.ndim == 2:
        beta = [fields.map_table_fission_weighted(kinetics.beta[:, i])
                for i in range(kinetics.n_families)]
    else:
        beta = [float(kinetics.beta[i]) for i in range(kinetics.n_families)]
    return inv_velocity, beta


def project_effective_kinetics(solver, forward, adjoint, kinetics: Kinetics,
                               *, k0: float | None = None) -> EffectiveKinetics:
    """Project the current diffusion operator onto a forward/adjoint shape.

    ``solver`` supplies the *current* material/control/temperature operator.
    ``forward`` and ``adjoint`` are the most recent shape anchor and may be
    either solver :class:`Result` objects or group-first arrays. ``k0`` is the
    physical eigenvalue used for the transient critical adjustment; when
    omitted it is read from a forward ``Result``.

    Reusing an anchor against a nearby operator gives the local Rayleigh
    reactivity estimate used between shape corrections. Large black-absorber
    moves must be split into small cached frames and periodically re-anchored.
    """
    if not isinstance(kinetics, Kinetics):
        raise TypeError("kinetics must be a Kinetics object")
    if k0 is None:
        if not isinstance(forward, Result):
            raise ValueError("k0 is required when forward is not a Result")
        k0 = float(forward.k_eff)
    k0 = float(k0)
    if not np.isfinite(k0) or k0 <= 0.0:
        raise ValueError("k0 must be finite and positive")

    xp = solver.xp
    groups = solver.n_groups
    shape = tuple(solver.grid.shape)
    phi = _flux_array(forward, "forward flux", groups, shape, xp, solver.dtype)
    star = _flux_array(adjoint, "adjoint flux", groups, shape, xp, solver.dtype)
    weight = _source_weight(solver)
    inv_velocity, beta_fields = _mapped_kinetics(solver, kinetics)

    fission_source = solver.nu_sigma_f[0] * phi[0]
    for g in range(1, groups):
        fission_source = fission_source + solver.nu_sigma_f[g] * phi[g]

    loss = _loss_apply(solver, phi)
    loss_importance = 0.0
    fission_importance = 0.0
    time_importance = 0.0
    for g in range(groups):
        loss_importance = loss_importance + xp.sum(star[g] * loss[g])
        fission_importance = fission_importance + xp.sum(
            star[g] * weight * solver.chi[g] * fission_source)
        time_importance = time_importance + xp.sum(
            star[g] * weight * inv_velocity[g] * phi[g])
    loss_importance = float(loss_importance)
    fission_importance = float(fission_importance)
    time_importance = float(time_importance)
    if fission_importance <= 0.0 or time_importance <= 0.0:
        raise ValueError("forward/adjoint shapes have non-positive importance")

    beta_eff = []
    chi_delayed = kinetics.chi_delayed
    for i, beta_i in enumerate(beta_fields):
        delayed_importance = 0.0
        for g in range(groups):
            if chi_delayed is None:
                spectrum = solver.chi[g]
            elif chi_delayed.ndim == 1:
                spectrum = float(chi_delayed[g])
            else:
                spectrum = float(chi_delayed[i, g])
            delayed_importance = delayed_importance + xp.sum(
                star[g] * weight * spectrum * beta_i * fission_source)
        beta_eff.append(float(delayed_importance) / fission_importance)

    rho = 1.0 - k0 * loss_importance / fission_importance
    generation_time = k0 * time_importance / fission_importance
    return EffectiveKinetics(
        rho=float(rho), generation_time=float(generation_time),
        beta=np.asarray(beta_eff), decay=kinetics.decay.copy(),
        time_importance=time_importance,
        fission_importance=fission_importance)


def projected_shape_residual(solver, forward, adjoint) -> float:
    """Relative eigen-shape defect with the amplitude mode projected out.

    The current operator is applied to ``forward`` and its fission component is
    scaled so the residual is orthogonal to ``adjoint``. A pure normalization
    or eigenvalue change therefore reports zero; only spatial/energy-shape
    mismatch remains. This is the trigger used by the quasi-static driver.
    """
    xp = solver.xp
    groups = solver.n_groups
    shape = tuple(solver.grid.shape)
    phi = _flux_array(forward, "forward flux", groups, shape, xp, solver.dtype)
    star = _flux_array(adjoint, "adjoint flux", groups, shape, xp, solver.dtype)
    weight = _source_weight(solver)
    loss = _loss_apply(solver, phi)
    fission_source = solver.nu_sigma_f[0] * phi[0]
    for g in range(1, groups):
        fission_source = fission_source + solver.nu_sigma_f[g] * phi[g]
    fission = [weight * solver.chi[g] * fission_source
               for g in range(groups)]
    loss_importance = sum(xp.sum(star[g] * loss[g]) for g in range(groups))
    fission_importance = sum(
        xp.sum(star[g] * fission[g]) for g in range(groups))
    fission_importance = float(fission_importance)
    if fission_importance <= 0.0:
        raise ValueError("forward/adjoint shapes have non-positive importance")
    alpha = float(loss_importance) / fission_importance
    residual_norm = 0.0
    reference_norm = 0.0
    for g in range(groups):
        residual = loss[g] - alpha * fission[g]
        residual_norm = residual_norm + xp.sum(residual * residual)
        reference_norm = reference_norm + xp.sum(loss[g] * loss[g])
    reference_norm = float(reference_norm)
    if reference_norm <= 0.0:
        raise ValueError("shape has a non-positive operator norm")
    return float(np.sqrt(float(residual_norm) / reference_norm))


def _project_spatial_precursors(solver, adjoint, spatial_precursors,
                                kinetics, time_importance):
    """Adjoint-project spatial precursor fields into amplitude coordinates."""
    xp = solver.xp
    star = _flux_array(adjoint, "adjoint flux", solver.n_groups,
                       tuple(solver.grid.shape), xp, solver.dtype)
    fields = xp.asarray(spatial_precursors, dtype=solver.dtype)
    expected = (kinetics.n_families,) + tuple(solver.grid.shape)
    if fields.shape != expected:
        raise ValueError(f"spatial precursor shape {fields.shape} != {expected}")
    weight = _source_weight(solver)
    out = []
    for i in range(kinetics.n_families):
        importance = 0.0
        for g in range(solver.n_groups):
            if kinetics.chi_delayed is None:
                spectrum = solver.chi[g]
            elif kinetics.chi_delayed.ndim == 1:
                spectrum = float(kinetics.chi_delayed[g])
            else:
                spectrum = float(kinetics.chi_delayed[i, g])
            importance = importance + xp.sum(
                star[g] * weight * spectrum * fields[i])
        out.append(float(importance) / time_importance)
    projected = np.asarray(out)
    if np.any(~np.isfinite(projected)) or np.any(projected < 0.0):
        raise RuntimeError("spatial precursor projection is invalid")
    return projected


def equilibrium_precursors(parameters: EffectiveKinetics,
                           amplitude: float = 1.0) -> np.ndarray:
    """Effective precursor amplitudes for a critical steady population."""
    amplitude = float(amplitude)
    return (parameters.beta / (parameters.generation_time * parameters.decay)
            * amplitude)


def _step_count(t_end: float, dt: float) -> int:
    if not np.isfinite(t_end) or not np.isfinite(dt) or t_end < 0.0 or dt <= 0.0:
        raise ValueError("t_end must be non-negative and dt positive")
    steps = int(round(t_end / dt))
    if not np.isclose(steps * dt, t_end, rtol=1e-12,
                      atol=1e-14 * max(1.0, t_end)):
        raise ValueError("t_end must be an integer multiple of dt")
    return steps


def advance_point_kinetics(parameters: EffectiveKinetics, dt: float,
                           amplitude: float, precursors) -> tuple[float, np.ndarray]:
    """Advance one backward-Euler point-kinetics step.

    The state is deliberately host-resident: even with dozens of precursor
    families this is a tiny dense system, while the projected spatial fields
    and coupled temperature remain on the accelerator.
    """
    if not isinstance(parameters, EffectiveKinetics):
        raise TypeError("parameters must be EffectiveKinetics")
    dt = float(dt)
    amplitude = float(amplitude)
    precursors = np.asarray(precursors, dtype=float)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude must be finite and positive")
    if (precursors.shape != parameters.beta.shape
            or np.any(~np.isfinite(precursors))):
        raise ValueError("precursors must match the delayed-neutron families")

    families = len(parameters.beta)
    matrix = np.zeros((1 + families, 1 + families))
    matrix[0, 0] = ((parameters.rho - parameters.beta_total)
                    / parameters.generation_time)
    matrix[0, 1:] = parameters.decay
    matrix[1:, 0] = parameters.beta / parameters.generation_time
    matrix[1:, 1:] = -np.diag(parameters.decay)
    old = np.concatenate(([amplitude], precursors))
    state = np.linalg.solve(np.eye(len(old)) - dt * matrix, old)
    if np.any(~np.isfinite(state)) or state[0] <= 0.0:
        raise RuntimeError("point kinetics produced an invalid state")
    return float(state[0]), state[1:]


def integrate_point_kinetics(parameters_at: EffectiveKinetics | Callable,
                             t_end: float, dt: float, *, amplitude0: float = 1.0,
                             precursors0=None) -> PointKineticsResult:
    """Backward-Euler march of the projected amplitude/precursor equations.

    ``parameters_at`` is either one :class:`EffectiveKinetics` object or an
    end-of-step callable. Parameter changes do not reset precursors. This is
    intentionally the same first-order time convention as ``TransientSolver``
    so fixed-shape comparisons isolate the spatial approximation rather than a
    time-integrator difference.
    """
    steps = _step_count(t_end, dt)
    get = (parameters_at if callable(parameters_at)
           else lambda _t: parameters_at)
    p0 = get(0.0)
    if not isinstance(p0, EffectiveKinetics):
        raise TypeError("parameters_at must return EffectiveKinetics")
    amplitude0 = float(amplitude0)
    if not np.isfinite(amplitude0) or amplitude0 <= 0.0:
        raise ValueError("amplitude0 must be finite and positive")
    precursors = (equilibrium_precursors(p0, amplitude0) if precursors0 is None
                  else np.asarray(precursors0, dtype=float).copy())
    if precursors.shape != p0.beta.shape or np.any(~np.isfinite(precursors)):
        raise ValueError("precursors0 must match the delayed-neutron families")

    y = np.concatenate(([amplitude0], precursors))
    times = np.arange(steps + 1, dtype=float) * dt
    amplitude = [amplitude0]
    precursor_history = [precursors.copy()]
    rho_history = [p0.rho]
    for n in range(1, steps + 1):
        p = get(float(times[n]))
        if not isinstance(p, EffectiveKinetics):
            raise TypeError("parameters_at must return EffectiveKinetics")
        if p.beta.shape != p0.beta.shape or not np.array_equal(p.decay, p0.decay):
            raise ValueError("delayed family count/decay constants changed in time")
        next_amplitude, next_precursors = advance_point_kinetics(
            p, dt, y[0], y[1:])
        y = np.concatenate(([next_amplitude], next_precursors))
        amplitude.append(float(y[0]))
        precursor_history.append(y[1:].copy())
        rho_history.append(p.rho)
    return PointKineticsResult(
        times=times, amplitude=np.asarray(amplitude),
        precursors=np.asarray(precursor_history), rho=np.asarray(rho_history))


def _unpack_problem(spec, mix_material, mix_weight):
    if len(spec) == 2:
        materials, material_map = spec
        return materials, material_map, mix_material, mix_weight
    if len(spec) == 4:
        return tuple(spec)
    raise ValueError("problem_at must return (materials, material_map) or "
                     "(materials, material_map, mix_material, mix_weight), "
                     f"got {len(spec)} elements")


def _coupled_quasistatic(
        ctx, t_end, dt, *, problem_at=None, dt_thermal=None,
        initial_coupled=None, steady_kwargs=None, shape_kwargs=None,
        adjoint_kwargs=None,
        thermal_rtol=1e-8, thermal_maxiter=20000,
        thermal_check_every=4, thermal_precond_degree=0,
        thermal_diagnostics_every=0, profile=False, shape_dt=None,
        adjoint_every=1, shape_on_final=True, shape_method="adiabatic",
        residual_tol=None, fallback_residual=None, iqs_kwargs=None,
        iqs_precond_degree=1):
    """Shared fixed-shape, adiabatic, and IQS coupled implementation.

    An initial adjoint eigenvalue solve establishes the shape importance. At
    changed control or thermal-feedback states, the current operator is
    projected onto the anchor and the resulting point-kinetics amplitude
    drives the backward-Euler conduction solver. Adiabatic mode periodically
    replaces the anchor with an eigen shape; IQS mode instead advances the
    time-dependent spatial shape and precursor fields over each macro step.
    Optional projected-residual thresholds can shorten that macro step or
    replace a fine interval with the full spatial diffusion solve.

    ``problem_at(0)`` must describe the same physical core as ``ctx`` because
    the converged coupled state supplies the time-zero forward shape. Return
    cached material/map/blend objects between actual control changes so the
    driver can avoid redundant operator construction. ``initial_coupled`` can
    reuse a previously computed converged :class:`CoupledResult`.

    With ``shape_dt=None`` the time-zero spatial shape remains fixed. Otherwise
    ``shape_method`` selects either periodic adiabatic eigen re-anchoring or a
    time-dependent IQS corrector.
    """
    from .coupling import (CoupledResult, CoupledSolver, _CoupledPhaseProfiler,
                           _fuel_mask)
    from .thermal import ConductionSolver

    if ctx.kinetics is None:
        raise ValueError("coupled quasi-static transients require ctx.kinetics")
    steps = _step_count(float(t_end), float(dt))
    t_end, dt = float(t_end), float(dt)
    dt_thermal = float(dt if dt_thermal is None else dt_thermal)
    if not np.isfinite(dt_thermal) or dt_thermal <= 0.0:
        raise ValueError("dt_thermal must be finite and positive")
    every = int(round(dt_thermal / dt))
    if every < 1 or not np.isclose(every * dt, dt_thermal, rtol=1e-12,
                                   atol=1e-14 * max(1.0, dt_thermal)):
        raise ValueError("dt_thermal must be an integer multiple of dt")
    if shape_dt is None:
        shape_every = None
    else:
        shape_dt = float(shape_dt)
        if not np.isfinite(shape_dt) or shape_dt <= 0.0:
            raise ValueError("shape_dt must be finite and positive")
        shape_every = int(round(shape_dt / dt))
        if (shape_every < 1
                or not np.isclose(shape_every * dt, shape_dt, rtol=1e-12,
                                  atol=1e-14 * max(1.0, shape_dt))):
            raise ValueError("shape_dt must be an integer multiple of dt")
    adjoint_every = int(adjoint_every)
    if adjoint_every < 1:
        raise ValueError("adjoint_every must be positive")
    shape_on_final = bool(shape_on_final)
    shape_method = str(shape_method).lower()
    if shape_method not in ("adiabatic", "iqs"):
        raise ValueError("shape_method must be 'adiabatic' or 'iqs'")
    if shape_every is None and shape_method == "iqs":
        raise ValueError("IQS treatment requires shape_dt")
    residual_tol = (None if residual_tol is None else float(residual_tol))
    fallback_residual = (None if fallback_residual is None
                         else float(fallback_residual))
    for name, value in (("residual_tol", residual_tol),
                        ("fallback_residual", fallback_residual)):
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if (residual_tol is not None and fallback_residual is not None
            and fallback_residual < residual_tol):
        raise ValueError("fallback_residual must be >= residual_tol")
    iqs_precond_degree = int(iqs_precond_degree)
    if iqs_precond_degree < 0:
        raise ValueError("iqs_precond_degree must be non-negative")
    thermal_maxiter = int(thermal_maxiter)
    thermal_check_every = int(thermal_check_every)
    thermal_precond_degree = int(thermal_precond_degree)
    thermal_diagnostics_every = int(thermal_diagnostics_every)
    if thermal_maxiter < 1 or thermal_check_every < 1:
        raise ValueError("thermal iteration controls must be positive")
    if thermal_precond_degree < 0 or thermal_diagnostics_every < 0:
        raise ValueError("thermal precondition/diagnostic controls must be non-negative")
    if not np.isfinite(thermal_rtol) or thermal_rtol <= 0.0:
        raise ValueError("thermal_rtol must be finite and positive")
    reserved = {"adjoint", "state0"} & (shape_kwargs or {}).keys()
    if reserved:
        raise ValueError("shape_kwargs cannot set "
                         + ", ".join(sorted(reserved)))

    xp = ctx.thermal_solver().xp
    profiler = _CoupledPhaseProfiler(xp, enabled=profile)
    counters = profiler.counters
    if initial_coupled is None:
        with profiler.region("initial_coupled_solve"):
            options = {"tol": 1e-8, "anderson_depth": 5}
            options.update(steady_kwargs or {})
            steady = CoupledSolver(ctx).solve(**options)
    else:
        if not isinstance(initial_coupled, CoupledResult):
            raise TypeError("initial_coupled must be a CoupledResult")
        steady = initial_coupled
        counters["initial_state_reuses"] += 1
    if not steady.converged:
        raise RuntimeError(f"coupled steady state did not converge: {steady}")

    synchronize(xp)
    initialization_start = time.perf_counter()

    stationary = lambda _t: (ctx.materials, ctx.material_map,
                              ctx.mix_material, ctx.mix_weight)
    get_problem = problem_at or stationary
    spec0 = _unpack_problem(get_problem(0.0), ctx.mix_material, ctx.mix_weight)
    temperature = xp.asarray(steady.temperature, dtype=ctx.dtype)
    feedback_hook = ctx.feedback.hook(temperature)

    def build_solver(spec, hook):
        mats, mmap, mix_m, mix_w = spec
        return ctx.solver_cls(
            ctx.grid, mats, mmap, bc=ctx.bc, active=ctx.active,
            mask_bc=ctx.mask_bc, mix_material=mix_m, mix_weight=mix_w,
            device=ctx.device, dtype=ctx.dtype, xs_update=hook)

    with profiler.region("operator_rebuild"):
        anchor_solver = build_solver(spec0, feedback_hook)
    counters["operator_rebuilds"] += 1
    adjoint_options = dict(ctx.eigen_kwargs)
    adjoint_options.update(adjoint_kwargs or {})
    reserved = {"adjoint", "state0"} & adjoint_options.keys()
    if reserved:
        raise ValueError("adjoint_kwargs cannot set "
                         + ", ".join(sorted(reserved)))
    with profiler.region("initial_adjoint_solve"):
        adjoint = anchor_solver.solve(adjoint=True, **adjoint_options)
    counters["adjoint_eigen_solves"] += 1
    if not adjoint.converged:
        raise RuntimeError(f"initial adjoint solve did not converge: {adjoint}")

    forward_flux = steady.neutronics.flux
    with profiler.region("reactivity_projection"):
        anchor_raw = project_effective_kinetics(
            anchor_solver, forward_flux, adjoint.flux, ctx.kinetics,
            k0=steady.k_eff)
    counters["reactivity_projections"] += 1
    # The final fixed-point temperature and the neutron result retained by the
    # coupled steady solver differ only by its convergence tolerance. Remove
    # that tiny projected mismatch so the supplied equilibrium is exactly the
    # zero-reactivity amplitude initial condition.
    rho_reference = anchor_raw.rho
    parameters = replace(anchor_raw, rho=0.0)
    amplitude = 1.0
    precursors = equilibrium_precursors(parameters, amplitude)

    metric = getattr(ctx.grid, "cylindrical_metrics", lambda: None)()
    volume_weight = (None if metric is None
                     else xp.asarray(metric[0], dtype=ctx.dtype))

    def normalized_neutron_shape(solver, flux):
        phi = xp.asarray(flux, dtype=ctx.dtype)
        source = solver.fields.fission_source([phi[g]
                                               for g in range(solver.n_groups)])
        source = source / steady.k_eff
        total = xp.sum(source if volume_weight is None
                       else source * volume_weight)
        total = float(total)
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("shape has no finite positive fission source")
        return phi / total, source / total, total

    forward_flux, initial_source_shape, _ = normalized_neutron_shape(
        anchor_solver, forward_flux)
    _, initial_beta_fields = _mapped_kinetics(anchor_solver, ctx.kinetics)
    spatial_precursor_shape = xp.stack([
        initial_beta_fields[i] / ctx.kinetics.decay[i] * initial_source_shape
        for i in range(ctx.kinetics.n_families)])

    def time_dependent_corrector(start_spec, start_hook, end_spec, end_hook,
                                 width, start_flux, start_precursors,
                                 substeps=1):
        """One IQS macro solve, or a fine-step full-diffusion fallback."""
        from .transient import TransientSolver

        substeps = int(substeps)
        if substeps < 1:
            raise ValueError("corrector substeps must be positive")
        start_result = copy.copy(steady.neutronics)
        start_result.k_eff = steady.k_eff
        start_result.flux = xp.asarray(start_flux, dtype=ctx.dtype)

        def local_problem(t):
            return start_spec if t <= 0.0 else end_spec

        def local_feedback(t):
            return start_hook if t <= 0.0 else end_hook

        corrector = TransientSolver(
            ctx.grid, local_problem, ctx.kinetics, bc=ctx.bc,
            active=ctx.active, mask_bc=ctx.mask_bc,
            mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
            group_operator=ctx.group_operator, eig_solver=ctx.solver_cls,
            device=ctx.device, dtype=ctx.dtype,
            precond_degree=iqs_precond_degree,
            xs_update_at=local_feedback)
        groups = ctx.materials[0].n_groups
        options = ({"rebalance": True, "anderson_depth": 1,
                    "max_sweeps": 4000, "scatter_subsweeps": 6}
                   if groups > 2 else {"rebalance": False})
        options["linsolve_kwargs"] = {"check_every": 4}
        options.update(iqs_kwargs or {})
        result = corrector.solve(
            t_end=width, dt=width / substeps,
            initial_steady=start_result,
            initial_precursors=start_precursors, **options)
        predicted = float(result.power[-1])
        if not np.isfinite(predicted) or predicted <= 0.0:
            raise RuntimeError("IQS corrector produced invalid amplitude")
        return (result, result.flux / predicted,
                result.precursors / predicted, predicted)

    def thermal_solver_for(width):
        return ConductionSolver(
            ctx.grid, ctx.thermal_materials, ctx.material_map,
            bc=ctx.thermal_bc, active=ctx.active,
            mask_bc=ctx.thermal_mask_bc,
            ambient_temperature=ctx.ambient_temperature,
            mix_material=ctx.mix_material, mix_weight=ctx.mix_weight,
            device=ctx.device, dtype=ctx.dtype, time_step=width,
            precond_degree=thermal_precond_degree)

    thermal = thermal_solver_for(dt_thermal)
    volume = thermal.cell_volume
    fuel = _fuel_mask(ctx)
    fuel_dev = xp.asarray(fuel)
    active_dev = (None if ctx.active is None
                  else xp.asarray(ctx.active).astype(bool))

    def normalized_power_shape(spec, flux):
        from .blend import MaterialBlend
        from .power import fission_energy_xs

        mats, mmap, mix_m, mix_w = spec
        table, _ = fission_energy_xs(mats)
        blend = MaterialBlend(
            xp, ctx.grid.shape, mmap, len(mats), dtype=thermal.dtype,
            mix_material=mix_m, mix_weight=mix_w)
        kappa_xs = xp.stack(
            [blend.linear(table[:, g]) for g in range(table.shape[1])])
        flux_stack = flux if hasattr(flux, "shape") else xp.stack(flux)
        raw = xp.sum(kappa_xs * flux_stack, axis=0)
        if active_dev is not None:
            raw = xp.where(active_dev, raw, 0.0)
        integral = float(xp.sum(raw * volume))
        if not np.isfinite(integral) or integral <= 0.0:
            raise RuntimeError("no finite positive fission power in the anchor flux")
        return raw / integral

    with profiler.region("power_shape"):
        unit_power_shape = normalized_power_shape(spec0, forward_flux)

    steady_temperature = np.asarray(steady.temperature)
    cached_peak = float(steady_temperature[fuel].max())
    cached_mean = float(steady_temperature[fuel].mean())
    times = np.arange(steps + 1, dtype=float) * dt
    powers = [amplitude]
    rhos = [0.0]
    peaks = [cached_peak]
    means = [cached_mean]
    power_accum = None
    window_steps = 0
    shape_times = []
    shape_reasons = []
    shape_ks = [float(steady.k_eff)]
    shape_state = getattr(ctx, "_state", None)
    adjoint_state = anchor_solver.state
    shape_updates = 0
    residual_times = []
    residual_values = []
    fallback_times = []
    spatial_precursors = spatial_precursor_shape.copy()
    anchor_spec = spec0
    anchor_hook = feedback_hook
    anchor_flux = forward_flux
    anchor_precursors = spatial_precursor_shape
    anchor_time = 0.0
    anchor_amplitude = amplitude
    current_solver = anchor_solver
    previous_spec = spec0
    previous_hook = feedback_hook
    # Object identity is the cache contract used by TransientSolver too.
    current_key = tuple(id(item) for item in spec0) + (id(feedback_hook),)

    synchronize(xp)
    initialization_seconds = time.perf_counter() - initialization_start
    march_start = time.perf_counter()
    march_context = (profiler.region("quasistatic_total")
                     if profile else nullcontext())
    with march_context:
        for n in range(1, steps + 1):
            t = float(times[n])
            force_shape = False
            fallback_step = False
            spec = _unpack_problem(get_problem(t), ctx.mix_material,
                                   ctx.mix_weight)
            key = tuple(id(item) for item in spec) + (id(feedback_hook),)
            if key != current_key:
                with profiler.region("operator_rebuild"):
                    current_solver = build_solver(spec, feedback_hook)
                counters["operator_rebuilds"] += 1
                with profiler.region("reactivity_projection"):
                    raw = project_effective_kinetics(
                        current_solver, forward_flux, adjoint.flux,
                        ctx.kinetics, k0=steady.k_eff)
                counters["reactivity_projections"] += 1
                parameters = replace(raw, rho=raw.rho - rho_reference)
                if residual_tol is not None or fallback_residual is not None:
                    with profiler.region("shape_residual"):
                        residual = projected_shape_residual(
                            current_solver, forward_flux, adjoint.flux)
                    residual_times.append(t)
                    residual_values.append(residual)
                    counters["residual_evaluations"] += 1
                    force_shape = (residual_tol is not None
                                   and residual >= residual_tol)
                    fallback_step = (fallback_residual is not None
                                     and residual >= fallback_residual)
                if (parameters.beta.shape != precursors.shape
                        or not np.array_equal(parameters.decay,
                                              anchor_raw.decay)):
                    raise ValueError("delayed family count/decay constants "
                                     "changed in time")
                current_key = key

            amplitude_before = amplitude
            if fallback_step:
                # The residual says the frozen shape is outside the trusted
                # QS neighborhood. Advance this fine interval with the full
                # spatial diffusion equations and carry their precursor field.
                with profiler.region("full_diffusion_fallback"):
                    fallback, new_flux, new_precursor_shape, predicted = \
                        time_dependent_corrector(
                            previous_spec, previous_hook, spec, feedback_hook,
                            dt, forward_flux,
                            spatial_precursors / amplitude_before, substeps=1)
                _, precursors = advance_point_kinetics(
                    parameters, dt, amplitude_before, precursors)
                amplitude = amplitude_before * predicted
                forward_flux, _, precursor_scale = normalized_neutron_shape(
                    current_solver, new_flux)
                spatial_precursor_shape = new_precursor_shape / precursor_scale
                spatial_precursors = amplitude * spatial_precursor_shape
                with profiler.region("power_shape"):
                    unit_power_shape = normalized_power_shape(spec, forward_flux)
                fallback_times.append(t)
                counters["full_diffusion_fallbacks"] += 1
                counters["fallback_inner_iterations"] += \
                    fallback.total_inner_iterations
                counters["fallback_fixed_point_sweeps"] += sum(
                    fallback.step_iterations)
            else:
                with profiler.region("amplitude_solve"):
                    amplitude, precursors = advance_point_kinetics(
                        parameters, dt, amplitude, precursors)
                # Maintain an actual spatial precursor field between IQS
                # corrections. The macro corrector replaces this approximation
                # with its implicit spatial history at the next shape event.
                source_shape = current_solver.fields.fission_source(
                    [forward_flux[g] for g in range(current_solver.n_groups)])
                source_shape = source_shape / steady.k_eff
                _, beta_fields = _mapped_kinetics(current_solver, ctx.kinetics)
                for i in range(ctx.kinetics.n_families):
                    spatial_precursors[i] = (
                        spatial_precursors[i]
                        + dt * beta_fields[i] * amplitude * source_shape
                    ) / (1.0 + ctx.kinetics.decay[i] * dt)
            counters["amplitude_steps"] += 1
            powers.append(amplitude)
            rhos.append(parameters.rho)
            with profiler.region("power_edit"):
                weighted_shape = amplitude * unit_power_shape
                power_accum = (weighted_shape if power_accum is None
                               else power_accum + weighted_shape)
            window_steps += 1

            final_window = n == steps
            if window_steps == every or final_window:
                width = window_steps * dt
                stepper = (thermal if window_steps == every
                           else thermal_solver_for(width))
                power_density = ctx.total_power * power_accum / window_steps
                next_thermal = counters["thermal_steps"] + 1
                diagnostics = (thermal_diagnostics_every > 0 and
                               next_thermal % thermal_diagnostics_every == 0)
                with profiler.region("thermal_solve"):
                    thermal_result = stepper.step(
                        power_density, temperature, rtol=thermal_rtol,
                        maxiter=thermal_maxiter,
                        check_every=thermal_check_every,
                        diagnostics=diagnostics,
                        synchronize_timing=False)
                temperature = thermal_result.temperature
                counters["thermal_steps"] += 1
                counters["thermal_iterations"] += thermal_result.iterations
                counters["thermal_diagnostics"] += int(diagnostics)
                with profiler.region("feedback_update"):
                    feedback_hook = ctx.feedback.hook(temperature)
                counters["feedback_updates"] += 1
                with profiler.region("telemetry_transfer"):
                    packet = asnumpy(xp.stack((
                        xp.max(temperature[fuel_dev]),
                        xp.mean(temperature[fuel_dev]))))
                counters["telemetry_transfers"] += 1
                cached_peak, cached_mean = map(float, packet)
                power_accum = None
                window_steps = 0

            due_interval = (shape_every is not None
                            and n % shape_every == 0)
            due_final = (shape_every is not None and final_window
                         and shape_on_final and not due_interval)
            due_residual = force_shape and not fallback_step
            if due_interval or due_final or due_residual or fallback_step:
                # Re-anchor against the end-of-step control and temperature
                # state. Amplitude and effective precursor values remain
                # continuous; normalizing the new power shape to unit integral
                # makes total fission power continuous as well.
                with profiler.region("shape_operator_rebuild"):
                    shape_solver = build_solver(spec, feedback_hook)
                counters["operator_rebuilds"] += 1
                predicted = 1.0
                precursors_follow_flux = True
                if fallback_step:
                    # The fine-step fallback already supplied the new shape and
                    # spatial precursor history. Only re-anchor its projection
                    # against feedback updated by the thermal half-step.
                    reason = "full_diffusion_fallback"
                    shape_state = [forward_flux[g].copy()
                                   for g in range(shape_solver.n_groups)]
                elif shape_method == "iqs":
                    width = t - anchor_time
                    with profiler.region("iqs_shape_solve"):
                        corrected, forward_flux, _corrected_precursors, predicted = \
                            time_dependent_corrector(
                                anchor_spec, anchor_hook, spec, feedback_hook,
                                width, anchor_flux, anchor_precursors,
                                substeps=1)
                    counters["iqs_shape_solves"] += 1
                    counters["iqs_inner_iterations"] += \
                        corrected.total_inner_iterations
                    counters["iqs_fixed_point_sweeps"] += sum(
                        corrected.step_iterations)
                    # The corrector's precursor field followed its independent
                    # coarse amplitude prediction.  Preserve the inventory
                    # accumulated with the accepted fine point-amplitude
                    # history instead of replacing it with that different
                    # history merely because the spatial shape was corrected.
                    spatial_precursor_shape = spatial_precursors / amplitude
                    precursors_follow_flux = False
                    shape_state = [forward_flux[g].copy()
                                   for g in range(shape_solver.n_groups)]
                    predicted_end = anchor_amplitude * predicted
                    disagreement = abs(predicted_end - amplitude) / amplitude
                    counters["iqs_predictor_checks"] += 1
                    counters["iqs_max_amplitude_error_ppm"] = max(
                        counters["iqs_max_amplitude_error_ppm"],
                        int(round(1e6 * disagreement)))
                    reason = ("residual" if due_residual else
                              "maximum_interval" if due_interval else
                              "final_state")
                else:
                    precursors_follow_flux = False
                    forward_options = dict(ctx.eigen_kwargs)
                    forward_options.update(shape_kwargs or {})
                    with profiler.region("forward_shape_solve"):
                        forward = shape_solver.solve(
                            state0=shape_state, **forward_options)
                    counters["forward_shape_solves"] += 1
                    counters["forward_shape_outer_iterations"] += \
                        forward.outer_iterations
                    counters["forward_shape_inner_iterations"] += \
                        forward.inner_iterations
                    if not forward.converged:
                        raise RuntimeError(
                            f"quasi-static forward shape did not converge: {forward}")
                    shape_state = shape_solver.state
                    forward_flux = forward.flux
                    spatial_precursor_shape = spatial_precursors / amplitude
                    reason = ("residual" if due_residual else
                              "maximum_interval" if due_interval else
                              "final_state")
                # Feedback can change at the thermal exchange immediately
                # before this re-anchor. Normalize flux and precursor shape
                # together under that exact end operator.
                forward_flux, _, precursor_scale = normalized_neutron_shape(
                    shape_solver, forward_flux)
                if precursors_follow_flux:
                    spatial_precursor_shape = (spatial_precursor_shape
                                               / precursor_scale)
                spatial_precursors = amplitude * spatial_precursor_shape
                shape_updates += 1

                if (fallback_step
                        or shape_updates % adjoint_every == 0):
                    with profiler.region("adjoint_shape_solve"):
                        adjoint = shape_solver.solve(
                            adjoint=True, state0=adjoint_state,
                            **adjoint_options)
                    counters["adjoint_eigen_solves"] += 1
                    counters["adjoint_shape_outer_iterations"] += \
                        adjoint.outer_iterations
                    counters["adjoint_shape_inner_iterations"] += \
                        adjoint.inner_iterations
                    if not adjoint.converged:
                        raise RuntimeError(
                            f"quasi-static adjoint shape did not converge: {adjoint}")
                    adjoint_state = shape_solver.state

                with profiler.region("reactivity_projection"):
                    raw = project_effective_kinetics(
                        shape_solver, forward_flux, adjoint.flux,
                        ctx.kinetics, k0=steady.k_eff)
                counters["reactivity_projections"] += 1
                parameters = replace(raw, rho=raw.rho - rho_reference)
                precursors = _project_spatial_precursors(
                    shape_solver, adjoint.flux, spatial_precursors,
                    ctx.kinetics, raw.time_importance)
                with profiler.region("power_shape"):
                    unit_power_shape = normalized_power_shape(spec, forward_flux)
                current_key = (tuple(id(item) for item in spec)
                               + (id(feedback_hook),))
                shape_times.append(t)
                shape_reasons.append(reason)
                denominator = 1.0 - raw.rho
                shape_ks.append(steady.k_eff / denominator
                                if denominator > 0.0 else float("nan"))
                anchor_spec = spec
                anchor_hook = feedback_hook
                anchor_flux = forward_flux
                anchor_precursors = spatial_precursor_shape
                anchor_time = t
                anchor_amplitude = amplitude
            peaks.append(cached_peak)
            means.append(cached_mean)
            previous_spec = spec
            previous_hook = feedback_hook

    synchronize(xp)
    seconds = time.perf_counter() - march_start
    with profiler.region("result_transfer"):
        final_temperature = asnumpy(temperature)
    counters["result_transfers"] += 1
    counters["shape_updates"] = shape_updates
    for name in ("forward_shape_solves", "iqs_shape_solves",
                 "full_diffusion_fallbacks", "residual_evaluations",
                 "iqs_predictor_checks"):
        counters[name] += 0
    phase_seconds = profiler.seconds()
    phase_seconds.pop("neutronics_solve", None)
    return QuasiStaticResult(
        times=times, power=np.asarray(powers), rho=np.asarray(rhos),
        peak_temperature=np.asarray(peaks),
        mean_temperature=np.asarray(means), temperature=final_temperature,
        flux=forward_flux, adjoint_flux=adjoint.flux, k0=steady.k_eff,
        steady=steady, seconds=seconds,
        initialization_seconds=initialization_seconds,
        device=ctx.device, steps=steps,
        effective_precursors=np.asarray(precursors),
        spatial_precursors=spatial_precursors,
        shape_update_times=np.asarray(shape_times),
        shape_update_reasons=shape_reasons,
        shape_k_eff=np.asarray(shape_ks),
        residual_times=np.asarray(residual_times),
        shape_residual=np.asarray(residual_values),
        fallback_times=np.asarray(fallback_times),
        phase_seconds=phase_seconds, counters=dict(counters))


def fixed_shape_coupled_transient(ctx, t_end, dt, **kwargs):
    """Coupled quasi-static prototype with the time-zero shape frozen.

    Control and temperature changes are represented through adjoint-projected
    effective kinetics, but no forward shape correction is made. This is exact
    for shape-preserving perturbations and useful as the cheapest comparison
    mode. Use :func:`quasistatic_coupled_transient` for control-drum motion.
    """
    forbidden = {"shape_dt", "adjoint_every", "shape_on_final", "shape_method",
                 "residual_tol", "fallback_residual", "iqs_kwargs",
                 "iqs_precond_degree"} & kwargs.keys()
    if forbidden:
        raise TypeError("fixed_shape_coupled_transient does not accept "
                        + ", ".join(sorted(forbidden)))
    return _coupled_quasistatic(
        ctx, t_end, dt, shape_dt=None, adjoint_every=1,
        shape_on_final=False, **kwargs)


def quasistatic_coupled_transient(ctx, t_end, dt, *, shape_dt=2.0,
                                  adjoint_every=1, shape_on_final=True,
                                  shape_method="iqs", residual_tol=None,
                                  fallback_residual=None,
                                  **kwargs):
    """Run an improved or adiabatic quasi-static coupled transient.

    The amplitude and effective delayed precursors advance every ``dt`` while
    ``shape_method='iqs'`` advances a time-dependent spatial shape and its
    precursor fields every ``shape_dt``. ``'adiabatic'`` selects the cheaper
    instantaneous eigen shape. The adjoint is refreshed every
    ``adjoint_every`` corrections.

    ``residual_tol`` forces an early shape correction when the amplitude-free
    spatial defect grows too large. ``fallback_residual`` is the hard safety
    threshold: that fine interval is advanced by the full spatial diffusion
    equations instead of point kinetics.
    """
    return _coupled_quasistatic(
        ctx, t_end, dt, shape_dt=shape_dt, adjoint_every=adjoint_every,
        shape_on_final=shape_on_final, shape_method=shape_method,
        residual_tol=residual_tol, fallback_residual=fallback_residual,
        **kwargs)
