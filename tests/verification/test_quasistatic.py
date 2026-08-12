"""Quasi-static projection and amplitude equations against exact limits."""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, EffectiveKinetics, Grid, Kinetics,
                   Material, ThermalFeedback, ThermalMaterial, TransientSolver,
                   equilibrium_precursors, fixed_shape_coupled_transient,
                   integrate_point_kinetics, project_effective_kinetics,
                   projected_adjoint_residual, projected_shape_residual,
                   quasistatic_coupled_transient)
from ndgpu.coupling import CoupledSolver, CouplingContext, coupled_transient
from ndgpu.quasistatic import (_match_spatial_precursor_importance,
                               _project_spatial_precursors,
                               _shape_time_importance)


D, SA, NF, V = 1.3, 0.030, 0.035, 2.2e5
BETA, LAM = 0.0065, 0.08
GRID = Grid(shape=(6, 1, 1), size=(60.0, 1.0, 1.0))
BASE = Material(name="base", diffusion=[D], sigma_a=[SA], nu_sigma_f=[NF])
KIN = Kinetics(velocities=[V], beta=[BETA], decay=[LAM])


def coupled_context():
    material_map = np.zeros(GRID.shape, dtype=np.int32)
    feedback = ThermalFeedback(t_ref=[600.0], doppler=[0.0])
    thermal = ThermalMaterial(
        conductivity=0.2, sink_coeff=0.04, sink_temperature=600.0,
        heat_capacity=2.0, name="homogeneous fuel")
    return CouplingContext(
        grid=GRID, materials=[BASE], material_map=material_map,
        thermal_materials=[thermal], feedback=feedback, total_power=120.0,
        bc="reflective", thermal_bc="adiabatic", device="cpu",
        kinetics=KIN, solver_cls=DiffusionEigenSolver,
        eigen_kwargs={"tol_k": 1e-11, "tol_source": 1e-11})


def localized_context():
    other = Material(name="second fuel", diffusion=[D], sigma_a=[SA],
                     nu_sigma_f=[NF])
    material_map = np.zeros(GRID.shape, dtype=np.int32)
    feedback = ThermalFeedback(t_ref=[600.0, 600.0], doppler=[0.0, 0.0])
    thermal = [ThermalMaterial(
        conductivity=0.2, sink_coeff=0.04, sink_temperature=600.0,
        heat_capacity=2.0, name=f"fuel {i}") for i in range(2)]
    return CouplingContext(
        grid=GRID, materials=[BASE, other], material_map=material_map,
        thermal_materials=thermal, feedback=feedback, total_power=120.0,
        bc="reflective", thermal_bc="adiabatic", device="cpu",
        kinetics=KIN, solver_cls=DiffusionEigenSolver,
        eigen_kwargs={"tol_k": 1e-11, "tol_source": 1e-11})


def shapes(material=BASE):
    solver = DiffusionEigenSolver(
        GRID, [material], bc="reflective", device="cpu")
    forward = solver.solve(tol_k=1e-10, tol_source=1e-10)
    adjoint = solver.solve(tol_k=1e-10, tol_source=1e-10, adjoint=True)
    return solver, forward, adjoint


def test_homogeneous_projection_recovers_exact_kinetics():
    solver, forward, adjoint = shapes()
    p = project_effective_kinetics(solver, forward, adjoint, KIN)
    assert p.rho == pytest.approx(0.0, abs=2e-14)
    assert p.generation_time == pytest.approx(1.0 / (V * SA), rel=2e-13)
    np.testing.assert_allclose(p.beta, [BETA], rtol=0, atol=2e-14)
    np.testing.assert_allclose(p.decay, [LAM], rtol=0, atol=0)


def test_projection_is_invariant_to_forward_and_adjoint_normalization():
    solver, forward, adjoint = shapes()
    a = project_effective_kinetics(solver, forward, adjoint, KIN)
    b = project_effective_kinetics(
        solver, 17.0 * forward.flux, 0.031 * adjoint.flux, KIN,
        k0=forward.k_eff)
    assert b.rho == pytest.approx(a.rho, abs=2e-14)
    assert b.generation_time == pytest.approx(a.generation_time, rel=2e-13)
    np.testing.assert_allclose(b.beta, a.beta, rtol=2e-13)


def test_iqs_precursor_shape_matching_preserves_accepted_projection():
    solver, _, adjoint = shapes()
    candidate = np.linspace(0.2, 1.1, np.prod(GRID.shape)).reshape(
        (1,) + GRID.shape)
    target = np.array([3.75])
    matched, factors = _match_spatial_precursor_importance(
        solver, adjoint.flux, candidate, target, KIN,
        time_importance=0.42)
    projected = _project_spatial_precursors(
        solver, adjoint.flux, matched, KIN, time_importance=0.42)

    np.testing.assert_allclose(projected, target, rtol=2e-14)
    np.testing.assert_allclose(matched, candidate * factors[0], rtol=2e-14)


def test_projected_shape_residual_removes_only_the_amplitude_mode():
    ref, forward, adjoint = shapes()
    assert projected_shape_residual(ref, forward, adjoint) < 2e-13
    local = Material(name="local absorber", diffusion=[D],
                     sigma_a=[SA * 1.02], nu_sigma_f=[NF])
    mmap = np.zeros(GRID.shape, dtype=np.int32)
    mmap[:2] = 1
    current = DiffusionEigenSolver(
        GRID, [BASE, local], mmap, bc="reflective", device="cpu")
    residual = projected_shape_residual(current, forward, adjoint.flux)
    assert residual > 1e-3


def test_projected_adjoint_residual_removes_only_the_amplitude_mode():
    ref, forward, adjoint = shapes()
    assert projected_adjoint_residual(ref, forward, adjoint) < 2e-13
    assert projected_adjoint_residual(
        ref, 7.0 * forward.flux, 0.03 * adjoint.flux) < 2e-13
    local = Material(name="local absorber", diffusion=[D],
                     sigma_a=[SA * 1.02], nu_sigma_f=[NF])
    mmap = np.zeros(GRID.shape, dtype=np.int32)
    mmap[:2] = 1
    current = DiffusionEigenSolver(
        GRID, [BASE, local], mmap, bc="reflective", device="cpu")
    residual = projected_adjoint_residual(current, forward, adjoint.flux)
    assert residual > 1e-3


def test_local_rayleigh_projection_recovers_uniform_absorption_worth():
    ref, forward, adjoint = shapes()
    delta = 0.5 * BETA * SA
    perturbed = Material(name="perturbed", diffusion=[D], sigma_a=[SA - delta],
                         nu_sigma_f=[NF])
    current = DiffusionEigenSolver(
        GRID, [perturbed], bc="reflective", device="cpu")
    p = project_effective_kinetics(
        current, forward, adjoint, KIN, k0=forward.k_eff)
    assert p.rho == pytest.approx(delta / SA, rel=2e-13)
    assert p.rho / p.beta_total == pytest.approx(0.5, rel=2e-13)


def test_critical_point_kinetics_equilibrium_is_stationary():
    p = EffectiveKinetics(
        rho=0.0, generation_time=1.5e-4,
        beta=np.array([0.0015, 0.005]), decay=np.array([0.08, 0.2]),
        time_importance=1.0, fission_importance=1.0)
    c0 = equilibrium_precursors(p)
    r = integrate_point_kinetics(p, t_end=10.0, dt=0.1)
    np.testing.assert_allclose(r.power, 1.0, rtol=0, atol=2e-13)
    np.testing.assert_allclose(r.precursors, np.broadcast_to(c0, r.precursors.shape),
                               rtol=0, atol=2e-12)


def test_projected_amplitude_matches_full_uniform_spatial_transient():
    ref, forward, adjoint = shapes()
    delta = 0.5 * BETA * (NF / forward.k_eff)
    perturbed = Material(name="perturbed", diffusion=[D], sigma_a=[SA - delta],
                         nu_sigma_f=[NF])
    current = DiffusionEigenSolver(
        GRID, [perturbed], bc="reflective", device="cpu")
    p0 = project_effective_kinetics(ref, forward, adjoint, KIN)
    p1 = project_effective_kinetics(
        current, forward, adjoint, KIN, k0=forward.k_eff)
    dt, t_end = 2e-4, 0.02
    qs = integrate_point_kinetics(
        lambda t: p0 if t <= 0.0 else p1, t_end=t_end, dt=dt)
    full = TransientSolver(
        GRID, lambda t: (([BASE] if t <= 0.0 else [perturbed]), None),
        KIN, bc="reflective", device="cpu").solve(
            t_end=t_end, dt=dt, tol_step=1e-9)
    np.testing.assert_allclose(qs.power, full.power, rtol=0, atol=2e-11)


def test_point_kinetics_rejects_family_changes():
    p1 = EffectiveKinetics(0.0, 1e-4, np.array([0.0065]), np.array([0.08]),
                           1.0, 1.0)
    p2 = EffectiveKinetics(0.0, 1e-4, np.array([0.003, 0.0035]),
                           np.array([0.08, 0.2]), 1.0, 1.0)
    with pytest.raises(ValueError, match="family count"):
        integrate_point_kinetics(lambda t: p1 if t <= 0 else p2,
                                 t_end=0.1, dt=0.1)


def test_fixed_shape_coupling_holds_a_stationary_equilibrium():
    ctx = coupled_context()
    steady = CoupledSolver(ctx).solve(tol=1e-10)
    result = fixed_shape_coupled_transient(
        ctx, t_end=0.2, dt=0.05, dt_thermal=0.1,
        initial_coupled=steady, profile=True)
    np.testing.assert_allclose(result.power, 1.0, rtol=0, atol=2e-13)
    assert np.ptp(result.mean_temperature) < 2e-10
    assert np.ptp(result.peak_temperature) < 2e-10
    assert result.counters["initial_state_reuses"] == 1
    assert result.counters["shape_updates"] == 0
    assert result.counters["adjoint_eigen_solves"] == 1
    assert result.counters["reactivity_projections"] == 2


def test_fixed_shape_coupling_matches_full_shape_preserving_insertion():
    ctx = coupled_context()
    material_map = ctx.material_map
    delta = 0.35 * BETA * SA
    perturbed = Material(
        name="uniform insertion", diffusion=[D], sigma_a=[SA - delta],
        nu_sigma_f=[NF])
    base_state = ([BASE], material_map)
    perturbed_state = ([perturbed], material_map)

    def problem_at(t):
        return base_state if t <= 0.0 else perturbed_state

    full = coupled_transient(
        ctx, t_end=0.02, dt=2e-4, dt_thermal=0.002,
        problem_at=problem_at)
    accelerated = fixed_shape_coupled_transient(
        coupled_context(), t_end=0.02, dt=2e-4, dt_thermal=0.002,
        problem_at=problem_at)
    np.testing.assert_allclose(accelerated.power, full.power,
                               rtol=0, atol=3e-10)
    np.testing.assert_allclose(accelerated.mean_temperature,
                               full.mean_temperature, rtol=0, atol=2e-9)


def test_adiabatic_shape_updates_preserve_exact_uniform_solution():
    ctx = coupled_context()
    material_map = ctx.material_map
    delta = 0.35 * BETA * SA
    perturbed = Material(
        name="uniform insertion", diffusion=[D], sigma_a=[SA - delta],
        nu_sigma_f=[NF])
    base_state = ([BASE], material_map)
    perturbed_state = ([perturbed], material_map)

    def problem_at(t):
        return base_state if t <= 0.0 else perturbed_state

    fixed = fixed_shape_coupled_transient(
        ctx, t_end=0.02, dt=2e-4, dt_thermal=0.002,
        problem_at=problem_at)
    adiabatic = quasistatic_coupled_transient(
        coupled_context(), t_end=0.02, dt=2e-4, dt_thermal=0.002,
        shape_dt=0.005, adjoint_every=2, shape_method="adiabatic",
        problem_at=problem_at)
    np.testing.assert_allclose(adiabatic.power, fixed.power,
                               rtol=0, atol=3e-11)
    assert adiabatic.counters["shape_updates"] == 4
    assert adiabatic.counters["forward_shape_solves"] == 4
    assert adiabatic.counters["adjoint_eigen_solves"] == 3
    np.testing.assert_allclose(
        adiabatic.shape_update_times, [0.005, 0.010, 0.015, 0.020])
    assert adiabatic.shape_update_reasons == ["maximum_interval"] * 4


def test_adiabatic_quasistatic_tracks_reduced_hpmr_drum_ramp():
    """The intended use case: cached drum frames plus coupled feedback."""
    from ndgpu.benchmarks.hpmr import build_hpmr2d
    from ndgpu.benchmarks.hpmr_thermal import (build_hpmr_coupling,
                                               hpmr_drum_ramp)

    problem = build_hpmr2d(
        refine=2, drum_angle_deg=150.0, absorber="polar")
    problem_at = hpmr_drum_ramp(
        problem, angle_from=150.0, angle_to=154.0,
        t_start=0.1, t_ramp=0.2, n_angles=5, refine=2)
    full = coupled_transient(
        build_hpmr_coupling(problem), t_end=0.5, dt=0.05,
        dt_thermal=0.1, problem_at=problem_at)
    qs = quasistatic_coupled_transient(
        build_hpmr_coupling(problem), t_end=0.5, dt=0.05,
        dt_thermal=0.1, shape_dt=0.1, adjoint_every=2,
        shape_method="adiabatic", problem_at=problem_at)
    iqs = quasistatic_coupled_transient(
        build_hpmr_coupling(problem), t_end=0.5, dt=0.05,
        dt_thermal=0.1, shape_dt=0.1, adjoint_every=2,
        shape_method="iqs", problem_at=problem_at)

    power_error = np.max(np.abs(qs.power - full.power) / full.power)
    assert power_error < 5e-3, (power_error, full.power, qs.power)
    assert abs(qs.mean_temperature[-1] - full.mean_temperature[-1]) < 0.02
    assert qs.counters["shape_updates"] == 5
    assert qs.counters["forward_shape_solves"] == 5
    assert qs.counters["adjoint_eigen_solves"] == 3
    iqs_power_error = np.max(np.abs(iqs.power - full.power) / full.power)
    assert iqs_power_error < 5e-3, (iqs_power_error, full.power, iqs.power)
    assert abs(iqs.mean_temperature[-1] - full.mean_temperature[-1]) < 0.02
    assert iqs.counters["iqs_shape_solves"] == 5


def test_iqs_keeps_accepted_precursor_history_across_shape_corrections():
    """A coarse shape predictor must not replace the fine amplitude history."""
    from ndgpu.benchmarks.hpmr import build_hpmr2d
    from ndgpu.benchmarks.hpmr_thermal import (build_hpmr_coupling,
                                               hpmr_drum_ramp)

    problem = build_hpmr2d(
        refine=2, drum_angle_deg=90.0, absorber="polar")
    problem_at = hpmr_drum_ramp(
        problem, angle_from=90.0, angle_to=95.0,
        t_start=0.0, t_ramp=0.5, n_angles=6, refine=2)
    options = dict(t_end=1.0, dt=0.05, dt_thermal=0.25,
                   problem_at=problem_at)
    full = coupled_transient(build_hpmr_coupling(problem), **options)
    adiabatic = quasistatic_coupled_transient(
        build_hpmr_coupling(problem), shape_dt=0.25, adjoint_every=2,
        shape_method="adiabatic", **options)
    iqs = quasistatic_coupled_transient(
        build_hpmr_coupling(problem), shape_dt=0.25, adjoint_every=2,
        shape_method="iqs", **options)
    adaptive = quasistatic_coupled_transient(
        build_hpmr_coupling(problem), shape_dt=0.25, adjoint_every=2,
        shape_method="iqs", iqs_predictor_tol=0.02, **options)
    guarded = quasistatic_coupled_transient(
        build_hpmr_coupling(problem), shape_dt=0.25, adjoint_every=2,
        shape_method="iqs", residual_tol=1e-8, fallback_residual=1.0,
        **options)

    # The deliberately coarse independent predictor differs by several
    # percent. Its spatial shape is useful, but adopting its precursor field
    # used to leave the accepted point amplitude about 10% low after the ramp.
    assert iqs.counters["iqs_max_amplitude_error_ppm"] > 10_000
    assert adaptive.counters["iqs_predictor_interval_reductions"] > 0
    assert adaptive.counters["shape_updates"] > iqs.counters["shape_updates"]
    assert abs(iqs.power[-1] / full.power[-1] - 1.0) < 1e-3
    assert np.max(np.abs(guarded.power - full.power) / full.power) < 1e-2
    assert (np.max(np.abs(guarded.power - full.power) / full.power)
            < np.max(np.abs(adiabatic.power - full.power) / full.power))
    assert guarded.counters["full_diffusion_fallbacks"] == 0
    assert "residual" in guarded.shape_update_reasons


def test_residual_trigger_falls_back_to_full_diffusion_interval():
    perturbed = Material(
        name="local absorber", diffusion=[D], sigma_a=[SA * 1.02],
        nu_sigma_f=[NF])
    base_map = np.zeros(GRID.shape, dtype=np.int32)
    changed_map = base_map.copy()
    changed_map[:2] = 1
    base_state = ([BASE, BASE], base_map)
    changed_state = ([BASE, perturbed], changed_map)

    def problem_at(t):
        return base_state if t <= 0.0 else changed_state

    full = coupled_transient(
        localized_context(), t_end=0.002, dt=0.002,
        dt_thermal=0.002, problem_at=problem_at)
    guarded = quasistatic_coupled_transient(
        localized_context(), t_end=0.002, dt=0.002,
        dt_thermal=0.002, shape_dt=1.0, problem_at=problem_at,
        residual_tol=1e-8, fallback_residual=1e-6)
    assert guarded.counters["full_diffusion_fallbacks"] == 1
    assert guarded.shape_update_reasons == ["full_diffusion_fallback"]
    np.testing.assert_allclose(guarded.fallback_times, [0.002])
    np.testing.assert_allclose(guarded.power, full.power,
                               rtol=0, atol=2e-10)


def test_adjoint_residual_refreshes_before_maximum_age():
    perturbed = Material(
        name="local absorber", diffusion=[D], sigma_a=[SA * 1.02],
        nu_sigma_f=[NF])
    base_map = np.zeros(GRID.shape, dtype=np.int32)
    changed_map = base_map.copy()
    changed_map[:2] = 1
    base_state = ([BASE, BASE], base_map)
    changed_state = ([BASE, perturbed], changed_map)

    def problem_at(t):
        return base_state if t <= 0.0 else changed_state

    adaptive = quasistatic_coupled_transient(
        localized_context(), t_end=0.004, dt=0.002,
        dt_thermal=0.004, shape_dt=0.002, shape_method="adiabatic",
        adjoint_every=100, adjoint_residual_tol=1e-6,
        problem_at=problem_at)
    assert adaptive.counters["adjoint_residual_evaluations"] == 2
    assert adaptive.counters["adjoint_residual_refreshes"] >= 1
    assert adaptive.counters["adjoint_eigen_solves"] >= 2
    assert adaptive.counters["max_adjoint_residual_ppm"] > 1000
    assert adaptive.adjoint_residual_times.shape == adaptive.adjoint_residual.shape


def test_adjoint_residual_tolerance_must_be_positive():
    with pytest.raises(ValueError, match="adjoint_residual_tol"):
        quasistatic_coupled_transient(
            localized_context(), t_end=0.002, dt=0.002,
            shape_dt=0.002, adjoint_residual_tol=0.0)


def test_iqs_time_dependent_shape_is_closer_than_instantaneous_eigen_shape():
    perturbed = Material(
        name="local absorber", diffusion=[D], sigma_a=[SA * 1.02],
        nu_sigma_f=[NF])
    base_map = np.zeros(GRID.shape, dtype=np.int32)
    changed_map = base_map.copy()
    changed_map[:2] = 1
    base_state = ([BASE, BASE], base_map)
    changed_state = ([BASE, perturbed], changed_map)

    def problem_at(t):
        return base_state if t <= 0.0 else changed_state

    options = dict(t_end=0.02, dt=0.002, dt_thermal=0.01,
                   problem_at=problem_at)
    full = coupled_transient(localized_context(), **options)
    adiabatic = quasistatic_coupled_transient(
        localized_context(), shape_dt=0.01,
        shape_method="adiabatic", **options)
    iqs = quasistatic_coupled_transient(
        localized_context(), shape_dt=0.01, shape_method="iqs", **options)

    reference = np.asarray(full.flux).ravel()
    reference /= np.linalg.norm(reference)

    def error(result):
        shape = np.asarray(result.flux).ravel()
        shape /= np.linalg.norm(shape)
        return np.linalg.norm(shape - reference)

    assert error(iqs) < error(adiabatic)
    assert iqs.counters["iqs_shape_solves"] == 2
    assert iqs.counters["iqs_predictor_checks"] == 2
    assert iqs.counters["iqs_precursor_shape_corrections"] == 2
    assert iqs.counters["iqs_max_precursor_shape_change_ppm"] > 0
    assert np.all(np.isfinite(iqs.spatial_precursors))
    assert np.all(iqs.effective_precursors >= 0.0)


def test_iqs_power_includes_adjoint_weighted_shape_derivative():
    """A changing fission-normalized shape must not make amplitude equal power.

    With a fixed adjoint over this single macro interval, integrating
    ``<phi*, V^-1 dpsi/dt>`` is exactly the old/new time-importance ratio.
    This is a coordinate correction: the accepted precursor inventory remains
    untouched and the point amplitude is converted to physical fission power.
    """
    changed = Material(
        name="weak local fuel", diffusion=[D], sigma_a=[SA * 1.02],
        nu_sigma_f=[0.8 * NF])
    base_map = np.zeros(GRID.shape, dtype=np.int32)
    changed_map = base_map.copy()
    changed_map[:2] = 1
    states = (([BASE, BASE], base_map), ([BASE, changed], changed_map))
    problem_at = lambda t: states[0] if t <= 0.0 else states[1]

    result = quasistatic_coupled_transient(
        localized_context(), t_end=0.005, dt=0.005, dt_thermal=0.005,
        shape_dt=0.005, adjoint_every=99, shape_method="iqs",
        problem_at=problem_at)

    final_solver = DiffusionEigenSolver(
        GRID, states[1][0], states[1][1], bc="reflective", device="cpu")
    initial_solver = DiffusionEigenSolver(
        GRID, states[0][0], states[0][1], bc="reflective", device="cpu")
    initial_flux = np.asarray(result.steady.neutronics.flux).copy()
    initial_source = initial_solver.fields.fission_source(initial_flux)
    initial_flux /= np.sum(initial_source / result.k0)
    old_t = _shape_time_importance(
        final_solver, initial_flux, result.adjoint_flux, KIN)
    new_t = _shape_time_importance(
        final_solver, result.flux, result.adjoint_flux, KIN)

    assert result.power_shape_factor[-1] == pytest.approx(
        old_t / new_t, rel=2e-12)
    assert abs(result.power_shape_factor[-1] - 1.0) > 1e-3
    np.testing.assert_allclose(
        result.power, result.amplitude * result.power_shape_factor,
        rtol=2e-14, atol=0.0)
    assert result.counters["shape_derivative_corrections"] == 1
    assert result.counters["max_shape_derivative_correction_ppm"] > 1000


def test_residual_trigger_forces_early_iqs_correction():
    perturbed = Material(
        name="local absorber", diffusion=[D], sigma_a=[SA * 1.02],
        nu_sigma_f=[NF])
    base_map = np.zeros(GRID.shape, dtype=np.int32)
    changed_map = base_map.copy()
    changed_map[:2] = 1
    states = (([BASE, BASE], base_map), ([BASE, perturbed], changed_map))
    result = quasistatic_coupled_transient(
        localized_context(), t_end=0.004, dt=0.002, dt_thermal=0.004,
        shape_dt=1.0, shape_on_final=False, residual_tol=1e-6,
        problem_at=lambda t: states[0] if t <= 0.0 else states[1])
    assert result.shape_update_reasons == ["residual"]
    np.testing.assert_allclose(result.shape_update_times, [0.002])
    assert result.counters["iqs_shape_solves"] == 1
    assert result.counters["full_diffusion_fallbacks"] == 0
