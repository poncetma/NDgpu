"""The coupled fixed point: exact limits, contraction, and conservation.

These are the structural properties the coupling must have before any number
it produces is worth reading. They run on the fast 2-group placeholder core --
adequate here because none of them is a physics *value*, only a relation that
must hold whatever the cross sections are.
"""

import numpy as np
import pytest

from ndgpu.benchmarks.hpmr import build_hpmr2d
from ndgpu.benchmarks.hpmr_thermal import (AMBIENT_K, build_hpmr_coupling,
                                           hpmr_feedback)
from ndgpu.coupling import (CoupledSolver, coupled_map, neutronics_step,
                            temperature_defect_pcm, thermal_step, uncoupled_k)
from ndgpu.feedback import ThermalFeedback


def _problem(refine=3, angle=180.0):
    return build_hpmr2d(refine=refine, drum_angle_deg=angle, absorber="polar")


def _ctx(problem, **kw):
    return build_hpmr_coupling(problem, **kw)


def test_zero_feedback_reproduces_the_uncoupled_eigenvalue():
    """With the coefficients zeroed the temperature cannot affect k, so the
    coupled solve must return exactly the isothermal answer -- and get there in
    two iterations, since the second merely confirms the first."""
    p = _problem()
    n = len(p.materials)
    inert = ThermalFeedback(t_ref=np.full(n, 800.0), doppler=np.zeros(n))
    ctx = _ctx(p, feedback=inert, warm_start=False)
    res = CoupledSolver(ctx).solve(tol=1e-9)
    plain = uncoupled_k(ctx)
    assert res.k_eff == pytest.approx(plain, abs=2e-8)
    assert res.iterations == 2


def test_zero_power_leaves_no_fission_heat_in_the_core():
    """At zero power there is no source, so the temperature field is whatever
    the heat pipes and the vessel agree on -- strictly between the two, and
    reached in one iteration since nothing feeds back."""
    from ndgpu.benchmarks.hpmr_thermal import SINK_TEMPERATURE_K

    p = _problem()
    ctx = _ctx(p, power_w=0.0, warm_start=False)
    res = CoupledSolver(ctx).solve(tol=1e-9)
    fuel = p.material_map == 1
    assert res.thermal.source_watts == 0.0
    assert np.all(res.temperature[fuel] < SINK_TEMPERATURE_K)
    assert np.all(res.temperature[fuel] > AMBIENT_K)
    # No fission heat means the fuel cannot be hotter than its heat pipes, so
    # the pipes must be feeding the vessel rather than the other way round.
    assert res.thermal.sink_watts < 0.0
    assert res.thermal.balance_residual < 1e-10
    assert res.k_eff == pytest.approx(
        uncoupled_k(ctx, res.temperature), abs=2e-8)


def test_the_map_is_strongly_contractive():
    """Because the k-eigenvalue solve pins the power LEVEL, feedback can only
    redistribute power -- so plain Picard converges fast on its own and
    acceleration is a convenience, not a crutch. If this ratio ever exceeds 1
    the physics is wrong, not the accelerator."""
    p = _problem()
    ctx = _ctx(p, warm_start=False)
    res = CoupledSolver(ctx).solve(tol=1e-9, relaxation=1.0, anderson_depth=0)
    r = res.residual_history
    assert res.converged
    ratios = [b / a for a, b in zip(r[:-1], r[1:]) if a > 0]
    assert all(q < 0.5 for q in ratios), ratios


def test_power_normalization_and_energy_balance_hold_at_the_fixed_point():
    p = _problem()
    ctx = _ctx(p, warm_start=False)
    res = CoupledSolver(ctx).solve(tol=1e-9)
    th = res.thermal
    assert th.source_watts == pytest.approx(ctx.total_power, rel=1e-12)
    assert th.balance_residual < 1e-10
    assert th.sink_watts > 0.9 * ctx.total_power     # heat pipes do the work


def test_acceleration_and_relaxation_reach_the_same_fixed_point():
    """Anderson, plain Picard and under-relaxed Picard are three routes to one
    answer. A change that moves k is a bug, not a trade-off."""
    p = _problem()
    runs = {}
    for label, kw in (("picard", dict(relaxation=1.0, anderson_depth=0)),
                      ("relaxed", dict(relaxation=0.5, anderson_depth=0)),
                      ("anderson", dict(relaxation=1.0, anderson_depth=5))):
        res = CoupledSolver(_ctx(p, warm_start=False)).solve(tol=1e-9, **kw)
        assert res.converged, label
        runs[label] = res
    ks = [r.k_eff for r in runs.values()]
    assert max(ks) - min(ks) < 2e-8, runs
    peaks = [r.peak_temperature for r in runs.values()]
    assert max(peaks) - min(peaks) < 1e-5


def test_warm_start_does_not_move_the_answer():
    """The warm start is an accelerator, so it must change the cost and not the
    result. (Carrying k_guess across as well DID move it -- the power iteration
    stops on the change between outers, so a seeded k made the first |dk|
    vanish and the solve exited before the flux responded.)"""
    p = _problem()
    cold = CoupledSolver(_ctx(p, warm_start=False)).solve(tol=1e-9)
    warm = CoupledSolver(_ctx(p, warm_start=True)).solve(tol=1e-9)
    assert warm.k_eff == pytest.approx(cold.k_eff, abs=2e-8)
    assert warm.peak_temperature == pytest.approx(cold.peak_temperature, abs=1e-4)


def test_heating_costs_reactivity():
    p = _problem()
    ctx = _ctx(p, warm_start=False)
    res = CoupledSolver(ctx).solve(tol=1e-9)
    k_cold = uncoupled_k(ctx, np.full(ctx.shape, AMBIENT_K))
    assert res.k_eff < k_cold
    assert temperature_defect_pcm(k_cold, res.k_eff) < 0.0


def test_the_map_is_a_pure_function_of_temperature_without_warm_start():
    """What makes a lockstep comparison against an external coupling tool
    meaningful: G(T) must depend on T alone, not on what was solved before."""
    p = _problem()
    ctx = _ctx(p, warm_start=False)
    T0 = ctx.initial_temperature()
    a, _ = coupled_map(T0, ctx)
    b, _ = coupled_map(T0, ctx)
    np.testing.assert_array_equal(a, b)


def test_coupled_transient_holds_the_steady_state():
    """The sharpest structural check on the transient coupling: started from
    the converged coupled steady state and left alone, BOTH physics must sit
    still. Power stays at 1 and the temperature does not drift -- any error in
    the feedback hook, the power normalization or the thermal capacity shows up
    here as motion where there should be none."""
    from ndgpu.coupling import coupled_transient

    p = _problem()
    ctx = _ctx(p)
    r = coupled_transient(ctx, t_end=2.0, dt=0.05, dt_thermal=0.25)
    # The rebalanced fixed-source solve is stopped at a relative source
    # tolerance, not iterated to machine epsilon; this is its measured
    # stationary round-off envelope on the HP-MR mesh.
    np.testing.assert_allclose(r.power, 1.0, rtol=0, atol=2e-9)
    assert np.ptp(r.peak_temperature) < 1e-6
    assert np.ptp(r.mean_temperature) < 1e-6


def test_coupled_transient_matches_the_uncoupled_one_without_feedback():
    """With the feedback coefficients zeroed the coupling cannot act, so the
    power history must be the plain transient's, step for step."""
    from ndgpu.benchmarks.hpmr_thermal import hpmr_drum_ramp
    from ndgpu.coupling import coupled_transient
    from ndgpu.transient import TransientSolver
    from ndgpu.tri import TriGroupOperator

    p = _problem()
    n = len(p.materials)
    inert = ThermalFeedback(t_ref=np.full(n, 800.0), doppler=np.zeros(n))
    ctx = _ctx(p, feedback=inert)
    pa = hpmr_drum_ramp(p, angle_from=180.0, angle_to=170.0, t_start=0.05,
                        t_ramp=0.2, n_angles=5, refine=3)

    coupled = coupled_transient(ctx, t_end=0.5, dt=0.05, problem_at=pa)
    plain = TransientSolver(p.grid, pa, p.kinetics, bc=p.bc, active=p.active,
                            mask_bc=p.mask_bc, mix_material=p.mix_material,
                            mix_weight=p.mix_weight,
                            group_operator=TriGroupOperator,
                            eig_solver=ctx.solver_cls,
                            precond_degree=1,
                            device="cpu").solve(
                                t_end=0.5, dt=0.05,
                                initial_steady=coupled.steady.neutronics,
                                linsolve_kwargs={"check_every": 4})
    np.testing.assert_allclose(coupled.power, plain.power, rtol=1e-9, atol=1e-11)


def _light_thermal(ctx, factor=0.005):
    """The same context with the thermal mass scaled down, so the fuel responds
    in seconds instead of the HP-MR's real 268 s.

    Needed to test the SIGN of the feedback loop in a fast test: on the real
    heat capacity a 3 s run heats the fuel by ~0.05 K, and the resulting
    reactivity is smaller than the difference between two runs' distinct
    starting states -- so a short run on real properties cannot see the loop at
    all, whichever way it points. Shortening the time constant is the honest
    way to isolate the sign; it is not a claim about the reactor.
    """
    from dataclasses import replace as dc_replace

    from ndgpu.thermal import ThermalMaterial

    light = [ThermalMaterial(m.conductivity, m.sink_coeff, m.sink_temperature,
                             m.heat_capacity * factor, m.name)
             for m in ctx.thermal_materials]
    return dc_replace(ctx, thermal_materials=light, _thermal=None)


def test_a_drum_withdrawal_raises_power_and_feedback_pushes_back():
    """Direction check on the whole loop: withdrawing absorber adds reactivity,
    power rises, the fuel heats, and the Doppler term opposes it. Getting any
    sign wrong in that chain reverses one of these."""
    from ndgpu.benchmarks.hpmr_thermal import hpmr_drum_ramp
    from ndgpu.coupling import coupled_transient

    p = _problem(angle=150.0)
    pa = hpmr_drum_ramp(p, angle_from=150.0, angle_to=156.0, t_start=0.1,
                        t_ramp=0.4, n_angles=7, refine=3)
    hot = coupled_transient(_light_thermal(_ctx(p)), t_end=3.0, dt=0.05,
                            dt_thermal=0.25, problem_at=pa)
    assert hot.power[-1] > 1.05                    # withdrawal => power up
    assert hot.mean_temperature[-1] > hot.mean_temperature[0] + 1.0  # fuel heats

    n = len(p.materials)
    inert = ThermalFeedback(t_ref=np.full(n, 800.0), doppler=np.zeros(n))
    free = coupled_transient(_light_thermal(_ctx(p, feedback=inert)), t_end=3.0,
                             dt=0.05, dt_thermal=0.25, problem_at=pa)
    # Same insertion, no feedback: the excursion must run further.
    assert free.power[-1] > hot.power[-1]


def test_on_the_real_thermal_mass_the_feedback_is_slow():
    """The corollary, stated as a fact about the reactor rather than left
    implicit: with rho*cp/h = 268 s, a few seconds of transient heats the fuel
    by well under a kelvin, so the excursion is arrested late. This is why the
    sign test above has to shorten the time constant."""
    from ndgpu.benchmarks.hpmr_thermal import hpmr_drum_ramp
    from ndgpu.coupling import coupled_transient

    p = _problem(angle=150.0)
    pa = hpmr_drum_ramp(p, angle_from=150.0, angle_to=156.0, t_start=0.1,
                        t_ramp=0.4, n_angles=7, refine=3)
    r = coupled_transient(_ctx(p), t_end=3.0, dt=0.05, dt_thermal=0.25,
                          problem_at=pa)
    assert r.power[-1] > 1.05                                  # power responds
    assert r.mean_temperature[-1] - r.mean_temperature[0] < 1.0   # heat does not


def test_thermal_substepping_changes_little():
    """dt_thermal is a cost knob, not a physics one: taking the thermal step
    5x coarser must not move the answer meaningfully, because the temperature
    moves on a 268 s constant while the neutronics step is 50 ms."""
    from ndgpu.benchmarks.hpmr_thermal import hpmr_drum_ramp
    from ndgpu.coupling import coupled_transient

    p = _problem(angle=150.0)
    pa = hpmr_drum_ramp(p, angle_from=150.0, angle_to=154.0, t_start=0.1,
                        t_ramp=0.4, n_angles=5, refine=3)
    fine = coupled_transient(_ctx(p), t_end=2.0, dt=0.05, dt_thermal=0.05,
                             problem_at=pa)
    coarse = coupled_transient(_ctx(p), t_end=2.0, dt=0.05, dt_thermal=0.25,
                               problem_at=pa)
    assert abs(coarse.power[-1] - fine.power[-1]) / fine.power[-1] < 1e-3
    assert abs(coarse.mean_temperature[-1] - fine.mean_temperature[-1]) < 1e-3


def test_coupled_transient_flushes_a_partial_thermal_window():
    """The final neutronics interval must heat the core even below dt_thermal."""
    from ndgpu.benchmarks.hpmr_thermal import hpmr_drum_ramp
    from ndgpu.coupling import coupled_transient

    p = _problem(angle=150.0)
    pa = hpmr_drum_ramp(p, angle_from=150.0, angle_to=156.0, t_start=0.0,
                        t_ramp=0.2, n_angles=5, refine=3)
    r = coupled_transient(
        _light_thermal(_ctx(p)), t_end=0.30, dt=0.05,
        dt_thermal=0.25, problem_at=pa, thermal_check_every=2,
        thermal_precond_degree=1, thermal_diagnostics_every=1, profile=True)
    # Five neutron steps make the first 0.25-s thermal window. The sixth must
    # advance a 0.05-s final window, rather than leaving its heat in an unused
    # accumulator.
    assert r.mean_temperature[-1] > r.mean_temperature[-2] + 1e-6
    assert r.counters["neutronics_steps"] == 6
    assert r.counters["thermal_steps"] == 2
    assert r.counters["thermal_diagnostics"] == 2
    assert r.counters["telemetry_transfers"] == 2
    assert r.counters["neutron_inner_iterations"] > 0
    assert r.counters["neutron_fixed_point_sweeps"] >= 6
    assert r.counters["initial_eigen_outer_iterations"] > 0
    assert r.counters["initial_eigen_inner_iterations"] > 0
    assert r.counters["initial_state_reuses"] == 1
    # Four changed ramp states plus the feedback-driven rebuild on the sixth
    # neutron step. The final feedback update has no following step to rebuild.
    assert r.counters["operator_rebuilds"] == 5
    expected_phases = {
        "initial_eigen_solve", "neutronics_solve", "operator_rebuild", "power_edit",
        "thermal_solve", "feedback_update", "telemetry_transfer",
        "result_transfer", "transient_total",
    }
    assert expected_phases <= r.phase_seconds.keys()
    assert all(r.phase_seconds[k] >= 0.0 for k in expected_phases)


def test_the_halves_compose_into_the_map():
    """coupled_map must be exactly thermal_step o neutronics_step -- the
    property that lets two separate processes reproduce it between them."""
    p = _problem()
    ctx = _ctx(p, warm_start=False)
    T0 = ctx.initial_temperature()
    q, _ = neutronics_step(T0, ctx)
    t_split, _ = thermal_step(q, ctx)
    t_map, info = coupled_map(T0, ctx)
    np.testing.assert_array_equal(t_split, t_map)
    np.testing.assert_array_equal(np.asarray(q), np.asarray(info["power"]))
