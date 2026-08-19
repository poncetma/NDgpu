"""Transient solver against exact references.

The cornerstone is the point-kinetics comparison: for a *uniform* reactivity
perturbation of a bare homogeneous core, the flux shape never changes, so the
space-time diffusion equations reduce *exactly* to the point-kinetics ODEs — a
rigorous, independent reference for the whole transient stack (precursor
treatment, critical adjustment, time stepping).

The perturbation is a uniform **absorption** step (not a fission step): a small
drop in Sigma_a inserts positive reactivity while leaving nu*Sigma_f fixed, so
res.power (proportional to nu*Sigma_f * phi, i.e. the neutron population) tracks
the point-kinetics amplitude n(t) *exactly*. A fission (nu*Sigma_f) step would
instead make res.power a production-rate proxy that jumps by (1+eps) the instant
the cross section changes, offsetting it from n(t) by ~eps and masking the true
agreement. With a physical neutron speed the prompt jump to beta/(beta-rho) is
resolved in-window, and the match tightens to a few 1e-4 (backward-Euler O(dt)).
"""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, Grid, Kinetics, Material,
                   TransientSolver, TransientSDP1Solver)

D, SIGMA_A, NU_SIGMA_F = 1.3, 0.030, 0.035
BASE = Material(name="1g", diffusion=[D], sigma_a=[SIGMA_A], nu_sigma_f=[NU_SIGMA_F])
# Physical thermal-neutron speed (2200 m/s = 2.2e5 cm/s): the prompt generation
# time Lambda = 1/(v*a) ~ 1.3e-4 s is short enough that the prompt jump develops
# within the transient window.
V, BETA, LAM = 2.2e5, 0.0065, 0.08
KIN = Kinetics(velocities=[V], beta=[BETA], decay=[LAM])
GRID = Grid(shape=(12, 12, 12), size=(90.0, 90.0, 90.0))


def absorption_step_problem(delta_sigma_a):
    """Sigma_a -> Sigma_a - delta everywhere for t > 0 (positive reactivity)."""
    m0 = [BASE]
    m1 = [Material(name="pert", diffusion=[D], sigma_a=[SIGMA_A - delta_sigma_a],
                   nu_sigma_f=[NU_SIGMA_F])]
    return lambda t: ((m0 if t <= 0 else m1), None)


def test_unperturbed_transient_stays_steady():
    solver = TransientSolver(GRID, lambda t: ([BASE], None), KIN, device="cpu")
    res = solver.solve(t_end=0.5, dt=0.05)
    assert np.allclose(res.power, 1.0, atol=1e-5), res.power


def test_diffusion_bdf_orders_preserve_equilibrium_and_report_startup_order():
    """BDF1--6 share the diffusion path and must not manufacture reactivity."""
    steady = DiffusionEigenSolver(GRID, [BASE], device="cpu").solve(
        tol_k=1e-9, tol_source=1e-8)
    for order in range(1, 7):
        result = TransientSolver(
            GRID, lambda _t: ([BASE], None), KIN, device="cpu").solve(
                t_end=0.06, dt=0.01, initial_steady=steady,
                time_scheme=f"bdf{order}", tol_step=1e-9)
        assert result.time_scheme == ("backward-euler" if order == 1
                                      else f"bdf{order}")
        assert result.time_orders == [min(step, order)
                                      for step in range(1, 7)]
        np.testing.assert_allclose(result.power, 1.0, rtol=0, atol=2e-9)


def test_bdf3_monolithic_and_groupwise_diffusion_solve_same_equations():
    """High-order time history is independent of the in-step solver choice."""
    prob = absorption_step_problem(0.0005)
    steady = DiffusionEigenSolver(GRID, [BASE], device="cpu").solve(
        tol_k=1e-9, tol_source=1e-8)
    common = dict(t_end=0.008, dt=0.002, initial_steady=steady,
                  time_scheme="bdf3", tol_step=1e-9)
    fixed = TransientSolver(GRID, prob, KIN, device="cpu").solve(**common)
    monolithic = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        **common, step_solver="monolithic",
        multigroup_kwargs={"rtol": 1e-11})
    assert fixed.time_orders == monolithic.time_orders == [1, 2, 3, 3]
    np.testing.assert_allclose(monolithic.power, fixed.power,
                               rtol=3e-8, atol=2e-11)
    np.testing.assert_allclose(monolithic.flux_numpy, fixed.flux_numpy,
                               rtol=3e-8, atol=2e-11)


def test_monolithic_feedback_repeats_without_advancing_bdf_history():
    """A rejected constituent iterate re-solves, but accepts the step once."""
    steady = DiffusionEigenSolver(GRID, [BASE], device="cpu").solve(
        tol_k=1e-9, tol_source=1e-8)

    def xs_update_at(_t):
        # New identity on every request forces the intended operator rebuild;
        # the no-op value keeps an exact reference solution available.
        return lambda _fields: None

    calls = []

    def feedback_iteration(t, _flux, _power, iteration):
        calls.append((t, iteration))
        return iteration == 2

    result = TransientSolver(
        GRID, lambda _t: ([BASE], None), KIN, device="cpu",
        xs_update_at=xs_update_at,
        feedback_iteration=feedback_iteration,
    ).solve(
        t_end=0.02, dt=0.01, initial_steady=steady,
        time_scheme="bdf2", step_solver="monolithic", tol_step=1e-9,
        multigroup_kwargs={"rtol": 1e-11})

    assert calls == [(0.01, 1), (0.01, 2), (0.02, 1), (0.02, 2)]
    assert result.feedback_iterations == [2, 2]
    # If rejected iterations had pushed history, the second accepted step
    # would no longer be the sole BDF2 startup transition.
    assert result.time_orders == [1, 2]
    np.testing.assert_allclose(result.power, 1.0, rtol=0, atol=1e-8)


def test_nonuniform_bdf_schedule_preserves_diffusion_equilibrium():
    widths = [0.007, 0.011, 0.004, 0.013, 0.009, 0.016]
    result = TransientSolver(
        GRID, lambda _t: ([BASE], None), KIN, device="cpu").solve(
            t_end=sum(widths), dt=widths, time_scheme="bdf5",
            tol_step=1e-9)
    np.testing.assert_allclose(result.times,
                               np.r_[0.0, np.cumsum(widths)], atol=2e-16)
    assert result.time_orders == [1, 2, 3, 4, 5, 5]
    np.testing.assert_allclose(result.power, 1.0, rtol=0, atol=2e-9)


def test_explicit_step_schedule_validation_is_strict():
    solver = TransientSolver(GRID, lambda _t: ([BASE], None), KIN,
                             device="cpu")
    with pytest.raises(ValueError, match="sum to t_end"):
        solver.solve(t_end=0.1, dt=[0.04, 0.04])
    with pytest.raises(ValueError, match="finite and positive"):
        solver.solve(t_end=0.1, dt=[0.05, -0.05, 0.1])
    with pytest.raises(NotImplementedError, match="BDF schemes only"):
        solver.solve(t_end=0.1, dt=[0.05, 0.05],
                     time_scheme="crank-nicolson")


def test_bdf_history_restart_reduces_order_without_disturbing_equilibrium():
    result = TransientSolver(
        GRID, lambda _t: ([BASE], None), KIN, device="cpu").solve(
            t_end=0.08, dt=0.01, time_scheme="bdf5",
            bdf_restart_times=[0.04], tol_step=1e-9)
    assert result.time_orders == [1, 2, 3, 1, 2, 3, 4, 5]
    np.testing.assert_allclose(result.power, 1.0, rtol=0, atol=2e-9)


def test_bdf_restart_must_be_aligned_with_the_step_schedule():
    solver = TransientSolver(GRID, lambda _t: ([BASE], None), KIN,
                             device="cpu")
    with pytest.raises(ValueError, match="coincide with a step end"):
        solver.solve(t_end=0.08, dt=0.01, time_scheme="bdf3",
                     bdf_restart_times=[0.045])


def test_adaptive_bdf_rejects_aligns_event_and_restarts_order():
    """Accepted history advances once; a ramp corner is hit exactly."""
    cache = {}

    def ramp_problem(t):
        fraction = min(float(t) / 0.01, 1.0)
        key = round(fraction, 12)
        if key not in cache:
            cache[key] = [Material(
                diffusion=[D], sigma_a=[SIGMA_A - 0.0005 * fraction],
                nu_sigma_f=[NU_SIGMA_F])]
        return cache[key], None

    committed_times = []
    adaptive_error_calls = []

    def adaptive_error_state(t, _dt):
        adaptive_error_calls.append(t)
        marker = np.asarray([t])
        return [marker], [marker]

    result = TransientSolver(
        GRID, ramp_problem, KIN, device="cpu",
        xs_update_at=lambda _t: (lambda _fields: None),
        feedback_iteration=lambda _t, _flux, _power, _iteration: True,
        on_step=lambda t, _flux, _power: committed_times.append(t),
        adaptive_error_state=adaptive_error_state,
    ).solve(
            t_end=0.02, dt=0.01, time_scheme="bdf3",
            bdf_restart_times=[0.01], step_solver="monolithic",
            adaptive_bdf={"rtol": 1e-2, "min_dt": 1e-6,
                          "max_dt": 0.01},
            tol_step=1e-9, multigroup_kwargs={"rtol": 1e-11})

    assert result.rejected_steps > 0
    assert len(result.rejected_errors) == len(result.rejected_step_iterations)
    assert len(result.rejected_errors) == len(result.rejected_times)
    assert len(result.rejected_errors) == len(result.rejected_step_widths)
    assert len(result.rejected_errors) == len(result.rejected_time_orders)
    assert all(width > 0.0 for width in result.rejected_step_widths)
    assert len(result.step_widths) == len(result.local_errors) == len(
        result.times) - 1
    assert max(result.local_errors) <= 1.0
    assert min(result.rejected_errors) > 1.0
    assert sum(result.step_widths) == pytest.approx(0.02, abs=2e-15)
    assert result.times[-1] == pytest.approx(0.02, abs=2e-15)
    np.testing.assert_allclose(committed_times, result.times[1:], atol=2e-15)
    assert len(adaptive_error_calls) == len(result.step_widths) + result.rejected_steps
    event = int(np.flatnonzero(np.isclose(result.times, 0.01))[0])
    # time_orders indexes steps, so the first step leaving the event is event.
    assert result.time_orders[event] == 1
    assert result.power[-1] == pytest.approx(3.634, abs=2e-2)


def test_adaptive_bdf_automatic_order_selects_and_reports_candidates():
    prob = absorption_step_problem(0.00005)
    result = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        t_end=0.04, dt=0.001, time_scheme="bdf4",
        step_solver="monolithic", tol_step=1e-9,
        adaptive_bdf={"rtol": 1e-2, "min_dt": 1e-6,
                      "max_dt": 0.005, "automatic_order": True},
        multigroup_kwargs={"rtol": 1e-11})
    assert len(result.next_time_orders) == len(result.step_widths)
    assert len(result.order_candidate_errors) == len(result.step_widths)
    assert set(result.time_orders) <= {1, 2, 3, 4}
    assert max(result.time_orders) > 1
    assert all(order in candidates for order, candidates in zip(
        result.time_orders, result.order_candidate_errors))


def test_adaptive_backward_euler_uses_same_accept_reject_controller():
    result = TransientSolver(
        GRID, absorption_step_problem(0.00005), KIN, device="cpu").solve(
            t_end=0.02, dt=0.001, time_scheme="bdf1",
            step_solver="monolithic", tol_step=1e-9,
            adaptive_bdf={"rtol": 1e-2, "min_dt": 1e-6,
                          "max_dt": 0.005},
            multigroup_kwargs={"rtol": 1e-11})
    assert result.time_scheme == "backward-euler"
    assert set(result.time_orders) == {1}
    assert len(result.local_errors) == len(result.step_widths)
    assert max(result.local_errors) <= 1.0


@pytest.mark.parametrize("kwargs,match", [
    ({"adaptive_bdf": {"rtol": 1e-3}}, "monolithic"),
    ({"adaptive_bdf": {"rtol": 1e-3}, "step_solver": "monolithic",
      "time_scheme": "backward-euler"}, "bdf1..bdf6"),
    ({"adaptive_bdf": {"mystery": 1}, "step_solver": "monolithic",
      "time_scheme": "bdf2"}, "unknown adaptive_bdf"),
])
def test_adaptive_bdf_rejects_unsupported_configurations(kwargs, match):
    solver = TransientSolver(GRID, lambda _t: ([BASE], None), KIN,
                             device="cpu")
    with pytest.raises(ValueError, match=match):
        solver.solve(t_end=0.02, dt=0.01, **kwargs)


def test_rejects_a_horizon_that_is_not_an_integer_number_of_steps():
    """Constant-step solvers must not silently report a different end time."""
    solver = TransientSolver(GRID, lambda t: ([BASE], None), KIN, device="cpu")
    with pytest.raises(ValueError, match="integer multiple"):
        solver.solve(t_end=1.0, dt=0.06)


def test_default_step_acceleration_matches_the_documented_configuration():
    """Pin the public defaults used by existing transient input decks."""
    prob = absorption_step_problem(0.001)
    default = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        t_end=0.02, dt=0.002).power
    explicit = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        t_end=0.02, dt=0.002, anderson_depth=5, rebalance=False).power
    np.testing.assert_allclose(default, explicit, rtol=0, atol=1e-12)


def test_opt_in_monolithic_steps_match_fixed_point_history():
    """The direct block solve changes convergence, not transient equations."""
    prob = absorption_step_problem(0.001)
    steady = DiffusionEigenSolver(GRID, [BASE], device="cpu").solve(
        tol_k=1e-9, tol_source=1e-8)
    common = dict(t_end=0.006, dt=0.002, initial_steady=steady)
    fixed = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        **common, tol_step=1e-9)
    monolithic = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        **common, step_solver="monolithic", tol_step=1e-9,
        multigroup_kwargs={"rtol": 1e-11})
    assert fixed.step_method == "fixed-point"
    assert monolithic.step_method == "monolithic"
    np.testing.assert_allclose(monolithic.power, fixed.power,
                               rtol=2e-8, atol=1e-11)
    np.testing.assert_allclose(monolithic.flux_numpy, fixed.flux_numpy,
                               rtol=2e-8, atol=1e-11)


def test_monolithic_adjoint_coarse_correction_is_opt_in_and_counted():
    prob = absorption_step_problem(0.001)
    result = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        t_end=0.004, dt=0.002, step_solver="monolithic",
        multigroup_kwargs={"coarse_correction": True, "rtol": 1e-10})
    assert result.step_method == "monolithic"
    assert result.adjoint_coarse_setups == 2
    assert np.all(np.isfinite(result.power))


def test_monolithic_energy_anderson_is_opt_in_and_preserves_solution():
    prob = absorption_step_problem(0.001)
    common = dict(t_end=0.004, dt=0.002, step_solver="monolithic",
                  tol_step=1e-9)
    plain = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        **common, multigroup_kwargs={"rtol": 1e-11})
    accelerated = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        **common, multigroup_kwargs={"rtol": 1e-11,
                                     "scatter_sweeps": 3,
                                     "energy_anderson": 1})
    reduction_free = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        **common, multigroup_kwargs={"rtol": 1e-11,
                                     "scatter_sweeps": 4,
                                     "energy_anderson": 1,
                                     "inner_fixed_relaxations": 1})
    np.testing.assert_allclose(
        accelerated.power, plain.power, rtol=2e-9, atol=2e-12)
    np.testing.assert_allclose(
        accelerated.flux_numpy, plain.flux_numpy, rtol=2e-9, atol=2e-12)
    np.testing.assert_allclose(
        reduction_free.power, plain.power, rtol=2e-9, atol=2e-12)


def test_monolithic_step_options_fail_fast():
    solver = TransientSolver(
        GRID, lambda t: ([BASE], None), KIN, device="cpu")
    with pytest.raises(ValueError, match="step_solver"):
        solver.solve(t_end=0.01, dt=0.01, step_solver="direct")
    with pytest.raises(ValueError, match="unknown multigroup_kwargs"):
        solver.solve(t_end=0.01, dt=0.01, step_solver="monolithic",
                     multigroup_kwargs={"mystery": 1})
    with pytest.raises(ValueError, match="coarse_adjoint shape"):
        solver.solve(t_end=0.01, dt=0.01, step_solver="monolithic",
                     multigroup_kwargs={"coarse_adjoint": np.ones((2, 3))})


def test_reused_pcg_workspaces_preserve_transient_history_and_work():
    """Persistent storage is an execution optimization, not new numerics."""
    steady = DiffusionEigenSolver(GRID, [BASE], device="cpu").solve()
    prob = absorption_step_problem(0.001)
    results = {}
    for reuse in (False, True):
        results[reuse] = TransientSolver(
            GRID, prob, KIN, device="cpu").solve(
                t_end=0.006, dt=0.002, initial_steady=steady,
                reuse_krylov_workspaces=reuse)
    old, new = results[False], results[True]
    assert new.total_inner_iterations == old.total_inner_iterations
    assert new.step_iterations == old.step_iterations
    np.testing.assert_array_equal(new.power, old.power)
    np.testing.assert_array_equal(new.flux_numpy, old.flux_numpy)


def test_fp32_preconditioner_preserves_fp64_transient_accuracy():
    steady = DiffusionEigenSolver(GRID, [BASE], device="cpu").solve()
    prob = absorption_step_problem(0.001)
    common = dict(t_end=0.006, dt=0.002, initial_steady=steady)
    full = TransientSolver(
        GRID, prob, KIN, device="cpu", precond_degree=1).solve(**common)
    mixed = TransientSolver(
        GRID, prob, KIN, device="cpu", precond_degree=1,
        precond_dtype=np.float32).solve(**common)
    assert mixed.mixed_precision_fallbacks == 0
    assert mixed.step_iterations == full.step_iterations
    np.testing.assert_allclose(mixed.power, full.power, rtol=5e-8, atol=0.0)
    np.testing.assert_allclose(
        mixed.flux_numpy, full.flux_numpy, rtol=5e-8, atol=1e-12)


def test_mixed_preconditioner_dtype_is_conservative():
    prob = absorption_step_problem(0.001)
    with pytest.raises(ValueError, match="only float32"):
        TransientSolver(
            GRID, prob, KIN, device="cpu", precond_dtype=np.float16)
    with pytest.raises(ValueError, match="narrower"):
        TransientSolver(
            GRID, prob, KIN, device="cpu", dtype=np.float32,
            precond_dtype=np.float32)


def test_full_fp32_transient_stays_in_fp32():
    prob = absorption_step_problem(0.001)
    result = TransientSolver(
        GRID, prob, KIN, device="cpu", dtype=np.float32).solve(
            t_end=0.004, dt=0.002)
    assert result.flux.dtype == np.float32
    assert result.precursors.dtype == np.float32
    assert np.all(np.isfinite(result.power))


def test_compatible_initial_steady_skips_eigen_solve_without_moving_history():
    """A coupled equilibrium hand-off is an optimization, not new physics."""
    steady = DiffusionEigenSolver(GRID, [BASE], device="cpu").solve(
        tol_k=1e-8, tol_source=1e-7)
    prob = absorption_step_problem(0.001)
    reference = TransientSolver(GRID, prob, KIN, device="cpu").solve(
        t_end=0.02, dt=0.002)

    class NoSecondEigenSolve(DiffusionEigenSolver):
        def solve(self, **_kw):  # pragma: no cover - failure path is the test
            raise AssertionError("initial eigenvalue solve was not skipped")

    reused = TransientSolver(
        GRID, prob, KIN, eig_solver=NoSecondEigenSolve,
        device="cpu").solve(t_end=0.02, dt=0.002,
                            initial_steady=steady)
    assert reused.initial_state_reused
    assert not reference.initial_state_reused
    assert reused.k0 == pytest.approx(reference.k0, abs=1e-10)
    np.testing.assert_allclose(reused.power, reference.power, rtol=0, atol=2e-10)
    np.testing.assert_allclose(reused.flux_numpy, reference.flux_numpy,
                               rtol=0, atol=2e-9)


def test_initial_steady_handoff_validates_shape_and_convergence():
    import copy

    steady = DiffusionEigenSolver(GRID, [BASE], device="cpu").solve()
    solver = TransientSolver(GRID, lambda t: ([BASE], None), KIN, device="cpu")
    bad = copy.copy(steady)
    bad.converged = False
    with pytest.raises(ValueError, match="must be converged"):
        solver.solve(t_end=0.01, dt=0.01, initial_steady=bad)
    bad = copy.copy(steady)
    bad.flux = bad.flux[:, :-1]
    with pytest.raises(ValueError, match="flux shape"):
        solver.solve(t_end=0.01, dt=0.01, initial_steady=bad)


def test_spatial_precursor_handoff_restarts_without_losing_history():
    import copy

    eps = 0.001
    perturbed = Material(
        name="perturbed", diffusion=[D],
        sigma_a=[SIGMA_A - eps * NU_SIGMA_F],
        nu_sigma_f=[NU_SIGMA_F])
    problem = lambda t: ([BASE] if t <= 0.0 else [perturbed], None)
    full = TransientSolver(GRID, problem, KIN, device="cpu").solve(
        t_end=0.02, dt=0.002)
    first = TransientSolver(GRID, problem, KIN, device="cpu").solve(
        t_end=0.01, dt=0.002)

    restart = copy.copy(first.steady)
    restart.k_eff = first.k0
    restart.flux = first.flux / first.power[-1]
    second = TransientSolver(
        GRID, lambda _t: ([perturbed], None), KIN,
        device="cpu").solve(
            t_end=0.01, dt=0.002, initial_steady=restart,
            initial_precursors=first.precursors / first.power[-1])
    joined_power = np.concatenate(
        (first.power, first.power[-1] * second.power[1:]))
    np.testing.assert_allclose(joined_power, full.power, rtol=0, atol=3e-12)
    np.testing.assert_allclose(
        first.power[-1] * second.flux_numpy, full.flux_numpy,
        rtol=0, atol=3e-11)


def test_matches_point_kinetics_for_uniform_perturbation():
    from scipy.integrate import solve_ivp

    # Get the initial eigenvalue for the exact critical adjustment (a = nuSf/k0),
    # then size the absorption drop to a +$0.50 step (well below prompt critical).
    k0 = TransientSolver(GRID, lambda t: ([BASE], None), KIN, device="cpu").solve(
        t_end=1e-3, dt=1e-3).k0
    a = NU_SIGMA_F / k0
    rho_dollars = 0.5
    d_sigma_a = rho_dollars * BETA * a

    t_end, dt = 0.5, 1e-4
    res = TransientSolver(GRID, absorption_step_problem(d_sigma_a), KIN,
                          device="cpu").solve(t_end=t_end, dt=dt)

    # Exact point kinetics for this absorption step. With the critical
    # adjustment DB^2 + Sigma_a = a, dropping Sigma_a by d_sigma_a gives
    #   (1/v) dn/dt = (d_sigma_a - beta a) n + lam C,  dC/dt = beta a n - lam C.
    rhs = lambda t, y: [V * ((d_sigma_a - BETA * a) * y[0] + LAM * y[1]),
                        BETA * a * y[0] - LAM * y[1]]
    ref = solve_ivp(rhs, (0, t_end), [1.0, BETA * a / LAM], method="Radau",
                    t_eval=res.times, rtol=1e-11, atol=1e-13)

    err = np.max(np.abs(res.power - ref.y[0]) / ref.y[0])
    assert err < 5e-4, f"max relative deviation from point kinetics: {err:.2e}"
    # The prompt jump is real and resolved: power reaches the beta/(beta-rho)
    # plateau (= 2.0 at $0.50) well before the delayed rise carries it past.
    plateau = 1.0 / (1.0 - rho_dollars)
    i_plateau = np.searchsorted(res.times, 0.2)
    assert res.power[i_plateau] == pytest.approx(plateau, abs=0.05), res.power[i_plateau]
    assert res.power[-1] > plateau  # delayed supercritical rise continues


# --- transient SDP1 (simplified double-P1 kinetics) ----------------------------

def test_sdp1_unperturbed_transient_stays_steady():
    """The steady SDP1 state is an exact fixed point of the transient block: with
    no perturbation the power holds at 1, confirming the two-moment time terms
    and the block RHS are mutually consistent."""
    res = TransientSDP1Solver(GRID, lambda t: ([BASE], None), KIN,
                              device="cpu").solve(t_end=0.5, dt=0.05)
    assert np.allclose(res.power, 1.0, atol=1e-5), res.power


@pytest.mark.parametrize("variant,coeffs_name,row_scale",
                         [("sp3", "_SPN_C", 3.0), ("sdp1", None, 7.0 / 3.0)])
def test_transient_block_time_terms_match_uform_time_matrix(
        variant, coeffs_name, row_scale):
    """The dedicated two-moment transient block (SP3GroupOperator with theta)
    must act identically to the general U-form operator with theta, whose time
    matrix theta * sum_m c^(m) is pinned to first principles in
    test_sdpn_derivation.py. Variable map: Phi1 = U1, phi2 = U2/3; the
    dedicated (5x-scaled) row 1 is row_scale times the U-form row 2 (3 for
    SP3, 7/3 for SDP1 -- their U-form rows differ, but both map to the same
    dedicated-row time term theta*(9 phi2 - 2 Phi1)). This equivalence is
    exactly what the old, incomplete row-1 time term (theta*5*phi2) violated.
    """
    import ndgpu.operator as op_mod
    from ndgpu.operator import SDPNGroupOperator, SP3GroupOperator
    from ndgpu import Grid

    coeffs = getattr(op_mod, coeffs_name) if coeffs_name else None
    rng = np.random.default_rng(7)
    grid = Grid(shape=(6, 5, 1), size=(7.0, 6.0, 1.0))
    shp = grid.shape
    D1 = 0.5 + rng.random(shp)
    sigma_t = 1.0 + rng.random(shp)
    removal = 0.1 + 0.3 * rng.random(shp)
    for theta in (None, 0.37):
        ded = SP3GroupOperator(np, grid, D1, sigma_t, removal, bc="zero-flux",
                               variant=variant, theta=theta)
        gen = SDPNGroupOperator(np, grid, D1, sigma_t, removal, order=1,
                                bc="zero-flux", coeffs=coeffs, theta=theta)
        u_ded = rng.random((2,) + shp)                # (Phi1, phi2)
        u_gen = np.stack([u_ded[0], 3.0 * u_ded[1]])  # (U1, U2) = (Phi1, 3 phi2)
        out_ded = ded.apply(u_ded)
        out_gen = gen.apply(u_gen)
        assert np.allclose(out_ded[0], out_gen[0], atol=1e-12), (variant, theta)
        assert np.allclose(out_ded[1], row_scale * out_gen[1], atol=1e-12), \
            (variant, theta)


def test_sdpn_transient_order1_matches_dedicated_sdp1():
    """TransientSDPNSolver at order 1 solves the same equations as the
    dedicated TransientSDP1Solver (different variables and Krylov solver), so
    their power traces must agree to solver tolerance."""
    from ndgpu import TransientSDPNSolver

    d_sigma_a = 0.3 * BETA * NU_SIGMA_F
    prob = absorption_step_problem(d_sigma_a)
    kw = dict(t_end=0.02, dt=1e-3)
    p_ded = TransientSDP1Solver(GRID, prob, KIN, device="cpu").solve(**kw).power
    p_gen = TransientSDPNSolver(GRID, prob, KIN, order=1,
                                device="cpu").solve(**kw).power
    assert np.allclose(p_ded, p_gen, rtol=0, atol=2e-5), \
        np.max(np.abs(p_ded - p_gen))


def test_spn_transient_order0_matches_diffusion_transient():
    """TransientSPNSolver at order 0 is SP1 = P1/diffusion kinetics through
    the U-form block machinery, so its power trace must reproduce
    TransientSolver to solver tolerance."""
    from ndgpu import TransientSPNSolver

    d_sigma_a = 0.3 * BETA * NU_SIGMA_F
    prob = absorption_step_problem(d_sigma_a)
    kw = dict(t_end=0.02, dt=1e-3)
    p_dif = TransientSolver(GRID, prob, KIN, device="cpu").solve(**kw).power
    p_sp1 = TransientSPNSolver(GRID, prob, KIN, order=0,
                               device="cpu").solve(**kw).power
    assert np.allclose(p_dif, p_sp1, rtol=0, atol=2e-5), \
        np.max(np.abs(p_dif - p_sp1))


def test_sdp1_transient_tracks_point_kinetics_and_diffusion():
    """For a uniform absorption step the SDP1 power tracks point kinetics (with
    k0 from the SDP1 eigenvalue) to the same backward-Euler O(dt) level as the
    diffusion kinetics (measured ~3.5e-4 at dt = 1e-4). Before the exact
    even-moment time matrix (theta * sum_m c^(m)) this sat at a few 1e-3 -- the
    incomplete row-1 time term, not a transport effect."""
    from scipy.integrate import solve_ivp

    k0 = TransientSDP1Solver(GRID, lambda t: ([BASE], None), KIN,
                             device="cpu").solve(t_end=1e-3, dt=1e-3).k0
    a = NU_SIGMA_F / k0
    rho_dollars = 0.5
    d_sigma_a = rho_dollars * BETA * a

    t_end, dt = 0.5, 1e-4
    res = TransientSDP1Solver(GRID, absorption_step_problem(d_sigma_a), KIN,
                              device="cpu").solve(t_end=t_end, dt=dt)

    rhs = lambda t, y: [V * ((d_sigma_a - BETA * a) * y[0] + LAM * y[1]),
                        BETA * a * y[0] - LAM * y[1]]
    ref = solve_ivp(rhs, (0, t_end), [1.0, BETA * a / LAM], method="Radau",
                    t_eval=res.times, rtol=1e-11, atol=1e-13)
    err = np.max(np.abs(res.power - ref.y[0]) / ref.y[0])
    assert err < 5e-4, f"max relative deviation from point kinetics: {err:.2e}"
    # Prompt jump to beta/(beta-rho) is resolved.
    plateau = 1.0 / (1.0 - rho_dollars)
    i_plateau = np.searchsorted(res.times, 0.2)
    assert res.power[i_plateau] == pytest.approx(plateau, abs=0.05)


def test_sdp3_unperturbed_transient_stays_steady():
    """The steady SDP3 state is an exact fixed point of the 4-moment transient
    block: the theta terms and their backward-Euler sources cancel exactly in
    equilibrium at every moment."""
    from ndgpu import TransientSDP3Solver

    res = TransientSDP3Solver(GRID, lambda t: ([BASE], None), KIN,
                              device="cpu").solve(t_end=0.5, dt=0.05)
    assert np.allclose(res.power, 1.0, atol=1e-5), res.power


def test_sdp3_transient_tracks_point_kinetics():
    """SDP3 kinetics under the uniform absorption step: the 4-moment U-form
    block with the exact time matrix tracks point kinetics to the same
    backward-Euler O(dt) level as SDP1/diffusion (measured ~3.4e-4 at
    dt = 1e-4 over the full 0.5 s window; a shorter window here keeps the test
    fast -- it still covers the prompt jump and the start of the delayed
    rise)."""
    from scipy.integrate import solve_ivp
    from ndgpu import TransientSDP3Solver

    k0 = TransientSDP3Solver(GRID, lambda t: ([BASE], None), KIN,
                             device="cpu").solve(t_end=1e-3, dt=1e-3).k0
    a = NU_SIGMA_F / k0
    rho_dollars = 0.5
    d_sigma_a = rho_dollars * BETA * a

    t_end, dt = 0.1, 1e-4
    res = TransientSDP3Solver(GRID, absorption_step_problem(d_sigma_a), KIN,
                              device="cpu").solve(t_end=t_end, dt=dt)

    rhs = lambda t, y: [V * ((d_sigma_a - BETA * a) * y[0] + LAM * y[1]),
                        BETA * a * y[0] - LAM * y[1]]
    ref = solve_ivp(rhs, (0, t_end), [1.0, BETA * a / LAM], method="Radau",
                    t_eval=res.times, rtol=1e-11, atol=1e-13)
    err = np.max(np.abs(res.power - ref.y[0]) / ref.y[0])
    assert err < 5e-4, f"max relative deviation from point kinetics: {err:.2e}"
    # Most of the prompt jump toward beta/(beta-rho) = 2 develops in-window
    # (the approach time constant is (beta-rho)/Lambda ~ 0.04 s).
    assert res.power[-1] > 1.8
