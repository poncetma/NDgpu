"""Algebraic verification of the shared BDF history representation."""

import numpy as np
import pytest

from ndgpu.timescheme import BDF, BDFStepController, make_time_scheme


@pytest.mark.parametrize("order", range(1, 7))
def test_bdf_coefficients_differentiate_polynomials_exactly(order):
    """The q-step formula is exact for monomials through degree q."""
    a = np.asarray(BDF(order).coefficients(order))
    j = -np.arange(order + 1, dtype=float)
    assert np.sum(a) == pytest.approx(0.0, abs=2e-14)
    for degree in range(1, order + 1):
        got = np.sum(a * j**degree)
        expected = 1.0 if degree == 1 else 0.0
        assert got == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("max_order", range(1, 7))
def test_bdf_ramps_order_and_carried_field_matches_history(max_order):
    scheme = BDF(max_order)
    scheme.start([np.array(1.0)])
    accepted = [1.0]
    for step in range(1, max_order + 2):
        q = min(step, max_order)
        assert scheme.order_at(step) == q
        a = np.asarray(scheme.coefficients(step))
        expected_h = sum(-a[j] * accepted[-j] for j in range(1, q + 1))
        assert float(scheme.carried(step)[0]) == pytest.approx(
            expected_h / a[0])
        value = float(step + 1)
        scheme.push([np.array(value)])
        accepted.append(value)


@pytest.mark.parametrize("max_order", range(1, 7))
def test_bdf_precursor_elimination_preserves_equilibrium(max_order):
    """Flux and precursor histories must cancel at a critical equilibrium."""
    scheme = BDF(max_order)
    beta, lam, dt, source = [0.0065], [0.08], 0.125, 2.5
    equilibrium = beta[0] * source / lam[0]
    C_hist = [[np.array(equilibrium)]]
    for step in range(1, max_order + 3):
        bcoef = scheme.precursor_bcoef(step, beta, lam, dt)[0]
        decayed = scheme.precursor_decayed(
            step, C_hist, np.array(source), beta, lam, dt)[0]
        # The eliminated end-of-step precursor contribution removed from the
        # prompt/total fission weight is restored exactly by the history term.
        assert float(decayed) == pytest.approx(bcoef * source, abs=2e-15)
        C_new = scheme.precursor_update(
            step, C_hist, np.array(source), np.array(source),
            beta, lam, dt)
        assert float(C_new[0]) == pytest.approx(equilibrium, abs=2e-14)
        C_hist.insert(0, C_new)
        del C_hist[max_order:]


def test_bdf_names_are_publicly_resolved_and_invalid_orders_fail():
    for order in range(1, 7):
        scheme = make_time_scheme(f"bdf{order}")
        assert scheme.order == order
    with pytest.raises(ValueError, match="between 1 and 6"):
        make_time_scheme("bdf7")


def test_variable_step_bdf_coefficients_are_exact_on_nonuniform_nodes():
    widths = [0.07, 0.11, 0.04, 0.13, 0.09, 0.16]
    scheme = BDF(5)
    scheme.start([np.array(1.0)])
    accepted_widths = []
    for step, width in enumerate(widths, 1):
        scheme.prepare_step(step, width)
        q = scheme.order_at(step)
        a = np.asarray(scheme.coefficients(step))
        nodes = [0.0]
        distance = width
        for j in range(1, q + 1):
            nodes.append(-distance / width)
            if j < q:
                distance += accepted_widths[-j]
        nodes = np.asarray(nodes)
        for degree in range(q + 1):
            expected = 1.0 if degree == 1 else 0.0
            assert np.sum(a * nodes**degree) == pytest.approx(
                expected, abs=2e-10)
        scheme.push([np.array(step + 1.0)])
        accepted_widths.append(width)


def test_bdf_predictor_is_exact_for_nonuniform_polynomial_history():
    scheme = BDF(5)
    times = [0.0, 0.07, 0.18, 0.22, 0.35]

    def value(t):
        return 1.0 - 2.0*t + 0.5*t**2 + 0.25*t**3 - 0.1*t**4

    scheme.start([np.array(value(times[0]))])
    for step, (old, new) in enumerate(zip(times, times[1:]), 1):
        width = new - old
        scheme.prepare_step(step, width)
        scheme.push([np.array(value(new))])
    target_width = 0.09
    predicted = float(scheme.predict(target_width)[0])
    assert predicted == pytest.approx(value(times[-1] + target_width), abs=2e-13)


def test_bdf_predictor_retains_extra_state_for_degree_max_order():
    scheme = BDF(5)
    times = [0.0, 0.02, 0.07, 0.11, 0.20, 0.31]

    def value(t):
        return 1.0 + t - 2*t**2 + 0.3*t**3 - 0.2*t**4 + 0.1*t**5

    scheme.start([np.array(value(0.0))])
    for step, (old, new) in enumerate(zip(times, times[1:]), 1):
        scheme.prepare_step(step, new - old)
        scheme.push([np.array(value(new))])
    assert len(scheme._history_u) == 6
    assert float(scheme.predict(0.08)[0]) == pytest.approx(
        value(times[-1] + 0.08), abs=2e-12)


def test_bdf_predicts_an_external_history_on_the_same_time_nodes():
    scheme = BDF(3)
    times = [0.0, 0.03, 0.08, 0.17]
    scheme.start([np.array(1.0)])
    external = [[np.array(2.0 - t + 3.0*t**2 - 0.5*t**3)]
                for t in times]
    for step, (old, new) in enumerate(zip(times, times[1:]), 1):
        scheme.prepare_step(step, new - old)
        scheme.push([np.array(new)])
    history = list(reversed(external))
    target = times[-1] + 0.06
    expected = 2.0 - target + 3.0*target**2 - 0.5*target**3
    assert float(scheme.predict_history(history, 0.06)[0]) == pytest.approx(
        expected, abs=2e-13)


def test_bdf_predictor_corrector_error_matches_published_norm():
    corrected = [np.array([2.0, -4.0]), np.array([1.0])]
    predicted = [np.array([1.8, -4.4]), np.array([1.1])]
    rtol, atol = 2e-2, 1e-3
    numerator = np.linalg.norm(np.concatenate([
        (corrected[i] - predicted[i]).ravel() for i in range(2)]))
    denominator = np.linalg.norm(np.concatenate([
        (atol + rtol * np.abs(corrected[i])).ravel() for i in range(2)]))
    assert BDF.error_norm(corrected, predicted, rtol=rtol, atol=atol) \
        == pytest.approx(numerator / denominator)


def test_bdf_step_controller_rejects_by_half_and_bounds_growth():
    controller = BDFStepController(max_factor=4.0)
    assert controller.propose(0.08, 1.01, 5) == pytest.approx(0.04)
    assert controller.propose(0.08, np.inf, 5) == pytest.approx(0.04)
    assert controller.propose(0.08, 0.0, 5) == pytest.approx(0.32)
    # A just-passing error is conservatively reduced with safety > 1.
    assert controller.factor(1.0, 2) < 1.0


def test_bdf_step_controller_can_scale_rejection_to_the_error():
    controller = BDFStepController(rejection_strategy="error")
    assert controller.factor(1.0, 5, accepted=False) == pytest.approx(0.5)
    assert controller.factor(64.0, 5, accepted=False) == pytest.approx(0.4)
    assert controller.factor(np.inf, 5, accepted=False) == pytest.approx(0.2)


def test_bdf_automatic_order_selection_controls_prepared_recurrence():
    scheme = BDF(4)
    scheme.enable_order_selection()
    scheme.start([np.array(1.0)])
    assert scheme.order_at(1) == 1
    for step in range(1, 5):
        scheme.prepare_step(step, 0.1)
        scheme.push([np.array(1.0 + step)])
    scheme.select_order(3)
    assert scheme.order_at(5) == 3
    scheme.prepare_step(5, 0.1)
    assert len(scheme.coefficients(5)) == 4
    scheme.select_order(2)
    assert scheme.order_at(5) == 2
    with pytest.raises(ValueError, match="selected BDF order"):
        scheme.select_order(5)


def test_bdf_controller_selects_largest_width_with_hysteresis():
    controller = BDFStepController()
    order, width = controller.choose_order(
        0.01, {2: 0.5, 3: 0.05, 4: 0.8}, 3)
    proposals = {q: controller.propose(0.01, e, q)
                 for q, e in {2: 0.5, 3: 0.05, 4: 0.8}.items()}
    assert order == max(proposals, key=proposals.get)
    assert width == pytest.approx(proposals[order])

    # A negligible gain does not justify changing recurrence order.
    order, _ = controller.choose_order(
        0.01, {2: 0.2, 3: 0.2}, 2, hysteresis=100.0)
    assert order == 2


@pytest.mark.parametrize("kwargs", [
    {"safety": 1.0}, {"floor": -1.0},
    {"min_factor": 0.0}, {"max_factor": 0.5},
    {"rejection_strategy": "mystery"}, {"reject_max_factor": 1.0},
])
def test_bdf_step_controller_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        BDFStepController(**kwargs)


@pytest.mark.parametrize("max_order,min_ratio", [(1, 1.8), (2, 3.2),
                                                   (3, 5.0), (4, 7.0)])
def test_bdf_converges_on_smooth_stiff_decay(max_order, min_ratio):
    """CPU-only manufactured ODE gate for the complete history recurrence."""
    lam, t_end = -4.0, 2.0

    def integrate(dt):
        scheme = BDF(max_order)
        y = np.array(np.exp(lam * 0.0))
        scheme.start([y])
        # Supply exact startup values here to isolate the q-step recurrence;
        # startup ramping is tested independently above.
        for step in range(1, max_order):
            y = np.array(np.exp(lam * step * dt))
            scheme.push([y])
        for step in range(max_order, int(round(t_end / dt)) + 1):
            a0 = scheme.a0(step)
            carried = scheme.carried(step)[0]
            y = np.array(a0 * carried / (a0 - dt * lam))
            scheme.push([y])
        return float(y)

    exact = np.exp(lam * t_end)
    coarse = abs(integrate(0.05) - exact)
    fine = abs(integrate(0.025) - exact)
    assert coarse / fine > min_ratio
