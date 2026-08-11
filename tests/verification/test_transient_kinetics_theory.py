"""The transient against point-kinetics theory, on the HP-MR triangular mesh.

These are the checks that caught nothing wrong when a coupled run *looked*
wrong -- and that would have caught a real kinetics regression immediately.
Both are independent of the neutron generation time, and the reactivity they
compare against comes from two STATIC eigenvalue solves, so theory and
measurement share no machinery:

* a step insertion of reactivity rho makes the power jump promptly to
  beta / (beta - rho);
* it then grows on the period the inhour equation gives, which for a single
  delayed family and negligible Lambda/T is T = (beta - rho) / (lambda rho).

Run on the 2-group placeholder set, which is legitimate here: none of this is a
physics *value*, only a relation that must hold whatever the cross sections are.
"""

import numpy as np
import pytest

from ndgpu import TransientSolver, TriDiffusionEigenSolver
from ndgpu.benchmarks.hpmr import HPMR_KINETICS, build_hpmr2d
from ndgpu.tri import TriGroupOperator

REFINE = 2
BETA = float(HPMR_KINETICS.beta.sum())
LAM = float(HPMR_KINETICS.decay[0])


def _problem(angle):
    return build_hpmr2d(refine=REFINE, drum_angle_deg=float(angle),
                        absorber="polar")


def _k(p):
    return TriDiffusionEigenSolver(
        p.grid, p.materials, p.material_map, bc=p.bc, active=p.active,
        mask_bc=p.mask_bc, mix_material=p.mix_material,
        mix_weight=p.mix_weight, device="cpu").solve(tol_k=1e-10,
                                                     tol_source=1e-9).k_eff


def _solver(p, problem_at=None):
    return TransientSolver(
        p.grid, problem_at or (lambda t: (p.materials, p.material_map)),
        HPMR_KINETICS, bc=p.bc, active=p.active, mask_bc=p.mask_bc,
        mix_material=p.mix_material, mix_weight=p.mix_weight,
        group_operator=TriGroupOperator, eig_solver=TriDiffusionEigenSolver,
        device="cpu")


def _step(p_from, p_to, t_switch):
    """Materials fixed, blend jumps -- i.e. the drum rotates instantaneously."""
    a = (p_from.materials, p_from.material_map, p_from.mix_material, p_from.mix_weight)
    b = (p_to.materials, p_to.material_map, p_to.mix_material, p_to.mix_weight)
    return lambda t: (b if t >= t_switch else a)


def _insertion(angle_from=90.0, angle_to=94.0):
    p0, p1 = _problem(angle_from), _problem(angle_to)
    rho = 1e5 * (1.0 / _k(p0) - 1.0 / _k(p1))
    return p0, p1, rho


def test_unperturbed_transient_is_exactly_stationary():
    """The sharpest invariant available: an unperturbed core started from its
    own steady state must not move at all, to the last bit."""
    p = _problem(90.0)
    r = _solver(p).solve(t_end=2.0, dt=0.05)
    np.testing.assert_array_equal(r.power, np.ones_like(r.power))


def test_prompt_jump_matches_theory():
    p0, p1, rho = _insertion()
    assert rho > 0, "the test insertion should be positive"
    dt = 0.02
    r = _solver(p0, _step(p0, p1, 0.5 * dt)).solve(t_end=1.0, dt=dt)
    # Far enough past the step for the prompt transient to be over (the prompt
    # lifetime is microseconds) but early enough that delayed growth has not
    # yet moved the power appreciably.
    jump = float(r.power[int(np.argmin(np.abs(r.times - 0.2)))])
    theory = BETA / (BETA - rho / 1e5)
    assert jump == pytest.approx(theory, rel=0.02), (
        f"rho = {rho:.1f} pcm: jump {jump:.4f} vs theory {theory:.4f}")


def test_asymptotic_period_matches_the_inhour_equation():
    p0, p1, rho = _insertion()
    t_end = 30.0
    r = _solver(p0, _step(p0, p1, 0.01)).solve(t_end=t_end, dt=0.02)
    late = r.times > 0.6 * t_end
    period = 1.0 / np.polyfit(r.times[late], np.log(r.power[late]), 1)[0]
    theory = (BETA - rho / 1e5) / (LAM * rho / 1e5)
    assert period == pytest.approx(theory, rel=0.10), (
        f"rho = {rho:.1f} pcm: period {period:.1f} s vs theory {theory:.1f} s")


def test_a_negative_insertion_drops_the_power():
    """Sign check on the whole chain, including the blend swap."""
    p0, p1, rho = _insertion(90.0, 86.0)
    assert rho < 0
    dt = 0.02
    r = _solver(p0, _step(p0, p1, 0.5 * dt)).solve(t_end=1.0, dt=dt)
    jump = float(r.power[int(np.argmin(np.abs(r.times - 0.2)))])
    theory = BETA / (BETA - rho / 1e5)
    assert jump < 1.0
    assert jump == pytest.approx(theory, rel=0.02)


def test_the_answer_is_converged_in_time():
    """The step size must be a cost knob, not a physics one."""
    p0, p1, _ = _insertion()
    ends = []
    for dt in (0.1, 0.05, 0.02):
        r = _solver(p0, _step(p0, p1, 0.5 * dt)).solve(t_end=5.0, dt=dt)
        ends.append(float(r.power[-1]))
    assert abs(ends[-1] - ends[0]) / ends[-1] < 2e-3, ends


def test_drum_worth_is_measured_not_assumed():
    """Manoeuvres are specified in dollars because degrees are not a fixed
    reactivity -- the drum is nearly withdrawn above ~150 deg, so the whole
    travel from there to 180 is worth a fraction of what a few degrees near
    90 deg buys."""
    from ndgpu.benchmarks.hpmr_thermal import (hpmr_angle_for_dollars,
                                               hpmr_drum_worth)

    near = hpmr_drum_worth(90.0, [96.0], refine=REFINE)[96.0]
    far = hpmr_drum_worth(150.0, [156.0], refine=REFINE)[156.0]
    # Same six degrees, very different reactivity. The bar is set for this
    # deliberately coarse test mesh, where the ratio measures 2.9x; it widens
    # sharply with refinement (at refine 4 the far end is worth only a few pcm),
    # which is itself the reason a manoeuvre in degrees is not portable.
    assert near > 2.0 * far, (near, far)

    # The helper reports what it actually inserted, which need not equal the
    # request: on a coarse mesh the polar arc's area fraction changes only when
    # it crosses a cell boundary, so the worth curve is a staircase (91 and
    # 92 deg are both 17.4 pcm at refine 2). Reporting the achieved value is the
    # honest behaviour; silently returning the requested one would not be.
    angle, rho = hpmr_angle_for_dollars(90.0, 0.25, refine=REFINE,
                                        with_worth=True)
    assert 90.0 < angle < 110.0
    assert rho == pytest.approx(hpmr_drum_worth(90.0, [angle], refine=REFINE)[angle],
                                rel=1e-9)
    assert 0.10 * BETA * 1e5 < rho < 0.50 * BETA * 1e5

    # Asking for more than the drum has must fail loudly, not extrapolate a
    # curve that has flattened.
    with pytest.raises(ValueError, match="outside the worth available"):
        hpmr_angle_for_dollars(150.0, 1.0, refine=REFINE)
