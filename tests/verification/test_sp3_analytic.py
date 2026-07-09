"""SP3 solver validation against exact analytic SP3 bare-reactor solutions."""

import pytest

from ndgpu import (
    Grid,
    ONE_GROUP_DEMO,
    PWR_TWO_GROUP,
    SP3EigenSolver,
    k_bare_box,
    k_bare_box_sp3,
    k_infinite,
)

SIZE = (90.0, 90.0, 90.0)
TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


def solve_k(material, n, **kw):
    grid = Grid(shape=(n, n, n), size=SIZE)
    res = SP3EigenSolver(grid, material, device="cpu", **kw).solve(**TIGHT)
    assert res.converged, res
    return res.k_eff


def test_sp3_reflective_gives_k_infinity():
    # In an infinite medium phi2 = 0 and SP3 reduces exactly to diffusion.
    k = solve_k(PWR_TWO_GROUP, 8, bc="reflective")
    assert k == pytest.approx(k_infinite(PWR_TWO_GROUP), abs=1e-6)


def test_sp3_one_group_matches_analytic_and_converges_second_order():
    k_exact = k_bare_box_sp3(ONE_GROUP_DEMO, SIZE)
    err = {n: abs(solve_k(ONE_GROUP_DEMO, n) - k_exact) for n in (16, 32)}
    assert err[32] < 2e-4
    ratio = err[16] / err[32]
    assert 3.0 < ratio < 5.0, f"expected ~4x error reduction, got {ratio:.2f} ({err})"


def test_sp3_two_group_matches_analytic():
    k_exact = k_bare_box_sp3(PWR_TWO_GROUP, SIZE)
    assert solve_k(PWR_TWO_GROUP, 32) == pytest.approx(k_exact, abs=5e-4)


def test_sp3_is_a_distinct_approximation_from_diffusion():
    # The analytic SP3 and diffusion k differ for a finite core (transport
    # correction), and each solver converges to its own reference.
    d = abs(k_bare_box_sp3(ONE_GROUP_DEMO, SIZE) - k_bare_box(ONE_GROUP_DEMO, SIZE))
    assert d > 2e-5, "SP3 and diffusion analytic k unexpectedly identical"
