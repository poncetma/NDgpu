"""Cylindrical (r-z) geometry verification against analytic diffusion results.

A bare homogeneous finite cylinder with a zero-flux surface has the exact
one-group eigenvalue k = nu-Sigma_f / (Sigma_a + D * B^2) with buckling
B^2 = (j0 / R)^2 + (pi / H)^2, j0 the first zero of the Bessel function J0.
The cell-centered finite-volume stencil should converge to it at second
order, and the r = 0 axis must behave as a natural (regular) boundary.
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, Material

J0_ZERO = 2.404825557695773

MAT = Material(diffusion=[1.3], sigma_a=[0.030], nu_sigma_f=[0.035])
R, H = 80.0, 160.0
BC = (("reflective", "zero-flux"), "reflective", "zero-flux")


def bare_cylinder_k():
    B2 = (J0_ZERO / R) ** 2 + (np.pi / H) ** 2
    return 0.035 / (0.030 + 1.3 * B2)


def solve(n, bc=BC):
    grid = Grid(shape=(n, 1, 2 * n), size=(R, 1.0, H), geometry="cylindrical")
    s = DiffusionEigenSolver(grid, MAT, bc=bc, device="cpu")
    return s.solve(tol_k=1e-9, tol_source=1e-8)


def test_bare_cylinder_analytic_k_and_convergence_order():
    k_exact = bare_cylinder_k()
    err = [abs(solve(n).k_eff - k_exact) for n in (20, 40, 80)]
    assert err[-1] < 5e-6
    # second order: each refinement should cut the error ~4x
    assert err[0] / err[1] == pytest.approx(4.0, rel=0.3)
    assert err[1] / err[2] == pytest.approx(4.0, rel=0.3)


def test_axis_boundary_condition_is_inert():
    # The r = 0 face has zero area, so the bc spec there must not matter.
    k_refl = solve(40).k_eff
    k_zero = solve(40, bc=(("zero-flux", "zero-flux"), "reflective", "zero-flux")).k_eff
    assert k_zero == pytest.approx(k_refl, abs=1e-12)


def test_flux_follows_bessel_mode():
    res = solve(80)
    grid_r = (np.arange(80) + 0.5) * (R / 80)
    phi = res.flux_numpy[0][:, 0, 80]  # radial cut at the axial midplane
    phi = phi / phi[0]
    assert np.all(np.diff(phi) < 0)          # monotone decrease toward surface
    assert phi[-1] < 0.06                    # near zero at the zero-flux surface
    # half-max radius of J0 mode: J0(x) = 0.5 at x ~ 1.5211 -> r ~ R * 1.5211/j0
    r_half = grid_r[np.argmin(np.abs(phi - 0.5))]
    assert r_half == pytest.approx(R * 1.5211 / J0_ZERO, rel=0.02)


def test_cylindrical_grid_requires_single_y_cell():
    with pytest.raises(ValueError, match="ny must be 1"):
        Grid(shape=(10, 2, 10), size=(10.0, 2.0, 10.0), geometry="cylindrical")
