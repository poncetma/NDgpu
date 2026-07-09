"""Validation against exact analytic bare-reactor solutions.

For a homogeneous box with zero flux on the surface, k_eff and the flux shape
are known in closed form, so these tests check the discretization error itself
(2nd order in mesh spacing), not just self-consistency.
"""

import numpy as np
import pytest

from ndgpu import (
    DiffusionEigenSolver,
    Grid,
    Material,
    ONE_GROUP_DEMO,
    PWR_TWO_GROUP,
    k_bare_box,
    k_infinite,
)

SIZE = (90.0, 90.0, 90.0)  # cm, near-critical for both demo materials
TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


def solve_k(material, n, size=SIZE, **kw):
    grid = Grid(shape=(n, n, n), size=size)
    solver = DiffusionEigenSolver(grid, material, device="cpu", **kw)
    res = solver.solve(**TIGHT)
    assert res.converged, res
    return res


def test_one_group_matches_analytic_and_converges_second_order():
    k_exact = k_bare_box(ONE_GROUP_DEMO, SIZE)
    err = {n: abs(solve_k(ONE_GROUP_DEMO, n).k_eff - k_exact) for n in (16, 32)}
    assert err[32] < 2e-4
    ratio = err[16] / err[32]
    assert 3.0 < ratio < 5.0, f"expected ~4x error reduction, got {ratio:.2f} ({err})"


def test_two_group_matches_analytic_and_converges_second_order():
    k_exact = k_bare_box(PWR_TWO_GROUP, SIZE)
    err = {n: abs(solve_k(PWR_TWO_GROUP, n).k_eff - k_exact) for n in (16, 32)}
    assert err[32] < 5e-4
    ratio = err[16] / err[32]
    assert 3.0 < ratio < 5.0, f"expected ~4x error reduction, got {ratio:.2f} ({err})"


def test_reflective_bc_gives_k_infinity():
    # No leakage -> flat flux and k = k_inf exactly (to solver tolerance).
    res = solve_k(PWR_TWO_GROUP, 8, bc="reflective")
    assert res.k_eff == pytest.approx(k_infinite(PWR_TWO_GROUP), abs=1e-6)
    flux = res.flux_numpy
    assert np.allclose(flux, flux.mean(axis=(1, 2, 3), keepdims=True), rtol=1e-6)


def test_flux_shape_is_fundamental_mode():
    # Bare homogeneous box: flux ~ sin(pi x / Lx) sin(pi y / Ly) sin(pi z / Lz).
    n = 24
    grid = Grid(shape=(n, n, n), size=SIZE)
    res = DiffusionEigenSolver(grid, ONE_GROUP_DEMO, device="cpu").solve(**TIGHT)
    assert res.converged

    s = [np.sin(np.pi * grid.cell_centers(ax) / SIZE[ax]) for ax in range(3)]
    mode = np.einsum("i,j,k->ijk", *s)
    flux = res.flux_numpy[0]
    cos_sim = np.sum(flux * mode) / np.sqrt(np.sum(flux**2) * np.sum(mode**2))
    assert cos_sim > 0.99999


def test_heterogeneous_material_map_runs_and_is_physical():
    # Fissile core surrounded by a pure absorber/scatterer blanket: k must drop
    # below the bare-core value with the same outer dimensions but stay above
    # the value with a vacuum (zero-flux) boundary hugging the smaller core.
    reflector = Material(
        name="water-ish reflector",
        diffusion=[1.13, 0.16],
        sigma_a=[0.0004, 0.0197],
        nu_sigma_f=[0.0, 0.0],
        sigma_s=[[0.0, 0.0494], [0.0, 0.0]],
    )
    n = 24
    grid = Grid(shape=(n, n, n), size=(120.0, 120.0, 120.0))
    mmap = np.ones(grid.shape, dtype=np.int64)  # 1 = reflector
    lo, hi = n // 4, 3 * n // 4
    mmap[lo:hi, lo:hi, lo:hi] = 0  # 0 = fuel, central 60 cm cube

    res = DiffusionEigenSolver(
        grid, [PWR_TWO_GROUP, reflector], material_map=mmap, device="cpu"
    ).solve(**TIGHT)
    assert res.converged

    k_core_bare = solve_k(PWR_TWO_GROUP, 16, size=(60.0, 60.0, 60.0)).k_eff
    assert res.k_eff > k_core_bare  # reflector returns neutrons -> raises k
    assert res.k_eff < k_infinite(PWR_TWO_GROUP)  # still leaks -> below k_inf


def test_gpu_device_request_errors_cleanly_without_cuda():
    grid = Grid(shape=(4, 4, 4), size=SIZE)
    try:
        import cupy  # noqa: F401

        pytest.skip("cupy present; this test targets CUDA-less machines")
    except ImportError:
        with pytest.raises((ImportError, RuntimeError)):
            DiffusionEigenSolver(grid, ONE_GROUP_DEMO, device="gpu")
