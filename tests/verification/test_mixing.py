"""Per-cell two-material volume-mixing in the cross-section fields.

The mix blends a second material into a base cell with weight w: cross sections
blend linearly (exact reaction-rate averaging under a flat flux) and the
diffusion coefficient blends harmonically (its transport cross section 1/(3D)
volume-averages). These tests pin those blend laws exactly and confirm the
degenerate weights reduce to plain integer-map assignment, so a non-mixed run
is unchanged.
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, Material
from ndgpu.backend import get_backend
from ndgpu.solver import Fields

MAT_A = Material(name="A", diffusion=[1.5], sigma_a=[0.02], nu_sigma_f=[0.03])
MAT_B = Material(name="B", diffusion=[0.5], sigma_a=[0.10], nu_sigma_f=[0.00])
GRID = Grid(shape=(2, 2, 2), size=(20.0, 20.0, 20.0))


def _fields(mix_material=None, mix_weight=None):
    xp = get_backend("cpu")
    mmap = np.zeros(GRID.shape, dtype=int)          # all material A
    return Fields(xp, GRID, [MAT_A, MAT_B], mmap, np.float64,
                  mix_material=mix_material, mix_weight=mix_weight)


def test_blend_laws_at_a_mixed_cell():
    w = 0.3
    mm = np.full(GRID.shape, -1, dtype=int); mm[0, 0, 0] = 1     # blend B in
    ww = np.zeros(GRID.shape); ww[0, 0, 0] = w
    f = _fields(mm, ww)
    # linear for cross sections
    assert f.sigma_t[0][0, 0, 0] == pytest.approx((1 - w) * MAT_A.sigma_t[0] + w * MAT_B.sigma_t[0])
    assert f.removal[0][0, 0, 0] == pytest.approx((1 - w) * 0.02 + w * 0.10)
    assert f.nu_sigma_f[0][0, 0, 0] == pytest.approx((1 - w) * 0.03 + w * 0.0)
    # harmonic for the diffusion coefficient
    assert f.diffusion[0][0, 0, 0] == pytest.approx(1.0 / ((1 - w) / 1.5 + w / 0.5))
    # a non-mixed cell is exactly material A
    assert f.diffusion[0][1, 1, 1] == MAT_A.diffusion[0]
    assert f.removal[0][1, 1, 1] == 0.02


def test_weight_one_equals_direct_assignment():
    mm = np.full(GRID.shape, -1, dtype=int); mm[0, 0, 0] = 1
    ww = np.zeros(GRID.shape); ww[0, 0, 0] = 1.0
    f = _fields(mm, ww)
    assert f.diffusion[0][0, 0, 0] == pytest.approx(MAT_B.diffusion[0])
    assert f.removal[0][0, 0, 0] == pytest.approx(MAT_B.removal[0])


def test_negative_sentinel_is_exact_noop():
    # mix_material = -1 everywhere must be bit-identical to no mixing at all.
    mm = np.full(GRID.shape, -1, dtype=int)
    ww = np.full(GRID.shape, 0.7)                   # weight ignored where mm < 0
    base = _fields()
    mixed = _fields(mm, ww)
    for g in range(1):
        assert np.array_equal(mixed.diffusion[g], base.diffusion[g])
        assert np.array_equal(mixed.removal[g], base.removal[g])


def test_solver_weight_one_matches_relabelled_map():
    # A block of cells assigned to B directly vs. assigned to A and mixed to B
    # with weight 1 must give the same eigenvalue.
    n = 16
    grid = Grid(shape=(n, n, n), size=(80.0, 80.0, 80.0))
    fuel = Material(name="f", diffusion=[1.2], sigma_a=[0.025], nu_sigma_f=[0.035])
    absb = Material(name="b", diffusion=[0.6], sigma_a=[0.30], nu_sigma_f=[0.0])
    mats = [fuel, absb]
    direct = np.zeros(grid.shape, dtype=int); direct[:2] = 1        # a slab of B
    k_direct = DiffusionEigenSolver(grid, mats, direct, device="cpu").solve(
        tol_k=1e-9, tol_source=1e-8).k_eff
    base = np.zeros(grid.shape, dtype=int)                          # all A
    mm = np.full(grid.shape, -1, dtype=int); mm[:2] = 1
    ww = np.zeros(grid.shape); ww[:2] = 1.0
    k_mixed = DiffusionEigenSolver(grid, mats, base, device="cpu",
                                   mix_material=mm, mix_weight=ww).solve(
        tol_k=1e-9, tol_source=1e-8).k_eff
    assert k_mixed == pytest.approx(k_direct, abs=1e-8)
