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


def test_chi_keeps_fissile_spectrum_when_mixed_with_nonfissile():
    # chi is a spectrum, not a cross section: mixing moderator into a fuel
    # cell must NOT scale the emission spectrum by the fuel fraction (that
    # loses (1 - w) of the fission neutrons -- the bug that made pin-resolved
    # C5G7 appear to diverge at coarse mesh). Fuel + non-fissile partner keeps
    # the fuel spectrum exactly, at any weight, in either blend direction.
    fuel = Material(name="fuel", diffusion=[1.4, 0.4], sigma_a=[0.01, 0.09],
                    nu_sigma_f=[0.006, 0.12], chi=[1.0, 0.0],
                    sigma_s=[[0.0, 0.02], [0.0, 0.0]])
    wat = Material(name="water", diffusion=[1.2, 0.15], sigma_a=[0.0, 0.02],
                   nu_sigma_f=[0.0, 0.0], chi=[0.0, 0.0],
                   sigma_s=[[0.0, 0.04], [0.0, 0.0]])
    xp = get_backend("cpu")
    mm = np.full(GRID.shape, -1, dtype=int); mm[0, 0, 0] = 1
    ww = np.zeros(GRID.shape); ww[0, 0, 0] = 0.7
    for mats in ([fuel, wat], [wat, fuel]):        # base fissile / mix fissile
        f = Fields(xp, GRID, mats, np.zeros(GRID.shape, dtype=int),
                   np.float64, mix_material=mm, mix_weight=ww)
        assert f.chi[0][0, 0, 0] == pytest.approx(1.0)
        assert f.chi[1][0, 0, 0] == pytest.approx(0.0)
        # nu_sigma_f still blends linearly by volume
        w_fuel = 0.3 if mats[0] is fuel else 0.7
        assert f.nu_sigma_f[1][0, 0, 0] == pytest.approx(w_fuel * 0.12)


def test_chi_blend_is_production_weighted_for_two_fissile_materials():
    # Two fissile components: the merged spectrum weights each component's chi
    # by its share of the cell's fission production, w * sum_g nuSigma_f.
    m1 = Material(name="m1", diffusion=[1.4, 0.4], sigma_a=[0.01, 0.09],
                  nu_sigma_f=[0.01, 0.11], chi=[1.0, 0.0],
                  sigma_s=[[0.0, 0.02], [0.0, 0.0]])
    m2 = Material(name="m2", diffusion=[1.3, 0.3], sigma_a=[0.01, 0.10],
                  nu_sigma_f=[0.02, 0.28], chi=[0.8, 0.2],
                  sigma_s=[[0.0, 0.03], [0.0, 0.0]])
    xp = get_backend("cpu")
    w = 0.4
    mm = np.full(GRID.shape, -1, dtype=int); mm[0, 0, 0] = 1
    ww = np.zeros(GRID.shape); ww[0, 0, 0] = w
    f = Fields(xp, GRID, [m1, m2], np.zeros(GRID.shape, dtype=int),
               np.float64, mix_material=mm, mix_weight=ww)
    p1, p2 = (1 - w) * (0.01 + 0.11), w * (0.02 + 0.28)
    for g, (c1, c2) in enumerate([(1.0, 0.8), (0.0, 0.2)]):
        expect = (p1 * c1 + p2 * c2) / (p1 + p2)
        assert f.chi[g][0, 0, 0] == pytest.approx(expect)
    # the merged spectrum is still normalized
    assert f.chi[0][0, 0, 0] + f.chi[1][0, 0, 0] == pytest.approx(1.0)


def test_mixed_cell_k_matches_premixed_material():
    # A reflective cell of fuel volume-mixed with moderator at weight w must
    # reproduce the k_inf of the explicitly pre-mixed material (linear XS,
    # harmonic D, fuel chi) -- the same homogenization done by hand in
    # ndgpu.benchmarks.c5g7._homogenized_pin.
    fuel = Material(name="fuel", diffusion=[1.4, 0.4], sigma_a=[0.01, 0.09],
                    nu_sigma_f=[0.006, 0.12], chi=[1.0, 0.0],
                    sigma_s=[[0.0, 0.02], [0.0, 0.0]])
    wat = Material(name="water", diffusion=[1.2, 0.15], sigma_a=[0.0, 0.02],
                   nu_sigma_f=[0.0, 0.0], chi=[0.0, 0.0],
                   sigma_s=[[0.0, 0.04], [0.0, 0.0]])
    w = 0.6                                          # fuel fraction
    lin = lambda a, b: w * np.asarray(a) + (1 - w) * np.asarray(b)
    harm = lambda a, b: 1.0 / (w / np.asarray(a) + (1 - w) / np.asarray(b))
    premixed = Material(name="mix", diffusion=harm(fuel.diffusion, wat.diffusion),
                        sigma_a=lin(fuel.sigma_a, wat.sigma_a),
                        nu_sigma_f=lin(fuel.nu_sigma_f, wat.nu_sigma_f),
                        chi=fuel.chi, sigma_s=lin(fuel.sigma_s, wat.sigma_s))
    grid = Grid(shape=(1, 1, 1), size=(2.0, 2.0, 2.0))
    zero = np.zeros(grid.shape, dtype=int)
    k_pre = DiffusionEigenSolver(grid, [premixed], zero, bc="reflective",
                                 device="cpu").solve(tol_k=1e-9).k_eff
    mm = np.zeros(grid.shape, dtype=int)             # blend fuel (idx 0) ...
    ww = np.full(grid.shape, w)                      # ... into water at w
    k_mix = DiffusionEigenSolver(grid, [fuel, wat], np.ones(grid.shape, dtype=int),
                                 bc="reflective", device="cpu",
                                 mix_material=mm, mix_weight=ww).solve(
        tol_k=1e-9).k_eff
    assert k_mix == pytest.approx(k_pre, abs=1e-8)


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
