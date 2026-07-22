"""2D discrete-ordinates (S_N) transport solver -- the 2D extension of the 1D
Gauss-Legendre S_N reference that scores the SDPN family.

Four checks, mirroring how the 1D reference is trusted:

  * quadrature -- the product (polar x azimuthal) ordinate set integrates the
    isotropic moments exactly: sum(w) = 1, <mu> = <eta> = <mu eta> = 0,
    <mu^2> = 1/3;
  * composition -- a homogeneous medium with reflective boundaries gives exactly
    k_inf (flat flux, the flat/isotropic limit where transport, diffusion and
    SP3 all coincide), which only holds if the sweep, the reflective boundary
    fixed point and the eigenvalue outer all cohere;
  * normalization -- in a large, scattering-dominated (optically thick) box the
    transport correction is small, so S_N and diffusion agree to a few pcm; a
    normalization error would show up as a large offset here;
  * physics -- with a resolved absorber S_N gives a large transport correction
    over diffusion (steep-gradient self-shielding) and SP3 closes most of the
    diffusion -> transport gap, i.e. SP3 is a good transport approximation of the
    same problem the diffusion data defines.
"""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, Grid, Material, SP3EigenSolver,
                   SDP1EigenSolver, SNTransportSolver, k_infinite, quadrature_2d)

# 2D on the structured grid: vacuum in x, y and reflective (no leakage) in z.
BC3 = (("vacuum", "vacuum"), ("vacuum", "vacuum"), "reflective")
TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


def _kdiff(cls, grid, mats, mmap):
    return cls(grid, mats, material_map=mmap, bc=BC3,
               device="cpu").solve(**TIGHT).k_eff


def test_quadrature_integrates_isotropic_moments():
    mu, eta, w = quadrature_2d(3, 12)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.sum(w * mu) == pytest.approx(0.0, abs=1e-12)
    assert np.sum(w * eta) == pytest.approx(0.0, abs=1e-12)
    assert np.sum(w * mu * eta) == pytest.approx(0.0, abs=1e-12)
    assert np.sum(w * mu * mu) == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert np.sum(w * eta * eta) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_n_azi_must_be_multiple_of_four():
    with pytest.raises(ValueError, match="multiple of 4"):
        quadrature_2d(2, 6)


@pytest.mark.parametrize("mat,name", [
    (Material(diffusion=[1.0], sigma_a=[0.02], nu_sigma_f=[0.025],
              sigma_s=[[0.0]]), "1group"),
    (Material(diffusion=[1.4, 0.4], sigma_a=[0.01, 0.10],
              nu_sigma_f=[0.007, 0.13], sigma_s=[[0.0, 0.018], [0.0, 0.0]],
              chi=[1.0, 0.0]), "2group"),
])
def test_reflective_homogeneous_is_kinf(mat, name):
    grid = Grid(shape=(8, 8, 1), size=(16.0, 16.0, 1.0))
    r = SNTransportSolver(grid, mat, n_polar=2, n_azi=8,
                          bc="reflective").solve(tol_k=1e-9, tol_source=1e-9)
    assert r.converged
    assert r.k_eff == pytest.approx(k_infinite(mat), abs=2e-5)   # < 2 pcm
    for g in range(r.flux.shape[0]):                             # spatially flat per group
        assert r.flux[g].min() / r.flux[g].max() > 0.999


def test_diffusion_limit_matches_diffusion():
    # Optically thick, scattering-dominated (c = 0.8): transport corrections are
    # small, so S_N and diffusion must agree to a few tens of pcm. A wrong
    # angular normalization would produce a large offset instead.
    n = 12
    grid = Grid(shape=(n, n, 1), size=(48.0, 48.0, 1.0))          # ~24 mfp across
    m = Material(diffusion=[1.0 / 3.0], sigma_a=[0.2], nu_sigma_f=[0.26],
                 sigma_s=[[0.0]])                                 # Sigma_t=1, c=0.8
    kdif = _kdiff(DiffusionEigenSolver, grid, [m], np.zeros((n, n, 1), int))
    r = SNTransportSolver(grid, m, n_polar=3, n_azi=12,
                          bc="vacuum").solve(tol_k=1e-7, tol_source=1e-6)
    assert r.converged
    assert abs(r.k_eff - kdif) * 1e5 < 60.0                      # < 60 pcm


def test_transport_correction_and_sp3_bracket():
    # A resolved absorber block in fuel (vacuum). S_N is a genuine, large
    # transport correction over diffusion, and SP3 closes most of the
    # diffusion -> transport gap (a good approximation of the same problem).
    n = 40
    grid = Grid(shape=(n, n, 1), size=(40.0, 40.0, 1.0))
    fuel = Material(diffusion=[1.1], sigma_a=[0.012], nu_sigma_f=[0.026],
                    sigma_s=[[0.0]], name="fuel")
    absb = Material(diffusion=[0.9], sigma_a=[0.20], nu_sigma_f=[0.0],
                    sigma_s=[[0.0]], name="absorber")             # Sigma_t=0.37
    mmap = np.zeros((n, n, 1), int)
    lo, hi = int(n * 0.4), int(n * 0.6)
    mmap[lo:hi, lo:hi, 0] = 1
    kdiff = _kdiff(DiffusionEigenSolver, grid, [fuel, absb], mmap)
    ksp3 = _kdiff(SP3EigenSolver, grid, [fuel, absb], mmap)
    r = SNTransportSolver(grid, [fuel, absb], material_map=mmap, n_polar=2,
                          n_azi=8, bc="vacuum").solve(tol_k=1e-7, tol_source=1e-6)
    assert r.converged
    gap = r.k_eff - kdiff
    assert gap * 1e5 > 2000.0                                    # big transport effect
    assert ksp3 > kdiff                                          # SP3 corrects the same way
    # SP3 lands far closer to transport than diffusion does (closes > 80% of gap).
    assert abs(r.k_eff - ksp3) < 0.2 * abs(gap)
