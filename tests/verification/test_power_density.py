"""Fission power density: normalization, weighting and mixing.

The thermal coupling's source term. What must be exact: the integral over the
core equals the rated power, the result does not depend on the eigenvalue
flux's arbitrary normalization, and the cross-section weighting/mixing matches
what the neutronics solver's ``Fields`` does -- otherwise the two physics
disagree about where the fuel is.
"""

import numpy as np
import pytest

from ndgpu import Grid, Material
from ndgpu.power import fission_energy_xs, power_density


def _mats():
    """Fuel with kappa*Sigma_f deliberately NOT proportional to nu*Sigma_f, and
    an inert reflector. The disproportion is the whole point: it is what makes
    the kappa-vs-nu choice observable."""
    fuel = Material(name="fuel", diffusion=[1.5, 0.4], sigma_a=[0.01, 0.10],
                    nu_sigma_f=[0.007, 0.20], sigma_s=[[0.0, 0.02], [0.0, 0.0]],
                    chi=[1.0, 0.0], kappa_fission=[0.004, 0.05])
    refl = Material(name="refl", diffusion=[1.2, 0.9], sigma_a=[2e-4, 5e-3],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.04], [0.0, 0.0]])
    return [refl, fuel]


def _problem(nx=6, ny=5, nz=4, seed=0):
    rng = np.random.default_rng(seed)
    shape = (nx, ny, nz)
    mmap = np.zeros(shape, dtype=int)
    mmap[1:-1, 1:-1, 1:-1] = 1                      # fuel block inside reflector
    flux = rng.uniform(0.5, 2.0, size=(2, *shape))
    return shape, mmap, flux


def test_integral_equals_rated_power():
    shape, mmap, flux = _problem()
    grid = Grid(shape=shape, size=(60.0, 50.0, 40.0))
    q = power_density(flux, _mats(), mmap, total_power=2.0e6,
                      cell_volume=grid.cell_volume)
    assert np.sum(q) * grid.cell_volume == pytest.approx(2.0e6, rel=1e-12)


def test_normalization_is_independent_of_flux_scale():
    """The eigenvalue flux carries an arbitrary normalization (the power
    iteration rescales it every outer), so scaling it must not move q'''."""
    shape, mmap, flux = _problem()
    a = power_density(flux, _mats(), mmap, total_power=1.0, cell_volume=3.0)
    b = power_density(1.37e5 * flux, _mats(), mmap, total_power=1.0,
                      cell_volume=3.0)
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=0.0)


def test_kappa_fission_is_used_and_differs_from_nu_weighting():
    mats = _mats()
    table, key = fission_energy_xs(mats)
    assert key == "kappa_fission"
    np.testing.assert_allclose(table[1], mats[1].kappa_fission)

    shape, mmap, flux = _problem()
    q_kappa = power_density(flux, mats, mmap, total_power=1.0)
    stripped = [Material(name=m.name, diffusion=m.diffusion, sigma_a=m.sigma_a,
                         nu_sigma_f=m.nu_sigma_f, sigma_s=m.sigma_s, chi=m.chi)
                for m in mats]
    assert fission_energy_xs(stripped)[1] == "nu_sigma_f"
    q_nu = power_density(flux, stripped, mmap, total_power=1.0)
    # Both normalize to the same total, but the SHAPE differs -- nu-weighting
    # tilts the distribution by the local spectrum.
    assert np.sum(q_kappa) == pytest.approx(np.sum(q_nu), rel=1e-12)
    assert not np.allclose(q_kappa, q_nu, rtol=1e-6)


def test_volume_mixing_blends_linearly():
    """Cross sections blend linearly on mixed cells, as Fields does. A cell
    half-filled with fuel must produce exactly half a full cell's density."""
    mats = _mats()
    shape = (3, 1, 1)
    mmap = np.array([[[0]], [[1]], [[0]]])          # refl | fuel | refl
    flux = np.ones((2, *shape))
    mix_material = np.array([[[-1]], [[-1]], [[1]]])   # blend fuel into cell 2
    mix_weight = np.array([[[0.0]], [[0.0]], [[0.5]]])

    q = power_density(flux, mats, mmap, mix_material=mix_material,
                      mix_weight=mix_weight)
    assert q[0, 0, 0] == 0.0
    assert q[2, 0, 0] == pytest.approx(0.5 * q[1, 0, 0], rel=1e-13)


def test_active_mask_excludes_excised_cells():
    shape, mmap, flux = _problem()
    active = np.ones(shape, dtype=bool)
    active[0] = False
    q = power_density(flux, _mats(), mmap, total_power=5.0, cell_volume=1.0,
                      active=active)
    assert np.all(q[0] == 0.0)
    assert np.sum(q) == pytest.approx(5.0, rel=1e-12)


def test_cylindrical_volume_weight_changes_the_normalization():
    """On an r-z grid cells are annuli, so the scalar cell_volume is wrong; the
    radial metric must carry the weight or the outer ring is under-counted."""
    grid = Grid(shape=(8, 1, 4), size=(40.0, 1.0, 20.0), geometry="cylindrical")
    rc = grid.cylindrical_metrics()[0]
    mmap = np.ones(grid.shape, dtype=int)
    flux = np.ones((2, *grid.shape))
    q = power_density(flux, _mats(), mmap, total_power=1.0,
                      cell_volume=grid.cell_volume, volume_weight=rc)
    assert float(np.sum(q * grid.cell_volume * rc)) == pytest.approx(1.0, rel=1e-12)
    # A flat flux in an annular geometry is NOT a flat power density per rated
    # watt: the outer rings hold more volume, so each cm^3 gets less.
    flat = power_density(flux, _mats(), mmap, total_power=1.0,
                         cell_volume=grid.cell_volume)
    assert not np.allclose(q, flat)


def test_zero_flux_is_reported_not_silently_divided():
    shape, mmap, _ = _problem()
    with pytest.raises(ValueError, match="cannot normalize"):
        power_density(np.zeros((2, *shape)), _mats(), mmap, total_power=1.0)
