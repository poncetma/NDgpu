"""Superhomogenization pipeline (ndgpu.sph).

Step 1 -- flux-weighted homogenization. Collapsing a fine reference solution to
one Material per coarse region must preserve that region's reaction rates
exactly: Sigma_hom * <phi>_region * V_region equals the sum of the fine
cell reaction rates. This holds by construction of the flux-and-volume weighting
and is the property the SPH factor solve relies on to recover the reference
eigenvalue (not just its flux shape). Later steps (the SPH factor solve) append
their own tests here.
"""

import numpy as np

from ndgpu import Grid, Material, PWR_TWO_GROUP, SP3EigenSolver
from ndgpu.sph import flux_weighted_homogenize


def _reference():
    absorber = Material(name="poison", diffusion=[1.1, 0.5], sigma_a=[0.01, 0.20],
                        nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])
    mats = [PWR_TWO_GROUP, absorber]
    n = 24
    grid = Grid(shape=(n, n, 1), size=(60.0, 60.0, 1.0))
    dV = (60.0 / n) ** 2
    mmap = np.zeros((n, n, 1), dtype=np.int64)
    mmap[n // 2:, :, :] = 1                        # right half poison
    mmap[5:8, 5:8, :] = 1                          # a poison patch in the fuel half
    region = np.zeros((n, n, 1), dtype=np.int64)
    region[n // 2:, :, :] = 1                      # two homogenization regions
    res = SP3EigenSolver(grid, mats, material_map=mmap,
                         bc=("vacuum", "vacuum", "reflective"), device="cpu"
                         ).solve(tol_k=1e-9, tol_source=1e-8)
    assert res.converged
    return res.flux_numpy, mats, mmap, region, dV


def test_flux_weighted_homogenization_preserves_reaction_rates():
    flux, mats, mmap, region, dV = _reference()
    hmats, rflux, rvol = flux_weighted_homogenize(flux, mats, mmap, region, cell_volume=dV)
    G = 2
    sa = np.array([m.sigma_a for m in mats])
    nf = np.array([m.nu_sigma_f for m in mats])
    ss = np.array([m.sigma_s for m in mats])       # (M, G, G)
    fmap = mmap.reshape(-1)
    phi = flux.reshape(G, -1)

    for i in range(2):
        cells = region.reshape(-1) == i
        for g in range(G):
            w = phi[g][cells] * dV
            # absorption and fission rates
            for xs, hom in ((sa, hmats[i].sigma_a), (nf, hmats[i].nu_sigma_f)):
                fine = (xs[fmap[cells], g] * w).sum()
                homr = hom[g] * rflux[i, g] * rvol[i]
                assert abs(homr - fine) <= 1e-9 * max(fine, 1.0)
            # out-scatter rate from group g
            for gp in range(G):
                if gp == g:
                    continue
                fine = (ss[fmap[cells], g, gp] * w).sum()
                homr = hmats[i].sigma_s[g, gp] * rflux[i, g] * rvol[i]
                assert abs(homr - fine) <= 1e-9 * max(fine, 1.0)


def test_homogenized_region_is_a_valid_material():
    flux, mats, mmap, region, dV = _reference()
    hmats, _, _ = flux_weighted_homogenize(flux, mats, mmap, region, cell_volume=dV)
    assert len(hmats) == 2
    for m in hmats:
        assert np.all(m.diffusion > 0)
        assert np.all(m.sigma_a >= 0)
        assert np.isclose(m.chi.sum(), 1.0)
    # region 0 is fuel-dominated (fissile), region 1 is the poison half
    assert hmats[0].is_fissile
