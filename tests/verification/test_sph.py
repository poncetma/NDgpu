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

from ndgpu import (DiffusionEigenSolver, Grid, Material, PWR_TWO_GROUP,
                   SP3EigenSolver, flux_weighted_homogenize, region_average,
                   sph_correct)


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


def _reflective_assembly():
    # A heterogeneous assembly with a central absorber cluster (an intra-assembly
    # gradient), reflective -- the SPH *generation* geometry, where matching the
    # region reaction rates is equivalent to matching k_inf.
    poison = Material(name="poison", diffusion=[1.15, 0.55], sigma_a=[0.009, 0.12],
                      nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])
    mats = [PWR_TWO_GROUP, poison]
    n = 32
    grid = Grid(shape=(n, n, 1), size=(40.0, 40.0, 1.0))
    dV = (40.0 / n) ** 2
    mmap = np.zeros((n, n, 1), dtype=np.int64)
    c = n // 2
    for di in range(-3, 4):
        for dj in range(-3, 4):
            if abs(di) + abs(dj) <= 3:
                mmap[c + di, c + dj, 0] = 1
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    rr = np.sqrt((ii - c + 0.5) ** 2 + (jj - c + 0.5) ** 2)
    region = np.digitize(rr, [6, 12]).reshape(n, n, 1).astype(np.int64)   # 3 rings
    return grid, mats, mmap, region, dV


def test_sph_correction_preserves_the_sp3_eigenvalue():
    # Full pipeline: SP3 reference -> flux-weighted homogenize -> SPH-correct so
    # the coarse diffusion reproduces the SP3 k_inf. The mu*Phi = phi_ref
    # condition preserves the region reaction rates, hence the eigenvalue.
    grid, mats, mmap, region, dV = _reflective_assembly()
    ref = SP3EigenSolver(grid, mats, material_map=mmap, bc="reflective",
                         device="cpu").solve(tol_k=1e-9, tol_source=1e-8)
    hmats, rflux, _ = flux_weighted_homogenize(ref.flux_numpy, mats, mmap, region,
                                               cell_volume=dV)

    def coarse_solve(materials):
        res = DiffusionEigenSolver(grid, materials, material_map=region,
                                   bc="reflective", device="cpu"
                                   ).solve(tol_k=1e-10, tol_source=1e-9)
        return region_average(res.flux_numpy, region), res.k_eff

    k_homog = coarse_solve(hmats)[1]
    out = sph_correct(hmats, region, rflux, coarse_solve, tol=1e-9)

    assert out.converged
    err_homog = abs(k_homog - ref.k_eff) * 1e5
    err_sph = abs(out.k_eff - ref.k_eff) * 1e5
    assert err_homog > 20.0                      # homogenization alone has real error
    assert err_sph < 0.5                          # SPH restores the reference k
    assert np.allclose(out.factors, 1.0, atol=0.15)   # well-homogenized: factors near 1


def _leaky_colorset():
    # fuel | poisoned-fuel stripe | fuel, vacuum on the x ends (net leakage),
    # reflective y. Three column homogenization regions. Unlike the reflective
    # assembly this has genuine inter-region current, which is what makes the
    # naive SPH fixed point oscillate and what SPH cannot fully absorb.
    poison = Material(name="poison", diffusion=[1.15, 0.55], sigma_a=[0.009, 0.12],
                      nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])
    mats = [PWR_TWO_GROUP, poison]
    n = 36
    grid = Grid(shape=(n, 12, 1), size=(90.0, 30.0, 1.0))
    dV = (90.0 / n) * (30.0 / 12)
    mmap = np.zeros((n, 12, 1), dtype=np.int64)
    mmap[n // 3:2 * n // 3, 4:8, :] = 1
    region = np.zeros((n, 12, 1), dtype=np.int64)
    region[n // 3:2 * n // 3, :, :] = 1
    region[2 * n // 3:, :, :] = 2
    return grid, mats, mmap, region, dV, ("vacuum", "reflective", "reflective")


def test_sph_converges_and_improves_k_on_a_leaky_colorset():
    # On a leaky problem the plain fixed point oscillates; Anderson converges it.
    # SPH then greatly reduces the homogenization error, but -- unlike the
    # reflective case -- it does NOT reach the reference: a single per-region
    # factor matches reaction rates, not interface currents, so a leaky transport
    # eigenvalue is only approached (exactness there needs discontinuity factors,
    # or per-assembly reflective generation as real lattice codes do).
    grid, mats, mmap, region, dV, bc = _leaky_colorset()
    ref = SP3EigenSolver(grid, mats, material_map=mmap, bc=bc,
                         device="cpu").solve(tol_k=1e-9, tol_source=1e-8)
    hmats, rflux, _ = flux_weighted_homogenize(ref.flux_numpy, mats, mmap, region,
                                               cell_volume=dV)

    def coarse_solve(materials):
        res = DiffusionEigenSolver(grid, materials, material_map=region, bc=bc,
                                   device="cpu").solve(tol_k=1e-10, tol_source=1e-9)
        return region_average(res.flux_numpy, region), res.k_eff

    k_homog = coarse_solve(hmats)[1]
    out = sph_correct(hmats, region, rflux, coarse_solve, tol=1e-9, depth=5)

    err_homog = abs(k_homog - ref.k_eff) * 1e5
    err_sph = abs(out.k_eff - ref.k_eff) * 1e5
    assert out.converged                              # Anderson tames the oscillation
    assert out.iterations < 40                        # and does so quickly
    assert err_homog > 100.0                          # homogenization is well off
    assert err_sph < 0.25 * err_homog                 # SPH more than 4x closer
    assert err_sph > 1.0                              # but a leaky floor remains
