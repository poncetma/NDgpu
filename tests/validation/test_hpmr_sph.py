"""SPH on the 2D HP-MR, TriSP3 as the transport reference (slow, full-core).

The HP-MR cross sections are already assembly-homogenized, so on this mesh SPH
is not correcting a spatial homogenization -- it is folding the SP3-vs-diffusion
*angular* difference into the diffusion constants. That difference lives almost
entirely in the near-black B4C drum arc, whose self-shielding plain diffusion
misses (it over-counts the absorption). SPH generates a per-material factor that
makes coarse TriDiffusion reproduce the TriSP3 eigenvalue, and hence the drum
worth, to a few pcm -- where uncorrected diffusion is off by ~120 pcm.

Marked slow: a full-core SP3 solve plus an Anderson-iterated sequence of
diffusion solves. Run with `pytest -m slow` (or unmarked in the nightly suite).
"""
import numpy as np
import pytest

from ndgpu import (TriDiffusionEigenSolver, TriSP3EigenSolver,
                   flux_weighted_homogenize, region_average, sph_correct)
from ndgpu.benchmarks.hpmr import build_hpmr2d, DRUM_ABSORBER

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


@pytest.mark.slow
def test_sph_folds_sp3_self_shielding_into_hpmr_diffusion():
    # Drums inserted (angle=0): the arc faces the core, maximal self-shielding.
    p = build_hpmr2d(refine=4, drum_angle_deg=0.0, absorber="raster")
    dV = p.grid.cell_volume
    common = dict(active=p.active, mask_bc=p.mask_bc)

    ref = TriSP3EigenSolver(p.grid, p.materials, p.material_map, **common).solve(**TIGHT)
    dif = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, **common).solve(**TIGHT)
    assert ref.converged and dif.converged

    region = p.material_map                       # one region per material type
    hmats, rflux, _ = flux_weighted_homogenize(
        ref.flux_numpy, p.materials, p.material_map, region, cell_volume=dV)

    def coarse_solve(materials):
        r = TriDiffusionEigenSolver(p.grid, materials, region, **common).solve(**TIGHT)
        return region_average(r.flux_numpy, region), r.k_eff

    out = sph_correct(hmats, region, rflux, coarse_solve, tol=1e-7, depth=6)

    gap_dif = abs(dif.k_eff - ref.k_eff) * 1e5
    gap_sph = abs(out.k_eff - ref.k_eff) * 1e5
    assert out.converged
    assert gap_dif > 30.0                         # diffusion misses the arc self-shielding
    assert gap_sph < 15.0                         # SPH recovers the SP3 eigenvalue
    assert gap_sph < 0.3 * gap_dif                # a large fraction of the gap closed

    # the correction is carried by the absorber (its factor departs from 1),
    # while a transparent material like the fuel is left essentially untouched.
    mu = out.factors
    assert np.abs(mu[DRUM_ABSORBER] - 1.0).max() > 0.1
