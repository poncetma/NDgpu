"""SPH on the 2D HP-MR with the transport reference swapped across families.

The SPH pipeline (:mod:`ndgpu.sph`) is reference-agnostic: it consumes the
*physical scalar flux* phi0 of any ndgpu eigensolver Result and the region
reaction rates that flux implies. For the block transport solvers phi0 is the
reconstructed 0th angular moment (SP3: phi0 = Phi1 - 2 phi2; SDPN: the even-
moment vector dotted with the phi0 closure weights), so swapping the reference
between SP3, SDP1 and SDP2 feeds SPH three genuinely different angular
treatments of the near-black B4C drum arc -- not a re-extraction of the same
field.

This module checks that each family drives the pipeline to convergence and folds
its own transport self-shielding into coarse TriDiffusion to a few pcm, and that
the three families agree with one another far more tightly than any of them
agrees with uncorrected diffusion (they are all transport, differing only in
angular order). The companion example
``examples/hpmr_sph_reference_families.py`` prints the full comparison table.

Marked slow: three full-core transport solves plus three Anderson-iterated
sequences of diffusion solves. Run with ``pytest -m slow``.
"""
import numpy as np
import pytest

from ndgpu import (TriDiffusionEigenSolver, TriSDP1EigenSolver,
                   TriSDP2EigenSolver, TriSP3EigenSolver,
                   flux_weighted_homogenize, region_average, sph_correct)
from ndgpu.benchmarks.hpmr import build_hpmr2d, DRUM_ABSORBER

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)

# The three matched-DoF transport references SPH can fold into diffusion.
REFERENCE_FAMILIES = {
    "SP3": TriSP3EigenSolver,
    "SDP1": TriSDP1EigenSolver,
    "SDP2": TriSDP2EigenSolver,
}


def _run_sph(problem, reference_cls):
    """Full SPH pipeline on the HP-MR from one transport reference family.

    Returns (reference_k, corrected_k, factors, converged): the reference
    eigenvalue, the SPH-corrected coarse-diffusion eigenvalue, the per-material
    SPH factors, and whether the factor solve converged.
    """
    p = problem
    dV = p.grid.cell_volume
    common = dict(active=p.active, mask_bc=p.mask_bc)

    ref = reference_cls(p.grid, p.materials, p.material_map, **common).solve(**TIGHT)
    assert ref.converged, f"{reference_cls.__name__} reference did not converge"

    region = p.material_map                       # one region per material type
    hmats, rflux, _ = flux_weighted_homogenize(
        ref.flux_numpy, p.materials, p.material_map, region, cell_volume=dV)

    def coarse_solve(materials):
        r = TriDiffusionEigenSolver(p.grid, materials, region, **common).solve(**TIGHT)
        return region_average(r.flux_numpy, region), r.k_eff

    out = sph_correct(hmats, region, rflux, coarse_solve, tol=1e-7, depth=6)
    return ref.k_eff, out.k_eff, out.factors, out.converged


@pytest.fixture(scope="module")
def hpmr_inserted():
    # Drums inserted (angle=0): the arc faces the core, maximal self-shielding
    # -- the state where the transport-vs-diffusion angular gap is largest.
    return build_hpmr2d(refine=4, drum_angle_deg=0.0, absorber="raster")


@pytest.mark.slow
@pytest.mark.parametrize("family", list(REFERENCE_FAMILIES))
def test_sph_folds_each_transport_family_into_hpmr_diffusion(hpmr_inserted, family):
    p = hpmr_inserted
    common = dict(active=p.active, mask_bc=p.mask_bc)
    dif = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                  **common).solve(**TIGHT)
    assert dif.converged

    ref_k, sph_k, mu, converged = _run_sph(p, REFERENCE_FAMILIES[family])

    gap_dif = abs(dif.k_eff - ref_k) * 1e5        # uncorrected diffusion vs transport
    gap_sph = abs(sph_k - ref_k) * 1e5            # SPH-corrected diffusion vs transport
    assert converged
    assert gap_dif > 30.0                         # diffusion misses the arc self-shielding
    assert gap_sph < 15.0                         # SPH recovers this family's eigenvalue
    assert gap_sph < 0.3 * gap_dif                # a large fraction of the gap closed
    # the correction is carried by the absorber; a transparent material like the
    # fuel is left essentially untouched.
    assert np.abs(mu[DRUM_ABSORBER] - 1.0).max() > 0.1


@pytest.mark.slow
def test_sph_reference_families_agree_more_than_diffusion(hpmr_inserted):
    # SP3, SDP1 and SDP2 are all transport treatments of the drum arc, so their
    # SPH-corrected eigenvalues track each other far more tightly than any tracks
    # uncorrected diffusion. This is the "compare across families" check: the
    # choice of angular order is a small, controlled correction on top of the
    # large, uncontrolled diffusion error.
    p = hpmr_inserted
    common = dict(active=p.active, mask_bc=p.mask_bc)
    dif = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                  **common).solve(**TIGHT)

    corrected = {}
    for name, cls in REFERENCE_FAMILIES.items():
        ref_k, sph_k, _, converged = _run_sph(p, cls)
        assert converged
        corrected[name] = sph_k

    ks = np.array(list(corrected.values()))
    spread = (ks.max() - ks.min()) * 1e5          # cross-family SPH spread, pcm
    # the largest disagreement between any corrected family and diffusion
    dif_gap = max(abs(k - dif.k_eff) for k in ks) * 1e5

    assert spread < 20.0                          # the three families agree tightly
    assert spread < 0.5 * dif_gap                 # far tighter than the diffusion gap
