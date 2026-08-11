"""Locally-refined HP-MR mesh: nonconforming FV coupling and drum resolution.

Local refinement splits only the coarse cells in each drum's absorber band into
four, meeting the surrounding coarse cells at 2:1 hanging nodes. The key
correctness check is conservation: a homogeneous medium with reflective
boundaries must give exactly k_inf whether or not the drums are refined -- if
the nonconforming coarse-to-fine coupling lost or double-counted current, it
would not. On top of that, refining the absorber band resolves the arc that a
coarse raster misses, so the drum worth appears.
"""

import numpy as np
import pytest

from ndgpu import k_infinite
from ndgpu.benchmarks.hpmr import _placeholder_materials, hpmr_locally_refined_mesh
from ndgpu.mesh import UnstructuredDiffusionSolver


def test_nonconforming_reflective_is_kinf():
    # Homogeneous medium (all six slots the same material) + reflective BC:
    # k must equal k_inf exactly, refined or not -- the conservation test.
    fuel = _placeholder_materials()[1]
    homog = [fuel] * 6
    kinf = k_infinite(fuel)
    for refine_drums in (False, True):
        mesh, cm, _, _ = hpmr_locally_refined_mesh(
            refine=3, drum_angle_deg=90.0, refine_drums=refine_drums, materials=homog)
        res = UnstructuredDiffusionSolver(mesh, homog, cm, alpha_boundary=0.0).solve(tol_k=1e-9)
        assert res.k_eff == pytest.approx(kinf, abs=1e-7), (refine_drums, res.k_eff)


def test_local_refinement_adds_fine_cells_in_the_band():
    m0, c0, _, _ = hpmr_locally_refined_mesh(refine=3, drum_angle_deg=120.0, refine_drums=False)
    m1, c1, _, _ = hpmr_locally_refined_mesh(refine=3, drum_angle_deg=120.0, refine_drums=True)
    assert m1.n_cells > m0.n_cells                 # the drum bands were refined
    # refinement is local: it stays a small fraction of the whole mesh
    assert m1.n_cells < 1.5 * m0.n_cells
    assert int((c1 == 5).sum()) >= int((c0 == 5).sum())   # >= absorber cells resolved


def test_hanging_node_refinement_is_consistent():
    # Consistency of the 2:1 coupling: in a homogeneous medium with vacuum
    # boundaries (so the flux has real curvature and leakage), the locally
    # refined k must converge toward an independent globally fine reference as
    # the base resolution rises. A broken hanging-node coefficient would either
    # not converge or converge to the wrong limit (an O(1) inconsistency);
    # instead the error shrinks monotonically and stays small. (The band
    # refinement itself buys no accuracy here -- there is no feature to resolve
    # -- so this isolates the interface treatment, not the refinement's payoff.)
    fuel = _placeholder_materials()[1]
    homog = [fuel] * 6

    def k(refine, refine_drums):
        mesh, cm, _, alpha = hpmr_locally_refined_mesh(
            refine=refine, drum_angle_deg=90.0, refine_drums=refine_drums, materials=homog)
        return UnstructuredDiffusionSolver(mesh, homog, cm, alpha).solve(tol_k=1e-9).k_eff

    k_ref = k(8, False)                                    # fine uniform reference
    err = [abs(k(r, True) - k_ref) * 1e5 for r in (2, 3, 4)]   # pcm
    assert err[0] < 40.0                                   # no O(1) inconsistency
    assert err[2] < err[1] < err[0]                        # converging, monotone
    assert err[2] < 15.0                                   # approaching the true k


def test_local_refine_gives_negative_drum_worth():
    def k(angle):
        mesh, cm, mats, alpha = hpmr_locally_refined_mesh(
            refine=3, drum_angle_deg=angle, refine_drums=True)
        r = UnstructuredDiffusionSolver(mesh, mats, cm, alpha).solve(tol_k=1e-7)
        assert r.converged
        return r.k_eff
    # Convention (hpmr._drum_geometry): 0 = the B4C arc faces the core centre
    # (inserted), 180 = outward (withdrawn). Verified geometrically -- at 0 deg
    # the absorber cells sit ~11 cm core-side of their drum centre, at 180 deg
    # ~12 cm outward.
    k_in, k_out = k(0.0), k(180.0)
    assert k_out > k_in                             # arcs toward core remove reactivity
    assert (1 / k_in - 1 / k_out) * 1e5 > 500       # worth, positive by insertion
