"""2D VVER-440 hexagonal core (Chao & Shatilla) against the FEMFFUSION
diffusion reference, on both geometry treatments.

The body-fitted triangular mesh (6 r^2 cells/assembly) and the unstructured
Gmsh mesh (FEMFFUSION's own VVER440.msh, when a checkout is present) must
both land near K_REFERENCE; the bulk transient exercises the tri operator
inside TransientSolver.
"""

import os

import numpy as np
import pytest

from ndgpu.benchmarks import build_vver440
from ndgpu.benchmarks.vver440 import K_REFERENCE
from ndgpu.tri import TriDiffusionEigenSolver


def test_vver440_triangular_geometry_and_k():
    # build_vver440 returns the corrected hexagonal geometry on a body-fitted
    # triangular mesh. k converges near the FEMFFUSION diffusion reference
    # (the old hand-placed hex was a sheared parallelogram, ~1900 pcm off;
    # this is a real hexagon).
    p = build_vver440(refine=1)
    assert int(p.active.sum()) == 421 * 6            # 6 triangles per assembly
    res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                                  mask_bc=p.mask_bc, device="cpu").solve(tol_k=1e-8, tol_source=1e-7)
    assert res.converged
    assert res.k_eff == pytest.approx(K_REFERENCE, abs=6e-3)


def test_vver440_bulk_transient():
    # A distributed +0.1$ step (material 1) far below prompt-critical: a benign,
    # well-resolved transient with a prompt jump and slow delayed rise to ~1.1.
    from ndgpu import TransientSolver
    from ndgpu.tri import TriGroupOperator
    p = build_vver440(refine=1, perturbation="bulk")
    res = TransientSolver(p.grid, p.problem_at, p.kinetics, active=p.active,
                          mask_bc=p.mask_bc, device="cpu",
                          group_operator=TriGroupOperator,
                          eig_solver=TriDiffusionEigenSolver).solve(t_end=1.0, dt=0.02)
    P = np.asarray(res.power)
    assert P[-1] == pytest.approx(P.max(), rel=1e-6)  # monotone rise, no peak/relax
    assert 1.08 < P[-1] < 1.20                        # ~0.1$ delayed-supercritical rise


_MSH = os.path.expanduser(
    "~/claude-tests/FEMFFUSION/examples/2D_VVER440/VVER440.msh")


@pytest.mark.skipif(not os.path.exists(_MSH), reason="FEMFFUSION VVER440.msh not present")
def test_vver440_msh_matches_femffusion():
    from ndgpu.benchmarks.vver440 import build_vver440_msh
    from ndgpu.mesh import UnstructuredDiffusionSolver
    mesh, mats, cm, alpha = build_vver440_msh(_MSH)
    assert mesh.n_cells == 1263
    res = UnstructuredDiffusionSolver(mesh, mats, cm, alpha).solve(tol_k=1e-7)
    assert res.k_eff == pytest.approx(K_REFERENCE, abs=1e-3)   # FEMFFUSION FE3 diffusion
