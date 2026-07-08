"""Unstructured finite-volume solver on a Gmsh-style mesh.

Validated two ways: (1) on a Cartesian grid of quads the unstructured FV must
reproduce the structured Cartesian solver on the same mesh, and (2) the VVER-440
Gmsh mesh (when a FEMFFUSION checkout is available) matches its diffusion k.
"""

import os

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, ONE_GROUP_DEMO
from ndgpu.mesh import UnstructuredDiffusionSolver, assemble_mesh, read_gmsh


def _cartesian_quad_mesh(n, L):
    dx = L / n
    nid = {}
    coords = []

    def gid(i, j):
        if (i, j) not in nid:
            nid[(i, j)] = len(coords)
            coords.append((i * dx, j * dx))
        return nid[(i, j)]

    cells = [(gid(i, j), gid(i + 1, j), gid(i + 1, j + 1), gid(i, j + 1))
             for i in range(n) for j in range(n)]
    return assemble_mesh(coords, cells, [0] * len(cells))


def test_unstructured_matches_structured_on_quads():
    # Bare homogeneous square, vacuum boundary. The unstructured FV on a quad
    # grid must equal the structured Cartesian FV (2D: reflective z) to tight
    # tolerance -- same discretization, same mesh.
    n, L = 24, 90.0
    mesh = _cartesian_quad_mesh(n, L)
    k_u = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.5).solve(tol_k=1e-9).k_eff
    k_s = DiffusionEigenSolver(
        Grid(shape=(n, n, 1), size=(L, L, L / n)), ONE_GROUP_DEMO,
        bc=("vacuum", "vacuum", "reflective"), device="cpu",
    ).solve(tol_k=1e-9, tol_source=1e-8).k_eff
    assert k_u == pytest.approx(k_s, abs=2e-4)


def test_unstructured_reflective_is_kinf():
    # Reflective (alpha=0) on all faces -> k = k_inf, independent of geometry.
    mesh = _cartesian_quad_mesh(10, 50.0)
    res = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.0).solve(tol_k=1e-10)
    from ndgpu import k_infinite
    assert res.k_eff == pytest.approx(k_infinite(ONE_GROUP_DEMO), abs=1e-6)


_MSH = os.path.expanduser(
    "~/claude-tests/FEMFFUSION/examples/2D_VVER440/VVER440.msh")


@pytest.mark.skipif(not os.path.exists(_MSH), reason="FEMFFUSION VVER440.msh not present")
def test_vver440_msh_matches_femffusion():
    from ndgpu.benchmarks.vver440 import build_vver440_msh
    mesh, mats, cm, alpha = build_vver440_msh(_MSH)
    assert mesh.n_cells == 1263
    res = UnstructuredDiffusionSolver(mesh, mats, cm, alpha).solve(tol_k=1e-7)
    assert res.k_eff == pytest.approx(1.00349, abs=1e-3)   # FEMFFUSION FE3 diffusion
