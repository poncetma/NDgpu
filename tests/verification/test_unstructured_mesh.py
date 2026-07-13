"""Unstructured finite-volume solver: cross-solver equivalence and exact
invariants.

On a Cartesian grid of quads the unstructured FV must reproduce the
structured Cartesian solver (same discretization, same mesh), and reflective
boundaries must give k = k_inf regardless of geometry. The VVER-440 Gmsh-mesh
validation lives in tests/validation/test_vver440.py.
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, ONE_GROUP_DEMO, k_infinite
from ndgpu.mesh import (UnstructuredDiffusionSolver, assemble_mesh,
                        assemble_mesh_3d, read_gmsh)


def _hex_grid(nx, ny, nz, Lx, Ly, Lz):
    """A structured nx*ny*nz grid of hexahedra as an unstructured 3D mesh."""
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    nid, coords = {}, []

    def gid(i, j, k):
        if (i, j, k) not in nid:
            nid[(i, j, k)] = len(coords); coords.append((i * dx, j * dy, k * dz))
        return nid[(i, j, k)]

    cells = [(gid(i, j, k), gid(i + 1, j, k), gid(i + 1, j + 1, k), gid(i, j + 1, k),
              gid(i, j, k + 1), gid(i + 1, j, k + 1), gid(i + 1, j + 1, k + 1), gid(i, j + 1, k + 1))
             for i in range(nx) for j in range(ny) for k in range(nz)]
    return assemble_mesh_3d(coords, cells, [0] * len(cells))


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


def test_unstructured_matches_tri_on_hpmr_staircase():
    # The two HP-MR paths -- structured triangular lattice (TriDiffusionEigenSolver)
    # and the assembled unstructured mesh (UnstructuredDiffusionSolver) -- rasterize
    # the SAME triangles with the same centroid-painted absorber, so they must
    # produce the same k. This ties the general-geometry solver to the fast
    # structured one on the geometry that actually matters, drums in and out.
    from ndgpu.benchmarks.hpmr import build_hpmr2d, hpmr_locally_refined_mesh
    from ndgpu.tri import TriDiffusionEigenSolver

    for angle in (0.0, 120.0):
        p = build_hpmr2d(refine=4, drum_angle_deg=angle, absorber="raster")
        k_tri = TriDiffusionEigenSolver(
            p.grid, p.materials, p.material_map, active=p.active, mask_bc=p.mask_bc
        ).solve(tol_k=1e-9, tol_source=1e-8).k_eff

        mesh, cm, mats, alpha = hpmr_locally_refined_mesh(
            refine=4, drum_angle_deg=angle, refine_drums=False)
        k_mesh = UnstructuredDiffusionSolver(mesh, mats, cm, alpha).solve(tol_k=1e-9).k_eff

        assert k_tri == pytest.approx(k_mesh, abs=1e-6), (angle, k_tri, k_mesh)


def test_matrix_free_apply_equals_assembled_operator():
    # The matrix-free within-group apply (a bincount scatter over the face list)
    # must equal an explicitly assembled dense operator built from the same
    # face/boundary weights -- this pins the one piece of new logic, the segment
    # sum, independent of the eigensolve. A locally-refined mesh exercises the
    # 2:1 hanging-node faces too.
    from ndgpu.benchmarks.hpmr import hpmr_locally_refined_mesh
    mesh, cm, mats, alpha = hpmr_locally_refined_mesh(
        refine=3, drum_angle_deg=90.0, refine_drums=True)
    solver = UnstructuredDiffusionSolver(mesh, mats, cm, alpha, device="cpu")
    n = mesh.n_cells

    D = np.array([mats[m].diffusion[0] for m in cm])
    A_dense = np.zeros((n, n))
    for i, j, L, d in mesh.faces:
        w = 2.0 * D[i] * D[j] / (D[i] + D[j]) * L / d
        A_dense[i, i] += w; A_dense[j, j] += w
        A_dense[i, j] -= w; A_dense[j, i] -= w
    for i, L, db in mesh.bfaces:
        A_dense[i, i] += alpha * D[i] * L / (db * alpha + D[i])
    A_dense[np.diag_indices(n)] += np.array([mats[m].removal[0] for m in cm]) * mesh.area

    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.standard_normal(n)
        assert np.allclose(solver.ops[0].apply(x), A_dense @ x, rtol=0, atol=1e-9)


def test_unstructured_reflective_is_kinf():
    # Reflective (alpha=0) on all faces -> k = k_inf, independent of geometry.
    mesh = _cartesian_quad_mesh(10, 50.0)
    res = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.0).solve(tol_k=1e-10)
    assert res.k_eff == pytest.approx(k_infinite(ONE_GROUP_DEMO), abs=1e-6)


def test_3d_cell_volumes_and_faces():
    # A 2x2x2 grid of unit hexes: each volume 1, total 8, and the six outer
    # sides carry 4 boundary faces each (24), the rest interior.
    mesh = _hex_grid(2, 2, 2, 2.0, 2.0, 2.0)
    assert mesh.n_cells == 8
    assert np.allclose(mesh.area, 1.0)
    assert mesh.area.sum() == pytest.approx(8.0)
    assert len(mesh.bfaces) == 24                       # 6 faces * 4 per face
    assert all(A == pytest.approx(1.0) for _, A, _ in mesh.bfaces)   # unit face area
    # a single regular tetrahedron: volume = 1/6 of the unit corner box
    tet = assemble_mesh_3d([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                           [(0, 1, 2, 3)], [0])
    assert tet.area[0] == pytest.approx(1.0 / 6.0)
    assert len(tet.bfaces) == 4 and len(tet.faces) == 0


def test_unstructured_3d_matches_structured_on_hexes():
    # Bare homogeneous cube, vacuum boundary: the 3D unstructured FV on a hex
    # grid must equal the structured Cartesian solver on the same grid.
    n, L = 12, 60.0
    mesh = _hex_grid(n, n, n, L, L, L)
    k_u = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.5).solve(tol_k=1e-9).k_eff
    k_s = DiffusionEigenSolver(Grid(shape=(n, n, n), size=(L, L, L)), ONE_GROUP_DEMO,
                               bc="vacuum", device="cpu").solve(tol_k=1e-9, tol_source=1e-8).k_eff
    assert k_u == pytest.approx(k_s, abs=1e-5)


def test_unstructured_3d_reflective_is_kinf():
    mesh = _hex_grid(6, 6, 6, 30.0, 30.0, 30.0)
    res = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.0).solve(tol_k=1e-10)
    assert res.k_eff == pytest.approx(k_infinite(ONE_GROUP_DEMO), abs=1e-6)


def test_read_gmsh_routes_3d_tets(tmp_path):
    # Two tetrahedra sharing the face {2,3,4}: read_gmsh must detect the 3D
    # volume elements, build a 3D Mesh, and find one interior face.
    msh = tmp_path / "twotet.msh"
    msh.write_text(
        "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
        "$Nodes\n5\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n5 1 1 1\n$EndNodes\n"
        "$Elements\n2\n1 4 2 1 1 1 2 3 4\n2 4 2 1 1 2 3 4 5\n$EndElements\n")
    mesh = read_gmsh(str(msh))
    assert mesh.n_cells == 2
    assert mesh.coords.shape[1] == 3
    assert np.all(mesh.area > 0)
    assert len(mesh.faces) == 1                         # the shared face {2,3,4}
