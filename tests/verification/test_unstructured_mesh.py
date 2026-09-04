"""Unstructured finite-volume solver: cross-solver equivalence and exact
invariants.

On a Cartesian grid of quads the unstructured FV must reproduce the
structured Cartesian solver (same discretization, same mesh), and reflective
boundaries must give k = k_inf regardless of geometry. The VVER-440 Gmsh-mesh
validation lives in tests/validation/test_vver440.py.
"""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, Grid, ONE_GROUP_DEMO, PWR_TWO_GROUP,
                   k_bare_box, k_infinite)
from ndgpu.mesh import (UnstructuredDiffusionSolver, assemble_mesh,
                        assemble_mesh_3d, read_gmsh)

# A large Robin coefficient makes the boundary flux vanish (Dirichlet limit),
# so a bare box converges to the zero-flux analytic k_bare_box.
_ZERO_FLUX = 1e8


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


def _tet_cube(n, L):
    """An n^3 lattice with each cube split into 6 tetrahedra (a mesh of tets)."""
    dx = L / n
    nid, coords = {}, []

    def gid(i, j, k):
        if (i, j, k) not in nid:
            nid[(i, j, k)] = len(coords); coords.append((i * dx, j * dx, k * dx))
        return nid[(i, j, k)]

    tets = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
            (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]   # 6-tet cube split
    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                corner = [gid(i, j, k), gid(i + 1, j, k), gid(i + 1, j + 1, k),
                          gid(i, j + 1, k), gid(i, j, k + 1), gid(i + 1, j, k + 1),
                          gid(i + 1, j + 1, k + 1), gid(i, j + 1, k + 1)]
                cells += [tuple(corner[t] for t in tet) for tet in tets]
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


def test_unstructured_matches_tri_with_polar_volume_mixing():
    from ndgpu.benchmarks.hpmr import build_hpmr2d, build_hpmr2d_local
    from ndgpu.tri import TriDiffusionEigenSolver

    for angle in (90.0, 95.0):
        structured = build_hpmr2d(
            refine=3, drum_angle_deg=angle, absorber="polar", samples=0)
        k_tri = TriDiffusionEigenSolver(
            structured.grid, structured.materials, structured.material_map,
            active=structured.active, mask_bc=structured.mask_bc,
            mix_material=structured.mix_material,
            mix_weight=structured.mix_weight, device="cpu").solve(
                tol_k=1e-9, tol_source=1e-8).k_eff

        local = build_hpmr2d_local(
            refine=3, drum_angle_deg=angle, local_refinement=False,
            absorber="polar", samples=0)
        k_mesh = UnstructuredDiffusionSolver(
            local.mesh, local.materials, local.cell_material,
            local.alpha_boundary, mix_material=local.mix_material,
            mix_weight=local.mix_weight, device="cpu").solve(
                tol_k=1e-9, tol_source=1e-8).k_eff

        assert k_mesh == pytest.approx(k_tri, abs=2e-6), (angle, k_tri, k_mesh)


def _dense_operator(mesh, cm, mats, alpha):
    """The within-group group-0 operator as an explicit dense matrix, assembled
    straight from the face/boundary lists (an independent reference for the
    matrix-free apply)."""
    n = mesh.n_cells
    D = np.array([mats[m].diffusion[0] for m in cm])
    A = np.zeros((n, n))
    for i, j, L, d in mesh.faces:
        w = 2.0 * D[i] * D[j] / (D[i] + D[j]) * L / d
        A[i, i] += w; A[j, j] += w
        A[i, j] -= w; A[j, i] -= w
    for i, L, db in mesh.bfaces:
        A[i, i] += alpha * D[i] * L / (db * alpha + D[i])
    A[np.diag_indices(n)] += np.array([mats[m].removal[0] for m in cm]) * mesh.area
    return A


def _assert_apply_matches_dense(mesh, cm, mats, alpha):
    solver = UnstructuredDiffusionSolver(mesh, mats, cm, alpha, device="cpu")
    A = _dense_operator(mesh, cm, mats, alpha)
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.standard_normal(mesh.n_cells)
        assert np.allclose(solver.ops[0].apply(x), A @ x, rtol=0, atol=1e-9)


def test_matrix_free_apply_equals_assembled_operator():
    # The matrix-free within-group apply (a gather over the ELLPACK adjacency)
    # must equal an explicitly assembled dense operator built from the same
    # face/boundary weights -- this pins the gather independent of the eigensolve.
    # A locally-refined mesh exercises the 2:1 hanging-node faces too.
    from ndgpu.benchmarks.hpmr import hpmr_locally_refined_mesh
    mesh, cm, mats, alpha = hpmr_locally_refined_mesh(
        refine=3, drum_angle_deg=90.0, refine_drums=True)
    _assert_apply_matches_dense(mesh, cm, mats, alpha)


def test_matrix_free_apply_equals_assembled_operator_3d():
    # Same equivalence in 3D, on a mesh mixing hexes and tetrahedra so the
    # polygon face areas (quad and triangle) both feed the operator correctly --
    # pins the 3D face-weight assembly, not only the volumes the geometry tests
    # already check.
    hexes = _hex_grid(3, 3, 3, 15.0, 15.0, 15.0)
    tets = _tet_cube(2, 10.0)
    for mesh in (hexes, tets):
        cm = np.zeros(mesh.n_cells, int)
        _assert_apply_matches_dense(mesh, cm, [ONE_GROUP_DEMO], 0.5)


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


def test_unstructured_3d_converges_to_analytic_second_order():
    # Bare homogeneous box with vanishing boundary flux: the 3D FV k_eff must
    # converge to the exact analytic k_bare_box at second order in the mesh
    # spacing (error drops ~4x when the cell count per side doubles) -- this
    # checks the discretization error itself, not just cross-solver agreement.
    L = 90.0
    k_exact = k_bare_box(ONE_GROUP_DEMO, (L, L, L))
    err = {}
    for n in (16, 32):
        mesh = _hex_grid(n, n, n, L, L, L)
        k = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                        alpha_boundary=_ZERO_FLUX).solve(tol_k=1e-9,
                                                                         tol_source=1e-8).k_eff
        err[n] = abs(k - k_exact)
    assert err[32] < 2e-4
    ratio = err[16] / err[32]
    assert 3.5 < ratio < 4.5, f"expected ~4x error drop (2nd order), got {ratio:.2f} ({err})"


def test_unstructured_3d_two_group_matches_structured():
    # Multigroup (fission + down-scatter) in 3D: the hex-mesh solver must equal
    # the structured Cartesian solver on the same grid, vacuum boundary.
    n, L = 12, 60.0
    mesh = _hex_grid(n, n, n, L, L, L)
    k_u = UnstructuredDiffusionSolver(mesh, [PWR_TWO_GROUP], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.5).solve(tol_k=1e-9, tol_source=1e-8).k_eff
    k_s = DiffusionEigenSolver(Grid(shape=(n, n, n), size=(L, L, L)), PWR_TWO_GROUP,
                               bc="vacuum", device="cpu").solve(tol_k=1e-9, tol_source=1e-8).k_eff
    assert k_u == pytest.approx(k_s, abs=1e-5)


def test_tet_mesh_reflective_is_kinf():
    # Physics on tetrahedra (not just their volumes): a cube split into tets,
    # reflective on all sides -> k = k_inf, independent of the (skewed) cell
    # shapes, so the tet face coupling and volumes are mutually consistent.
    mesh = _tet_cube(4, 20.0)
    res = UnstructuredDiffusionSolver(mesh, [ONE_GROUP_DEMO], np.zeros(mesh.n_cells, int),
                                      alpha_boundary=0.0).solve(tol_k=1e-10)
    assert res.k_eff == pytest.approx(k_infinite(ONE_GROUP_DEMO), abs=1e-6)


def test_mixed_element_mesh_reflective_is_kinf():
    # One hexahedron beside two prisms (a square split into two triangles, both
    # extruded), sharing a quad interface. Exercises face de-duplication across
    # element types -- a hex quad face matching a prism side face -- which the
    # single-type meshes never hit. Reflective -> k_inf, and the volumes sum.
    pts = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 1)]     # A B C D E F
    coords = [(x, y, z) for z in (0.0, 1.0) for (x, y) in pts]  # level0: 0-5, level1: 6-11
    hexc = (0, 1, 2, 3, 6, 7, 8, 9)                            # ABCD extruded
    wedge1 = (1, 4, 5, 7, 10, 11)                              # BEF extruded
    wedge2 = (1, 5, 2, 7, 11, 8)                               # BFC extruded
    mesh = assemble_mesh_3d(coords, [hexc, wedge1, wedge2], [0, 0, 0])
    assert mesh.area.sum() == pytest.approx(2.0)               # 1 + 0.5 + 0.5
    assert len(mesh.faces) == 2                                # hex|wedge2 and wedge1|wedge2
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
