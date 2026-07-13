"""3D unstructured mesh solver vs the structured tri-prism solver on the HP-MR.

The 2D check (tests/verification/test_unstructured_mesh.py) ties the unstructured
FV to the structured triangular lattice in-plane; this extends it to 3D. The 2D
HP-MR tri mesh is extruded into wedges with the same axial layering and
axial-reflector scheme build_hpmr3d uses, and the general-geometry mesh solver
must reproduce the structured tri-prism solver's k on the full core -- the two
share the same two-point-flux discretization, so they agree to ~solver tolerance.

Marked slow: a full-core 3D solve on both solvers (~50k wedges). Run with
`pytest -m slow`.
"""
import numpy as np
import pytest

from ndgpu.mesh import assemble_mesh_3d, UnstructuredDiffusionSolver
from ndgpu.tri import TriDiffusionEigenSolver
from ndgpu.benchmarks.hpmr import (hpmr_locally_refined_mesh, build_hpmr3d,
                                   _placeholder_materials, FUEL, CENTRAL,
                                   AXIAL_REFLECTOR, TOTAL_HEIGHT,
                                   AXIAL_REFLECTOR_HEIGHT)

TOL = dict(tol_k=1e-7, tol_source=1e-6)


def _extrude_wedges(coords2d, tris, cmat2d, nz, height):
    """Extrude 2D triangles into nz layers of 6-node prisms, applying the same
    fuel->axial-reflector remap in the top/bottom 20 cm that build_hpmr3d does."""
    coords2d = np.asarray(coords2d, float)
    n2 = len(coords2d)
    dz = height / nz
    coords3d = np.zeros(((nz + 1) * n2, 3))
    for lev in range(nz + 1):
        coords3d[lev * n2:(lev + 1) * n2, :2] = coords2d
        coords3d[lev * n2:(lev + 1) * n2, 2] = lev * dz
    zc = (np.arange(nz) + 0.5) * dz
    refl = (zc < AXIAL_REFLECTOR_HEIGHT) | (zc > height - AXIAL_REFLECTOR_HEIGHT)
    cells, cmat = [], []
    for t, (a, b, c) in enumerate(tris):
        for lev in range(nz):
            o0, o1 = lev * n2, (lev + 1) * n2
            cells.append((a + o0, b + o0, c + o0, a + o1, b + o1, c + o1))
            m = cmat2d[t]
            cmat.append(AXIAL_REFLECTOR if (m in (FUEL, CENTRAL) and refl[lev]) else m)
    return coords3d, cells, np.array(cmat)


@pytest.mark.slow
def test_unstructured_3d_matches_tri_prism_on_hpmr():
    refine, nz, angle = 4, 10, 180.0
    mesh2d, cmat2d, _, _ = hpmr_locally_refined_mesh(refine=refine, drum_angle_deg=angle,
                                                     refine_drums=False)
    mats3d = _placeholder_materials(three_d=True)
    coords3d, cells3d, cmat3d = _extrude_wedges(mesh2d.coords, mesh2d.cells, cmat2d,
                                                nz, TOTAL_HEIGHT)
    mesh3d = assemble_mesh_3d(coords3d, cells3d, cmat3d)

    ku = UnstructuredDiffusionSolver(mesh3d, mats3d, cmat3d,
                                     alpha_boundary=0.5).solve(**TOL).k_eff

    p = build_hpmr3d(refine=refine, drum_angle_deg=angle, nz=nz)
    ks = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                                 mask_bc=p.mask_bc, bc=p.bc).solve(**TOL).k_eff

    assert mesh3d.n_cells == int(p.active.sum())        # same unknown count
    assert ku == pytest.approx(ks, abs=1e-5), (ku, ks)  # same discretization -> same k
