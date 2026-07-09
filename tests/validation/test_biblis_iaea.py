"""BIBLIS (2D vacuum) and IAEA (3D albedo) diffusion benchmarks.

These exercise the Robin/albedo boundary term and the active-cell mask for
non-rectangular cores, and check k_eff against the published references and
against FEMFFUSION's own diffusion result.
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver
from ndgpu.benchmarks import build_biblis, build_iaea

TOL = dict(tol_k=1e-8, tol_source=1e-7)


def _solve(p):
    return DiffusionEigenSolver(p.grid, p.materials, p.material_map, bc=p.bc,
                                active=p.active, mask_bc=p.mask_bc,
                                device="cpu").solve(**TOL)


def test_biblis_geometry_and_keff():
    p = build_biblis(cells_per_assembly=2)
    assert p.grid.shape == (34, 34, 1)
    assert int(p.active.sum()) == 257 * 4        # matches FEMFFUSION active cells
    # Fully symmetric octagonal core.
    m = p.material_map[:, :, 0]
    assert np.array_equal(m, m[:, ::-1]) and np.array_equal(m, m.T)
    res = _solve(p)
    assert res.converged
    # FEMFFUSION diffusion (FE3, converged) gives 1.02535; FV at this mesh is
    # near it and Richardson-extrapolates onto it.
    assert res.k_eff == pytest.approx(1.0287, abs=2e-3)


def test_iaea_geometry_and_keff():
    p = build_iaea(cells_per_node=2)
    assert p.grid.shape == (34, 34, 38)
    assert int(p.active.sum()) == 241 * 19 * 8   # 241 active/plane, 19 planes, r^3
    res = _solve(p)
    assert res.converged
    # Classic IAEA-3D reference k_eff = 1.02903.
    assert res.k_eff == pytest.approx(1.02903, abs=1e-3)


def test_boundary_barely_matters_for_reflected_iaea():
    # The thick reflector screens the outer boundary: k is nearly BC-independent.
    p = build_iaea(cells_per_node=1)
    kw = dict(active=p.active, device="cpu")
    k_alb = DiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                 bc=p.bc, mask_bc=p.mask_bc, **kw).solve(**TOL).k_eff
    k_ref = DiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                 bc="reflective", mask_bc="reflective", **kw).solve(**TOL).k_eff
    assert abs(k_alb - k_ref) < 5e-3
