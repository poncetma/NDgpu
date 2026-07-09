"""Extruded (tri-z) triangular solver: analytic and consistency checks.

The in-plane stencil is untouched by the z extension, so these tests pin the
new physics only: the axial coupling, the z-face boundary conditions, and
their interplay with the active-mask machinery.
"""

import numpy as np
import pytest

from ndgpu import PWR_TWO_GROUP, k_from_buckling, k_infinite
from ndgpu.tri import TriDiffusionEigenSolver, TriGrid

H = 120.0
TOL = dict(tol_k=1e-10, tol_source=1e-9)


def _k(nz, zbc):
    grid = TriGrid(shape=(3, 3, 2, nz), side=5.0, height=H)
    res = TriDiffusionEigenSolver(grid, PWR_TWO_GROUP,
                                  bc=("reflective", "reflective", zbc),
                                  device="cpu").solve(**TOL)
    assert res.converged
    return res.k_eff


def test_all_reflective_gives_kinf():
    assert _k(4, "reflective") == pytest.approx(k_infinite(PWR_TWO_GROUP), abs=1e-9)


def test_zero_flux_z_converges_2nd_order_to_slab():
    # Reflective in-plane + zero-flux z = the exact 1D slab problem with
    # buckling (pi/H)^2; halving dz must cut the k error 4x.
    k_exact = k_from_buckling(PWR_TWO_GROUP, (np.pi / H) ** 2)
    e16 = abs(_k(16, "zero-flux") - k_exact)
    e32 = abs(_k(32, "zero-flux") - k_exact)
    assert e16 / e32 == pytest.approx(4.0, abs=0.3)
    assert e32 < 3e-5


def test_z_bc_ordering():
    ks = {zbc: _k(16, zbc) for zbc in ("reflective", "vacuum", "zero-flux")}
    assert ks["zero-flux"] < ks["vacuum"] < ks["reflective"]


def test_extruded_vver440_matches_2d():
    # The full VVER-440 core (heterogeneous, active mask, vacuum in-plane)
    # extruded with reflective z faces must reproduce the 2D k exactly:
    # the z direction adds no gradient.
    from ndgpu.benchmarks import build_vver440
    p = build_vver440(refine=1)
    solve = dict(tol_k=1e-9, tol_source=1e-8)
    k2d = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                  active=p.active, mask_bc=p.mask_bc,
                                  device="cpu").solve(**solve).k_eff
    nz = 2
    grid = TriGrid(shape=p.grid.shape + (nz,), side=p.grid.side, height=30.0)
    mmap = np.repeat(p.material_map[..., None], nz, axis=3)
    active = np.repeat(p.active[..., None], nz, axis=3)
    k3d = TriDiffusionEigenSolver(grid, p.materials, mmap, active=active,
                                  mask_bc=p.mask_bc,
                                  bc=("reflective", "reflective", "reflective"),
                                  device="cpu").solve(**solve).k_eff
    assert k3d == pytest.approx(k2d, abs=1e-7)


def test_grid_shape_validation():
    with pytest.raises(ValueError, match=r"nrows, ncols, 2"):
        TriGrid(shape=(3, 3, 4), side=1.0)
    g = TriGrid(shape=(3, 3, 2, 8), side=2.0, height=40.0)
    assert g.nz == 8 and g.dz == 5.0 and g.n_cells == 3 * 3 * 2 * 8
    g2 = TriGrid(shape=(3, 3, 2), side=2.0)
    assert g2.nz == 1 and g2.dz == 1.0
