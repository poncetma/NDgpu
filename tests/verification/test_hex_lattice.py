"""Structured hexagonal-lattice operator: topology and exact invariants.

The homogeneous 7-hex cluster with reflective boundaries must give k = k_inf
exactly (discretization-independent), which pins the hex topology and
interior coupling; vacuum/zero-flux must order correctly around it.
"""

import numpy as np
import pytest

from ndgpu import HexDiffusionEigenSolver, HexGrid, Material, offset_to_axial

TOL = dict(tol_k=1e-9, tol_source=1e-8)


def _hex7():
    mat = Material(name="hex", diffusion=[1 / (3 * 0.23751333043), 1 / (3 * 1.01359946401)],
                   sigma_a=[0.0278454, 0.107186], nu_sigma_f=[5.62285e-3, 0.145865],
                   sigma_s=[[0, 1.60795e-2], [0, 0]], chi=[1, 0])
    void = Material(name="void", diffusion=[1, 1], sigma_a=[0, 0], nu_sigma_f=[0, 0],
                    sigma_s=[[0, 0], [0, 0]])
    mmap = np.pad(offset_to_axial(np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]])), 1)[:, :, None]
    grid = HexGrid(shape=mmap.shape, pitch=23.6)
    return grid, [void, mat], mmap, mmap > 0


def test_offset_to_axial_gives_six_neighbours():
    # The centre of a 7-hex cluster must have all six axial neighbours present.
    ax = offset_to_axial(np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]]))
    rc = np.argwhere(ax > 0)
    # centre is the cell whose 6 axial neighbours are all active
    def nbrs(r, c):
        return [(r, c - 1), (r, c + 1), (r - 1, c), (r + 1, c), (r - 1, c + 1), (r + 1, c - 1)]
    active = {tuple(p) for p in rc}
    assert any(all(n in active for n in nbrs(r, c)) for r, c in rc)


def test_hex7_reflective_equals_kinf():
    grid, mats, mmap, active = _hex7()
    res = HexDiffusionEigenSolver(grid, mats, mmap, active=active,
                                  mask_bc="reflective", device="cpu").solve(**TOL)
    assert res.k_eff == pytest.approx(0.626177, abs=1e-5)   # FEMFFUSION reflective


def test_hex7_bc_ordering():
    grid, mats, mmap, active = _hex7()
    k = {}
    for bc in ("reflective", "vacuum", "zero-flux"):
        k[bc] = HexDiffusionEigenSolver(grid, mats, mmap, active=active,
                                        mask_bc=bc, device="cpu").solve(**TOL).k_eff
    assert k["zero-flux"] < k["vacuum"] < k["reflective"]
