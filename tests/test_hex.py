"""Structured hexagonal-lattice diffusion solver.

The homogeneous 7-hex cluster reproduces FEMFFUSION's reflective k exactly
(k = k_inf, discretization-independent), which validates the hex topology and
interior coupling; vacuum/zero-flux are correctly ordered. VVER-440 runs and
lands near the reference (a coarse-mesh, one-cell-per-assembly offset).
"""

import numpy as np
import pytest

from ndgpu import HexDiffusionEigenSolver, HexGrid, Material, offset_to_axial
from ndgpu.benchmarks import build_vver440

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


def test_vver440_triangular_geometry_and_k():
    # build_vver440 now returns the corrected hexagonal geometry on a body-fitted
    # triangular mesh (6 r^2 cells/assembly). k converges near the FEMFFUSION
    # diffusion reference 1.00349 (the old hand-placed hex was a sheared
    # parallelogram, ~1900 pcm off; this is a real hexagon).
    from ndgpu.tri import TriDiffusionEigenSolver
    p = build_vver440(refine=1)
    assert int(p.active.sum()) == 421 * 6            # 6 triangles per assembly
    res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                                  mask_bc=p.mask_bc, device="cpu").solve(tol_k=1e-8, tol_source=1e-7)
    assert res.converged
    assert res.k_eff == pytest.approx(1.0035, abs=6e-3)


def test_vver440_bulk_transient():
    # A distributed +0.1$ step (material 1) far below prompt-critical: a benign,
    # well-resolved transient with a prompt jump and slow delayed rise to ~1.1.
    from ndgpu import TransientSolver
    from ndgpu.tri import TriGroupOperator, TriDiffusionEigenSolver
    p = build_vver440(refine=1, perturbation="bulk")
    res = TransientSolver(p.grid, p.problem_at, p.kinetics, active=p.active,
                          mask_bc=p.mask_bc, device="cpu",
                          group_operator=TriGroupOperator,
                          eig_solver=TriDiffusionEigenSolver).solve(t_end=1.0, dt=0.02)
    P = np.asarray(res.power)
    assert P[-1] == pytest.approx(P.max(), rel=1e-6)  # monotone rise, no peak/relax
    assert 1.08 < P[-1] < 1.20                        # ~0.1$ delayed-supercritical rise
