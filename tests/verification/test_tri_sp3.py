"""Triangular simplified-P3 solver (TriSP3EigenSolver).

SP3 on the body-fitted hex/triangular mesh: the Cartesian SP3 angular block
(SP3GroupOperator) driven by the triangular spatial stencil (TriGroupOperator).
Two checks. First, composition: a homogeneous medium with reflective boundaries
must give exactly k_inf -- the flat-flux limit where the second SP3 moment
vanishes and SP3 collapses to diffusion -- which only holds if the block, the
triangular leakage operator and their coupling cohere. Second, physics: on the
HP-MR core SP3 must differ from diffusion, and specifically resolve *less* drum
worth, because it captures the flux self-shielding that makes the near-black
B4C arc greyer than diffusion assumes (the same transport effect SPH exists to
fold into diffusion constants).
"""

import numpy as np
import pytest

from ndgpu import (TriDiffusionEigenSolver, TriGrid, TriSP3EigenSolver,
                   k_infinite)
from ndgpu.benchmarks.hpmr import build_hpmr2d, _placeholder_materials

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


def test_tri_sp3_reflective_is_kinf():
    fuel = _placeholder_materials()[1]
    nr = nc = 6
    grid = TriGrid(shape=(nr, nc, 2), side=5.0)
    active = np.zeros((nr, nc, 2), dtype=bool)
    active[1:-1, 1:-1, :] = True                     # interior, one-cell void border
    cm = np.zeros((nr, nc, 2), dtype=np.int64)
    res = TriSP3EigenSolver(grid, [fuel], cm, active=active,
                            mask_bc="reflective").solve(**TIGHT)
    assert res.converged
    assert res.k_eff == pytest.approx(k_infinite(fuel), abs=1e-6)


def _hpmr_k(solver_cls, angle):
    p = build_hpmr2d(refine=4, drum_angle_deg=angle, absorber="polar")
    res = solver_cls(p.grid, p.materials, p.material_map, active=p.active,
                     mask_bc=p.mask_bc, mix_material=p.mix_material,
                     mix_weight=p.mix_weight).solve(**TIGHT)
    assert res.converged
    return res.k_eff


def test_tri_sp3_resolves_less_drum_worth_than_diffusion():
    # SP3 and diffusion disagree on the HP-MR (transport effect), by a physical
    # margin, and SP3 sees the near-black drum as less absorbing -- so the drum
    # worth it predicts is smaller in magnitude than diffusion's.
    # 0 = arc at the core centre (inserted), 180 = outward (withdrawn).
    kd_out, kd_in = _hpmr_k(TriDiffusionEigenSolver, 180.0), _hpmr_k(TriDiffusionEigenSolver, 0.0)
    ks_out, ks_in = _hpmr_k(TriSP3EigenSolver, 180.0), _hpmr_k(TriSP3EigenSolver, 0.0)

    assert abs(ks_out - kd_out) * 1e5 > 5.0          # SP3 is a distinct solution
    worth_diff = 1.0 / kd_out - 1.0 / kd_in          # both negative (arc inserts)
    worth_sp3 = 1.0 / ks_out - 1.0 / ks_in
    assert worth_diff < 0 and worth_sp3 < 0
    assert abs(worth_sp3) < abs(worth_diff)          # self-shielding -> less worth
