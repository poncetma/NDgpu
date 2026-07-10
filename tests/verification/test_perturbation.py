"""First-order perturbation theory: ndgpu.first_order_reactivity.

Two regimes, both physically important:

* Weak perturbations (feedback coefficients, cross-section sensitivities) -- the
  first-order estimate matches a direct re-solve to fractions of a percent, and
  the error shrinks quadratically as the perturbation does. This is the regime
  the tool is for.
* A strong localized black absorber (the HP-MR B4C drum arc) -- first-order PT
  over-predicts the worth several-fold, because it neglects the order-one flux
  self-shielding the absorber creates. We assert that breakdown so the
  limitation stays documented (and nobody mistakes PT for a drum-worth shortcut).
"""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, Grid, Material, PWR_TWO_GROUP,
                   first_order_reactivity)
from ndgpu.benchmarks.hpmr import build_hpmr2d, _placeholder_materials
from ndgpu.tri import TriDiffusionEigenSolver

TIGHT = dict(tol_k=1e-10, tol_source=1e-9)
GRID = Grid(shape=(20, 20, 20), size=(80.0, 80.0, 80.0))


def _poison(material, dsigma_a):
    return Material(name="perturbed", diffusion=material.diffusion,
                    sigma_a=material.sigma_a + np.asarray(dsigma_a),
                    nu_sigma_f=material.nu_sigma_f, sigma_s=material.sigma_s,
                    chi=material.chi)


def test_first_order_matches_resolve_and_converges_quadratically():
    ref = DiffusionEigenSolver(GRID, PWR_TWO_GROUP, device="cpu")
    fwd = ref.solve(**TIGHT)
    adj = ref.solve(adjoint=True, **TIGHT)

    def errors(eps):
        pert = DiffusionEigenSolver(GRID, _poison(PWR_TWO_GROUP, [0.0, eps]), device="cpu")
        drho_exact = 1.0 / fwd.k_eff - 1.0 / pert.solve(**TIGHT).k_eff
        drho_pt = first_order_reactivity(ref, fwd, adj, pert)
        return abs(drho_pt - drho_exact), abs(drho_exact)

    e_small, rho_small = errors(1e-4)
    e_big, rho_big = errors(1e-3)
    assert e_small / rho_small < 1e-3               # accurate for a weak perturbation
    # error is second order in the perturbation: 10x eps -> ~100x error
    assert 40 < e_big / e_small < 250, e_big / e_small


def _tri_hpmr(angle, materials=None):
    p = build_hpmr2d(refine=3, drum_angle_deg=angle, absorber="polar", materials=materials)
    return TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, active=p.active,
                                   mask_bc=p.mask_bc, mix_material=p.mix_material,
                                   mix_weight=p.mix_weight)


def test_first_order_reactivity_on_tri_weak_perturbation():
    # A weak absorption bump in the fuel: PT on the real HP-MR tri solver agrees
    # with a direct re-solve.
    base = _placeholder_materials()
    ref = _tri_hpmr(90.0)
    fwd = ref.solve(**TIGHT)
    adj = ref.solve(adjoint=True, **TIGHT)

    perturbed_mats = [base[0], _poison(base[1], [0.0, 1e-3])] + base[2:]
    pert = _tri_hpmr(90.0, materials=perturbed_mats)
    drho_exact = 1.0 / fwd.k_eff - 1.0 / pert.solve(**TIGHT).k_eff
    drho_pt = first_order_reactivity(ref, fwd, adj, pert)
    assert drho_exact < 0
    assert abs(drho_pt - drho_exact) / abs(drho_exact) < 0.05


def test_first_order_over_predicts_black_drum_worth():
    # Full drum swing (arc out -> arc into core). First-order PT badly
    # over-predicts because the near-black B4C arc self-shields: the neglected
    # flux depression is an order-one effect, not a small correction.
    ref = _tri_hpmr(0.0)
    fwd = ref.solve(**TIGHT)
    adj = ref.solve(adjoint=True, **TIGHT)

    pert = _tri_hpmr(180.0)
    worth_exact = 1.0 / fwd.k_eff - 1.0 / pert.solve(**TIGHT).k_eff
    worth_pt = first_order_reactivity(ref, fwd, adj, pert)
    assert worth_exact < 0 and worth_pt < 0
    # PT over-predicts the magnitude by more than 3x (self-shielding it ignores)
    assert worth_pt / worth_exact > 3.0
