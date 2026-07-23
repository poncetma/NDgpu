"""Hybrid S_N / diffusion on the triangular mesh (HybridTriSNDiffusionSolver):
full transport (SCB) in a masked drum region, diffusion in the bulk, coupled by
the interface net current -- the triangular counterpart of the Cartesian hybrid
and the payoff of the tri-S_N work.

Because S_N and diffusion are genuinely different discretizations, this is a real
domain decomposition (the drum is excised from the diffusion domain; its outgoing
current sources the adjacent bulk cells). The checks:

  * exact limits -- an empty drum mask reproduces the tri diffusion solver
    bit-for-bit, a full mask reproduces tri-S_N;
  * physics -- transport in a localized strong-absorber drum recovers essentially
    all of the diffusion -> S_N correction (the transport effect is confined to
    the drum), and always the same sign as full S_N.
"""

import numpy as np
import pytest

from ndgpu import Material
from ndgpu.tri import TriGrid, TriDiffusionEigenSolver
from ndgpu.tri_sn import TriSNTransportSolver
from ndgpu.hybrid_tri_sn import HybridTriSNDiffusionSolver

FUEL = Material(diffusion=[1.1], sigma_a=[0.012], nu_sigma_f=[0.030],
                sigma_s=[[0.0]], name="fuel")
ABSB = Material(diffusion=[0.9], sigma_a=[0.5], nu_sigma_f=[0.0],
                sigma_s=[[0.0]], name="absorber")
QUAD = dict(n_polar=2, n_azi=8)
TIGHT = dict(tol_k=1e-8, tol_source=1e-7)


def _problem():
    nr = nc = 12
    grid = TriGrid(shape=(nr, nc, 2), side=2.0)
    active = np.zeros((nr, nc, 2), bool)
    active[1:-1, 1:-1, :] = True                             # interior, void border
    drum = np.zeros((nr, nc, 2), bool)
    drum[5:8, 5:8, :] = True
    drum &= active
    mmap = np.zeros((nr, nc, 2), int)
    mmap[drum] = 1                                           # absorber in the drum
    return grid, mmap, active, drum


def _kdiff(grid, mmap, active):
    return TriDiffusionEigenSolver(grid, [FUEL, ABSB], mmap, active=active,
                                   mask_bc="vacuum", device="cpu").solve(
        tol_k=1e-9, tol_source=1e-8).k_eff


def _ksn(grid, mmap, active):
    return TriSNTransportSolver(grid, [FUEL, ABSB], mmap, active=active,
                                scheme="scb", bc="vacuum", **QUAD).solve(
        tol_k=1e-7, tol_source=1e-6).k_eff


def _khyb(grid, mmap, active, mask):
    return HybridTriSNDiffusionSolver(grid, [FUEL, ABSB], mmap, sn_mask=mask,
                                      active=active, mask_bc="vacuum",
                                      **QUAD).solve(**TIGHT).k_eff


def test_empty_mask_is_tri_diffusion():
    grid, mmap, active, _ = _problem()
    k_diff = _kdiff(grid, mmap, active)
    k_hyb = _khyb(grid, mmap, active, np.zeros(grid.shape, bool))
    assert k_hyb == pytest.approx(k_diff, abs=1e-6)


def test_full_mask_is_tri_sn():
    grid, mmap, active, _ = _problem()
    k_sn = _ksn(grid, mmap, active)
    k_hyb = _khyb(grid, mmap, active, active.copy())
    assert k_hyb == pytest.approx(k_sn, abs=1e-6)


def test_drum_hybrid_captures_the_localized_transport_correction():
    grid, mmap, active, drum = _problem()
    k_diff = _kdiff(grid, mmap, active)
    k_sn = _ksn(grid, mmap, active)
    k_hyb = _khyb(grid, mmap, active, drum)
    gap = k_sn - k_diff
    assert gap * 1e5 > 1000.0                                # real transport effect
    # the transport is confined to the drum, so the hybrid recovers ~all of it
    assert abs(k_hyb - k_sn) < 0.1 * abs(gap)


def test_krylov_coupling_matches_schwarz():
    # The monolithic Krylov interface solve (one fused drum sweep + one bulk
    # diffusion backsolve per matvec) shares the Schwarz fixed point: identical
    # k, fewer transport sweeps.
    grid, mmap, active, drum = _problem()

    def run(coupling):
        h = HybridTriSNDiffusionSolver(grid, [FUEL, ABSB], mmap, sn_mask=drum,
                                       active=active, mask_bc="vacuum",
                                       coupling=coupling, **QUAD)
        r = h.solve(**TIGHT)
        return r, h.sn._sweep_count

    r_s, sweeps_s = run("schwarz")
    r_k, sweeps_k = run("krylov")
    assert r_s.converged and r_k.converged
    assert r_k.k_eff == pytest.approx(r_s.k_eff, abs=1e-6)
    assert sweeps_k < sweeps_s


def test_levels_engine_matches_lu():
    # The level-scheduled sweep engine (the GPU path) drives the same krylov
    # interface coupling through _sweep_iface: identical k to the LU engine.
    grid, mmap, active, drum = _problem()

    def run(engine):
        h = HybridTriSNDiffusionSolver(grid, [FUEL, ABSB], mmap, sn_mask=drum,
                                       active=active, mask_bc="vacuum",
                                       engine=engine, **QUAD)
        return h.solve(**TIGHT)

    r_lu = run("lu")
    r_lv = run("levels")
    assert r_lu.converged and r_lv.converged
    assert r_lv.k_eff == pytest.approx(r_lu.k_eff, abs=1e-10)
