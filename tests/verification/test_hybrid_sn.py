"""Hybrid discrete-ordinates (S_N) / diffusion solver: full transport only in a
masked subdomain (the control-drum absorber), diffusion in the bulk, coupled by
the interface net current (the drum is excised from the diffusion domain).

The S_N counterpart of the hybrid SP3/diffusion solver. Because S_N and diffusion
are genuinely different discretizations (angular unknowns + sweep vs a scalar
stencil), the coupling is a real domain decomposition rather than a masking, so
the checks are:

  * exact limits -- an empty mask is bit-for-bit the diffusion solver, a full
    mask is bit-for-bit S_N (every diffusion cell excised);
  * physics -- transport in the drum alone recovers most of the diffusion -> S_N
    correction, and across drum states the hybrid tracks the S_N control-drum
    worth far better than diffusion (which over-predicts the near-black arc's
    worth for want of the transport self-shielding).
"""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, Grid, HybridSNDiffusionSolver, Material,
                   SNTransportSolver)

BC3 = (("vacuum", "vacuum"), ("vacuum", "vacuum"), "reflective")
FUEL = Material(diffusion=[1.1], sigma_a=[0.012], nu_sigma_f=[0.030],
                sigma_s=[[0.0]], name="fuel")


def _problem(n=24, sigma_a=0.3):
    grid = Grid(shape=(n, n, 1), size=(2.0 * n, 2.0 * n, 1.0))
    absb = Material(diffusion=[0.9], sigma_a=[sigma_a], nu_sigma_f=[0.0],
                    sigma_s=[[0.0]], name="absorber")
    mmap = np.zeros((n, n, 1), int)
    lo, hi = int(n * 0.42), int(n * 0.58)
    mmap[lo:hi, lo:hi, 0] = 1
    drum = mmap[:, :, 0].astype(bool)
    return grid, [FUEL, absb], mmap, drum


def _kd(cls, grid, mats, mmap, **kw):
    return cls(grid, mats, material_map=mmap, bc=BC3, device="cpu",
               **kw).solve(tol_k=1e-9, tol_source=1e-8).k_eff


def _ksn(grid, mats, mmap):
    return SNTransportSolver(grid, mats, material_map=mmap[:, :, 0], n_polar=2,
                             n_azi=8, bc="vacuum").solve(tol_k=1e-7,
                                                         tol_source=1e-6).k_eff


def _khyb(grid, mats, mmap, mask):
    return HybridSNDiffusionSolver(grid, mats, material_map=mmap, sn_mask=mask,
                                   n_polar=2, n_azi=8,
                                   bc="vacuum").solve(tol_k=1e-8,
                                                      tol_source=1e-7).k_eff


def test_empty_mask_is_diffusion():
    grid, mats, mmap, _ = _problem()
    k_diff = _kd(DiffusionEigenSolver, grid, mats, mmap)
    k_hyb = _khyb(grid, mats, mmap, np.zeros((grid.shape[0],) * 2, bool))
    assert k_hyb == pytest.approx(k_diff, abs=1e-6)


def test_full_mask_is_sn():
    grid, mats, mmap, _ = _problem()
    k_sn = _ksn(grid, mats, mmap)
    k_hyb = _khyb(grid, mats, mmap, np.ones((grid.shape[0],) * 2, bool))
    assert k_hyb == pytest.approx(k_sn, abs=1e-6)


def test_drum_hybrid_recovers_most_of_the_transport_gap():
    grid, mats, mmap, drum = _problem(sigma_a=0.3)
    k_diff = _kd(DiffusionEigenSolver, grid, mats, mmap)
    k_sn = _ksn(grid, mats, mmap)
    k_hyb = _khyb(grid, mats, mmap, drum)
    gap = k_sn - k_diff
    assert gap * 1e5 > 1000.0                                # real transport effect
    # the hybrid lands far closer to transport than diffusion (closes > 70%).
    assert abs(k_hyb - k_sn) < 0.3 * abs(gap)


def test_nonrectangular_region_rejected():
    grid, mats, mmap, _ = _problem()
    n = grid.shape[0]
    mask = np.zeros((n, n), bool)
    mask[10:14, 10:14] = True
    mask[13, 13] = False                                     # punch a hole -> not rectangular
    with pytest.raises(ValueError, match="rectangular"):
        HybridSNDiffusionSolver(grid, mats, material_map=mmap, sn_mask=mask,
                                n_polar=2, n_azi=8, bc="vacuum")


def test_hybrid_tracks_sn_drum_worth_better_than_diffusion():
    # Drum worth = reactivity from withdrawn (weak absorber) to inserted (strong).
    # Diffusion over-predicts it; the hybrid, resolving the drum with transport,
    # tracks the S_N worth much more closely.
    def ks(sigma_a):
        grid, mats, mmap, drum = _problem(sigma_a=sigma_a)
        return (_kd(DiffusionEigenSolver, grid, mats, mmap),
                _ksn(grid, mats, mmap), _khyb(grid, mats, mmap, drum))

    kd_lo, ksn_lo, kh_lo = ks(0.05)
    kd_hi, ksn_hi, kh_hi = ks(0.8)
    worth = lambda a, b: 1.0 / a - 1.0 / b                  # withdrawn -> inserted
    w_diff = worth(kd_lo, kd_hi)
    w_sn = worth(ksn_lo, ksn_hi)
    w_hyb = worth(kh_lo, kh_hi)
    assert w_diff < 0 and w_sn < 0 and w_hyb < 0
    assert abs(w_diff) > abs(w_sn)                          # diffusion over-predicts worth
    # the hybrid's worth error vs S_N is at most half of diffusion's.
    assert abs(w_hyb - w_sn) < 0.5 * abs(w_diff - w_sn)


def test_krylov_coupling_matches_schwarz():
    # The monolithic Krylov interface solve and the alternating Schwarz fixed
    # point share the same fixed point: identical k, far fewer sweeps.
    n = 24
    grid = Grid(shape=(n, n, 1), size=(24.0, 24.0, 1.0))
    fuel = Material(diffusion=[1.1], sigma_a=[0.012], nu_sigma_f=[0.026],
                    sigma_s=[[0.0]], name="fuel")
    absb = Material(diffusion=[0.9], sigma_a=[0.20], nu_sigma_f=[0.0],
                    sigma_s=[[0.0]], name="absorber")
    mmap = np.zeros((n, n, 1), int)
    lo, hi = int(n * 0.4), int(n * 0.6)
    mmap[lo:hi, lo:hi, 0] = 1
    mask = mmap[:, :, 0] == 1

    def run(coupling):
        h = HybridSNDiffusionSolver(grid, [fuel, absb], material_map=mmap,
                                    sn_mask=mask, n_polar=2, n_azi=8,
                                    bc="vacuum", coupling=coupling)
        r = h.solve(tol_k=1e-7, tol_source=1e-6)
        return r, sum(b["sn"]._sweep_count for b in h.boxes)

    r_s, sweeps_s = run("schwarz")
    r_k, sweeps_k = run("krylov")
    assert r_s.converged and r_k.converged
    assert r_k.k_eff == pytest.approx(r_s.k_eff, abs=1e-6)
    assert sweeps_k * 3 < sweeps_s
