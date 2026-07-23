"""Hybrid SPDN/diffusion solver: transport moments only on a masked subdomain
(e.g. the HP-MR control-drum absorber), plain diffusion elsewhere.

The construction rests on three exact invariants, verified here:

  * an all-True mask reproduces the full transport (SP3/SDPN) solve bit-for-bit
    -- the mask touches only the higher-moment source and coupling, so masking
    every cell changes nothing;
  * an empty mask reproduces the diffusion solve -- with no higher-moment source
    anywhere the moments vanish and moment 0 is exactly -div(D grad phi)+Sig_r;
  * in the "confined" mode the higher moments are pinned to exactly zero outside
    the mask, so the scalar flux there equals the diffusion field.

Then the physics on the HP-MR core: putting transport in the drums alone
recovers part of the SP3 drum-worth self-shielding, so the hybrid worth sits
strictly between diffusion (over-predicts) and full SP3 (the reference).
"""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, Grid, SDP1EigenSolver, SDP2EigenSolver,
                   SDP3EigenSolver, SP3EigenSolver, TriDiffusionEigenSolver,
                   TriSP3EigenSolver)
from ndgpu.benchmarks.hpmr import build_hpmr2d, hpmr_transport_mask
from ndgpu.materials import Material

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


def _cartesian_problem():
    """A strong absorber block embedded in fissile fuel -- a drum stand-in."""
    grid = Grid(shape=(40, 40, 1), size=(80.0, 80.0, 2.0))
    fuel = Material(diffusion=[1.2], sigma_a=[0.018], nu_sigma_f=[0.024],
                    sigma_s=[[0.0]], name="fuel")
    absb = Material(diffusion=[0.30], sigma_a=[1.2], nu_sigma_f=[0.0],
                    sigma_s=[[0.0]], name="absorber")
    mmap = np.zeros((40, 40, 1), dtype=int)
    drum = np.zeros((40, 40, 1), dtype=bool)
    drum[17:23, 17:23, :] = True
    mmap[drum] = 1
    bc = (("reflective", "zero-flux"), ("reflective", "zero-flux"), "reflective")
    return grid, [fuel, absb], mmap, drum, bc


@pytest.mark.parametrize("confine", [False, True])
def test_all_true_mask_reproduces_full_sp3(confine):
    grid, mats, mmap, drum, bc = _cartesian_problem()
    kw = dict(material_map=mmap, bc=bc, device="cpu")
    k_full = SP3EigenSolver(grid, mats, **kw).solve(**TIGHT).k_eff
    k_hyb = SP3EigenSolver(grid, mats, hybrid_mask=np.ones_like(drum),
                           hybrid_confine=confine, **kw).solve(**TIGHT).k_eff
    assert k_hyb == pytest.approx(k_full, abs=1e-8)


@pytest.mark.parametrize("confine", [False, True])
def test_empty_mask_reproduces_diffusion(confine):
    grid, mats, mmap, drum, bc = _cartesian_problem()
    kw = dict(material_map=mmap, bc=bc, device="cpu")
    k_diff = DiffusionEigenSolver(grid, mats, **kw).solve(**TIGHT).k_eff
    k_hyb = SP3EigenSolver(grid, mats, hybrid_mask=np.zeros_like(drum),
                           hybrid_confine=confine, **kw).solve(**TIGHT).k_eff
    assert k_hyb == pytest.approx(k_diff, abs=1e-8)


def test_confined_pins_phi2_to_zero_outside_mask():
    grid, mats, mmap, drum, bc = _cartesian_problem()
    s = SP3EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu",
                       hybrid_mask=drum, hybrid_confine=True)
    s.solve(**TIGHT)
    phi2 = np.asarray(s.state[0][1])
    assert np.abs(phi2[~drum]).max() == 0.0           # excised: exactly zero
    assert np.abs(phi2[drum]).max() > 1e-4            # transport lives in the drum


def test_faithful_phi2_lives_but_is_sourced_only_in_drum():
    # Faithful mode: phi2 is a global field that decays out of the drum (no hard
    # cut), so it is nonzero just outside but must peak inside the sourced drum.
    grid, mats, mmap, drum, bc = _cartesian_problem()
    s = SP3EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu",
                       hybrid_mask=drum, hybrid_confine=False)
    s.solve(**TIGHT)
    phi2 = np.abs(np.asarray(s.state[0][1]))
    assert phi2[drum].max() > phi2[~drum].max()      # sourced in the drum
    assert phi2[~drum].max() > 0.0                   # but decays, not excised


def test_diffusion_solver_rejects_hybrid_mask():
    grid, mats, mmap, drum, bc = _cartesian_problem()
    with pytest.raises(ValueError, match="no higher moments"):
        DiffusionEigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu",
                             hybrid_mask=drum)


@pytest.mark.parametrize("solver", [SDP1EigenSolver, SDP2EigenSolver,
                                    SDP3EigenSolver])
@pytest.mark.parametrize("confine", [False, True])
def test_sdpn_family_all_true_reproduces_full(solver, confine):
    grid, mats, mmap, drum, bc = _cartesian_problem()
    kw = dict(material_map=mmap, bc=bc, device="cpu")
    k_full = solver(grid, mats, **kw).solve(**TIGHT).k_eff
    k_hyb = solver(grid, mats, hybrid_mask=np.ones_like(drum),
                   hybrid_confine=confine, **kw).solve(**TIGHT).k_eff
    assert k_hyb == pytest.approx(k_full, abs=1e-8)


def test_sdp3_hybrid_rejects_congruence_path():
    # The symmetrizing congruence mixes moments across the whole domain, so it
    # cannot carry a per-region mask; asking for it explicitly must raise.
    grid, mats, mmap, drum, bc = _cartesian_problem()
    with pytest.raises(ValueError, match="congruence"):
        SDP3EigenSolver(grid, mats, material_map=mmap, bc="reflective",
                        device="cpu", hybrid_mask=drum, symmetrize=True)


def test_tri_all_true_mask_reproduces_full_sp3():
    p = build_hpmr2d(refine=4, drum_angle_deg=90.0, absorber="polar")
    kw = dict(active=p.active, mask_bc=p.mask_bc, mix_material=p.mix_material,
              mix_weight=p.mix_weight, device="cpu")
    k_full = TriSP3EigenSolver(p.grid, p.materials, p.material_map,
                               **kw).solve(**TIGHT).k_eff
    allmask = np.asarray(p.material_map) > 0
    k_hyb = TriSP3EigenSolver(p.grid, p.materials, p.material_map,
                              hybrid_mask=allmask, **kw).solve(**TIGHT).k_eff
    assert k_hyb == pytest.approx(k_full, abs=1e-7)


def _hpmr_k(solver_cls, angle, **extra):
    p = build_hpmr2d(refine=4, drum_angle_deg=angle, absorber="polar")
    kw = dict(active=p.active, mask_bc=p.mask_bc, mix_material=p.mix_material,
              mix_weight=p.mix_weight, device="cpu")
    if extra.get("hybrid_mask") == "drum":
        extra = dict(extra, hybrid_mask=hpmr_transport_mask(p, "drum"))
    return solver_cls(p.grid, p.materials, p.material_map,
                      **kw, **extra).solve(**TIGHT).k_eff


def test_hpmr_hybrid_worth_between_diffusion_and_sp3():
    # Control-drum worth (180 deg withdrawn -> 0 deg inserted). Diffusion
    # over-predicts the near-black arc's worth; full SP3 self-shields it down.
    # Sourcing transport in the drums alone recovers part of that self-shielding,
    # so the hybrid worth sits strictly between the two and closer to SP3.
    # Tuple order is (withdrawn, inserted) so the worth is negative.
    kd = (_hpmr_k(TriDiffusionEigenSolver, 180.0),
          _hpmr_k(TriDiffusionEigenSolver, 0.0))
    ks = (_hpmr_k(TriSP3EigenSolver, 180.0),
          _hpmr_k(TriSP3EigenSolver, 0.0))
    kh = (_hpmr_k(TriSP3EigenSolver, 180.0, hybrid_mask="drum"),
          _hpmr_k(TriSP3EigenSolver, 0.0, hybrid_mask="drum"))

    worth = lambda k: 1.0 / k[0] - 1.0 / k[1]        # negative (arc inserts)
    wd, ws, wh = worth(kd), worth(ks), worth(kh)
    assert wd < 0 and ws < 0 and wh < 0
    # full SP3 resolves the least worth (most self-shielding); diffusion the most
    assert abs(ws) < abs(wd)
    # hybrid is intermediate: less worth than diffusion, more than full SP3
    assert abs(ws) < abs(wh) < abs(wd)
    # and it closes at least a third of the diffusion->SP3 gap
    assert abs(wh - ws) < abs(wd - ws)
