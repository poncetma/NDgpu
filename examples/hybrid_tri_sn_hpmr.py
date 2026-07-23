"""Hybrid S_N / diffusion on the HP-MR triangular core.

The triangular-mesh culmination of the S_N work: full discrete-ordinates
transport (second-order SCB) runs only in the control-drum cells and diffusion in
the bulk, on the real body-fitted HP-MR mesh (ndgpu.HybridTriSNDiffusionSolver).
The drums are excised from the diffusion domain and the two regions are coupled
by the interface net current.

The B4C arc is represented by **polar volume-mixing** (absorber="polar"): the
exact arc area fraction is diluted into the drum cells, so the reference
diffusion, the full S_N, and the hybrid are all spatially unbiased (the thin 1 cm
arc is otherwise thinner than a triangle). Drum-angle convention: 0 deg = arc at
the core centre (inserted), 180 deg = outward (withdrawn); every drum is measured
from its own radial line.

For inserted and withdrawn states the core is solved three ways -- triangular
diffusion, full triangular S_N (SCB), and the hybrid -- and the control-drum
worth (reactivity of inserting the drums) is compared. Transport self-shields the
near-black arc, so it resolves less worth than diffusion; the hybrid captures
that self-shielding with transport in ~1/5 of the cells.

    python examples/hybrid_tri_sn_hpmr.py [refine]   (default 4)

Cross sections are illustrative placeholders, not predictive.
"""

import sys
import time

import numpy as np

from ndgpu.benchmarks import build_hpmr2d, hpmr_transport_mask
from ndgpu.tri import TriDiffusionEigenSolver
from ndgpu.tri_sn import TriSNTransportSolver
from ndgpu.hybrid_tri_sn import HybridTriSNDiffusionSolver

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TOL = dict(tol_k=5e-7, tol_source=5e-6, max_outer=200)


def solve_all(angle):
    p = build_hpmr2d(refine=refine, drum_angle_deg=float(angle), absorber="polar")
    mix = dict(mix_material=p.mix_material, mix_weight=p.mix_weight)
    kw = dict(active=p.active, mask_bc=p.mask_bc)
    kd = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, device="cpu",
                                 **kw, **mix).solve(tol_k=1e-8, tol_source=1e-7).k_eff
    mask = hpmr_transport_mask(p, "drum").reshape(p.grid.shape)
    t = time.perf_counter()
    ksn = TriSNTransportSolver(p.grid, p.materials, p.material_map, active=p.active,
                               n_polar=2, n_azi=8, bc="vacuum", scheme="scb",
                               **mix).solve(**TOL).k_eff
    t_sn = time.perf_counter() - t
    t = time.perf_counter()
    kh = HybridTriSNDiffusionSolver(p.grid, p.materials, p.material_map,
                                    sn_mask=mask, n_polar=2, n_azi=8,
                                    **kw, **mix).solve(**TOL).k_eff
    t_h = time.perf_counter() - t
    frac = 100.0 * int(mask.sum()) / int((np.asarray(p.material_map) > 0).sum())
    return kd, ksn, kh, frac, t_sn, t_h


print(f"HP-MR 2D hybrid S_N/diffusion, refine={refine}, polar (volume-mixed) drums")
K, t_sn_tot, t_h_tot, frac = {}, 0.0, 0.0, 0.0
for label, a in (("inserted", 0.0), ("withdrawn", 180.0)):
    kd, ksn, kh, frac, t_sn, t_h = solve_all(a)
    K[a] = (kd, ksn, kh)
    t_sn_tot += t_sn
    t_h_tot += t_h
    print(f"  {label:>9} ({a:5.0f}deg):  diffusion {kd:.5f}   full S_N "
          f"{(ksn - kd) * 1e5:+.0f}   hybrid {(kh - kd) * 1e5:+.0f} pcm vs diff")

# drum worth = reactivity of insertion (withdrawn -> inserted); negative.
worth = lambda x: (1.0 / K[180.0][x] - 1.0 / K[0.0][x]) * 1e5
wd, ws, wh = worth(0), worth(1), worth(2)
print(f"\ncontrol-drum worth (inserting the drums):")
print(f"  diffusion  {wd:>8.0f} pcm")
print(f"  full S_N   {ws:>8.0f} pcm   (transport correction {ws - wd:+.0f} pcm, "
      f"{t_sn_tot:.0f} s)")
print(f"  hybrid     {wh:>8.0f} pcm   (transport correction {wh - wd:+.0f} pcm, "
      f"transport in {frac:.0f}% of cells, {t_h_tot:.0f} s)")
print("\n  Transport self-shields the black arc, resolving less worth than")
print("  diffusion; the hybrid captures that with transport confined to the drums.")
print("  (The isotropic interface reconstruction over-predicts the magnitude for")
print("  the 12 tightly-arranged drums; the limits and isolated-drum case are")
print("  exact, and the full-S_N worth resolves for refine >= 6 -- see hpmr_tri_sn.py.)")
print("\n(placeholder cross sections: illustrative, not predictive.)")
