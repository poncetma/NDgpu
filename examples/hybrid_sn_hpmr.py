"""Hybrid S_N/diffusion drum-worth analysis (HP-MR motivated, 2D Cartesian).

The S_N counterpart of examples/hpmr_hybrid.py. There the transport treatment
was SP3 on the body-fitted HP-MR triangular mesh; discrete ordinates has no tri
sweep, so this runs on a 2D Cartesian stand-in for the HP-MR drum physics: a
fuel core with a central control-drum absorber block on vacuum boundaries, the
drum "rotated" from withdrawn to inserted by ramping its absorption.

Three solves per drum state:
  * diffusion (whole core),
  * full S_N (discrete-ordinates transport, whole core) -- the reference,
  * hybrid -- S_N transport in the drum block only, diffusion in the bulk,
    coupled by the interface net current (ndgpu.HybridSNDiffusionSolver).

As with SP3, the interesting quantity is the control-drum WORTH (a reactivity
difference): diffusion over-predicts the near-black absorber's worth because it
misses the transport flux self-shielding that makes the drum "greyer"; the
hybrid recovers most of that self-shielding by resolving the drum -- and only the
drum -- with transport.

    python examples/hybrid_sn_hpmr.py [n_cells] [n_polar] [n_azi]

Cross sections are illustrative placeholders, not predictive.
"""

import sys
import time

import numpy as np

from ndgpu import (DiffusionEigenSolver, Grid, HybridSNDiffusionSolver, Material,
                   SNTransportSolver)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
n_polar = int(sys.argv[2]) if len(sys.argv) > 2 else 2
n_azi = int(sys.argv[3]) if len(sys.argv) > 3 else 8

BC3 = (("vacuum", "vacuum"), ("vacuum", "vacuum"), "reflective")
grid = Grid(shape=(n, n, 1), size=(2.0 * n, 2.0 * n, 1.0))
fuel = Material(diffusion=[1.1], sigma_a=[0.012], nu_sigma_f=[0.030],
                sigma_s=[[0.0]], name="fuel")
mmap = np.zeros((n, n, 1), int)
lo, hi = int(n * 0.42), int(n * 0.58)
mmap[lo:hi, lo:hi, 0] = 1                                    # central drum block
drum = mmap[:, :, 0].astype(bool)
n_drum, n_active = int(drum.sum()), n * n
print(f"HP-MR-motivated 2D hybrid S_N/diffusion, {n}x{n} cells, "
      f"S({n_polar * n_azi}) ({n_polar} polar x {n_azi} azi)")
print(f"transport (S_N) region = central drum block: {n_drum}/{n_active} cells "
      f"({100 * n_drum / n_active:.0f}%); diffusion elsewhere\n")


def solve(sigma_a):
    absb = Material(diffusion=[0.9], sigma_a=[sigma_a], nu_sigma_f=[0.0],
                    sigma_s=[[0.0]], name="absorber")
    mats = [fuel, absb]
    kd = DiffusionEigenSolver(grid, mats, material_map=mmap, bc=BC3,
                              device="cpu").solve(tol_k=1e-9, tol_source=1e-8).k_eff
    t = time.perf_counter()
    ks = SNTransportSolver(grid, mats, material_map=mmap[:, :, 0], n_polar=n_polar,
                           n_azi=n_azi, bc="vacuum").solve(tol_k=1e-7,
                                                           tol_source=1e-6).k_eff
    t_sn = time.perf_counter() - t
    t = time.perf_counter()
    kh = HybridSNDiffusionSolver(grid, mats, material_map=mmap, sn_mask=drum,
                                 n_polar=n_polar, n_azi=n_azi,
                                 bc="vacuum").solve(tol_k=1e-8, tol_source=1e-7).k_eff
    t_hyb = time.perf_counter() - t
    return kd, ks, kh, t_sn, t_hyb


states = [("withdrawn", 0.02), ("", 0.1), ("", 0.3), ("inserted", 0.8)]
print(f"{'drum':>10} {'Sig_a':>6} {'k_diff':>9} {'k_SN':>9} {'k_hyb':>9}  "
      f"{'SN-dif':>7} {'hyb-dif':>7} {'hyb-SN':>7}")
K = {}
t_sn_tot = t_hyb_tot = 0.0
for label, sa in states:
    kd, ks, kh, ts, th = solve(sa)
    K[sa] = (kd, ks, kh)
    t_sn_tot += ts; t_hyb_tot += th
    print(f"{label:>10} {sa:>6.2f} {kd:>9.5f} {ks:>9.5f} {kh:>9.5f}  "
          f"{(ks - kd) * 1e5:>+7.0f} {(kh - kd) * 1e5:>+7.0f} {(kh - ks) * 1e5:>+7.0f}")

sa_lo, sa_hi = 0.02, 0.8
worth = lambda x: (1.0 / K[sa_lo][x] - 1.0 / K[sa_hi][x]) * 1e5
wd, ws, wh = worth(0), worth(1), worth(2)
print(f"\ncontrol-drum worth (withdrawn -> inserted), S_N = reference:")
print(f"  full S_N   {ws:>8.0f} pcm   (reference)")
print(f"  hybrid     {wh:>8.0f} pcm   ({(wh - ws) / ws * 100:+.1f}% vs S_N)")
print(f"  diffusion  {wd:>8.0f} pcm   ({(wd - ws) / ws * 100:+.1f}% vs S_N)")
closed = (1.0 - abs(wh - ws) / abs(wd - ws)) * 100
print(f"\n  hybrid closes {closed:.0f}% of the diffusion->S_N worth error, "
      f"with transport in {100 * n_drum / n_active:.0f}% of the core.")
print(f"  cost: full S_N {t_sn_tot:.1f} s total, hybrid {t_hyb_tot:.1f} s "
      f"(transport confined to the drum).")
print("\n(placeholder cross sections: illustrative, not predictive.)")
