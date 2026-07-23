"""Hybrid S_N / diffusion on the HP-MR triangular core.

The triangular-mesh culmination of the S_N work: full discrete-ordinates
transport (second-order SCB) runs only in the control-drum cells, diffusion in
the bulk, on the real body-fitted HP-MR mesh -- so the drum self-shielding is
resolved with genuine transport where it matters, and the fuel bulk stays cheap
diffusion. The drum is excised from the diffusion domain and the two regions are
coupled by the interface net current (ndgpu.HybridTriSNDiffusionSolver), the same
coupling the Cartesian hybrid established.

For each drum angle the core is solved three ways -- triangular diffusion, full
triangular S_N (SCB), and the hybrid -- and the control-drum worth (reactivity,
withdrawn 0 deg -> inserted 180 deg) is compared. The transport self-shields the
near-black B4C arc, so it resolves less worth than diffusion, and the hybrid
captures that self-shielding (its worth correction has the right sign) with
transport in only ~1/5 of the cells.

Two honest caveats for the 12-drum core. (1) The full-S_N worth correction is
small and only resolves for refine >= 6 (see examples/hpmr_tri_sn.py), so at the
default refine=4 the S_N reference here is itself under-resolved. (2) The
isotropic incoming-flux reconstruction at the drum interface over-predicts the
worth magnitude when many drums are tightly arranged -- the same limitation the
Cartesian HybridSNDiffusionSolver shows; it is exact for an isolated drum and the
limits (empty mask = diffusion, full mask = S_N) are exact regardless. A P1
incoming or a buffer ring around each drum would tighten the multi-drum case.

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
    p = build_hpmr2d(refine=refine, drum_angle_deg=float(angle), absorber="raster")
    kw = dict(active=p.active, mask_bc=p.mask_bc)
    kd = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map, device="cpu",
                                 **kw).solve(tol_k=1e-8, tol_source=1e-7).k_eff
    mask = hpmr_transport_mask(p, "drum").reshape(p.grid.shape)
    t = time.perf_counter()
    ksn = TriSNTransportSolver(p.grid, p.materials, p.material_map, active=p.active,
                               n_polar=2, n_azi=8, bc="vacuum",
                               scheme="scb").solve(**TOL).k_eff
    t_sn = time.perf_counter() - t
    t = time.perf_counter()
    kh = HybridTriSNDiffusionSolver(p.grid, p.materials, p.material_map,
                                    sn_mask=mask, n_polar=2, n_azi=8,
                                    **kw).solve(**TOL).k_eff
    t_h = time.perf_counter() - t
    frac = 100.0 * int(mask.sum()) / int((np.asarray(p.material_map) > 0).sum())
    return kd, ksn, kh, frac, t_sn, t_h


print(f"HP-MR 2D hybrid S_N/diffusion, refine={refine}, raster drums")
K, t_sn_tot, t_h_tot, frac = {}, 0.0, 0.0, 0.0
for a in (0.0, 180.0):
    kd, ksn, kh, frac, t_sn, t_h = solve_all(a)
    K[a] = (kd, ksn, kh)
    t_sn_tot += t_sn
    t_h_tot += t_h
    print(f"  {a:5.0f} deg:  diffusion {kd:.5f}   full S_N {(ksn - kd) * 1e5:+.0f}"
          f"   hybrid {(kh - kd) * 1e5:+.0f} pcm")

worth = lambda x: (1.0 / K[0.0][x] - 1.0 / K[180.0][x]) * 1e5
wd, ws, wh = worth(0), worth(1), worth(2)
print(f"\ncontrol-drum worth (withdrawn -> inserted):")
print(f"  diffusion  {wd:>8.0f} pcm")
print(f"  full S_N   {ws:>8.0f} pcm   (transport correction {ws - wd:+.0f} pcm, "
      f"{t_sn_tot:.0f} s)")
print(f"  hybrid     {wh:>8.0f} pcm   (transport correction {wh - wd:+.0f} pcm, "
      f"transport in {frac:.0f}% of cells, {t_h_tot:.0f} s)")
print("\n  The hybrid's worth correction has the right (self-shielding) sign but")
print("  over-predicts the magnitude for the 12 tightly-arranged drums (isotropic")
print("  interface reconstruction); the limits and the isolated-drum case are")
print("  exact. See the module docstring.")
print("\n(placeholder cross sections: illustrative, not predictive.)")
