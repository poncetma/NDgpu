"""HP-MR hybrid S_N/diffusion: cost of the acceleration schemes.

Runs the ``hybrid_tri_sn_hpmr`` pipeline -- tri diffusion, full tri-S_N (SCB)
and the hybrid S_N/diffusion, drums inserted (0 deg) and withdrawn (180 deg),
polar volume-mixed absorber -- twice:

  * original    : the pre-acceleration schemes -- within-group GMRES everywhere
                  and the Anderson power outer (full S_N and hybrid boxes);
  * accelerated : the current defaults -- DSA within groups everywhere, plus
                  the CMFD outer on the full-S_N solves.

Both configurations discretize the identical problem, so k and the drum worth
must agree to the convergence tolerance; the cost columns are the point.
Following the benchmark protocol, accuracy (k, worth vs the original scheme as
the named reference) is paired with cost (outers, transport sweeps, wall time).

    python examples/hpmr_sn_accel_benchmark.py [refine]   (default 4)

Cross sections are illustrative placeholders, not predictive.
"""

import sys
import time

import numpy as np

from ndgpu.benchmarks import build_hpmr2d, hpmr_transport_mask
from ndgpu.hybrid_tri_sn import HybridTriSNDiffusionSolver
from ndgpu.tri import TriDiffusionEigenSolver
from ndgpu.tri_sn import TriSNTransportSolver

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TOL = dict(tol_k=5e-7, tol_source=5e-6, max_outer=200)

CONFIGS = {
    "original": dict(sn=dict(acceleration="gmres", outer_acceleration="power"),
                     hyb=dict(acceleration="gmres")),
    "accelerated": dict(sn=dict(acceleration="dsa", outer_acceleration="cmfd"),
                        hyb=dict(acceleration="dsa-gmres")),
}


def run_config(name, cfg):
    out = {}
    for angle in (0.0, 180.0):
        p = build_hpmr2d(refine=refine, drum_angle_deg=angle, absorber="polar")
        mix = dict(mix_material=p.mix_material, mix_weight=p.mix_weight)
        kd = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                     device="cpu", active=p.active,
                                     mask_bc=p.mask_bc, **mix).solve(
                                         tol_k=1e-8, tol_source=1e-7).k_eff
        mask = hpmr_transport_mask(p, "drum").reshape(p.grid.shape)
        t = time.perf_counter()
        rs = TriSNTransportSolver(p.grid, p.materials, p.material_map,
                                  active=p.active, n_polar=2, n_azi=8,
                                  bc="vacuum", scheme="scb", **mix,
                                  **cfg["sn"]).solve(**TOL)
        t_sn = time.perf_counter() - t
        t = time.perf_counter()
        hyb = HybridTriSNDiffusionSolver(p.grid, p.materials, p.material_map,
                                         sn_mask=mask, n_polar=2, n_azi=8,
                                         active=p.active, mask_bc=p.mask_bc,
                                         **mix, **cfg["hyb"])
        rh = hyb.solve(**TOL)
        t_h = time.perf_counter() - t
        n_hyb_sweeps = hyb.sn._sweep_count if hyb._has_drum else 0
        out[angle] = dict(kd=kd, sn=rs, kh=rh.k_eff, t_sn=t_sn, t_h=t_h,
                          h_sweeps=n_hyb_sweeps)
    return out


def worth(k_with, k_out):
    return (1.0 / k_out - 1.0 / k_with) * 1e5


print(f"HP-MR 2D (refine={refine}, SCB drums, polar volume-mixed), "
      f"inserted + withdrawn solves")
results = {}
for name, cfg in CONFIGS.items():
    results[name] = run_config(name, cfg)

ref = results["original"]
print(f"\n{'config':<12} {'solver':<9} {'k(ins)':>9} {'k(wdn)':>9} "
      f"{'worth':>7} {'outers':>7} {'sweeps':>7} {'time [s]':>9}")
for name, res in results.items():
    r0, r180 = res[0.0], res[180.0]
    wsn = worth(r0["sn"].k_eff, r180["sn"].k_eff)
    whb = worth(r0["kh"], r180["kh"])
    print(f"{name:<12} {'full S_N':<9} {r0['sn'].k_eff:>9.5f} "
          f"{r180['sn'].k_eff:>9.5f} {wsn:>7.0f} "
          f"{r0['sn'].outer_iterations + r180['sn'].outer_iterations:>7d} "
          f"{r0['sn'].n_sweeps + r180['sn'].n_sweeps:>7d} "
          f"{r0['t_sn'] + r180['t_sn']:>9.1f}")
    print(f"{'':<12} {'hybrid':<9} {r0['kh']:>9.5f} {r180['kh']:>9.5f} "
          f"{whb:>7.0f} {'':>7} "
          f"{r0['h_sweeps'] + r180['h_sweeps']:>7d} "
          f"{r0['t_h'] + r180['t_h']:>9.1f}")

kd0, kd180 = ref[0.0]["kd"], ref[180.0]["kd"]
print(f"\ndiffusion reference: k(ins)={kd0:.5f} k(wdn)={kd180:.5f} "
      f"worth={worth(kd0, kd180):.0f} pcm")
print("(placeholder cross sections: illustrative, not predictive.)")
