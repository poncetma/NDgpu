"""Hybrid SP3/diffusion HP-MR: transport in the control drums, diffusion in the
bulk.

The near-black B4C drum arcs drive steep, near-discontinuous angular flux that
diffusion cannot resolve -- it over-predicts the drum worth. Full SP3 fixes this
but pays ~2x everywhere. The hybrid solver runs the SP3 second moment *only*
where a mask marks it (here the rotating drum bodies, ~1/5 of the core) and
plain diffusion elsewhere: the transport correction is generated at the drums
and decays into the surrounding graphite. This recovers a large part of the SP3
drum-worth self-shielding at close to diffusion's simplicity.

The control-drum WORTH is a reactivity difference, so the angle-independent part
of the transport correction (outer-boundary and reflector leakage) cancels and
the drum-local self-shielding -- exactly what the hybrid captures -- dominates.

    python examples/hpmr_hybrid.py [refine] [device]

Cross sections are placeholders; worths are illustrative, not predictive.
"""

import sys
import time

from ndgpu.benchmarks import build_hpmr2d, hpmr_transport_mask
from ndgpu.tri import TriDiffusionEigenSolver, TriSP3EigenSolver

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 6
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
angles = (0, 30, 60, 90, 120, 150, 180)

TOL = dict(tol_k=1e-8, tol_source=1e-7)


def run(cls, angle, **extra):
    p = build_hpmr2d(refine=refine, drum_angle_deg=float(angle), absorber="polar")
    common = dict(active=p.active, mask_bc=p.mask_bc, mix_material=p.mix_material,
                  mix_weight=p.mix_weight, device=device)
    if extra.get("hybrid_mask") == "drum":
        extra = dict(extra, hybrid_mask=hpmr_transport_mask(p, "drum"))
    t = time.perf_counter()
    res = cls(p.grid, p.materials, p.material_map, **common, **extra).solve(**TOL)
    if not res.converged:
        raise SystemExit(f"not converged at {angle} deg: {res}")
    return res.k_eff, time.perf_counter() - t, p


# Report the transport fraction once (geometry is angle-independent).
_, _, p0 = run(TriDiffusionEigenSolver, 0)
mask = hpmr_transport_mask(p0, "drum")
n_tr = int(mask.sum())
n_act = int((p0.material_map > 0).sum())
print(f"HP-MR 2D hybrid SP3/diffusion, refine={refine} "
      f"({6 * refine**2} tri/assembly), device={device}")
print(f"transport (SP3) region = drum bodies: {n_tr}/{n_act} active cells "
      f"({100 * n_tr / n_act:.0f}%); diffusion elsewhere\n")

kd, ks, kh = {}, {}, {}
td = ts = th = 0.0
for a in angles:
    kd[a], dt, _ = run(TriDiffusionEigenSolver, a); td += dt
    ks[a], dt, _ = run(TriSP3EigenSolver, a); ts += dt
    kh[a], dt, _ = run(TriSP3EigenSolver, a, hybrid_mask="drum"); th += dt


def rho(k, a):
    return (1.0 / k[0] - 1.0 / k[a]) * 1e5           # pcm worth vs withdrawn


print(f"{'drum':>5} {'--- k_eff ---':^26} {'--- worth vs 0 (pcm) ---':^28}")
print(f"{'angle':>5} {'diffusion':>8} {'hybrid':>8} {'SP3':>8}   "
      f"{'diff':>7} {'hybrid':>7} {'SP3':>7}  {'hyb err':>8} {'diff err':>8}")
for a in angles:
    rd, rh, rs = rho(kd, a), rho(kh, a), rho(ks, a)
    print(f"{a:>4}° {kd[a]:>8.5f} {kh[a]:>8.5f} {ks[a]:>8.5f}   "
          f"{rd:>7.0f} {rh:>7.0f} {rs:>7.0f}  {rh - rs:>+8.0f} {rd - rs:>+8.0f}")

ws, wd, wh = rho(ks, 180), rho(kd, 180), rho(kh, 180)
print(f"\nfull-insertion drum worth (0 -> 180 deg), SP3 = reference:")
print(f"  full SP3   {ws:>7.0f} pcm   ({ts:.1f} s total)")
print(f"  hybrid     {wh:>7.0f} pcm   ({(wh - ws) / ws * 100:+.1f}% vs SP3)   "
      f"({th:.1f} s total)")
print(f"  diffusion  {wd:>7.0f} pcm   ({(wd - ws) / ws * 100:+.1f}% vs SP3)   "
      f"({td:.1f} s total)")
err_closed = (1 - abs(wh - ws) / abs(wd - ws)) * 100
print(f"\n  hybrid closes {err_closed:.0f}% of the diffusion->SP3 worth error "
      f"with transport in {100 * n_tr / n_act:.0f}% of cells.")
print("\n(placeholder cross sections: illustrative, not predictive.)")
