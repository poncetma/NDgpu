"""Radial localization of a failed heat pipe from detectors ABOVE the core, and
how few of them you need (3D HP-MR).

A peripheral reflector ring resolves a fault's azimuth but barely its radius:
its importance (adjoint sensitivity) grows toward the core edge, so a ring
reading is dominated by peripheral faults and is nearly blind to how deep an
inner fault sits. Detectors in the TOP axial reflector fix that. The right
radial sensor is an azimuthally-symmetric ANNULUS at radius r in the top
reflector: its importance peaks in the core column beneath it (radius ~r) at
every azimuth, so it reports "is the fault at my radius?" independent of angle.

Two questions, both answered by cheap adjoint/importance solves (one per
detector; a fault's reading is then the reciprocity inner product <psi*, S>, so
one solve gives the detector's response to a fault in every assembly):

  1. mechanism -- each top annulus peaks at its own radius; the ring peaks at
     the edge.
  2. minimum count -- how many top annuli are needed to classify a fault's
     radial ring (here 4 rings, at r ~ 27/46/54/71 cm) reliably under noise, at
     ANY azimuth.

    python examples/noise_hpmr_localization_3d.py [freq_Hz] [device]
"""

import sys
import time
from itertools import combinations

import numpy as np

from ndgpu import NoiseSolver
from ndgpu.benchmarks.hpmr import (build_hpmr3d, PITCH, FUEL, BE_REFLECTOR,
                                   _FUEL_SITES, _BE_SITES, _DRUM_SITES)
from ndgpu.hexraster import rasterize_hex_sites, hex_site_xy

f_hz = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
refine, nz = 3, 10
THERMAL = 1
w = 2.0 * np.pi * f_hz

p = build_hpmr3d(refine=refine, nz=nz, absorber="polar", drum_angle_deg=180.0)
mmap = np.asarray(p.material_map)
nzt = mmap.shape[3]
TOPLAYER = nzt - 1
allsites = {(0, 0)} | set(_FUEL_SITES) | set(_BE_SITES) | set(_DRUM_SITES)


def id_map3d(sites, base):
    d = {t: 9999 for t in allsites}
    for i, t in enumerate(sites):
        d[t] = base + i
    return np.repeat(rasterize_hex_sites(d, PITCH, refine).material_map[..., None],
                     nzt, axis=3)


asm = id_map3d(_FUEL_SITES, 1)
N = len(_FUEL_SITES)
fuel_xy = np.array([hex_site_xy(R, C, PITCH) for (R, C) in _FUEL_SITES])
fuel_r = np.hypot(fuel_xy[:, 0], fuel_xy[:, 1])
radii = np.array(sorted(set(np.round(fuel_r, 0))))   # 4 radial rings
ring_of = np.array([int(np.argmin(np.abs(radii - r))) for r in fuel_r])  # bin per assembly
is_fuel = mmap == FUEL
at_layer_top = (np.arange(nzt) == TOPLAYER)[None, None, None, :]

ns = NoiseSolver(p.grid, p.materials, p.material_map, kinetics=p.kinetics, bc=p.bc,
                 active=p.active, mask_bc=p.mask_bc, mix_material=p.mix_material,
                 mix_weight=p.mix_weight, device=device)
phi0 = np.asarray(ns.flux0[THERMAL])
print(f"3D HP-MR: grid {p.grid.shape}, {int(np.asarray(p.active).sum())} active "
      f"cells, k={ns.k_eff:.4f}, f={f_hz} Hz")
print(f"{len(radii)} radial rings at r = {radii} cm; "
      f"{[int((ring_of==b).sum()) for b in range(len(radii))]} assemblies each\n")


def kernel(mask):
    r = ns.adjoint_importance([np.zeros(p.grid.shape), mask.astype(float)], w, tol=1e-7)
    assert r.converged, r
    return np.asarray(r.d_flux_numpy()[THERMAL])


def reading_per_assembly(k):                          # complex <psi*, S_j> for all j
    return np.array([-np.sum(k[(asm == j + 1) & is_fuel] * phi0[(asm == j + 1) & is_fuel])
                     for j in range(N)])


t0 = time.perf_counter()
# peripheral reference: full-height Be radial-reflector annulus (one solve)
ring_g = np.abs(reading_per_assembly(kernel((mmap == BE_REFLECTOR))))
# top annuli: top-reflector cells above ALL assemblies at each radius (one each)
top_G = []
for b, r in enumerate(radii):
    mask = np.isin(asm, np.where(ring_of == b)[0] + 1) & at_layer_top
    top_G.append(reading_per_assembly(kernel(mask)))
top_G = np.array(top_G)                               # (n_rings, N) complex
print(f"{1 + len(radii)} adjoint solves in {time.perf_counter()-t0:.0f}s "
      f"(1 peripheral ref + {len(radii)} top annuli)\n")


def radial_profile(g):
    prof = np.array([np.abs(g)[ring_of == b].mean() for b in range(len(radii))])
    return prof / prof.max()


print("RAW radial sensitivity (normalized), by fault ring -- note every detector")
print("peaks at the CENTRE: a fault's noise source is dSigma*phi0 and phi0 peaks")
print("at the core centre, so central faults radiate most regardless of detector.")
print("  radius[cm]:  " + "  ".join(f"{r:5.0f}" for r in radii))
print("  periph.ring: " + "  ".join(f"{v:5.2f}" for v in radial_profile(ring_g)))
for b, r in enumerate(radii):
    print(f"  top@{r:3.0f}cm : " + "  ".join(f"{v:5.2f}" for v in radial_profile(top_G[b])))

# Radius lives in the *ratio* between an inner and an outer top annulus, which
# divides out the common phi0 magnitude and is monotone in radius:
inner_b, outer_b = 0, len(radii) - 1
gauge = np.array([(np.abs(top_G[inner_b]) / np.abs(top_G[outer_b]))[ring_of == b].mean()
                  for b in range(len(radii))])
print(f"\nRADIUS GAUGE = |top@{radii[inner_b]:.0f}| / |top@{radii[outer_b]:.0f}| "
      "(magnitude divides out; monotone => radius-sensitive):")
print("  radius[cm]:  " + "  ".join(f"{r:5.0f}" for r in radii))
print("  gauge     :  " + "  ".join(f"{v:5.2f}" for v in gauge / gauge[0]))

# --- minimum number of top annuli for reliable radial classification ----------
# Radius is carried by the *pattern* of annulus amplitudes, not their scale, so
# the feature is the magnitude vector normalized to unit sum (phi0-independent).
# Template per ring = mean noiseless feature; classify a noisy fault to nearest.
rng = np.random.default_rng(0)


def accuracy(sel, bins, noise=0.3, trials=200):
    F = np.abs(top_G[list(sel)])                      # (T, N) amplitudes
    feat = F / (F.sum(axis=0, keepdims=True) + 1e-30)
    lab = np.array([bins[b] for b in ring_of])        # coarse/fine label per assembly
    templ = np.array([feat[:, lab == c].mean(axis=1) for c in range(max(bins) + 1)])
    ok = 0
    for j in range(N):
        f0 = F[:, j]
        for _ in range(trials):
            m = np.abs(f0 + noise * np.linalg.norm(f0) / np.sqrt(len(sel)) * (
                rng.standard_normal(len(sel)) + 1j * rng.standard_normal(len(sel))))
            m = m / (m.sum() + 1e-30)
            ok += np.argmin(np.linalg.norm(templ - m, axis=1)) == lab[j]
    return ok / (N * trials)


n_rings = len(radii)
FINE = list(range(n_rings))                           # 4 rings, each its own class
COARSE = [0, 0, 1, 1]                                 # inner half (27/46) vs outer (54/71)


def best_accuracy(T, bins, noise):
    return max(accuracy(s, bins, noise=noise)
               for s in combinations(range(n_rings), T))


for noise in (0.1, 0.3):
    print(f"\nRadial classification accuracy vs # top annuli "
          f"({noise:.0%} measurement noise, any azimuth):")
    print(f"  {'#annuli':>8} | {'4 rings (chance 25%)':>20} | "
          f"{'inner/outer (chance 50%)':>24}")
    for T in range(1, n_rings + 1):
        print(f"  {T:>8} | {best_accuracy(T, FINE, noise):>19.0%} | "
              f"{best_accuracy(T, COARSE, noise):>23.0%}")

# minimum annuli to reach a "reliable" target, per task and SNR
TARGET = 0.90
print(f"\nMinimum # top annuli to reach {TARGET:.0%} accuracy:")
for name, bins in (("inner/outer (coarse)", COARSE), ("4 rings (fine)", FINE)):
    for noise in (0.1, 0.3):
        hit = next((T for T in range(1, n_rings + 1)
                    if best_accuracy(T, bins, noise) >= TARGET), None)
        msg = f"{hit} annuli" if hit else f"not reached (max {best_accuracy(n_rings, bins, noise):.0%})"
        print(f"  {name:22s} @ {noise:.0%} noise: {msg}")

print("\n-> Radius is NOT in any single detector's magnitude (phi0-dominated); it")
print("   lives in the inner/outer annulus RATIO, and TWO top annuli (innermost +")
print("   outermost) extract essentially all of it -- a 3rd/4th barely move the")
print("   accuracy, so 2 is the count that matters. Whether 2 is 'reliable' is")
print("   set by SNR and how fine you slice: with clean measurements 2 annuli")
print("   separate inner vs outer core well, but the adjacent 46/54 cm rings")
print("   (8 cm apart, and phi0/axial-smeared) need much better SNR. The binding")
print("   limit past 2 detectors is radial information, not detector count.")

# --- figure -------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    ax.plot(radii, gauge / gauge[0], "o-", lw=2.5, color="C3")
    ax.set_xlabel("fault radius [cm]")
    ax.set_ylabel(f"inner/outer annulus ratio\n|top@{radii[inner_b]:.0f}| / |top@{radii[outer_b]:.0f}|")
    ax.set_title("Radius gauge from two top annuli\n(monotone; but 46 & 54 cm nearly tie)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    fine = [max(accuracy(s, FINE) for s in combinations(range(n_rings), T))
            for T in range(1, n_rings + 1)]
    coarse = [max(accuracy(s, COARSE) for s in combinations(range(n_rings), T))
              for T in range(1, n_rings + 1)]
    xs = range(1, n_rings + 1)
    ax.plot(xs, np.array(fine) * 100, "o-", lw=2.5, label="4 rings (fine)")
    ax.plot(xs, np.array(coarse) * 100, "s-", lw=2.5, label="inner/outer (coarse)")
    ax.axhline(25, color="grey", ls=":"); ax.axhline(50, color="grey", ls="--")
    ax.set_xlabel("number of top annuli"); ax.set_ylabel("radial accuracy [%]")
    ax.set_xticks(list(xs)); ax.set_ylim(0, 105)
    ax.set_title(f"How many above-core detectors?\n(30% noise, f={f_hz} Hz)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "noise_hpmr_localization_3d.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved figure -> {out}")
except Exception as e:
    print(f"\n(figure skipped: {e})")
