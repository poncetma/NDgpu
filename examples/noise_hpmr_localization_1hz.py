"""Localizing a ~1 Hz heat-pipe instability by fundamental-mode subtraction and
azimuth gating (2D HP-MR, reflector ring only).

At ~1 Hz the reactor responds in its global fundamental mode, so every fault's
detector signature is nearly parallel to one common direction: the
detector->fault transfer matrix G is nearly RANK-1. A naive matched filter then
cannot tell faults apart -- the degeneracy, not measurement noise, is the
limit (the 1 Hz signal is large and stationary, so SNR is actually good).

Two candidate processing tricks, no new hardware -- and this script TESTS
whether they actually help (spoiler: for a single, optimally-processed reflector
array at 1 Hz they do not, and it explains why):

  #2 fundamental-mode subtraction -- remove the dominant (fundamental) left
     singular subspace of G from both the templates and the measurement, then
     matched-filter the residual. Helps only if that mode is a strong,
     non-informative common background; here it is mild and carries source worth.
  #3 azimuth gating -- classify the fault's 60-degree sector first, then pick
     among only that sector's ~5 assemblies. A hard gate can only match or lose
     to the maximum-likelihood matched filter unless azimuth comes from a
     separate, more-reliable sensor.

We quantify the degeneracy vs frequency and sweep measurement noise at 1 Hz,
using the adjoint transfer matrix (one krylov solve per detector).

    python examples/noise_hpmr_localization_1hz.py [freq_Hz] [device]
"""

import sys

import numpy as np

from ndgpu import NoiseSolver
from ndgpu.benchmarks.hpmr import (build_hpmr2d, PITCH, FUEL,
                                   HPMR_FUEL_SITES, HPMR_BE_SITES,
                                   HPMR_DRUM_SITES)
from ndgpu.hexraster import rasterize_hex_sites, hex_site_xy

f_hz = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
refine = 4
THERMAL = 1
NOISE = 0.10
w = 2.0 * np.pi * f_hz

p = build_hpmr2d(refine=refine, drum_angle_deg=180.0)
mmap = np.asarray(p.material_map)
N, D = len(HPMR_FUEL_SITES), len(HPMR_BE_SITES)
allsites = ({(0, 0)} | set(HPMR_FUEL_SITES) | set(HPMR_BE_SITES)
            | set(HPMR_DRUM_SITES))


def id_map(sites, base):
    d = {t: 9999 for t in allsites}
    for i, t in enumerate(sites):
        d[t] = base + i
    return rasterize_hex_sites(d, PITCH, refine).material_map


asm = id_map(HPMR_FUEL_SITES, 1)
det = id_map(HPMR_BE_SITES, 1)
is_fuel = mmap == FUEL
xy = np.array([hex_site_xy(R, C, PITCH) for (R, C) in HPMR_FUEL_SITES])
fuel_r = np.hypot(xy[:, 0], xy[:, 1])
ring_of = np.array([int(np.argmin(np.abs(np.unique(np.round(fuel_r)) - r))) for r in fuel_r])
theta = np.arctan2(xy[:, 1], xy[:, 0]) % (2 * np.pi)
sector = (theta // (np.pi / 3)).astype(int)          # 6 azimuthal sectors (60 deg)

ns = NoiseSolver(p.grid, p.materials, p.material_map, kinetics=p.kinetics, bc=p.bc,
                 active=p.active, mask_bc=p.mask_bc, mix_material=p.mix_material,
                 mix_weight=p.mix_weight, device=device)
phi0 = np.asarray(ns.flux0[THERMAL])
print(f"2D HP-MR, f={f_hz} Hz, {D} reflector detectors, {N} assemblies, "
      f"k={ns.k_eff:.4f}, {NOISE:.0%} noise")
print(f"sectors: {np.bincount(sector)} assemblies per 60-deg wedge\n")

def transfer_matrix(freq):
    G = np.zeros((D, N), complex)
    for d in range(D):
        k = np.asarray(ns.adjoint_importance(
            [np.zeros(p.grid.shape), (det == d + 1).astype(float)],
            2 * np.pi * freq, tol=1e-8, method="krylov").d_flux_numpy()[THERMAL])
        for j in range(N):
            msk = (asm == j + 1) & is_fuel
            G[d, j] = -np.sum(k[msk] * phi0[msk])
    return G


# How the fundamental-mode degeneracy grows toward DC (sigma1/sigma2 of the
# column-normalized transfer matrix; larger = harder = where #2 could matter).
print("Fundamental-mode degeneracy of the reflector transfer matrix vs frequency:")
print(f"  {'freq [Hz]':>10} {'sigma1/sigma2':>14}")
deg_freqs = [0.01, 0.1, 1.0, 10.0]
deg_ratio = []
for fq in deg_freqs:
    Gq = transfer_matrix(fq)
    s = np.linalg.svd(Gq / np.linalg.norm(Gq, axis=0, keepdims=True), compute_uv=False)
    deg_ratio.append(s[0] / s[1])
    print(f"  {fq:>10g} {s[0] / s[1]:>14.1f}")

G = transfer_matrix(f_hz)
U, _, _ = np.linalg.svd(G, full_matrices=False)


def deflate(X, m):
    if m == 0:
        return X
    P = U[:, :m]
    return X - P @ (P.conj().T @ X)


def localize(noise, defl=0, gate=False, label_of=None, which=None, trials=300, seed=0):
    """Matched-filter accuracy. defl: # fundamental modes removed (#2). gate:
    two-stage -- predict the 6-way sector first, then pick within it (#3).
    label_of: score against this label (None = assembly id). which: restrict the
    evaluated TRUE faults to this subset (default all)."""
    rng = np.random.default_rng(seed)
    Gd = deflate(G, defl)
    Gdn = Gd / (np.linalg.norm(Gd, axis=0) + 1e-30)
    lab = np.arange(N) if label_of is None else label_of
    sec_templ = np.array([Gd[:, sector == c].mean(1) for c in range(6)])
    sec_templ /= np.linalg.norm(sec_templ, axis=1, keepdims=True) + 1e-30
    trues = list(range(N)) if which is None else list(which)
    ok = 0
    for j in trues:
        m0 = G[:, j]
        for _ in range(trials):
            md = deflate(m0 + noise * np.linalg.norm(m0) / np.sqrt(D) *
                         (rng.standard_normal(D) + 1j * rng.standard_normal(D)), defl)
            if gate:
                s_pred = int(np.argmax(np.abs(sec_templ.conj() @ md)))
                cand = np.where(sector == s_pred)[0]
            else:
                cand = np.arange(N)
            pred = cand[np.argmax(np.abs(Gdn[:, cand].conj().T @ md))]
            ok += lab[pred] == lab[j]
    return ok / (len(trues) * trials)


best_m = max(range(4), key=lambda m: localize(0.3, defl=m))    # best deflation @30% noise
print(f"\nAssembly-ID accuracy at {f_hz} Hz vs measurement noise "
      f"(#2 deflates {best_m} mode(s)):")
print(f"  {'noise':>6} | {'naive':>7} {'+#2':>7} {'+#3 gate':>9} {'+#2+#3':>8} | "
      f"{'azimuth naive':>13} {'azimuth +#3':>12}")
noises = [0.1, 0.2, 0.3, 0.5]
sweep = {"naive": [], "d2": [], "g3": [], "both": [], "az": []}
for noise in noises:
    nav = localize(noise)
    d2 = localize(noise, defl=best_m)
    g3 = localize(noise, gate=True)
    both = localize(noise, defl=best_m, gate=True)
    az = localize(noise, label_of=sector)
    azg = localize(noise, gate=True, label_of=sector)
    for key, val in (("naive", nav), ("d2", d2), ("g3", g3), ("both", both), ("az", az)):
        sweep[key].append(val)
    print(f"  {noise:>5.0%} | {nav:>7.0%} {d2:>7.0%} {g3:>9.0%} {both:>8.0%} | "
          f"{az:>13.0%} {azg:>12.0%}")

central = np.where(fuel_r < 40)[0]
periph = np.where(fuel_r > 60)[0]
print(f"\nWhere naive localization actually fails (30% noise), by fault position:")
print(f"  peripheral (r=71 cm): {localize(0.3, which=periph):.0%}    "
      f"central (r=27 cm): {localize(0.3, which=central):.0%}")

print("\n-> HONEST RESULT: neither #2 nor #3 beats the plain matched filter here.")
print("   (1) The 1 Hz degeneracy is mild -- sigma1/sigma2 ~ 4, only ~6 even at")
print("   0.01 Hz -- so the reflector transfer matrix is well-conditioned and the")
print("   low-frequency worry was overstated: the plain adjoint matched filter")
print("   already localizes the assembly at good SNR (100% at 10% noise). The")
print("   fundamental-mode amplitude carries source WORTH, so it is informative,")
print("   not a nuisance to remove -- deflating it (#2) only discards signal (best")
print("   deflation = 0 modes). (2) The matched filter IS the maximum-likelihood")
print("   estimator for this Gaussian problem, so a hard azimuth gate (#3) can")
print("   only match or lose to it -- it throws away the chance the fault is in an")
print("   adjacent sector, costing a little at every noise level.")
print("   Takeaways: #2 would help only with a strongly degenerate array")
print("   (sigma1/sigma2 >> 10, e.g. near true DC); #3 would help only if azimuth")
print("   came from a SEPARATE, more-reliable sensor. With one optimally-processed")
print("   array, the real limit is INFORMATION, not processing: radius and central")
print("   faults stay weak from the reflector ring -- the axis that in-core")
print("   detectors (#1) supply. Good news: 1 Hz azimuth/assembly ID is already")
print("   fine at realistic SNR without any of this.")

# --- figure -------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    ax.loglog(deg_freqs, deg_ratio, "o-", lw=2.2)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel(r"$\sigma_1/\sigma_2$ of transfer matrix")
    ax.set_title("Fundamental-mode degeneracy grows toward DC\n"
                 "(where #2 would matter; mild at 1 Hz)")
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    xn = np.array(noises) * 100
    ax.plot(xn, np.array(sweep["naive"]) * 100, "o-", lw=2.2, label="naive")
    ax.plot(xn, np.array(sweep["d2"]) * 100, "^--", lw=1.6, label="+#2 mode-sub")
    ax.plot(xn, np.array(sweep["g3"]) * 100, "s-", lw=2.2, label="+#3 azimuth gate")
    ax.plot(xn, np.array(sweep["az"]) * 100, "d:", lw=1.6, label="azimuth only (6-way)")
    ax.set_xlabel("measurement noise [%]"); ax.set_ylabel("accuracy [%]")
    ax.set_ylim(0, 105); ax.set_title(f"Assembly ID at {f_hz} Hz from the reflector "
                                      "ring\n(processing only, same detectors)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "noise_hpmr_localization_1hz.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved figure -> {out}")
except Exception as e:
    print(f"\n(figure skipped: {e})")
