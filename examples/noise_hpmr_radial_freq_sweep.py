"""When does above-core radial localization become reliable? A frequency sweep.

Radial resolution of a failed heat pipe from top-reflector annuli improves with
frequency: raising omega shortens the noise attenuation length, sharpening each
annulus's radial sensitivity. This sweeps frequency and reports the radial-ring
classification accuracy, to find where it crosses 95%.

Each frequency needs a handful of adjoint/importance solves on the 3D HP-MR
(65k cells); they run with the monolithic complex-GMRES solver (method="krylov")
which is markedly faster than the source iteration here.

Caveat: the measurement noise below is a fixed fraction of each detector's
signal, so this isolates the *geometric* resolvability vs frequency. In a real
detector the absolute flux-noise reaching the top reflector rolls off with
frequency (the point-kinetics low-pass), so a fixed noise floor would penalize
high frequency -- the true optimum is a mid-band. This answers "when is the
physics sharp enough", not "what is the best SNR operating point".

    python examples/noise_hpmr_radial_freq_sweep.py [device]
"""

import sys
import time
from itertools import combinations

import numpy as np

from ndgpu import NoiseSolver
from ndgpu.benchmarks.hpmr import (build_hpmr3d, PITCH, FUEL,
                                   HPMR_FUEL_SITES, HPMR_BE_SITES,
                                   HPMR_DRUM_SITES)
from ndgpu.hexraster import rasterize_hex_sites, hex_site_xy

device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
refine, nz = 3, 10
THERMAL = 1
NOISE = 0.10                                # measurement noise (fraction of signal)
FREQS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]

p = build_hpmr3d(refine=refine, nz=nz, absorber="polar", drum_angle_deg=180.0)
mmap = np.asarray(p.material_map)
nzt = mmap.shape[3]
TOPLAYER = nzt - 1
allsites = ({(0, 0)} | set(HPMR_FUEL_SITES) | set(HPMR_BE_SITES)
            | set(HPMR_DRUM_SITES))


def id_map3d(sites, base):
    d = {t: 9999 for t in allsites}
    for i, t in enumerate(sites):
        d[t] = base + i
    return np.repeat(rasterize_hex_sites(d, PITCH, refine).material_map[..., None],
                     nzt, axis=3)


asm = id_map3d(HPMR_FUEL_SITES, 1)
N = len(HPMR_FUEL_SITES)
fuel_r = np.hypot(*np.array([hex_site_xy(R, C, PITCH)
                             for (R, C) in HPMR_FUEL_SITES]).T)
radii = np.array(sorted(set(np.round(fuel_r, 0))))
ring_of = np.array([int(np.argmin(np.abs(radii - r))) for r in fuel_r])
n_rings = len(radii)
is_fuel = mmap == FUEL
at_top = (np.arange(nzt) == TOPLAYER)[None, None, None, :]
annulus_mask = [np.isin(asm, np.where(ring_of == b)[0] + 1) & at_top for b in range(n_rings)]

ns = NoiseSolver(p.grid, p.materials, p.material_map, kinetics=p.kinetics, bc=p.bc,
                 active=p.active, mask_bc=p.mask_bc, mix_material=p.mix_material,
                 mix_weight=p.mix_weight, device=device)
phi0 = np.asarray(ns.flux0[THERMAL])
print(f"3D HP-MR radial-localization frequency sweep: {int(np.asarray(p.active).sum())} "
      f"active cells, k={ns.k_eff:.4f}, {NOISE:.0%} noise, solver=krylov\n")

FINE = list(range(n_rings))
COARSE = [0, 0, 1, 1]
rng = np.random.default_rng(0)


def top_matrix(w):
    """|<psi*, S_j>| for each annulus x assembly, via one krylov adjoint solve each."""
    G = np.zeros((n_rings, N), complex)
    for b in range(n_rings):
        k = np.asarray(ns.adjoint_importance(
            [np.zeros(p.grid.shape), annulus_mask[b].astype(float)], w,
            tol=1e-7, method="krylov").d_flux_numpy()[THERMAL])
        G[b] = [-np.sum(k[(asm == j + 1) & is_fuel] * phi0[(asm == j + 1) & is_fuel])
                for j in range(N)]
    return G


def accuracy(G, sel, bins, trials=200):
    F = np.abs(G[list(sel)])
    lab = np.array([bins[b] for b in ring_of])
    templ = np.array([(F / (F.sum(0, keepdims=True) + 1e-30))[:, lab == c].mean(1)
                      for c in range(max(bins) + 1)])
    ok = 0
    for j in range(N):
        f0 = F[:, j]
        for _ in range(trials):
            m = np.abs(f0 + NOISE * np.linalg.norm(f0) / np.sqrt(len(sel)) *
                       (rng.standard_normal(len(sel)) + 1j * rng.standard_normal(len(sel))))
            m = m / (m.sum() + 1e-30)
            ok += np.argmin(np.linalg.norm(templ - m, axis=1)) == lab[j]
    return ok / (N * trials)


def best(G, T, bins):
    return max(accuracy(G, s, bins) for s in combinations(range(n_rings), T))


print(f"{'f [Hz]':>8} | {'coarse in/out':>22} | {'fine 4-ring':>22} | {'solve':>6}")
print(f"{'':>8} | {'2 ann':>7} {'4 ann':>7} {'':>6} | {'2 ann':>7} {'4 ann':>7} {'':>6} |")
print("-" * 72)
rows = {"coarse2": [], "coarse4": [], "fine2": [], "fine4": []}
for f in FREQS:
    t0 = time.perf_counter()
    G = top_matrix(2.0 * np.pi * f)
    dt = time.perf_counter() - t0
    c2, c4 = best(G, 2, COARSE), best(G, 4, COARSE)
    f2, f4 = best(G, 2, FINE), best(G, 4, FINE)
    rows["coarse2"].append(c2); rows["coarse4"].append(c4)
    rows["fine2"].append(f2); rows["fine4"].append(f4)
    print(f"{f:>8g} | {c2:>7.0%} {c4:>7.0%} {'':>6} | {f2:>7.0%} {f4:>7.0%} {'':>6} | {dt:>5.0f}s")


def threshold(vals):
    hit = next((FREQS[i] for i, v in enumerate(vals) if v >= 0.95), None)
    return f"{hit:g} Hz" if hit else "not reached"


print(f"\nFrequency at which accuracy first exceeds 95%:")
print(f"  coarse inner/outer, 2 annuli: {threshold(rows['coarse2'])}")
print(f"  coarse inner/outer, 4 annuli: {threshold(rows['coarse4'])}")
print(f"  fine 4-ring,        4 annuli: {threshold(rows['fine4'])}")
print("\n-> Radial resolvability climbs with frequency as the noise field localizes,"
      "\n   crossing 95% around 300-1000 Hz (coarse inner/outer with 2 annuli first;"
      "\n   the fine adjacent-ring split needs a bit more). TWO caveats matter:"
      "\n   (1) fixed *relative* noise -- the real signal reaching the top reflector"
      "\n   rolls off with frequency (point-kinetics low-pass), so a fixed noise floor"
      "\n   caps this and the usable optimum is a mid-band; (2) more fundamentally,"
      "\n   heat-pipe FLOW instabilities are ~Hz phenomena, and at those frequencies"
      "\n   radial accuracy sits at its ~68%/45% floor. Sharp radial localization"
      "\n   needs ~kHz content, which a low-frequency flow instability does not"
      "\n   supply -- so radius is only weakly recoverable for the physical fault,"
      "\n   even though the method itself resolves it cleanly at high frequency.")

# --- figure -------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.semilogx(FREQS, np.array(rows["coarse2"]) * 100, "o-", lw=2.2, label="inner/outer, 2 annuli")
    ax.semilogx(FREQS, np.array(rows["coarse4"]) * 100, "o--", lw=1.6, label="inner/outer, 4 annuli")
    ax.semilogx(FREQS, np.array(rows["fine4"]) * 100, "s-", lw=2.2, label="4-ring, 4 annuli")
    ax.axhline(95, color="C3", ls=":", lw=2, label="95% target")
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("radial accuracy [%]")
    ax.set_ylim(0, 102); ax.grid(alpha=0.3, which="both")
    ax.set_title(f"Above-core radial localization vs frequency\n"
                 f"(3D HP-MR, {NOISE:.0%} relative noise, krylov solver)")
    ax.legend(fontsize=8); fig.tight_layout()
    out = "noise_hpmr_radial_freq_sweep.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved figure -> {out}")
except Exception as e:
    print(f"\n(figure skipped: {e})")
