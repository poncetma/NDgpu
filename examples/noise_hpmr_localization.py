"""Locating a failed heat pipe from reflector detectors -- the adjoint way.

A failed heat pipe perturbs the cross sections in one fuel assembly, radiating a
flux-noise field the whole core over. Detectors in the Be reflector each read a
complex phasor delta-phi(r_d, omega) (amplitude AND phase). Which assembly
failed? This is the neutron-noise *inverse* problem, and the tri NoiseSolver is
its forward model.

Efficient forward model by ADJOINT/importance. By reciprocity, a detector with
response psi_d reads  R = <psi_d, delta-phi> = <psi*_d, S_noise>, where the
adjoint noise flux psi*_d solves A^T psi*_d = psi_d -- ONE solve per detector.
psi*_d,thermal(r) * phi0,thermal(r) is then that detector's sensitivity to a
thermal cross-section fluctuation anywhere in the core, so the full
detector-vs-location transfer ("Green's") matrix G[d, j] comes from D detector
solves regardless of how many candidate locations j are scanned -- versus one
forward solve per location. Here D = 12 reflector detectors vs N = 30
assemblies (and the kernel localizes a fault placed *anywhere*, not just the 30
assembly centres).

Localization is then a complex matched filter: the measured detector pattern m
is assigned to the assembly whose signature G[:, j] best correlates with it,
    score(j) = |<G[:, j], m>| / (||G[:, j]|| ||m||)   (amplitude and phase).

    python examples/noise_hpmr_localization.py [freq_Hz] [device]

Placeholder cross sections: the method and the peripheral-vs-central contrast
are the point, not absolute amplitudes.
"""

import sys
import time

import numpy as np

from ndgpu import NoiseSolver, NoiseSource
from ndgpu.benchmarks.hpmr import (build_hpmr2d, hpmr_raster, PITCH, FUEL,
                                   HPMR_FUEL_SITES, HPMR_BE_SITES,
                                   HPMR_DRUM_SITES)
from ndgpu.hexraster import rasterize_hex_sites, hex_site_xy

f_hz = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
refine = 4
THERMAL = 1
DSIGMA = 1.0e-4                # unit fault amplitude (cancels in the matched filter)
w = 2.0 * np.pi * f_hz

# --- geometry: assembly ids (fuel) and detector ids (reflector), cell-aligned -
p = build_hpmr2d(refine=refine, drum_angle_deg=180.0)
N, D = len(HPMR_FUEL_SITES), len(HPMR_BE_SITES)
allsites = ({(0, 0)} | set(HPMR_FUEL_SITES) | set(HPMR_BE_SITES)
            | set(HPMR_DRUM_SITES))


def id_map(sites, base):
    d = {t: 9999 for t in allsites}                # identical rasterization frame
    for i, t in enumerate(sites):
        d[t] = base + i
    return rasterize_hex_sites(d, PITCH, refine).material_map


asm = id_map(HPMR_FUEL_SITES, 1)                    # asm == j+1 -> assembly j's cells
det = id_map(HPMR_BE_SITES, 1)                      # det == d+1 -> detector d's cells
raster = hpmr_raster(refine, np.zeros(D))           # cell geometry for plotting
fuel_r = np.array([np.hypot(*hex_site_xy(R, C, PITCH))
                   for (R, C) in HPMR_FUEL_SITES])

ns = NoiseSolver(p.grid, p.materials, p.material_map, kinetics=p.kinetics, bc=p.bc,
                 active=p.active, mask_bc=p.mask_bc, mix_material=p.mix_material,
                 mix_weight=p.mix_weight, device=device)
phi0 = np.asarray(ns.flux0[THERMAL])
print(f"HP-MR 2D localization: {N} assemblies, {D} reflector detectors, "
      f"f={f_hz} Hz, k={ns.k_eff:.4f}, device={device}\n")

# --- build the transfer matrix G[D, N] by D adjoint (importance) solves --------
t0 = time.perf_counter()
kernels = []                                        # psi*_d,thermal per detector
for d in range(D):
    response = [np.zeros(p.grid.shape), (det == d + 1).astype(float)]   # unit thermal response
    psi = ns.adjoint_importance(response, w, tol=1e-9)
    assert psi.converged
    kernels.append(np.asarray(psi.d_flux_numpy()[THERMAL]))
t_adj = time.perf_counter() - t0

# a fault at assembly j is S_j,thermal = -DSIGMA * phi0 on its cells; reciprocity
# gives G[d,j] = <psi*_d, S_j> = -DSIGMA * sum_{cells in j} psi*_d * phi0.
G = np.zeros((D, N), complex)
for j in range(N):
    m = asm == j + 1
    wpj = phi0[m]
    for d in range(D):
        G[d, j] = -DSIGMA * np.sum(kernels[d][m] * wpj)

# --- validate the adjoint transfer matrix against direct forward solves --------
t0 = time.perf_counter()
worst = 0.0
for j in (5, 17):                                   # one inner-ish, one outer assembly
    field = np.zeros(p.grid.shape, complex)
    field[asm == j + 1] = DSIGMA
    res = ns.solve(NoiseSource(d_sigma_a=[np.zeros(p.grid.shape), field]), w, tol=1e-9)
    dphi = np.asarray(res.d_flux_numpy()[THERMAL])
    g_fwd = np.array([np.sum(dphi[det == d + 1]) for d in range(D)])   # <psi_d, dphi>
    worst = max(worst, np.max(np.abs(g_fwd - G[:, j])) / np.max(np.abs(G[:, j])))
t_fwd2 = time.perf_counter() - t0
print(f"adjoint transfer matrix: {D} solves in {t_adj:.1f}s  (forward would be "
      f"{N} solves)")
print(f"reciprocity check vs forward: max rel.diff = {worst:.1e}  "
      f"(2 forward solves took {t_fwd2:.1f}s)\n")
assert worst < 1e-6, "adjoint transfer matrix must match forward to solver tol"


# --- matched-filter localization ----------------------------------------------
def localize(m):
    return np.abs(G.conj().T @ m) / (np.linalg.norm(G, axis=0) * np.linalg.norm(m))


rng = np.random.default_rng(0)


def demo_fault(j_true, noise_frac=0.2):
    m0 = G[:, j_true]
    m = m0 + noise_frac * np.linalg.norm(m0) / np.sqrt(D) * (
        rng.standard_normal(D) + 1j * rng.standard_normal(D))
    s = localize(m)
    order = np.argsort(s)[::-1]
    return s, order


inner = int(np.where(fuel_r < 40)[0][0])            # a central assembly
outer = int(np.where(fuel_r > 60)[0][0])            # a peripheral assembly
print(f"Matched-filter localization with 20% measurement noise "
      f"(true -> top-3 guesses):")
scores = {}
for label, jt in (("peripheral", outer), ("central", inner)):
    s, order = demo_fault(jt)
    scores[label] = (jt, s)
    top = ", ".join(f"asm{k}({s[k]:.3f})" for k in order[:3])
    hit = "OK" if order[0] == jt else "MISS"
    print(f"  {label:10s} fault asm{jt} (r={fuel_r[jt]:.0f}cm): {top}   [{hit}] "
          f"margin over runner-up = {s[jt] - s[order[1] if order[0]==jt else order[0]]:+.3f}")
print("\n-> Peripheral faults give a sharp, well-separated peak; central faults")
print("   correlate strongly with their whole (same-radius) ring, so the margin")
print("   collapses -- the reflector ring resolves azimuth, not radius.")

# --- figure: an adjoint sensitivity kernel + two localization score maps -------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    act = np.asarray(p.active)
    nr, nc, _ = raster.material_map.shape
    verts, keep = [], []
    for a in range(nr):
        for b in range(nc):
            for t in range(2):
                if act[a, b, t]:
                    verts.append(raster.cell_vertices(a, b, t))
                    keep.append((a, b, t))
    keep_idx = tuple(np.array(keep).T)
    det_xy = [np.mean([hex_site_xy(*HPMR_BE_SITES[d], PITCH)], axis=0)[0]
              for d in range(D)]

    def paint(ax, cellvals, title, cmap, detector=None):
        pc = PolyCollection(verts, array=cellvals[keep_idx], cmap=cmap)
        pc.set_clim(0, np.percentile(cellvals[keep_idx], 99.5))
        ax.add_collection(pc)
        if detector is not None:
            x, y = hex_site_xy(*HPMR_BE_SITES[detector], PITCH)
            ax.plot(x, y, "c*", ms=18, mec="k")
        ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)

    def score_cells(s):
        out = np.zeros(p.grid.shape)
        for j in range(N):
            out[asm == j + 1] = s[j]
        return out

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    d0 = int(np.argmax([np.max(np.abs(k)) for k in kernels]))   # a representative detector
    paint(axes[0], np.abs(kernels[d0]) * phi0,
          f"adjoint sensitivity of one detector\n|psi*.phi0| (star = detector)",
          "viridis", detector=d0)
    for ax, label in zip(axes[1:], ("peripheral", "central")):
        jt, s = scores[label]
        cells = score_cells(s)
        pc = PolyCollection(verts, array=cells[keep_idx], cmap="inferno")
        pc.set_clim(np.percentile(s, 5), 1.0)
        ax.add_collection(pc)
        cx, cy = hex_site_xy(*HPMR_FUEL_SITES[jt], PITCH)
        ax.plot(cx, cy, "co", ms=13, mfc="none", mew=2.5)
        ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"localization score, {label} fault\n(circle = true; f={f_hz} Hz)")
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("HP-MR failed-heat-pipe localization from reflector detectors "
                 "(adjoint transfer matrix)")
    fig.tight_layout()
    out = "noise_hpmr_localization.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved figure -> {out}")
except Exception as e:
    print(f"\n(figure skipped: {e})")
