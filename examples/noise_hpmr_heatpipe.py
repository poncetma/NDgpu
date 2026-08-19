"""HP-MR heat-pipe flow-instability neutron noise on the body-fitted tri mesh.

Each fuel assembly of the heat-pipe microreactor is cooled by sodium heat pipes.
A *flow instability* in an assembly's heat pipes drives a temperature
oscillation localized to that assembly, which -- through moderator/Doppler
feedback -- perturbs the thermal absorption cross section by a small complex
phasor delta-Sigma_a2 = A exp(i theta) (amplitude and phase at the instability
frequency). A high-fidelity heat-pipe / thermal solver supplies that transfer
function; here we simply *impose* per-assembly phasors and let the tri-geometry
NoiseSolver propagate them to the flux noise everywhere in the core.

Two scenarios, each a physically distinct heat-pipe fault:

  PART A -- CORE-WIDE instability, coherent vs incoherent across assemblies.
    * coherent   : all assemblies oscillate in phase (theta = 0). The
                   perturbations add to a net reactivity oscillation -> a global,
                   fundamental-mode flux tilt; a core-average detector sees a
                   large signal.
    * incoherent : each assembly has an independent random phase. The net
                   reactivity nearly cancels, so the core-average signal is
                   suppressed by ~1/sqrt(N_assemblies). This ~sqrt(N) ratio is
                   the fingerprint that tells synchronized from independent
                   heat-pipe noise.

  PART B -- SINGLE assembly instability (one assembly's heat pipes go unstable).
    Shows the global-to-local transition: at low frequency the whole core
    responds in its fundamental mode (a global detector and a local one read
    alike); as frequency rises the response localizes around the faulty
    assembly (local detector >> global) -- the reactor spatially low-pass
    filters the noise.

    python examples/noise_hpmr_heatpipe.py [refine] [device]

Cross sections are the repo's placeholder HP-MR set, so absolute amplitudes are
illustrative, not predictive; the method and the qualitative signatures are the
point. The problem is linear, so every ratio below is independent of the
phasor amplitude.
"""

import sys

import numpy as np

from ndgpu import NoiseSolver, NoiseSource
from ndgpu.benchmarks.hpmr import (build_hpmr2d, hpmr_raster, PITCH, FUEL,
                                   HPMR_FUEL_SITES, HPMR_BE_SITES,
                                   HPMR_DRUM_SITES)
from ndgpu.hexraster import rasterize_hex_sites

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

THERMAL = 1                      # thermal group index (2-group model)
AMP = 1.0e-4                    # delta-Sigma_a2 phasor amplitude [1/cm] (sets scale only)
FREQS = (0.01, 0.1, 1.0, 10.0, 100.0)   # Hz (heat-pipe instabilities are low frequency)
N_REAL = 8                     # random-phase realizations for the incoherent statistics
SEED = 12345

# --- geometry: the core and a fuel-assembly id map aligned to its cells -------
angle = 0.0                      # drums withdrawn
p = build_hpmr2d(refine=refine, drum_angle_deg=angle)
mmap = np.asarray(p.material_map)
fuel = mmap == FUEL
act = np.asarray(p.active)

# Re-rasterize the SAME site set (identical frame) with each fuel assembly given
# a unique id 1..N and every other site a single "other" bucket, so asm_map is
# cell-aligned with the material map: asm_map == a+1 selects assembly a's cells.
N_ASM = len(HPMR_FUEL_SITES)
OTHER = N_ASM + 1
site_asm = {(0, 0): OTHER}
site_asm.update({s: i + 1 for i, s in enumerate(HPMR_FUEL_SITES)})
site_asm.update({s: OTHER for s in HPMR_BE_SITES})
site_asm.update({s: OTHER for s in HPMR_DRUM_SITES})
asm_map = rasterize_hex_sites(site_asm, PITCH, refine).material_map
assert asm_map.shape == mmap.shape, "assembly map must align with material map"
assert np.array_equal((asm_map >= 1) & (asm_map <= N_ASM), fuel), \
    "assembly cells must be exactly the fuel cells"
raster = hpmr_raster(refine, np.full(len(HPMR_DRUM_SITES), angle))  # cell geometry

REF_ASM = 1                      # reference assembly for the "local detector"
ref_cells = asm_map == REF_ASM

print(f"HP-MR 2D heat-pipe noise: refine={refine}, {p.grid.n_cells} cells, "
      f"{N_ASM} fuel assemblies, {int(fuel.sum())} fuel cells, device={device}")

# --- the noise solver: one static solve, then one complex solve per query -----
ns = NoiseSolver(p.grid, p.materials, p.material_map, kinetics=p.kinetics,
                 bc=p.bc, active=p.active, mask_bc=p.mask_bc,
                 mix_material=p.mix_material, mix_weight=p.mix_weight, device=device)
phi0 = np.asarray(ns.flux0[THERMAL])            # static thermal flux (real)
print(f"static core: k_eff = {ns.k_eff:.5f}\n")


def source(phase_per_asm, assemblies=None):
    """delta-Sigma_a2 field: amplitude AMP with each assembly's phasor.
    ``assemblies`` restricts the oscillation to a subset (default: all)."""
    field = np.zeros(p.grid.shape, dtype=complex)
    which = range(N_ASM) if assemblies is None else assemblies
    for a in which:
        field[asm_map == a + 1] = AMP * np.exp(1j * phase_per_asm[a])
    return NoiseSource(d_sigma_a=[np.zeros(p.grid.shape), field])


def metrics(res):
    """(global, local, shape-cos) from a NoiseResult, thermal group.
    global = |<dphi/phi>| over the whole fuel region (core-average / ex-core);
    local  = same over the reference assembly (in-core); cos = alignment of
    |dphi| with the static fundamental flux over the active core."""
    rel = np.asarray(res.relative()[THERMAL])
    glob = abs(complex(rel[fuel].mean()))
    local = abs(complex(rel[ref_cells].mean()))
    dmag = np.abs(np.asarray(res.d_flux_numpy()[THERMAL]))[act]
    cos = float(dmag @ phi0[act] / (np.linalg.norm(dmag) * np.linalg.norm(phi0[act])))
    return glob, local, cos


rng = np.random.default_rng(SEED)
incoh_phases = [rng.uniform(0.0, 2.0 * np.pi, N_ASM) for _ in range(N_REAL)]

# ============================ Part A: coherent vs incoherent =================
print("=" * 78)
print("PART A -- core-wide instability: coherent vs incoherent across assemblies")
print("=" * 78)
print("|global| = core-average thermal flux noise |<dphi/phi>|; cos = fundamental")
print(f"alignment. Incoherent = mean +/- std over {N_REAL} random-phase draws.\n")
hdr = (f"{'f [Hz]':>7} | {'|global|':>10} {'cos':>6} | "
       f"{'|global|':>10} {'+/-':>9} {'cos':>6} | {'ratio':>6}")
print(f"{'':>7} | {'COHERENT':^17} | {'INCOHERENT':^28} | {'C/I':>6}")
print(hdr); print("-" * len(hdr))
fig_maps = {}
for f in FREQS:
    w = 2.0 * np.pi * f
    gc, _, cc = metrics(res_c := ns.solve(source(np.zeros(N_ASM)), w, tol=1e-7))
    gi, ci, first = [], [], None
    for r, ph in enumerate(incoh_phases):
        g, _, c = metrics(res := ns.solve(source(ph), w, tol=1e-7))
        gi.append(g); ci.append(c)
        if r == 0:
            first = res
    gi, ci = np.array(gi), np.array(ci)
    print(f"{f:>7g} | {gc:>10.3e} {cc:>6.3f} | "
          f"{gi.mean():>10.3e} {gi.std():>9.2e} {ci.mean():>6.3f} | "
          f"{gc / gi.mean():>6.1f}")
    if f == 10.0:
        fig_maps["coherent (10 Hz)"] = res_c
        fig_maps["incoherent (10 Hz)"] = first
print(f"\n-> The coherent core-average signal runs ~{np.sqrt(N_ASM):.1f}x (=sqrt(N_asm)) "
      "above the")
print("   incoherent one: in-phase perturbations sum to a net reactivity swing")
print("   (fundamental mode, cos~1), random phases cancel in the average. A")
print("   core-average / ex-core detector therefore reads out how synchronized")
print("   the heat-pipe instabilities are. |global| also rolls off with")
print("   frequency: the reactor is a low-pass filter (zero-power transfer fn).")

# ============================ Part B: single-assembly ========================
print("\n" + "=" * 78)
print(f"PART B -- single assembly (#{REF_ASM}) unstable: global-to-local transition")
print("=" * 78)
print("|local| = flux noise averaged over the faulty assembly (in-core detector).\n")
hdr = f"{'f [Hz]':>7} | {'|global|':>10} {'|local|':>10} {'local/global':>13} {'cos':>7}"
print(hdr); print("-" * len(hdr))
for f in FREQS:
    g, l, c = metrics(res := ns.solve(source(np.zeros(N_ASM), assemblies=[REF_ASM - 1]),
                                      2.0 * np.pi * f, tol=1e-7))
    print(f"{f:>7g} | {g:>10.3e} {l:>10.3e} {l / g:>13.1f} {c:>7.3f}")
    if f == 100.0:
        fig_maps["single assembly (100 Hz)"] = res
print("\n-> With one faulty assembly the response starts global (low f: local/global")
print("   ~2, cos~1 -- the whole core tilts) and localizes as frequency rises")
print("   (local/global grows, cos falls): the flux noise concentrates around the")
print("   fault. A local in-core detector sees it strongly at all frequencies; a")
print("   core-average detector sees it wash out. This spatial low-pass, and the")
print("   coherence ratio of Part A, are the observables an external heat-pipe")
print("   transfer function feeds into once its delta-Sigma phasors are supplied.")

# --- figure: |dphi/phi| maps for the three scenarios --------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    nr, nc, _ = raster.material_map.shape
    verts, keep = [], []
    for a in range(nr):
        for b in range(nc):
            for t in range(2):
                if act[a, b, t]:
                    verts.append(raster.cell_vertices(a, b, t))
                    keep.append((a, b, t))
    keep_idx = tuple(np.array(keep).T)

    titles = list(fig_maps)
    fig, axes = plt.subplots(1, len(titles), figsize=(5.4 * len(titles), 5.2))
    for ax, key in zip(np.atleast_1d(axes), titles):
        vals = np.abs(np.asarray(fig_maps[key].relative()[THERMAL]))[keep_idx]
        pc = PolyCollection(verts, array=vals, cmap="inferno")
        pc.set_clim(0, np.percentile(vals, 99))
        ax.add_collection(pc)
        ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{key}\n|dphi/phi| (thermal)")
        fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("HP-MR heat-pipe flow-instability neutron noise")
    fig.tight_layout()
    out = "noise_hpmr_heatpipe.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved flux-noise map figure -> {out}")
except Exception as e:      # matplotlib missing / headless: the tables still stand
    print(f"\n(figure skipped: {e})")
