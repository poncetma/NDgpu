"""HP-MR SPH factors from three transport references: SP3 vs SDP1 vs SDP2.

The SPH pipeline (:mod:`ndgpu.sph`) folds a transport reference's angular
treatment of the near-black B4C control-drum arc into the coarse diffusion
constants, so plain TriDiffusion reproduces the transport eigenvalue -- and
hence the drum worth -- that it otherwise misses by ~100 pcm at insertion. The
reference is any ndgpu eigensolver's *physical* scalar flux phi0; the three
matched-DoF simplified-transport families extract that flux differently:

    SP3  (Brantley-Larsen)   phi0 = Phi1 - 2 phi2      2-moment block
    SDP1 (double-P1)         same block, half-range 2nd-moment coefficient
    SDP2 (double-P2)         3-moment block (the DoF of SP5)

so swapping the reference feeds SPH three genuinely different angular closures,
not a re-extraction of one field. This example runs the whole pipeline from each
family at two drum states and compares:

  1. eigenvalue closure at insertion (arc toward the core, maximal self-
     shielding): reference k, uncorrected diffusion k, SPH-corrected k, and the
     drum-absorber SPH factor that carries the correction;
  2. control-drum worth (withdrawn -> inserted) computed three ways -- plain
     diffusion, each transport reference, and each family's SPH-corrected
     diffusion -- showing SPH recovers each family's transport worth and that
     the three families agree far more tightly than any agrees with diffusion.

    python examples/hpmr_sph_reference_families.py [refine] [device]

refine (default 4; the raster absorber's floor) sets triangles per hex. Cross
sections are placeholders, so the worths are illustrative, not predictive.
"""

import sys

import numpy as np

from ndgpu import (TriDiffusionEigenSolver, TriSDP1EigenSolver,
                   TriSDP2EigenSolver, TriSP3EigenSolver,
                   flux_weighted_homogenize, region_average, sph_correct)
from ndgpu.benchmarks.hpmr import build_hpmr2d, DRUM_ABSORBER

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)
FAMILIES = {"SP3": TriSP3EigenSolver, "SDP1": TriSDP1EigenSolver,
            "SDP2": TriSDP2EigenSolver}
# Drum-arc rotation. These were swapped until verified against the eigenvalue:
# 0 deg gives the LOWER k (more absorption), i.e. the arc faces the core and the
# drum is INSERTED; 180 deg turns it outward (withdrawn). Confirmed for both the
# raster and polar absorbers.
INSERTED, WITHDRAWN = 0.0, 180.0


def solve_state(angle):
    """Run diffusion + every SPH family on the HP-MR at one drum angle.

    Returns (diffusion_k, {family: (reference_k, sph_k, factors)}).
    """
    p = build_hpmr2d(refine=refine, drum_angle_deg=angle, absorber="raster")
    dV = p.grid.cell_volume
    common = dict(active=p.active, mask_bc=p.mask_bc, device=device)
    region = p.material_map                 # one homogenization region per material

    dif = TriDiffusionEigenSolver(p.grid, p.materials, region, **common).solve(**TIGHT)
    if not dif.converged:
        raise SystemExit(f"diffusion not converged at {angle} deg")

    def coarse_solve(materials):
        r = TriDiffusionEigenSolver(p.grid, materials, region, **common).solve(**TIGHT)
        return region_average(r.flux_numpy, region), r.k_eff

    out = {}
    for name, cls in FAMILIES.items():
        ref = cls(p.grid, p.materials, region, **common).solve(**TIGHT)
        if not ref.converged:
            raise SystemExit(f"{name} reference not converged at {angle} deg")
        hmats, rflux, _ = flux_weighted_homogenize(
            ref.flux_numpy, p.materials, region, region, cell_volume=dV)
        sph = sph_correct(hmats, region, rflux, coarse_solve, tol=1e-7, depth=6)
        if not sph.converged:
            raise SystemExit(f"{name} SPH did not converge at {angle} deg")
        out[name] = (ref.k_eff, sph.k_eff, sph.factors)
    return dif.k_eff, out


print(f"HP-MR 2D SPH reference-family comparison, refine={refine} "
      f"({6 * refine**2} tri/assembly), device={device}\n")

states = {}
for angle in (WITHDRAWN, INSERTED):
    states[angle] = solve_state(angle)

# --- 1. eigenvalue closure at insertion (largest transport-vs-diffusion gap) ---
dif_k, fam = states[INSERTED]
print(f"Drums inserted ({INSERTED:.0f} deg, arc toward core -- maximal "
      f"self-shielding):")
print(f"  uncorrected diffusion  k = {dif_k:.6f}\n")
print(f"  {'family':>6} {'reference k':>12} {'SPH k':>10} {'ref-dif':>9} "
      f"{'SPH-dif':>9} {'|SPH-ref|':>10} {'drum mu (g1,g2)':>18}")
for name, (ref_k, sph_k, mu) in fam.items():
    d_ref = (ref_k - dif_k) * 1e5           # transport correction diffusion misses
    d_sph = (sph_k - dif_k) * 1e5           # correction SPH restores
    gap = abs(sph_k - ref_k) * 1e5          # residual SPH-vs-reference error
    mud = np.atleast_1d(mu[DRUM_ABSORBER])
    mus = ", ".join(f"{v:.3f}" for v in mud)
    print(f"  {name:>6} {ref_k:>12.6f} {sph_k:>10.6f} {d_ref:>+8.0f}p "
          f"{d_sph:>+8.0f}p {gap:>9.1f}p   {mus:>16}")
print("  (ref-dif = transport worth of the angular self-shielding; SPH-dif = "
      "how much\n   of it the corrected diffusion recovers; |SPH-ref| should be "
      "a few pcm)")

# --- 2. control-drum worth (withdrawn -> inserted), three ways ----------------
dif_w, fam_w = states[WITHDRAWN]
dif_i, fam_i = states[INSERTED]


def worth(k_out, k_in):                     # reactivity swing, pcm (positive = worth)
    return (1.0 / k_out - 1.0 / k_in) * 1e5


print(f"\nControl-drum worth ({WITHDRAWN:.0f} -> {INSERTED:.0f} deg), pcm:")
print(f"  {'method':>18} {'k withdrawn':>12} {'k inserted':>11} {'worth':>9}")
print(f"  {'plain diffusion':>18} {dif_w:>12.6f} {dif_i:>11.6f} "
      f"{worth(dif_w, dif_i):>8.0f}p")
sph_worths = []
for name in FAMILIES:
    ref_w, ref_i = fam_w[name][0], fam_i[name][0]
    sph_w, sph_i = fam_w[name][1], fam_i[name][1]
    print(f"  {name + ' reference':>18} {ref_w:>12.6f} {ref_i:>11.6f} "
          f"{worth(ref_w, ref_i):>8.0f}p")
    print(f"  {name + ' SPH':>18} {sph_w:>12.6f} {sph_i:>11.6f} "
          f"{worth(sph_w, sph_i):>8.0f}p")
    sph_worths.append(worth(sph_w, sph_i))

spread = max(sph_worths) - min(sph_worths)
dw = worth(dif_w, dif_i)
print(f"\n  SPH-corrected worth spans {spread:.0f} pcm across the three "
      f"families,\n  vs a {abs(max(sph_worths, key=lambda w: abs(w - dw)) - dw):.0f}"
      f" pcm shift from plain diffusion -- the angular order is a small,\n"
      f"  controlled correction on top of the large diffusion error it removes.")
print("\n(placeholder cross sections: worths are illustrative, not predictive;"
      "\n see ndgpu/benchmarks/hpmr.py)")
