"""HP-MR SPH factors from an S_N transport reference, against the SP3 reference.

The SPH pipeline (:mod:`ndgpu.sph`) folds a reference's angular treatment of the
near-black B4C control-drum arc into the coarse diffusion constants so plain
TriDiffusion reproduces the reference eigenvalue -- and hence the drum worth --
that it otherwise misses. examples/hpmr_sph_reference_families.py drives it from
the *simplified*-transport families (SP3, SDP1, SDP2); this one drives it from
full discrete ordinates (TriSNTransportSolver), which is the reference those
families are themselves approximating.

That matters because SP3 and S_N disagree about the drum arc by far more than
the SP*N* families disagree among themselves: a moment closure and a genuine
angular sweep treat a near-black absorber differently. So this is the test of
whether SPH is really reference-agnostic, or whether it has only ever been
exercised on references that share diffusion's stencil.

    python examples/hpmr_sph_sn_reference.py [refine] [n_polar] [n_azi]

Uses absorber="polar": the drum arc is volume-mixed into the cells it crosses,
and flux_weighted_homogenize splits each blended cell into fractional entries so
the arc is homogenized at its true volume fraction. A rasterized arc would need
a very fine mesh before a 1 cm B4C annulus is resolved at all.

Cross sections are placeholders, so worths are illustrative, not predictive.
"""

import sys
import time

import numpy as np

from ndgpu import (TriDiffusionEigenSolver, TriSDP1EigenSolver,
                   TriSDP2EigenSolver, TriSP3EigenSolver,
                   flux_weighted_homogenize, region_average, sph_correct)
from ndgpu.benchmarks.hpmr import build_hpmr2d, DRUM_ABSORBER
from ndgpu.tri_sn import TriSNTransportSolver

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
n_polar = int(sys.argv[2]) if len(sys.argv) > 2 else 2
n_azi = int(sys.argv[3]) if len(sys.argv) > 3 else 8

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)
TIGHT_SN = dict(tol_k=1e-8, tol_source=1e-7)
# Drum-arc rotation. Verified by eigenvalue, not by assumption: 0 deg gives the
# LOWER k, i.e. the arc faces the core and the drum is inserted.
INSERTED, WITHDRAWN = 0.0, 180.0


# The matched-DoF simplified-transport families, plus full discrete ordinates.
MOMENT_FAMILIES = {"SP3": TriSP3EigenSolver, "SDP1": TriSDP1EigenSolver,
                   "SDP2": TriSDP2EigenSolver}
REFERENCES = ("SP3", "SDP1", "SDP2", "S_N")


def reference(name, p, common):
    """(k_eff, scalar flux, converged) for one reference family.

    The moment families take mask_bc and expose .flux_numpy; tri-S_N takes only
    the active mask (its masked faces are vacuum, matching HP-MR's mask_bc) and
    its Result.flux is already numpy.
    """
    if name == "S_N":
        r = TriSNTransportSolver(
            p.grid, p.materials, p.material_map, active=p.active,
            mix_material=p.mix_material, mix_weight=p.mix_weight,
            n_polar=n_polar, n_azi=n_azi, bc="vacuum",
            scheme="scb").solve(**TIGHT_SN)
        return r.k_eff, np.asarray(r.flux), r.converged
    r = MOMENT_FAMILIES[name](p.grid, p.materials, p.material_map,
                              **common).solve(**TIGHT)
    return r.k_eff, r.flux_numpy, r.converged


def solve_state(angle):
    """Diffusion + SPH-from-each-reference at one drum angle."""
    p = build_hpmr2d(refine=refine, drum_angle_deg=angle, absorber="polar")
    dV = p.grid.cell_volume
    common = dict(active=p.active, mask_bc=p.mask_bc,
                  mix_material=p.mix_material, mix_weight=p.mix_weight)
    mixkw = dict(mix_material=p.mix_material, mix_weight=p.mix_weight)
    region = p.material_map

    dif = TriDiffusionEigenSolver(p.grid, p.materials, region,
                                  **common).solve(**TIGHT)
    if not dif.converged:
        raise SystemExit(f"diffusion not converged at {angle} deg")

    def coarse_solve(materials):
        r = TriDiffusionEigenSolver(p.grid, materials, region,
                                    **common).solve(**TIGHT)
        return region_average(r.flux_numpy, region, **mixkw), r.k_eff

    out = {}
    for name in REFERENCES:
        t0 = time.perf_counter()
        k_ref, flux, ok = reference(name, p, common)
        t_ref = time.perf_counter() - t0
        if not ok:
            raise SystemExit(f"{name} reference not converged at {angle} deg")
        hmats, rflux, _ = flux_weighted_homogenize(
            flux, p.materials, region, region, cell_volume=dV, **mixkw)
        sph = sph_correct(hmats, region, rflux, coarse_solve, tol=1e-7,
                          depth=6, max_iter=300)
        out[name] = dict(k_ref=k_ref, k_sph=sph.k_eff, mu=sph.factors,
                         its=sph.iterations, t_ref=t_ref, ok=sph.converged)
    return dif.k_eff, out


print(f"HP-MR 2D SPH from an S_N reference, refine={refine} "
      f"({6 * refine ** 2} tri/assembly), S_N = {n_polar}x{n_azi} scb\n")

states = {a: solve_state(a) for a in (INSERTED, WITHDRAWN)}

for angle, tag in ((INSERTED, "INSERTED (0 deg, arc toward core)"),
                   (WITHDRAWN, "WITHDRAWN (180 deg)")):
    dif_k, fam = states[angle]
    print(f"{tag}:   uncorrected diffusion k = {dif_k:.6f}")
    print(f"  {'reference':>9} {'ref k':>10} {'SPH k':>10} {'ref-dif':>9} "
          f"{'SPH-dif':>9} {'|SPH-ref|':>10} {'its':>4} {'t_ref':>7} "
          f"{'drum mu':>16}")
    for name, d in fam.items():
        mu = ", ".join(f"{v:.3f}" for v in np.atleast_1d(d["mu"][DRUM_ABSORBER]))
        print(f"  {name:>9} {d['k_ref']:>10.6f} {d['k_sph']:>10.6f} "
              f"{1e5 * (d['k_ref'] - dif_k):>+8.0f}p "
              f"{1e5 * (d['k_sph'] - dif_k):>+8.0f}p "
              f"{1e5 * abs(d['k_sph'] - d['k_ref']):>9.1f}p "
              f"{(str(d['its']) if d['ok'] else 'DIV'):>4} "
              f"{d['t_ref']:>6.1f}s   {mu:>14}")
    print()

# --- drum worth: withdrawn -> inserted, four ways ------------------------------
dif_i, fam_i = states[INSERTED]
dif_w, fam_w = states[WITHDRAWN]
print("Control-drum worth (withdrawn -> inserted), pcm in Delta k:")
print(f"  {'plain diffusion':>22} {1e5 * (dif_w - dif_i):>9.0f}")
for name in REFERENCES:
    w_ref = 1e5 * (fam_w[name]["k_ref"] - fam_i[name]["k_ref"])
    w_sph = 1e5 * (fam_w[name]["k_sph"] - fam_i[name]["k_sph"])
    print(f"  {name + ' reference':>22} {w_ref:>9.0f}")
    print(f"  {name + ' SPH diffusion':>22} {w_sph:>9.0f}   "
          f"(recovers {100 * (w_sph - 1e5 * (dif_w - dif_i)) / (w_ref - 1e5 * (dif_w - dif_i)):.0f}% "
          f"of the {name}-vs-diffusion gap)")
