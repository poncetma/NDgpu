"""HP-MR heterogeneous assembly: S_N reference, form function, pin reconstruction.

The full-core HP-MR model (:mod:`ndgpu.benchmarks.hpmr`) homogenizes each fuel
hex into one material. That discards the flux shape INSIDE the assembly, and
equivalence factoring does not bring it back -- SPH and GET correct
assembly-integrated reaction rates, not the distribution within a region. This
example runs the pipeline that does recover it:

  1. solve the pin-resolved assembly with S_N in an infinite lattice
     (periodic rhombic unit cell) -> k_inf and the heterogeneous flux;
  2. reduce that to the intra-assembly FORM FUNCTION, per pin, normalized to
     mean 1 over the fuel pins;
  3. solve the homogenized full core with diffusion;
  4. reconstruct pin power as  P_pin = S(x_pin) * f(pin), sampling the coarse
     shape at each pin's own position so an assembly in a flux gradient gets a
     tilted reconstruction rather than a uniform scaling.

    python examples/hpmr_assembly_reference.py [refine] [core_refine] [drum_deg]

Everything is self-contained: the pin lattice is transcribed from the VTB
Serpent deck into ``VTB_LATTICE_MAP``, and the real 11-group ENDF/B-8 pin cross
sections are vendored in ``benchmarks/data/hpmr_pin_xs_g11.npz`` (10 KB). No
download, no external library file.

Cost: refine is cells per PIN PITCH, and the assembly solve dominates. refine=4
is a fast smoke test (~20 s); refine=8 is a usable reference (~2 min, 21 632
cells); refine=10 (~34 000 cells) needs roughly 10 GB of RAM, and refine=12 is
beyond a 11 GB machine -- size it before launching.
"""

import sys
import time

import numpy as np

from ndgpu import TriDiffusionEigenSolver, flux_weighted_homogenize
from ndgpu.benchmarks.hpmr import (build_hpmr2d, hpmr_materials_builtin,
                                   HPMR_FUEL_SITES, PITCH)
from ndgpu.benchmarks.hpmr_assembly import (build_hpmr_assembly2d, pin_fluxes,
                                            pin_materials_builtin, pin_powers)
from ndgpu.hexraster import hex_site_xy
from ndgpu.materials import Material
from ndgpu.pin_power import peaking, reconstruct_pin_powers
from ndgpu.tri_sn import TriSNTransportSolver

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 6
core_refine = int(sys.argv[2]) if len(sys.argv) > 2 else 6
drum_deg = float(sys.argv[3]) if len(sys.argv) > 3 else 180.0

# --- 1. heterogeneous assembly, S_N in an infinite lattice --------------------
p = build_hpmr_assembly2d(refine=refine, materials=pin_materials_builtin())
print(f"Assembly: {p.material_map.size} cells, {len(p.pin_kind)} pins "
      f"({sum(k == 'fuel' for k in p.pin_kind)} fuel, "
      f"{sum(k == 'mod' for k in p.pin_kind)} moderator, "
      f"{sum(k == 'hp' for k in p.pin_kind)} heat pipe)")
print("  achieved volume fractions: "
      + ", ".join(f"{k} {v:.4f}" for k, v in p.volume_fractions.items()))

t0 = time.perf_counter()
sn = TriSNTransportSolver(p.grid, p.materials, p.material_map,
                          mix_material=p.mix_material, mix_weight=p.mix_weight,
                          n_polar=2, n_azi=8, bc="periodic", scheme="scb")
r = sn.solve(tol_k=1e-7, tol_source=1e-6, max_outer=400)
if not r.converged:
    raise SystemExit("assembly S_N did not converge")
print(f"  S_N k_inf = {r.k_eff:.6f}  ({r.outer_iterations} outers, "
      f"{time.perf_counter() - t0:.0f} s)")

# --- 2. form function + the flux-weighted homogenized assembly material -------
flux = r.flux_numpy                    # device -> host for postprocessing
power, form = pin_powers(p, flux)
_, flux_form = pin_fluxes(p, flux)
fuel = np.array([k == "fuel" for k in p.pin_kind])
print(f"  power form function over fuel pins: {form[fuel].min():.4f} .. "
      f"{form[fuel].max():.4f}")

region = np.zeros(p.material_map.shape, dtype=np.int64)   # one region: the assembly
hm, _, _ = flux_weighted_homogenize(
    flux, p.materials, p.material_map, region, cell_volume=p.grid.cell_volume,
    mix_material=p.mix_material, mix_weight=p.mix_weight)

# --- 3. homogenized full core -------------------------------------------------
mats = hpmr_materials_builtin(hm[0])
core = build_hpmr2d(refine=core_refine, drum_angle_deg=drum_deg,
                    absorber="polar", materials=mats)
c = TriDiffusionEigenSolver(core.grid, core.materials, core.material_map,
                            active=core.active, mask_bc=core.mask_bc,
                            mix_material=core.mix_material,
                            mix_weight=core.mix_weight).solve(tol_k=1e-9,
                                                              tol_source=1e-8)
print(f"Core (drums {drum_deg:.0f} deg): k_eff = {c.k_eff:.6f}")

# --- 4. reconstruct pin power over every fuel assembly ------------------------
sites = np.array([hex_site_xy(R, C, PITCH) for R, C in HPMR_FUEL_SITES])
centres = p.pin_centres - p.pin_centres.mean(axis=0)
pw, xy = reconstruct_pin_powers(core.raster, c.flux_numpy, core.materials,
                                core.material_map, sites, centres, form,
                                pin_kind=p.pin_kind,
                                mix_material=core.mix_material,
                                mix_weight=core.mix_weight, active=core.active)
pw /= pw[:, fuel].mean()
pk, _, _ = peaking(pw, p.pin_kind)
asm = pw[:, fuel].sum(axis=1)
asm /= asm.mean()
print(f"  {len(sites)} assemblies x {int(fuel.sum())} fuel pins "
      f"= {len(sites) * int(fuel.sum())} pins")
print(f"  pin peaking factor      = {pk:.4f}")
print(f"  assembly radial peaking = {asm.max():.4f} (min {asm.min():.4f})")

# How much does sampling the coarse shape per PIN buy over one value per assembly?
flat = np.array([pw[s][fuel].sum() / form[fuel].sum() for s in range(len(sites))])
d = np.abs(pw[:, fuel] - (flat[:, None] * form[None, :])[:, fuel])
d /= np.maximum(pw[:, fuel], 1e-30)
print(f"  intra-assembly tilt vs per-assembly scaling: max {100 * d.max():.1f}%, "
      f"rms {100 * d.std():.1f}%")
