"""Coupled neutronics/thermal steady state of the HP-MR microreactor.

    python examples/hpmr_coupled.py [refine] [drum_deg] [groups] [device] [nz]

Solves the core at rated power (2 MWt) with the fuel temperature and the cross
sections made consistent with each other: fission heats the fuel, conduction
and the heat pipes set the temperature, Doppler broadening puts the temperature
back into the absorption cross sections, and the loop is iterated to a fixed
point.

The headline number is the **temperature defect** -- the reactivity the core
loses between cold zero power and hot full power. It is invisible to an
isothermal calculation, and it is what the control drums have to pay for.

`groups` selects the cross sections: "11" (default) is the real ENDF/B-8 set
from the VTB Griffin library, whose kappaFission gives a properly
energy-weighted power distribution; "2" is the placeholder set, fast but with
no kappaFission (the power shape falls back to nu-weighting) and a fabricated
spectrum -- structural demos only, never a quoted number.

`nz` > 0 extrudes the core to the full 200 cm (160 cm fueled + 2 x 20 cm
axial Be reflectors) and reports the axial temperature profile, which is where
the interesting thermal shape lives: in 2D the model is a 1 cm slice and every
assembly has the same temperature top to bottom.
"""

import sys

import numpy as np

from ndgpu.benchmarks.hpmr import build_hpmr2d, build_hpmr3d
from ndgpu.benchmarks.hpmr_thermal import (AMBIENT_K, RATED_POWER_W,
                                           SINK_TEMPERATURE_K,
                                           build_hpmr_coupling,
                                           hpmr_endfb8_builtin)
from ndgpu.coupling import CoupledSolver, temperature_defect_pcm, uncoupled_k

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
drum_deg = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
groups = sys.argv[3] if len(sys.argv) > 3 else "11"
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
nz = int(sys.argv[5]) if len(sys.argv) > 5 else 0

mats = hpmr_endfb8_builtin(three_d=nz > 0) if groups == "11" else None
if nz > 0:
    problem = build_hpmr3d(refine=refine, nz=nz, drum_angle_deg=drum_deg,
                           absorber="polar", materials=mats)
else:
    problem = build_hpmr2d(refine=refine, drum_angle_deg=drum_deg,
                           absorber="polar", materials=mats)
ctx = build_hpmr_coupling(problem, device=device)

n_cells = int(np.count_nonzero(problem.active))
print(f"HP-MR {'3D' if nz else '2D'} coupled neutronics/thermal — refine {refine}"
      f"{f', nz {nz}' if nz else ''} ({n_cells:,} active cells), "
      f"drums at {drum_deg:g}°, {ctx.materials[1].n_groups}-group, {device}")
if nz:
    print(f"  rated power {RATED_POWER_W/1e6:g} MWt over the full core")
else:
    print(f"  rated power {RATED_POWER_W/1e6:g} MWt over the full 160 cm core; "
          f"this 2D slice carries {ctx.total_power:,.0f} W")
print(f"  heat pipes at {SINK_TEMPERATURE_K:g} K, vessel at {AMBIENT_K:g} K\n")

res = CoupledSolver(ctx).solve(tol=1e-8, anderson_depth=5, verbose=True)
if not res.converged:
    raise SystemExit(f"coupled iteration did not converge: {res}")

k_cold = uncoupled_k(ctx, np.full(ctx.shape, AMBIENT_K))
defect = temperature_defect_pcm(k_cold, res.k_eff)

fuel = problem.material_map == 1
T = res.temperature
th = res.thermal

print(f"\n  {f'k_eff (cold, {AMBIENT_K:g} K isothermal)':<34}: {k_cold:.6f}")
print(f"  {'k_eff (hot, coupled at power)':<34}: {res.k_eff:.6f}")
print(f"  {'temperature defect':<34}: {defect:+.0f} pcm")
print(f"\n  fuel temperature   : {T[fuel].min():.1f} / {T[fuel].mean():.1f} / "
      f"{T[fuel].max():.1f} K   (min / mean / max)")
print(f"  peak anywhere      : {res.peak_temperature:.1f} K")
print(f"\n  energy balance     : {th.source_watts:,.0f} W in = "
      f"{th.sink_watts:,.0f} W to the heat pipes + "
      f"{th.leakage_watts:,.0f} W through the vessel")
print(f"                       closure {th.balance_residual:.1e}")
if nz:
    # Axial profile, fuel only: the 3D result the 2D slice cannot show.
    dz = problem.grid.dz
    print("\n  axial fuel temperature (z at layer centre):")
    for iz in range(nz):
        layer = fuel[..., iz]
        if not layer.any():
            continue
        col = T[..., iz][layer]
        bar = "#" * int(round(40 * (col.mean() - T[fuel].min())
                              / max(T[fuel].max() - T[fuel].min(), 1e-9)))
        print(f"    z = {(iz + 0.5) * dz:6.1f} cm   mean {col.mean():7.2f} K   "
              f"max {col.max():7.2f} K  {bar}")

print(f"\n  converged in {res.iterations} coupling iterations, {res.seconds:.1f} s")
print("  k per coupling iteration: "
      + ", ".join(f"{k:.6f}" for k in res.k_history))

if groups != "11":
    print("\n(placeholder 2-group cross sections: shapes are illustrative, not"
          "\n predictive, and carry no kappaFission — see hpmr_endfb8_builtin)")
