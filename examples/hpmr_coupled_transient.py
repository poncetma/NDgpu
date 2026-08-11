"""Coupled neutronics/thermal transient of the HP-MR: a control-drum manoeuvre.

    python examples/hpmr_coupled_transient.py [--refine 6] [--groups 11]
        [--t-end 120] [--dt 0.05] [--dt-thermal 0.5] [--device auto] [--nz 0]

Starts from the converged coupled steady state at rated power, withdraws the
control drums a few degrees over a few seconds, and marches both physics
together: the neutronics on the prompt/delayed clock, the temperature on its
own much slower one.

**What the answer looks like, and why.** The fuel's thermal time constant is
rho*cp/h ~ 268 s, while the drum manoeuvre takes seconds. So the power runs up
on the delayed-neutron clock long before the fuel has warmed enough for Doppler
to answer, and the excursion is arrested slowly over minutes rather than
promptly. That separation of time scales is the whole reason the coupling is
worth doing -- and the reason ``--dt-thermal`` can be several neutronics steps
without changing the answer (backward Euler on the conduction side is
unconditionally stable, so the thermal step is chosen by accuracy, not by
stability).

This is also the script to time on a GPU: it reports wall time per neutronics
step, which is the number that decides whether a realistic transient is minutes
or hours. See ``notebooks/colab_hpmr_coupled_transient.ipynb``.
"""

import argparse
import time

import numpy as np

from ndgpu.benchmarks.hpmr import HPMR_KINETICS, build_hpmr2d, build_hpmr3d
from ndgpu.benchmarks.hpmr_thermal import (RATED_POWER_W,
                                           hpmr_angle_for_dollars,
                                           build_hpmr_coupling,
                                           hpmr_drum_ramp, hpmr_endfb8_builtin,
                                           sink_coefficient)
from ndgpu.coupling import coupled_transient

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--refine", type=int, default=4)
ap.add_argument("--nz", type=int, default=0, help="0 = the 2D radial core")
ap.add_argument("--groups", choices=("2", "11"), default="11")
ap.add_argument("--t-end", type=float, default=60.0, help="seconds")
ap.add_argument("--dt", type=float, default=0.05, help="neutronics step, s")
ap.add_argument("--dt-thermal", type=float, default=0.5, help="thermal step, s")
ap.add_argument("--drum-from", type=float, default=90.0,
                help="starting drum angle (0 deg = arc inserted)")
ap.add_argument("--dollars", type=float, default=0.25,
                help="reactivity to insert, in dollars. The end angle is found "
                     "by measuring the worth curve, because degrees are not a "
                     "fixed reactivity: the drum has almost no worth left above "
                     "~150 deg, so the old 150->153 deg default was 0.05 $ and "
                     "moved the fuel by hundredths of a kelvin.")
ap.add_argument("--drum-to", type=float, default=None,
                help="override the end angle and ignore --dollars")
ap.add_argument("--t-start", type=float, default=1.0, help="manoeuvre start, s")
ap.add_argument("--t-ramp", type=float, default=4.0, help="manoeuvre duration, s")
ap.add_argument("--n-angles", type=int, default=13)
ap.add_argument("--device", default="auto")
ap.add_argument("--profile", action="store_true",
                help="collect CUDA-event/CPU phase timings and counters")
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()

three_d = args.nz > 0
mats = hpmr_endfb8_builtin(three_d=three_d) if args.groups == "11" else None
build = ((lambda a: build_hpmr3d(refine=args.refine, nz=args.nz,
                                 drum_angle_deg=a, absorber="polar",
                                 materials=mats))
         if three_d else
         (lambda a: build_hpmr2d(refine=args.refine, drum_angle_deg=a,
                                 absorber="polar", materials=mats)))

t_build = time.perf_counter()
if args.drum_to is None:
    args.drum_to, achieved_pcm = hpmr_angle_for_dollars(
        args.drum_from, args.dollars, refine=args.refine, nz=args.nz,
        materials=mats, device=args.device, with_worth=True)
    args.dollars = achieved_pcm / 1e5 / float(HPMR_KINETICS.beta.sum())
problem = build(args.drum_from)
ctx = build_hpmr_coupling(problem, device=args.device)
problem_at = hpmr_drum_ramp(problem, angle_from=args.drum_from,
                            angle_to=args.drum_to, t_start=args.t_start,
                            t_ramp=args.t_ramp, n_angles=args.n_angles,
                            refine=args.refine, nz=args.nz, materials=mats)
t_build = time.perf_counter() - t_build

n_cells = int(np.count_nonzero(problem.active))
n_steps = int(round(args.t_end / args.dt))
tau = ctx.thermal_materials[1].heat_capacity / sink_coefficient()

print(f"HP-MR coupled transient — {'3D' if three_d else '2D'}, refine {args.refine}"
      f"{f', nz {args.nz}' if three_d else ''}, "
      f"{ctx.materials[1].n_groups} groups, {n_cells:,} active cells")
print(f"  drums {args.drum_from:g}° → {args.drum_to:.2f}° over {args.t_ramp:g} s "
      f"starting at t = {args.t_start:g} s   "
      f"({args.dollars:+.3f} $ actually inserted)   "
      f"({RATED_POWER_W/1e6:g} MWt rated)")
print(f"  {args.t_end:g} s at dt = {args.dt:g} s → {n_steps:,} neutronics steps; "
      f"thermal step {args.dt_thermal:g} s")
print(f"  fuel thermal time constant rho*cp/h = {tau:.0f} s "
      f"({tau/args.t_end:.1f}x the run length)\n")

t0 = time.perf_counter()
res = coupled_transient(ctx, t_end=args.t_end, dt=args.dt,
                        dt_thermal=args.dt_thermal, problem_at=problem_at,
                        verbose=not args.quiet, profile=args.profile)
wall = time.perf_counter() - t0

print(f"\n  {'steady state (before t=0)':<32}: {res.steady.iterations} coupling "
      f"iterations, {res.steady.seconds:.1f} s")
print(f"  {'k_eff at the initial state':<32}: {res.k0:.6f}")
print(f"  {'peak P/P0':<32}: {res.power.max():.4f} at t = "
      f"{res.times[int(np.argmax(res.power))]:.2f} s")
print(f"  {'final P/P0':<32}: {res.power[-1]:.4f}")
print(f"  {'fuel temperature rise':<32}: "
      f"{res.mean_temperature[-1] - res.mean_temperature[0]:+.2f} K mean, "
      f"{res.peak_temperature[-1] - res.peak_temperature[0]:+.2f} K peak")
print(f"  {'peak fuel temperature':<32}: {res.peak_temperature.max():.1f} K")

print(f"\n  {'problem build + drum frames':<32}: {t_build:.1f} s")
print(f"  {'transient':<32}: {wall:.1f} s  ({res.steps:,} steps, "
      f"{1000 * wall / max(res.steps, 1):.1f} ms/step)")
print(f"  {'device':<32}: {res.device}")
if res.phase_seconds:
    print("\n  transient phase timings (overlap-free CUDA events on GPU):")
    for name, value in sorted(res.phase_seconds.items()):
        print(f"    {name:<28} {value:9.3f} s")
    print("  counters:")
    for name, value in sorted(res.counters.items()):
        print(f"    {name:<28} {value:,}")

# A compact trace, so a headless run still shows the shape.
print()
lo, hi = res.power.min(), res.power.max()
span = (hi - lo) or 1.0
for j in np.linspace(0, len(res.times) - 1, min(21, len(res.times))).astype(int):
    bar = "#" * int(round(46 * (res.power[j] - lo) / span))
    print(f"  t = {res.times[j]:7.2f} s   P/P0 = {res.power[j]:7.4f}   "
          f"T_fuel = {res.mean_temperature[j]:7.2f} K  {bar}")

if args.groups != "11":
    print("\n(placeholder 2-group cross sections: illustrative, not predictive)")
