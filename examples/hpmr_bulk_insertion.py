"""Coupled HP-MR transient driven by a BULK reactivity insertion.

    python examples/hpmr_bulk_insertion.py [--dollars 0.5] [--refine 4]
        [--groups 11] [--t-end 30] [--dt 0.02] [--device auto]

A uniform scaling of every material's absorption cross section, sized to insert
a requested reactivity. Compared with rotating a control drum this is a much
better-posed experiment:

* **Shape preserving.** The perturbation is uniform, so the flux shape barely
  moves and the answer is the point-kinetics ODE -- prompt jump beta/(beta-rho),
  then the inhour period. Theory and solver can be compared directly.
* **Mesh insensitive.** Drum worth is a *differential* of two eigenvalues with
  the absorber arc placed differently against the mesh each time, and it does
  not converge: 90->95 deg measures +67, +205, +88, +137, +97 pcm at refine
  2/3/4/6/8. A bulk perturbation has no such geometry, so the reactivity it
  inserts is essentially mesh-independent.
* **Monotone.** rho is smooth and monotone in the scale factor, so solving for
  a target insertion is a two-solve secant rather than a search over a
  staircase.

The manoeuvre is a step at t = 0 (nothing is being modelled as physically
rotating), which makes the timing representative of the hardest steps: every
step after the insertion has a fixed point that starts far from converged.
"""

import argparse
import time

import numpy as np

from ndgpu import Material, TriDiffusionEigenSolver
from ndgpu.benchmarks.hpmr import HPMR_KINETICS, build_hpmr2d, build_hpmr3d
from ndgpu.benchmarks.hpmr_thermal import (RATED_POWER_W, build_hpmr_coupling,
                                           hpmr_endfb8_builtin,
                                           hpmr_kinetics_11g, sink_coefficient)
from ndgpu.coupling import coupled_transient

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--dollars", type=float, default=0.5)
ap.add_argument("--refine", type=int, default=4)
ap.add_argument("--nz", type=int, default=0)
ap.add_argument("--groups", choices=("2", "11"), default="11")
ap.add_argument("--t-end", type=float, default=30.0)
ap.add_argument("--dt", type=float, default=0.02)
ap.add_argument("--dt-thermal", type=float, default=0.2)
ap.add_argument("--device", default="auto")
ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
ap.add_argument("--precond-degree", type=int, default=0)
ap.add_argument("--quiet", action="store_true")
args = ap.parse_args()

three_d = args.nz > 0
mats0 = hpmr_endfb8_builtin(three_d=three_d) if args.groups == "11" else None
DTYPE = np.float32 if args.dtype == "float32" else np.float64


def build(materials, angle=150.0):
    if three_d:
        return build_hpmr3d(refine=args.refine, nz=args.nz, drum_angle_deg=angle,
                            absorber="polar", materials=materials)
    return build_hpmr2d(refine=args.refine, drum_angle_deg=angle,
                        absorber="polar", materials=materials)


def scale_absorption(materials, factor):
    """Uniform bulk perturbation: every material's Sigma_a times `factor`."""
    return [Material(name=m.name, diffusion=m.diffusion,
                     sigma_a=m.sigma_a * factor, nu_sigma_f=m.nu_sigma_f,
                     sigma_s=m.sigma_s, chi=m.chi if m.is_fissile else None,
                     kappa_fission=m.kappa_fission)
            for m in materials]


def k_of(materials):
    p = build(materials)
    return TriDiffusionEigenSolver(
        p.grid, p.materials, p.material_map, bc=p.bc, active=p.active,
        mask_bc=p.mask_bc, mix_material=p.mix_material,
        mix_weight=p.mix_weight, device=args.device,
        dtype=DTYPE).solve(tol_k=1e-10, tol_source=1e-9).k_eff


t_setup = time.perf_counter()
p_base = build(mats0)
base_materials = p_base.materials
kin = hpmr_kinetics_11g() if args.groups == "11" else HPMR_KINETICS
beta = float(kin.beta.sum())
target = args.dollars * beta * 1e5              # pcm

k0 = k_of(base_materials)
# rho is monotone and near-linear in the scale factor, so secant converges in
# two or three solves. Bracket downward: less absorption is more reactivity.
f, f_prev, rho_prev = 0.999, 1.0, 0.0
for _ in range(6):
    rho = 1e5 * (1.0 / k0 - 1.0 / k_of(scale_absorption(base_materials, f)))
    if abs(rho - target) <= 0.01 * abs(target):
        break
    slope = (rho - rho_prev) / (f - f_prev) if f != f_prev else 1.0
    f_prev, rho_prev = f, rho
    f = f - (rho - target) / slope
mats1 = scale_absorption(base_materials, f)
t_setup = time.perf_counter() - t_setup

ctx = build_hpmr_coupling(p_base, device=args.device)
ctx.dtype = DTYPE
ctx.kinetics = kin
step = (lambda t: ((mats1 if t >= 0.5 * args.dt else base_materials),
                   p_base.material_map, p_base.mix_material, p_base.mix_weight))

n_cells = int(np.count_nonzero(p_base.active))
n_steps = int(round(args.t_end / args.dt))
tau = ctx.thermal_materials[1].heat_capacity / sink_coefficient()
print(f"HP-MR bulk insertion — {'3D' if three_d else '2D'}, refine {args.refine}"
      f"{f', nz {args.nz}' if three_d else ''}, {ctx.materials[1].n_groups} groups, "
      f"{n_cells:,} active cells, {args.dtype}")
print(f"  Sigma_a x {f:.6f} everywhere  ->  {rho:+.1f} pcm = "
      f"{rho/1e5/beta:+.3f} $   (beta = {1e5*beta:.0f} pcm)")
print(f"  step at t = 0;  {args.t_end:g} s at dt = {args.dt:g} s -> "
      f"{n_steps:,} steps;  thermal step {args.dt_thermal:g} s")
print(f"  fuel thermal time constant = {tau:.0f} s\n")

t0 = time.perf_counter()
res = coupled_transient(ctx, t_end=args.t_end, dt=args.dt,
                        dt_thermal=args.dt_thermal, problem_at=step,
                        verbose=not args.quiet)
wall = time.perf_counter() - t0

# Read the power AFTER the prompt transient, not one step after the insertion.
# The jump develops over Lambda/(beta-rho) ~ 5 ms at half a dollar, so a single
# step of duration dt captures only part of it -- and less of it the smaller dt
# is (measured: P after one step goes 1.617 -> 1.277 -> 1.073 -> 1.021 as dt
# falls 0.02 -> 0.00025, while P at a fixed 0.1 s converges to 2.064). Compare
# at a fixed physical time, many prompt constants later but short against the
# period, and carry the delayed growth over that interval in the theory.
jump_theory = beta / (beta - rho / 1e5)
period = (beta - rho / 1e5) / (float(kin.decay.min()) * rho / 1e5)
t_read = min(0.2, 0.5 * args.t_end)
i_jump = int(np.argmin(np.abs(res.times - t_read)))
expected = jump_theory * np.exp(res.times[i_jump] / period)
print(f"\n  {f'P at t = {res.times[i_jump]:.2f} s (solver / theory)':<34}: "
      f"{res.power[i_jump]:.4f} / {expected:.4f}")
print(f"  {'  prompt jump beta/(beta-rho)':<34}: {jump_theory:.4f}, "
      f"then a {period:.1f} s period")
print(f"  {'peak P/P0':<34}: {res.power.max():.3f}")
print(f"  {'final P/P0':<34}: {res.power[-1]:.3f}")
print(f"  {'fuel temperature rise':<34}: "
      f"{res.mean_temperature[-1] - res.mean_temperature[0]:+.1f} K mean, "
      f"{res.peak_temperature[-1] - res.peak_temperature[0]:+.1f} K peak")
print(f"\n  {'setup (worth search + builds)':<34}: {t_setup:.1f} s")
print(f"  {'coupled steady state':<34}: {res.steady.seconds:.1f} s "
      f"({res.steady.iterations} iterations)")
print(f"  {'transient':<34}: {wall:.1f} s "
      f"({res.steps:,} steps, {1000*wall/max(res.steps,1):.0f} ms/step)")
print(f"  {'device':<34}: {res.device}")

lo, hi = res.power.min(), res.power.max()
span = (hi - lo) or 1.0
print()
for j in np.linspace(0, len(res.times) - 1, min(16, len(res.times))).astype(int):
    print(f"  t = {res.times[j]:6.2f} s   P/P0 = {res.power[j]:8.3f}   "
          f"T_fuel = {res.mean_temperature[j]:7.2f} K  "
          + "#" * int(round(40 * (res.power[j] - lo) / span)))
