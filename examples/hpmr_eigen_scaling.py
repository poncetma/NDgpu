"""Time one 2-D HP-MR diffusion eigenvalue solve on CPU or GPU."""

from __future__ import annotations

import argparse
import json
import platform
import time

from ndgpu.backend import device_name, get_backend, synchronize
from ndgpu.benchmarks import build_hpmr2d
from ndgpu.tri import TriDiffusionEigenSolver


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("refine", type=int)
parser.add_argument("device", choices=("cpu", "gpu"))
parser.add_argument("--angle", type=float, default=90.0)
parser.add_argument("--warmup-outers", type=int, default=3)
parser.add_argument("--tol-k", type=float, default=1e-8)
parser.add_argument("--tol-source", type=float, default=1e-7)
args = parser.parse_args()
if args.refine < 1 or args.warmup_outers < 0:
    parser.error("refine must be positive and warmup-outers non-negative")

xp = get_backend(args.device)
print(f"backend={device_name(xp)} host={platform.node()} refine={args.refine}",
      flush=True)

started = time.perf_counter()
problem = build_hpmr2d(refine=args.refine, drum_angle_deg=args.angle,
                       absorber="polar")
build_seconds = time.perf_counter() - started
active_cells = int(problem.active.sum())
solver = TriDiffusionEigenSolver(
    problem.grid, problem.materials, problem.material_map,
    active=problem.active, mask_bc=problem.mask_bc,
    mix_material=problem.mix_material, mix_weight=problem.mix_weight,
    device=args.device)
n_groups = solver.n_groups
print(f"active_cells={active_cells:,} flux_unknowns="
      f"{active_cells * n_groups:,} build_seconds={build_seconds:.6f}",
      flush=True)

warmup_seconds = 0.0
if args.warmup_outers:
    synchronize(xp)
    started = time.perf_counter()
    solver.solve(max_outer=args.warmup_outers, tol_k=0.0, tol_source=0.0)
    synchronize(xp)
    warmup_seconds = time.perf_counter() - started
    print(f"warmup_seconds={warmup_seconds:.6f}", flush=True)

synchronize(xp)
started = time.perf_counter()
result = solver.solve(tol_k=args.tol_k, tol_source=args.tol_source)
synchronize(xp)
solve_seconds = time.perf_counter() - started
summary = {
    "active_cells": active_cells,
    "angle_deg": args.angle,
    "backend": args.device,
    "build_seconds": build_seconds,
    "converged": bool(result.converged),
    "flux_unknowns": active_cells * n_groups,
    "inner_iterations": int(result.inner_iterations),
    "k_eff": float(result.k_eff),
    "n_groups": n_groups,
    "outer_iterations": int(result.outer_iterations),
    "refine": args.refine,
    "solve_seconds": solve_seconds,
    "solver_reported_seconds": float(result.solve_seconds),
    "tol_k": args.tol_k,
    "tol_source": args.tol_source,
    "warmup_seconds": warmup_seconds,
}
print("RESULT " + json.dumps(summary, sort_keys=True), flush=True)
if not result.converged:
    raise SystemExit("eigenvalue solve did not converge")
