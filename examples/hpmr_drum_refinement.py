"""Single-GPU HP-MR drum-worth global/local mesh convergence study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ndgpu.benchmarks.hpmr import (DRUM_ABSORBER_INNER, DRUM_ARC_HALF_DEG,
                                   DRUM_RADIUS, build_hpmr2d,
                                   build_hpmr2d_local)
from ndgpu.benchmarks.hpmr_thermal import hpmr_endfb8_builtin
from ndgpu.mesh import UnstructuredDiffusionSolver
from ndgpu.tri import TriDiffusionEigenSolver


def _numbers(value, cast=float):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _write(path, result):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


def _absorber_area(problem, local):
    if local:
        return float(np.dot(problem.mesh.area, problem.mix_weight))
    return float(problem.grid.cell_volume * np.sum(problem.mix_weight))


def _solve(args, materials, refine, level, angle):
    local = args.mode == "local"
    started = time.perf_counter()
    if local:
        problem = build_hpmr2d_local(
            refine=refine, drum_angle_deg=angle,
            local_refinement=level > 0, drum_refine_levels=level,
            materials=materials, absorber="polar", samples=args.samples)
        build_seconds = time.perf_counter() - started
        solver = UnstructuredDiffusionSolver(
            problem.mesh, problem.materials, problem.cell_material,
            problem.alpha_boundary, device=args.device,
            precond_degree=args.precond_degree,
            mix_material=problem.mix_material,
            mix_weight=problem.mix_weight)
        cells = problem.mesh.n_cells
        effective_refine = problem.drum_refine
        drum_cell_side = float(np.sqrt(
            4.0 * np.min(problem.mesh.area[problem.mix_weight > 0.0])
            / np.sqrt(3.0)))
    else:
        problem = build_hpmr2d(
            refine=refine, drum_angle_deg=angle, materials=materials,
            absorber="polar", samples=args.samples)
        build_seconds = time.perf_counter() - started
        solver = TriDiffusionEigenSolver(
            problem.grid, problem.materials, problem.material_map,
            active=problem.active, mask_bc=problem.mask_bc,
            mix_material=problem.mix_material,
            mix_weight=problem.mix_weight, device=args.device,
            precond_degree=args.precond_degree)
        cells = int(np.count_nonzero(problem.active))
        effective_refine = refine
        drum_cell_side = problem.grid.side

    print(f"solve start mode={args.mode} global_r={refine} local_level={level} "
          f"effective_drum_r={effective_refine} angle={angle:g} cells={cells:,}",
          flush=True)
    result = solver.solve(tol_k=args.tol_k, tol_source=args.tol_source,
                          max_outer=args.max_outer)
    if not result.converged:
        raise RuntimeError(f"eigen solve did not converge: r={refine}, "
                           f"level={level}, angle={angle}")
    record = {
        "angle_deg": angle,
        "k_eff": result.k_eff,
        "cells": cells,
        "unknowns": cells * len(materials[0].diffusion),
        "drum_cell_side_cm": drum_cell_side,
        "build_seconds": build_seconds,
        "solve_seconds": result.solve_seconds,
        "outer_iterations": result.outer_iterations,
        "inner_iterations": result.inner_iterations,
        "absorber_area_cm2": _absorber_area(problem, local),
    }
    print("HPMR_DRUM_REFINEMENT_SOLVE " + json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "global"), default="local")
    parser.add_argument("--refine", type=int, default=4,
                        help="fixed global refinement for local mode")
    parser.add_argument("--refines", default="4,6,8,12,16",
                        help="global-mode refinement ladder")
    parser.add_argument("--local-levels", default="0,1,2,3")
    parser.add_argument("--angles", default="90,90.5,95")
    parser.add_argument("--samples", type=int, default=0,
                        help="0 for exact curved-cell intersections")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--precond-degree", type=int, default=1)
    parser.add_argument("--tol-k", type=float, default=1e-8)
    parser.add_argument("--tol-source", type=float, default=1e-7)
    parser.add_argument("--max-outer", type=int, default=3000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    angles = _numbers(args.angles)
    if len(angles) < 2:
        parser.error("--angles requires a reference and at least one endpoint")
    cases = ([(args.refine, level) for level in _numbers(args.local_levels, int)]
             if args.mode == "local" else
             [(refine, 0) for refine in _numbers(args.refines, int)])
    materials = hpmr_endfb8_builtin(three_d=False)
    exact_area = (12.0 * np.deg2rad(DRUM_ARC_HALF_DEG)
                  * (DRUM_RADIUS**2 - DRUM_ABSORBER_INNER**2))
    report = {
        "status": "running",
        "mode": args.mode,
        "device": args.device,
        "groups": len(materials[0].diffusion),
        "angles_deg": angles,
        "samples": args.samples,
        "exact_absorber_area_cm2": exact_area,
        "cases": [],
    }
    _write(args.output, report)

    previous = {angle: None for angle in angles[1:]}
    for refine, level in cases:
        solves = [_solve(args, materials, refine, level, angle)
                  for angle in angles]
        k0 = solves[0]["k_eff"]
        worth = {str(angle): 1e5 * (1.0 / k0 - 1.0 / solve["k_eff"])
                 for angle, solve in zip(angles[1:], solves[1:])}
        interval_worth = {
            str(right["angle_deg"]): 1e5 * (
                1.0 / left["k_eff"] - 1.0 / right["k_eff"])
            for left, right in zip(solves[:-1], solves[1:])
        }
        increments = {}
        for angle in angles[1:]:
            value = worth[str(angle)]
            increments[str(angle)] = (None if previous[angle] is None
                                       else value - previous[angle])
            previous[angle] = value
        case = {
            "global_refine": refine,
            "local_level": level,
            "effective_drum_refine": refine * 2**level,
            "solves": solves,
            "worth_pcm": worth,
            "interval_worth_pcm": interval_worth,
            "monotonic_k": all(value > 0.0 for value in interval_worth.values()),
            "worth_increment_pcm": increments,
        }
        report["cases"].append(case)
        _write(args.output, report)
        print("HPMR_DRUM_REFINEMENT_CASE " + json.dumps(case), flush=True)

    report["status"] = "completed"
    _write(args.output, report)
    print("HPMR_DRUM_REFINEMENT_RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
