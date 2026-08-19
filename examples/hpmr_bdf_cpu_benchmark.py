"""CPU accuracy/work gate for high-order BDF on moving HP-MR drums.

The fine reference and every candidate see the same pre-rasterized,
weight-interpolated control trajectory.  The reference defaults to backward
Euler; candidates use larger steps and BDF1--6.  This isolates temporal error
and stability from mesh/raster differences while exercising the real 11-group
tri-grid operator.

Run from the repository root::

    python examples/hpmr_bdf_cpu_benchmark.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ndgpu.benchmarks.hpmr import build_hpmr2d
from ndgpu.benchmarks.hpmr_transient_bench import build_case
from ndgpu.transient import TransientSolver
from ndgpu.tri import TriDiffusionEigenSolver, TriGroupOperator


def _trajectory_frames(problem, refine, frame_count, asymmetric=False,
                       angle_from=150.0, angle_to=154.0, samples=6):
    frames = []
    for fraction in np.linspace(0.0, 1.0, frame_count):
        moving = angle_from + fraction * (angle_to - angle_from)
        if asymmetric:
            angles = np.full(12, angle_from)
            angles[:3] = moving
        else:
            angles = moving
        frame = build_hpmr2d(
            refine=refine, drum_angle_deg=angles, absorber="polar",
            materials=problem.materials, samples=samples)
        frames.append((frame.mix_material, frame.mix_weight))
    return frames


def _problem_at(problem, frames, duration):
    last = len(frames) - 1
    mix_material = frames[0][0]
    if not all(np.array_equal(mix_material, frame[0]) for frame in frames[1:]):
        raise ValueError("drum trajectory changes the mixture-material map; "
                         "weight interpolation is not valid")
    weights = [frame[1] for frame in frames]
    if not any(np.any(weight != weights[0]) for weight in weights[1:]):
        raise ValueError(
            "drum trajectory is unresolved on this mesh; increase --refine "
            "or polar sampling")

    def at(t):
        position = last * min(max(float(t) / duration, 0.0), 1.0)
        lower = min(int(np.floor(position)), last)
        upper = min(lower + 1, last)
        fraction = position - lower
        mix_weight = ((1.0 - fraction) * weights[lower]
                      + fraction * weights[upper])
        return (problem.materials, problem.material_map,
                mix_material, mix_weight)

    return at


def _run(problem, kinetics, steady, frames, duration, steps, scheme,
         adaptive_bdf=None, restart_at_knots=True):
    solver = TransientSolver(
        problem.grid, _problem_at(problem, frames, duration), kinetics,
        bc=problem.bc, active=problem.active, mask_bc=problem.mask_bc,
        mix_material=problem.mix_material, mix_weight=problem.mix_weight,
        group_operator=TriGroupOperator, eig_solver=TriDiffusionEigenSolver,
        precond_degree=1, device="cpu")
    start = time.perf_counter()
    result = solver.solve(
        t_end=duration, dt=duration / steps, initial_steady=steady,
        time_scheme=scheme, step_solver="monolithic", tol_step=1e-8,
        bdf_restart_times=((duration * np.arange(1, len(frames) - 1)
                            / (len(frames) - 1))
                           if restart_at_knots else ()),
        multigroup_kwargs={"scatter_sweeps": 3, "rtol": 1e-9,
                           "inner_rtol": 1e-3, "precond_degree": 1},
        adaptive_bdf=adaptive_bdf)
    return result, time.perf_counter() - start


def benchmark(refine=2, reference_steps=64, candidate_steps=(8, 16),
              schemes=("bdf1", "bdf2", "bdf3", "bdf5"), samples=10,
              control_intervals=4,
              cases=("slow-symmetric", "fast-symmetric", "fast-asymmetric"),
              adaptive_rtols=(), rejection_strategy="half",
              reject_max_factor=0.5, reference_scheme="bdf1",
              restart_at_knots=True, automatic_order=True):
    problem, kinetics = build_case(refine, nz=0, groups="11")
    steady_solver = TriDiffusionEigenSolver(
        problem.grid, problem.materials, problem.material_map,
        bc=problem.bc, active=problem.active, mask_bc=problem.mask_bc,
        mix_material=problem.mix_material, mix_weight=problem.mix_weight,
        precond_degree=1, device="cpu")
    steady = steady_solver.solve(tol_k=1e-11, tol_source=1e-10)
    if not steady.converged:
        raise RuntimeError("initial HP-MR eigenvalue solve did not converge")

    frame_count = control_intervals + 1
    symmetric = _trajectory_frames(
        problem, refine, frame_count, False, samples=samples)
    asymmetric = _trajectory_frames(
        problem, refine, frame_count, True, samples=samples)
    definitions = (("slow-symmetric", 0.8, symmetric),
                   ("fast-symmetric", 0.08, symmetric),
                   ("fast-asymmetric", 0.08, asymmetric))
    selected = set(cases)
    unknown = selected - {row[0] for row in definitions}
    if unknown:
        raise ValueError(f"unknown HP-MR BDF cases: {sorted(unknown)}")
    definitions = tuple(row for row in definitions if row[0] in selected)
    rows = []
    for case, duration, frames in definitions:
        reference, reference_wall = _run(
            problem, kinetics, steady, frames, duration,
            reference_steps, reference_scheme,
            restart_at_knots=restart_at_knots)
        reference_inner = reference.total_inner_iterations
        reference_outer = (sum(reference.step_iterations)
                           + sum(reference.rejected_step_iterations))

        def record(result, wall, steps, scheme, *, adaptive_rtol=None):
            candidate_on_reference = np.interp(
                reference.times, result.times, result.power)
            power_error = float(np.max(
                np.abs(candidate_on_reference - reference.power)
                / np.maximum(np.abs(reference.power), 1e-14)))
            final_power_error = float(abs(
                result.power[-1] - reference.power[-1])
                / max(abs(reference.power[-1]), 1e-14))
            flux_error = float(
                np.linalg.norm(result.flux_numpy - reference.flux_numpy)
                / np.linalg.norm(reference.flux_numpy))
            accepted = len(result.times) - 1
            rows.append(dict(
                case=case, duration=duration, scheme=result.time_scheme,
                reference_scheme=reference.time_scheme,
                restart_at_knots=bool(restart_at_knots),
                automatic_order=bool(automatic_order),
                adaptive=adaptive_rtol is not None,
                adaptive_rtol=adaptive_rtol,
                requested_steps=steps,
                steps=accepted, rejected_steps=result.rejected_steps,
                initial_dt=duration / steps,
                min_dt=min(result.step_widths, default=duration / steps),
                max_dt=max(result.step_widths, default=duration / steps),
                wall=wall, reference_wall=reference_wall,
                speedup=reference_wall / wall,
                reference_steps=len(reference.times) - 1,
                reference_inner=reference_inner,
                reference_outer=reference_outer,
                inner=result.total_inner_iterations,
                outer=(sum(result.step_iterations)
                       + sum(result.rejected_step_iterations)),
                rejected_outer=sum(result.rejected_step_iterations),
                inner_work_ratio=(reference_inner
                                  / max(result.total_inner_iterations, 1)),
                outer_work_ratio=(reference_outer / max(
                    sum(result.step_iterations)
                    + sum(result.rejected_step_iterations), 1)),
                max_power_error=power_error, flux_error=flux_error,
                final_power_error=final_power_error,
                final_power=float(result.power[-1]),
                min_power=float(result.power.min()),
                max_order=max(result.time_orders, default=1),
                order_counts={str(order): result.time_orders.count(order)
                              for order in sorted(set(result.time_orders))}))

        for steps in candidate_steps:
            if reference_steps % steps:
                raise ValueError("reference_steps must be divisible by candidates")
            if steps % control_intervals:
                raise ValueError("candidate steps must resolve every control event")
            for scheme in schemes:
                result, wall = _run(
                    problem, kinetics, steady, frames, duration, steps, scheme,
                    restart_at_knots=restart_at_knots)
                record(result, wall, steps, scheme)
            for rtol in adaptive_rtols:
                for scheme in schemes:
                    result, wall = _run(
                        problem, kinetics, steady, frames, duration, steps,
                        scheme, adaptive_bdf={
                            "rtol": rtol,
                            # Resolve the BDF1 recovery immediately after a
                            # discrete raster/control-frame change.
                            "min_dt": duration / reference_steps / 4096,
                            "max_dt": duration / control_intervals,
                            "automatic_order": (automatic_order
                                                and scheme != "bdf1"),
                            "rejection_strategy": rejection_strategy,
                            "reject_max_factor": reject_max_factor,
                        }, restart_at_knots=restart_at_knots)
                    record(result, wall, steps, scheme, adaptive_rtol=rtol)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--reference-steps", type=int, default=64)
    parser.add_argument("--reference-scheme", choices=("bdf1", "bdf2"),
                        default="bdf1")
    parser.add_argument("--candidate-steps", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--schemes", nargs="+",
                        choices=("bdf1", "bdf2", "bdf3", "bdf4", "bdf5",
                                 "bdf6"),
                        default=["bdf1", "bdf2", "bdf3", "bdf5"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--control-intervals", type=int, default=4)
    parser.add_argument("--cases", nargs="+",
                        choices=("slow-symmetric", "fast-symmetric",
                                 "fast-asymmetric"),
                        default=["slow-symmetric", "fast-symmetric",
                                 "fast-asymmetric"])
    parser.add_argument("--adaptive-rtols", type=float, nargs="+", default=[])
    parser.add_argument("--rejection-strategy", choices=("half", "error"),
                        default="half")
    parser.add_argument("--reject-max-factor", type=float, default=0.5)
    parser.add_argument("--no-knot-restarts", dest="restart_at_knots",
                        action="store_false",
                        help="let error/order control cross continuous "
                             "mixture-interpolation knots")
    parser.set_defaults(restart_at_knots=True)
    parser.add_argument("--fixed-adaptive-order", dest="automatic_order",
                        action="store_false",
                        help="ramp to and retain the requested BDF order")
    parser.set_defaults(automatic_order=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    rows = benchmark(args.refine, args.reference_steps,
                     tuple(args.candidate_steps), schemes=tuple(args.schemes),
                     samples=args.samples,
                     control_intervals=args.control_intervals,
                     cases=tuple(args.cases),
                     adaptive_rtols=tuple(args.adaptive_rtols),
                     rejection_strategy=args.rejection_strategy,
                     reject_max_factor=args.reject_max_factor,
                     reference_scheme=args.reference_scheme,
                     restart_at_knots=args.restart_at_knots,
                     automatic_order=args.automatic_order)
    print(f"{'case':>16} {'BDF':>6} {'N':>4} {'dt':>9} {'wall s':>8} "
          f"{'adapt':>8} {'speedup':>8} {'inner':>8} {'dPmax':>11} {'dPend':>11} "
          f"{'dflux':>11} {'P(end)':>10}")
    for row in rows:
        adaptive = ("fixed" if not row["adaptive"]
                    else f"{row['adaptive_rtol']:.0e}")
        print(f"{row['case']:>16} {row['scheme']:>6} {row['steps']:4d} "
              f"{row['initial_dt']:9.5f} {row['wall']:8.3f} {adaptive:>8} "
              f"{row['speedup']:8.2f} "
              f"{row['inner']:8d} {row['max_power_error']:11.3e} "
              f"{row['final_power_error']:11.3e} "
              f"{row['flux_error']:11.3e} {row['final_power']:10.6f}")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
