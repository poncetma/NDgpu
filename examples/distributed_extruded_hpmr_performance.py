"""Throughput benchmark for axially decomposed locally refined HP-MR."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

from mpi4py import MPI

from ndgpu import (DistributedContext,
                   DistributedExtrudedMeshDiffusionEigenSolver,
                   DistributedExtrudedMeshTransientSolver)
from ndgpu.benchmarks import (HPMR_KINETICS, build_hpmr3d_local,
                              with_hpmr3d_local_drum_angle)
from ndgpu.benchmarks.hpmr_thermal import (hpmr_endfb8_builtin,
                                           hpmr_kinetics_11g)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    parser.add_argument("--refine", type=int, default=8)
    parser.add_argument("--local-levels", type=int, default=3)
    parser.add_argument("--nz", type=int, default=10)
    parser.add_argument("--groups", choices=("2", "11"), default="11")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--initial-angle", type=float, default=90.0)
    parser.add_argument("--final-angle", type=float, default=93.88)
    parser.add_argument(
        "--drum-motion", choices=("step", "linear-ramp"), default="step")
    parser.add_argument("--tol-step", type=float, default=1e-7)
    parser.add_argument("--max-sweeps", type=int, default=1000)
    parser.add_argument("--precond-degree", type=int, default=0)
    parser.add_argument("--scatter-subsweeps", type=int, default=0)
    parser.add_argument("--check-every", type=int, default=1)
    parser.add_argument("--single-reduction", action="store_true")
    parser.add_argument("--batched-halos", action="store_true")
    parser.add_argument(
        "--step-solver", choices=("fixed-point", "monolithic"),
        default="fixed-point")
    parser.add_argument("--multigroup-scatter-sweeps", type=int, default=3)
    parser.add_argument("--multigroup-inner-rtol", type=float, default=1e-3)
    parser.add_argument("--multigroup-energy-anderson", type=int, default=0)
    parser.add_argument(
        "--multigroup-inner-fixed-relaxations", type=int, default=0)
    parser.add_argument(
        "--multigroup-inner-fixed-iterations", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.dt <= 0.0:
        raise ValueError("steps and dt must be positive")

    communicator = MPI.COMM_WORLD
    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication,
        batched_halos=args.batched_halos)
    if context.size > args.nz:
        raise ValueError("MPI rank count cannot exceed axial layer count")

    communicator.Barrier()
    setup_started = perf_counter()
    materials = (hpmr_endfb8_builtin(three_d=True)
                 if args.groups == "11" else None)
    kinetics = hpmr_kinetics_11g() if args.groups == "11" else HPMR_KINETICS
    build = dict(
        refine=args.refine, nz=args.nz,
        drum_refine_levels=args.local_levels,
        materials=materials, absorber="polar", samples=0)
    initial = build_hpmr3d_local(
        drum_angle_deg=args.initial_angle, **build)
    perturbed = None
    if args.drum_motion == "step":
        perturbed = with_hpmr3d_local_drum_angle(
            initial, args.final_angle, samples=build["samples"])
    communicator.Barrier()
    setup_seconds = communicator.reduce(
        perf_counter() - setup_started, op=MPI.MAX, root=0)
    if context.rank == 0:
        print("NDGPU_DISTRIBUTED_EXTRUDED_SETUP " + json.dumps({
            "mpi_size": context.size,
            "radial_cells": initial.grid.mesh.n_cells,
            "spatial_cells": initial.grid.n_cells,
            "setup_seconds": setup_seconds,
        }), flush=True)

    problem_update_seconds = 0.0
    problem_update_calls = 0
    cached_time = 0.0
    cached_problem = initial

    def problem_at(time):
        nonlocal problem_update_seconds, problem_update_calls
        nonlocal cached_time, cached_problem
        if time == 0.0:
            problem = initial
        elif args.drum_motion == "step":
            problem = perturbed
        elif time == cached_time:
            problem = cached_problem
        else:
            update_started = perf_counter()
            progress = min(max(time / (args.steps * args.dt), 0.0), 1.0)
            angle = (args.initial_angle
                     + progress * (args.final_angle - args.initial_angle))
            problem = with_hpmr3d_local_drum_angle(
                initial, angle, samples=build["samples"])
            problem_update_seconds += perf_counter() - update_started
            problem_update_calls += 1
            cached_time = time
            cached_problem = problem
        return (problem.materials, problem.material_map,
                problem.mix_material, problem.mix_weight)

    common = dict(
        bc=initial.bc, active=initial.active, mask_bc=initial.mask_bc,
        precond_degree=args.precond_degree)
    communicator.Barrier()
    steady_started = perf_counter()
    steady_solver = DistributedExtrudedMeshDiffusionEigenSolver(
        initial.grid, initial.materials, initial.material_map,
        mix_material=initial.mix_material, mix_weight=initial.mix_weight,
        context=context, decomposition="axial", **common)
    steady = steady_solver.solve(
        tol_k=1e-9, tol_source=1e-8, inner_rtol_floor=1e-11,
        verbose=context.rank == 0)
    communicator.Barrier()
    steady_seconds = communicator.reduce(
        perf_counter() - steady_started, op=MPI.MAX, root=0)

    transient_solver = DistributedExtrudedMeshTransientSolver(
        initial.grid, problem_at, kinetics,
        context=context, decomposition="axial", **common)
    context.reset_communication_stats()
    communicator.Barrier()
    transient_started = perf_counter()
    step_kwargs = dict(
        t_end=args.steps * args.dt, dt=args.dt,
        initial_steady=steady, tol_step=args.tol_step,
        max_sweeps=args.max_sweeps, verbose=context.rank == 0)
    if args.step_solver == "fixed-point":
        step_kwargs.update(
            anderson_depth=1, rebalance=True,
            scatter_subsweeps=args.scatter_subsweeps,
            linsolve_kwargs={
                "check_every": args.check_every,
                "single_reduction": args.single_reduction,
            })
    else:
        step_kwargs.update(
            step_solver="monolithic",
            multigroup_kwargs={
                "scatter_sweeps": args.multigroup_scatter_sweeps,
                "inner_rtol": args.multigroup_inner_rtol,
                "energy_anderson": args.multigroup_energy_anderson,
                "inner_fixed_relaxations": (
                    args.multigroup_inner_fixed_relaxations),
                "inner_fixed_iterations": (
                    args.multigroup_inner_fixed_iterations),
                "precond_degree": args.precond_degree,
                "rtol": max(0.1 * args.tol_step, 1e-12),
            })
    result = transient_solver.solve(**step_kwargs)
    communicator.Barrier()
    transient_seconds = communicator.reduce(
        perf_counter() - transient_started, op=MPI.MAX, root=0)

    rank_stats = communicator.gather({
        "rank": context.rank,
        "hostname": context.hostname,
        "device": context.device_identity,
        "owned_layers": [transient_solver.partition.start,
                         transient_solver.partition.stop],
        "local_shape": transient_solver.partition.local_shape,
        "problem_update_calls": problem_update_calls,
        "problem_update_seconds": problem_update_seconds,
        **context.communication_stats(),
    }, root=0)
    if context.rank == 0:
        groups = initial.materials[0].n_groups
        summary = {
            "status": "completed",
            "mpi_size": context.size,
            "device": args.device,
            "communication_mode": context.communication_mode,
            "global_refine": args.refine,
            "local_levels": args.local_levels,
            "effective_drum_refine": initial.drum_refine,
            "radial_cells": initial.grid.mesh.n_cells,
            "nz": args.nz,
            "spatial_cells": initial.grid.n_cells,
            "groups": groups,
            "unknowns": initial.grid.n_cells * groups,
            "steps": args.steps,
            "dt": args.dt,
            "drum_motion": args.drum_motion,
            "tol_step": args.tol_step,
            "step_solver": args.step_solver,
            "precond_degree": args.precond_degree,
            "scatter_subsweeps": (args.scatter_subsweeps
                                    if args.scatter_subsweeps else 6),
            "check_every": args.check_every,
            "single_reduction": args.single_reduction,
            "batched_halos": args.batched_halos,
            "multigroup_scatter_sweeps": args.multigroup_scatter_sweeps,
            "multigroup_inner_rtol": args.multigroup_inner_rtol,
            "multigroup_energy_anderson": args.multigroup_energy_anderson,
            "multigroup_inner_fixed_relaxations": (
                args.multigroup_inner_fixed_relaxations),
            "multigroup_inner_fixed_iterations": (
                args.multigroup_inner_fixed_iterations),
            "k0": result.k0,
            "initial_state_reused": result.initial_state_reused,
            "setup_seconds": setup_seconds,
            "steady_seconds": steady_seconds,
            "transient_seconds": transient_seconds,
            "seconds_per_step": transient_seconds / args.steps,
            "final_power": float(result.power[-1]),
            "step_iterations": result.step_iterations,
            "total_inner_iterations": result.total_inner_iterations,
            "maximum_problem_update_seconds": max(
                item["problem_update_seconds"] for item in rank_stats),
            "rank_communication": rank_stats,
        }
        print("NDGPU_DISTRIBUTED_EXTRUDED_PERFORMANCE "
              + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
