"""Long-transient performance benchmark for row-decomposed HPMR diffusion."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

from mpi4py import MPI

from ndgpu import (DistributedTriDiffusionEigenSolver,
                   DistributedTriTransientSolver)
from ndgpu.benchmarks import HPMR_KINETICS, build_hpmr2d, build_hpmr3d
from ndgpu.benchmarks.hpmr_thermal import (hpmr_endfb8_builtin,
                                           hpmr_kinetics_11g)
from ndgpu.distributed import DistributedContext


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    parser.add_argument("--refine", type=int, default=4)
    parser.add_argument("--nz", type=int, default=10)
    parser.add_argument("--groups", choices=("2", "11"), default="11")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--initial-angle", type=float, default=120.0)
    parser.add_argument("--final-angle", type=float, default=110.0)
    parser.add_argument("--tol-step", type=float, default=1e-8)
    parser.add_argument("--max-sweeps", type=int, default=1000)
    parser.add_argument("--precond-degree", type=int, default=0)
    parser.add_argument("--check-every", type=int, default=1)
    parser.add_argument("--single-reduction", action="store_true")
    parser.add_argument("--batched-halos", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.dt <= 0.0:
        raise ValueError("steps and dt must be positive")

    communicator = MPI.COMM_WORLD
    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication,
        batched_halos=args.batched_halos)
    three_d = args.nz > 0
    materials = (hpmr_endfb8_builtin(three_d=three_d)
                 if args.groups == "11" else None)
    kinetics = hpmr_kinetics_11g() if args.groups == "11" else HPMR_KINETICS
    builder = build_hpmr3d if three_d else build_hpmr2d
    build_kwargs = {
        "refine": args.refine,
        "absorber": "polar",
        "materials": materials,
    }
    if three_d:
        build_kwargs["nz"] = args.nz
    initial = builder(drum_angle_deg=args.initial_angle, **build_kwargs)
    perturbed = builder(drum_angle_deg=args.final_angle, **build_kwargs)

    def problem_at(time):
        problem = initial if time == 0.0 else perturbed
        return (problem.materials, problem.material_map,
                problem.mix_material, problem.mix_weight)

    common = dict(
        bc=initial.bc, active=initial.active, mask_bc=initial.mask_bc,
        precond_degree=args.precond_degree)
    communicator.Barrier()
    steady_started = perf_counter()
    steady_solver = DistributedTriDiffusionEigenSolver(
        initial.grid, initial.materials, initial.material_map,
        mix_material=initial.mix_material, mix_weight=initial.mix_weight,
        context=context, decomposition="rows", **common)
    steady = steady_solver.solve(
        tol_k=1e-10, tol_source=1e-9, inner_rtol_floor=1e-12,
        verbose=context.rank == 0)
    communicator.Barrier()
    steady_seconds = communicator.reduce(
        perf_counter() - steady_started, op=MPI.MAX, root=0)

    transient_solver = DistributedTriTransientSolver(
        initial.grid, problem_at, kinetics,
        context=context, decomposition="rows", **common)
    context.reset_communication_stats()
    communicator.Barrier()
    transient_started = perf_counter()
    result = transient_solver.solve(
        t_end=args.steps * args.dt, dt=args.dt,
        initial_steady=steady, tol_step=args.tol_step,
        max_sweeps=args.max_sweeps, anderson_depth=1, rebalance=True,
        linsolve_kwargs={
            "check_every": args.check_every,
            "single_reduction": args.single_reduction,
            "batched_halos": args.batched_halos,
        },
        verbose=context.rank == 0)
    communicator.Barrier()
    transient_seconds = communicator.reduce(
        perf_counter() - transient_started, op=MPI.MAX, root=0)

    rank_stats = communicator.gather({
        "rank": context.rank,
        "hostname": context.hostname,
        "device": context.device_identity,
        "owned_range": [transient_solver.partition.start,
                        transient_solver.partition.stop],
        "local_shape": transient_solver.partition.local_shape,
        **context.communication_stats(),
    }, root=0)
    if context.rank == 0:
        summary = {
            "status": "completed",
            "mpi_size": context.size,
            "device": args.device,
            "communication_mode": context.communication_mode,
            "global_shape": initial.grid.shape,
            "groups": args.groups,
            "steps": args.steps,
            "dt": args.dt,
            "tol_step": args.tol_step,
            "precond_degree": args.precond_degree,
            "check_every": args.check_every,
            "single_reduction": args.single_reduction,
            "k0": result.k0,
            "initial_state_reused": result.initial_state_reused,
            "steady_seconds": steady_seconds,
            "transient_seconds": transient_seconds,
            "seconds_per_step": transient_seconds / args.steps,
            "final_power": float(result.power[-1]),
            "step_iterations": result.step_iterations,
            "total_inner_iterations": result.total_inner_iterations,
            "rank_communication": rank_stats,
        }
        print("NDGPU_DISTRIBUTED_TRANSIENT_PERFORMANCE " +
              json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
