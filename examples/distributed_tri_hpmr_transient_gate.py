"""Multi-rank HPMR drum-step transient diffusion gate."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
from mpi4py import MPI

from ndgpu import (DistributedTriTransientSolver, TransientSolver, asnumpy)
from ndgpu.benchmarks import HPMR_KINETICS, build_hpmr2d, build_hpmr3d
from ndgpu.benchmarks.hpmr_thermal import (hpmr_endfb8_builtin,
                                           hpmr_kinetics_11g)
from ndgpu.distributed import DistributedContext
from ndgpu.tri import TriDiffusionEigenSolver, TriGroupOperator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--nz", type=int, default=0)
    parser.add_argument("--groups", choices=("2", "11"), default="2")
    parser.add_argument("--tol-step", type=float, default=1e-10)
    parser.add_argument("--max-sweeps", type=int, default=1000)
    parser.add_argument("--anderson-depth", type=int, default=1)
    parser.add_argument("--initial-angle", type=float, default=120.0)
    parser.add_argument("--final-angle", type=float, default=110.0)
    return parser.parse_args()


def relative_l2(actual, expected):
    return float(np.linalg.norm((actual - expected).ravel()) /
                 np.linalg.norm(expected.ravel()))


def main():
    args = parse_args()
    communicator = MPI.COMM_WORLD
    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication)
    if context.size < 2:
        raise RuntimeError("transient gate requires at least two MPI ranks")

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
        bc=initial.bc, active=initial.active, mask_bc=initial.mask_bc)
    solver = DistributedTriTransientSolver(
        initial.grid, problem_at, kinetics,
        context=context, decomposition="rows", **common)
    communicator.Barrier()
    distributed_start = perf_counter()
    result = solver.solve(
        t_end=0.02, dt=0.01, tol_step=args.tol_step,
        max_sweeps=args.max_sweeps, anderson_depth=args.anderson_depth,
        rebalance=True, verbose=context.rank == 0,
        steady_kwargs={
            "tol_k": 1e-10,
            "tol_source": 1e-9,
            "inner_rtol_floor": 1e-12,
            "verbose": context.rank == 0,
        })
    communicator.Barrier()
    distributed_seconds = communicator.reduce(
        perf_counter() - distributed_start, op=MPI.MAX, root=0)
    gathered_flux = result.gather_flux(root=0)
    gathered_precursors = result.gather_precursors(root=0)
    owned = solver.partition.owned_slice
    telemetry = communicator.gather({
        "rank": context.rank,
        "hostname": context.hostname,
        "device": context.device_identity,
        "owned_range": [solver.partition.start, solver.partition.stop],
        "local_shape": solver.partition.local_shape,
        "active_cells": int(np.count_nonzero(initial.active[owned])),
        "mixed_cells": int(np.count_nonzero(initial.mix_weight[owned])),
        "step_iterations": result.step_iterations,
        "total_inner_iterations": result.total_inner_iterations,
    }, root=0)

    passed = True
    failure = ""
    summary = None
    if context.rank == 0:
        try:
            serial_start = perf_counter()
            reference = TransientSolver(
                initial.grid, problem_at, kinetics,
                device=args.device, group_operator=TriGroupOperator,
                eig_solver=TriDiffusionEigenSolver, **common).solve(
                    t_end=0.02, dt=0.01, tol_step=args.tol_step,
                    max_sweeps=args.max_sweeps,
                    anderson_depth=args.anderson_depth, rebalance=True,
                    steady_kwargs={
                        "tol_k": 1e-10,
                        "tol_source": 1e-9,
                        "inner_rtol_floor": 1e-12,
                        "verbose": True,
                    })
            serial_seconds = perf_counter() - serial_start
            distributed_flux = np.asarray(asnumpy(gathered_flux))
            distributed_precursors = np.asarray(asnumpy(gathered_precursors))
            reference_flux = np.asarray(asnumpy(reference.flux))
            reference_precursors = np.asarray(asnumpy(reference.precursors))
            power_error = float(np.max(np.abs(result.power - reference.power)))
            flux_error = relative_l2(distributed_flux, reference_flux)
            precursor_error = relative_l2(
                distributed_precursors, reference_precursors)
            comparison = {
                "mpi_size": context.size,
                "device": args.device,
                "communication_mode": context.communication_mode,
                "refine": args.refine,
                "nz": args.nz,
                "groups": args.groups,
                "tol_step": args.tol_step,
                "anderson_depth": args.anderson_depth,
                "global_shape": initial.grid.shape,
                "distributed_k0": result.k0,
                "serial_k0": reference.k0,
                "initial_eigenvalue_error": abs(result.k0 - reference.k0),
                "times": result.times.tolist(),
                "distributed_power": result.power.tolist(),
                "serial_power": reference.power.tolist(),
                "maximum_power_error": power_error,
                "final_flux_relative_l2_error": flux_error,
                "final_precursor_relative_l2_error": precursor_error,
                "distributed_step_iterations": result.step_iterations,
                "serial_step_iterations": reference.step_iterations,
                "distributed_inner_iterations": result.total_inner_iterations,
                "serial_inner_iterations": reference.total_inner_iterations,
                "distributed_seconds": distributed_seconds,
                "serial_seconds": serial_seconds,
                "rank_telemetry": telemetry,
            }
            print("NDGPU_DISTRIBUTED_TRI_TRANSIENT_COMPARISON " +
                  json.dumps(comparison), flush=True)
            if power_error > 2e-7:
                raise AssertionError(f"power error {power_error} exceeds 2e-7")
            if flux_error > 2e-7 or precursor_error > 2e-7:
                raise AssertionError(
                    "final flux or precursor relative error exceeds 2e-7")
            summary = {
                **comparison,
                "status": "passed",
                "initial_angle_deg": args.initial_angle,
                "final_angle_deg": args.final_angle,
            }
        except Exception as exc:
            passed = False
            failure = f"{type(exc).__name__}: {exc}"

    passed, failure = communicator.bcast((passed, failure), root=0)
    if not passed:
        raise AssertionError(f"root transient comparison failed: {failure}")
    if context.rank == 0:
        print("NDGPU_DISTRIBUTED_TRI_TRANSIENT_GATE " + json.dumps(summary),
              flush=True)


if __name__ == "__main__":
    main()
