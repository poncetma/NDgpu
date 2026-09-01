"""Multi-rank HPMR drum-step transient diffusion gate."""

from __future__ import annotations

import argparse
import json

import numpy as np
from mpi4py import MPI

from ndgpu import (DistributedTriTransientSolver, TransientSolver, asnumpy)
from ndgpu.benchmarks import HPMR_KINETICS, build_hpmr2d
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

    initial = build_hpmr2d(
        refine=args.refine, drum_angle_deg=args.initial_angle,
        absorber="polar")
    perturbed = build_hpmr2d(
        refine=args.refine, drum_angle_deg=args.final_angle,
        absorber="polar")

    def problem_at(time):
        problem = initial if time == 0.0 else perturbed
        return (problem.materials, problem.material_map,
                problem.mix_material, problem.mix_weight)

    common = dict(
        bc=initial.bc, active=initial.active, mask_bc=initial.mask_bc)
    solver = DistributedTriTransientSolver(
        initial.grid, problem_at, HPMR_KINETICS,
        context=context, decomposition="rows", **common)
    result = solver.solve(
        t_end=0.02, dt=0.01, tol_step=1e-6, max_sweeps=300,
        rebalance=True, verbose=context.rank == 0)
    gathered_flux = result.gather_flux(root=0)
    gathered_precursors = result.gather_precursors(root=0)
    telemetry = communicator.gather({
        "rank": context.rank,
        "hostname": context.hostname,
        "device": context.device_identity,
        "owned_range": [solver.partition.start, solver.partition.stop],
        "local_shape": solver.partition.local_shape,
        "step_iterations": result.step_iterations,
        "total_inner_iterations": result.total_inner_iterations,
    }, root=0)

    passed = True
    failure = ""
    summary = None
    if context.rank == 0:
        try:
            reference = TransientSolver(
                initial.grid, problem_at, HPMR_KINETICS,
                device=args.device, group_operator=TriGroupOperator,
                eig_solver=TriDiffusionEigenSolver, **common).solve(
                    t_end=0.02, dt=0.01, tol_step=1e-6,
                    max_sweeps=300, rebalance=True)
            distributed_flux = np.asarray(asnumpy(gathered_flux))
            distributed_precursors = np.asarray(asnumpy(gathered_precursors))
            reference_flux = np.asarray(asnumpy(reference.flux))
            reference_precursors = np.asarray(asnumpy(reference.precursors))
            power_error = float(np.max(np.abs(result.power - reference.power)))
            flux_error = relative_l2(distributed_flux, reference_flux)
            precursor_error = relative_l2(
                distributed_precursors, reference_precursors)
            if power_error > 2e-7:
                raise AssertionError(f"power error {power_error} exceeds 2e-7")
            if flux_error > 2e-7 or precursor_error > 2e-7:
                raise AssertionError(
                    "final flux or precursor relative error exceeds 2e-7")
            summary = {
                "status": "passed",
                "mpi_size": context.size,
                "device": args.device,
                "communication_mode": context.communication_mode,
                "refine": args.refine,
                "global_shape": initial.grid.shape,
                "initial_angle_deg": args.initial_angle,
                "final_angle_deg": args.final_angle,
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
                "rank_telemetry": telemetry,
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
