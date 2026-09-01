"""Multi-rank triangular HPMR diffusion eigenvalue gate."""

from __future__ import annotations

import argparse
import json

import numpy as np
from mpi4py import MPI

from ndgpu import (DistributedTriDiffusionEigenSolver,
                   TriDiffusionEigenSolver, asnumpy)
from ndgpu.benchmarks import build_hpmr2d
from ndgpu.distributed import DistributedContext


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    parser.add_argument("--refine", type=int, default=4)
    parser.add_argument("--angle", type=float, default=120.0)
    return parser.parse_args()


def normalized(flux):
    norm = np.linalg.norm(flux.ravel())
    return flux / norm


def main():
    args = parse_args()
    communicator = MPI.COMM_WORLD
    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication)
    if context.size < 2:
        raise RuntimeError("triangular HPMR gate requires at least two MPI ranks")

    problem = build_hpmr2d(
        refine=args.refine, drum_angle_deg=args.angle, absorber="polar")
    solver = DistributedTriDiffusionEigenSolver(
        problem.grid, problem.materials, problem.material_map,
        active=problem.active, mask_bc=problem.mask_bc,
        mix_material=problem.mix_material, mix_weight=problem.mix_weight,
        context=context, decomposition="rows")
    local_shape = solver.partition.local_shape
    if solver.grid.shape != local_shape:
        raise AssertionError("solver grid is not rank-local")
    fields = (solver.fields.diffusion + solver.fields.removal +
              solver.fields.nu_sigma_f + solver.fields.chi)
    if any(tuple(field.shape) != local_shape for field in fields):
        raise AssertionError("a triangular cross-section field is not rank-local")

    result = solver.solve(
        tol_k=1e-8, tol_source=1e-7, inner_rtol_floor=1e-10,
        verbose=context.rank == 0)
    gathered_flux = result.gather_flux(root=0)
    telemetry = communicator.gather({
        "rank": context.rank,
        "hostname": context.hostname,
        "device": context.device_identity,
        "owned_range": [solver.partition.start, solver.partition.stop],
        "local_shape": local_shape,
        "active_cells": int(np.asarray(asnumpy(solver.active)).sum()),
        "mixed_cells": int(np.asarray(
            asnumpy(solver.fields.blend.active_mix)).sum()),
        "outer_iterations": result.outer_iterations,
        "inner_iterations": result.inner_iterations,
    }, root=0)

    passed = True
    failure = ""
    summary = None
    if context.rank == 0:
        try:
            reference = TriDiffusionEigenSolver(
                problem.grid, problem.materials, problem.material_map,
                active=problem.active, mask_bc=problem.mask_bc,
                mix_material=problem.mix_material,
                mix_weight=problem.mix_weight,
                device=args.device).solve(
                    tol_k=1e-8, tol_source=1e-7,
                    inner_rtol_floor=1e-10)
            distributed_flux = normalized(np.asarray(asnumpy(gathered_flux)))
            serial_flux = normalized(np.asarray(asnumpy(reference.flux)))
            difference = distributed_flux - serial_flux
            k_error = abs(result.k_eff - reference.k_eff)
            flux_l2 = float(np.linalg.norm(difference.ravel()))
            flux_max = float(np.max(np.abs(difference)))
            if not result.converged or not reference.converged:
                raise AssertionError("distributed or serial HPMR solve did not converge")
            if k_error > 2e-8:
                raise AssertionError(f"eigenvalue difference {k_error} exceeds 2e-8")
            if flux_l2 > 2e-7:
                raise AssertionError(f"flux L2 difference {flux_l2} exceeds 2e-7")
            summary = {
                "status": "passed",
                "mpi_size": context.size,
                "device": args.device,
                "communication_mode": context.communication_mode,
                "refine": args.refine,
                "drum_angle_deg": args.angle,
                "global_shape": problem.grid.shape,
                "distributed_k_eff": result.k_eff,
                "serial_k_eff": reference.k_eff,
                "eigenvalue_error": k_error,
                "normalized_flux_l2_error": flux_l2,
                "normalized_flux_max_error": flux_max,
                "distributed_outer_iterations": result.outer_iterations,
                "distributed_inner_iterations": result.inner_iterations,
                "serial_outer_iterations": reference.outer_iterations,
                "serial_inner_iterations": reference.inner_iterations,
                "rank_telemetry": telemetry,
            }
        except Exception as exc:
            passed = False
            failure = f"{type(exc).__name__}: {exc}"

    passed, failure = communicator.bcast((passed, failure), root=0)
    if not passed:
        raise AssertionError(f"root comparison failed: {failure}")
    if context.rank == 0:
        print("NDGPU_DISTRIBUTED_TRI_HPMR_GATE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
