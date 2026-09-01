"""Multi-rank Cartesian diffusion eigenvalue domain-decomposition gate."""

from __future__ import annotations

import argparse
import json

import numpy as np
from mpi4py import MPI

from ndgpu import (DiffusionEigenSolver, DistributedDiffusionEigenSolver, Grid,
                   PWR_TWO_GROUP, asnumpy, k_bare_box)
from ndgpu.distributed import DistributedContext


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    parser.add_argument("--shape", default="17,13,9")
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=0)
    return parser.parse_args()


def normalized_flux(flux):
    norm = np.linalg.norm(flux.ravel())
    if norm == 0.0:
        raise AssertionError("zero flux cannot be normalized")
    return flux / norm


def main():
    args = parse_args()
    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 3:
        raise ValueError("--shape must contain three comma-separated dimensions")

    communicator = MPI.COMM_WORLD
    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication)
    if context.size < 2:
        raise RuntimeError("eigenvalue gate requires at least two MPI ranks")

    grid = Grid(shape, (90.0, 70.0, 50.0))
    solver = DistributedDiffusionEigenSolver(
        grid, PWR_TWO_GROUP, context=context, decomposition="slab",
        axis=args.axis)
    local_shape = solver.partition.local_shape
    if solver.grid.shape != local_shape:
        raise AssertionError("solver grid is not rank-local")
    field_arrays = (
        solver.fields.diffusion + solver.fields.removal +
        solver.fields.nu_sigma_f + solver.fields.chi)
    if any(tuple(field.shape) != local_shape for field in field_arrays):
        raise AssertionError("a cross-section field is not rank-local")
    if any(operator.shape != local_shape for operator in solver.ops):
        raise AssertionError("an operator is not rank-local")

    result = solver.solve(
        tol_k=1e-9, tol_source=1e-8, inner_rtol_floor=1e-11,
        verbose=context.rank == 0)
    gathered_flux = result.gather_flux(root=0)
    telemetry = communicator.gather({
        "rank": context.rank,
        "hostname": context.hostname,
        "device": context.device_identity,
        "owned_range": [solver.partition.start, solver.partition.stop],
        "local_shape": local_shape,
        "local_flux_cells": int(result.local_flux.size),
        "outer_iterations": result.outer_iterations,
        "inner_iterations": result.inner_iterations,
    }, root=0)

    passed = True
    failure = ""
    summary = None
    if context.rank == 0:
        try:
            reference = DiffusionEigenSolver(
                grid, PWR_TWO_GROUP, device=args.device).solve(
                    tol_k=1e-9, tol_source=1e-8,
                    inner_rtol_floor=1e-11)
            gathered_host = np.asarray(asnumpy(gathered_flux))
            reference_host = np.asarray(asnumpy(reference.flux))
            flux_error = normalized_flux(gathered_host) - normalized_flux(
                reference_host)
            relative_l2 = float(np.linalg.norm(flux_error.ravel()))
            maximum_flux_error = float(np.max(np.abs(flux_error)))
            eigenvalue_error = abs(result.k_eff - reference.k_eff)
            if not result.converged or not reference.converged:
                raise AssertionError("distributed or serial solve did not converge")
            if eigenvalue_error > 2e-9:
                raise AssertionError(
                    f"eigenvalue difference {eigenvalue_error} exceeds 2e-9")
            if relative_l2 > 2e-8:
                raise AssertionError(
                    f"normalized flux L2 difference {relative_l2} exceeds 2e-8")
            summary = {
                "status": "passed",
                "mpi_size": context.size,
                "device": args.device,
                "communication_mode": context.communication_mode,
                "global_shape": shape,
                "partition_axis": args.axis,
                "distributed_k_eff": result.k_eff,
                "serial_k_eff": reference.k_eff,
                "analytic_k_eff": k_bare_box(PWR_TWO_GROUP, grid.size),
                "eigenvalue_error": eigenvalue_error,
                "normalized_flux_l2_error": relative_l2,
                "normalized_flux_max_error": maximum_flux_error,
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
        print("NDGPU_DISTRIBUTED_EIGEN_GATE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
