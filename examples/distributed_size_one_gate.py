"""Phase 1 acceptance gate under a real size-one MPI communicator."""

from __future__ import annotations

import argparse
import json

import numpy as np
from mpi4py import MPI

from ndgpu import (DiffusionEigenSolver, DistributedDiffusionEigenSolver,
                   DistributedTriDiffusionEigenSolver, Grid, PWR_TWO_GROUP,
                   TriDiffusionEigenSolver, TriGrid, asnumpy)
from ndgpu.distributed import DistributedContext


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    return parser.parse_args()


def assert_same(reference, distributed, label):
    if reference.k_eff != distributed.k_eff:
        raise AssertionError(
            f"{label} eigenvalue differs: "
            f"{reference.k_eff} != {distributed.k_eff}")
    if reference.k_history != distributed.k_history:
        raise AssertionError(f"{label} eigenvalue history differs")
    if reference.source_error_history != distributed.source_error_history:
        raise AssertionError(f"{label} source-error history differs")
    if reference.outer_iterations != distributed.outer_iterations:
        raise AssertionError(f"{label} outer iteration count differs")
    if reference.inner_iterations != distributed.inner_iterations:
        raise AssertionError(f"{label} inner iteration count differs")
    np.testing.assert_array_equal(
        asnumpy(reference.flux), asnumpy(distributed.local_flux))
    if distributed.gather_flux(root=0) is not distributed.local_flux:
        raise AssertionError(f"{label} size-one gather did not preserve ownership")


def main():
    args = parse_args()
    communicator = MPI.COMM_WORLD
    if communicator.Get_size() != 1:
        raise RuntimeError(
            f"Phase 1 gate requires exactly one MPI rank, got "
            f"{communicator.Get_size()}")

    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication)

    grid = Grid(shape=(5, 4, 3), size=(50.0, 40.0, 30.0))
    cartesian_reference = DiffusionEigenSolver(
        grid, PWR_TWO_GROUP, device=args.device).solve(
            tol_k=1e-9, tol_source=1e-8, verbose=True)
    cartesian = DistributedDiffusionEigenSolver(
        grid, PWR_TWO_GROUP, context=context, decomposition="slab").solve(
            tol_k=1e-9, tol_source=1e-8, verbose=True)
    assert_same(cartesian_reference, cartesian, "Cartesian")

    tri_grid = TriGrid(shape=(5, 4, 2), side=2.0)
    triangular_reference = TriDiffusionEigenSolver(
        tri_grid, PWR_TWO_GROUP, device=args.device).solve(
            tol_k=1e-9, tol_source=1e-8, verbose=True)
    triangular = DistributedTriDiffusionEigenSolver(
        tri_grid, PWR_TWO_GROUP, context=context, decomposition="rows").solve(
            tol_k=1e-9, tol_source=1e-8, verbose=True)
    assert_same(triangular_reference, triangular, "triangular")

    summary = {
        "status": "passed",
        "mpi_size": context.size,
        "mpi_rank": context.rank,
        "mpi_library": context.mpi_library_version,
        "device": args.device,
        "device_identity": context.device_identity,
        "communication_mode": context.communication_mode,
        "cartesian_k_eff": cartesian.k_eff,
        "cartesian_outer_iterations": cartesian.outer_iterations,
        "cartesian_inner_iterations": cartesian.inner_iterations,
        "triangular_k_eff": triangular.k_eff,
        "triangular_outer_iterations": triangular.outer_iterations,
        "triangular_inner_iterations": triangular.inner_iterations,
    }
    print("NDGPU_PHASE1_GATE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
