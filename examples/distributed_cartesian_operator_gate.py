"""Phase 2 Cartesian operator gate under a real MPI communicator."""

from __future__ import annotations

import argparse
import json

import numpy as np
from mpi4py import MPI

from ndgpu import (CartesianSlabPartition, DistributedCartesianGroupOperator,
                   DistributedContext, Grid)
from ndgpu.stencil import GroupOperator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    parser.add_argument("--axis", type=int, choices=(0, 1, 2), default=0)
    return parser.parse_args()


def fields(shape, axis, start=0):
    index = np.indices(shape, dtype=np.int64)
    index[axis] += start
    i, j, k = index
    diffusion = 0.2 + ((7 * i + 3 * j + 5 * k) % 23) / 19.0
    removal = 0.015 + ((2 * i + 11 * j + k) % 17) / 80.0
    flux = 0.1 + ((13 * i + 5 * j + 2 * k) % 29) / 31.0
    active = ((i + 2 * j + 3 * k) % 11) != 0
    return diffusion, removal, flux, active


def main():
    args = parse_args()
    communicator = MPI.COMM_WORLD
    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication)
    if context.size < 2:
        raise RuntimeError("Phase 2 operator gate requires at least two MPI ranks")

    shape = (11, 7, 5)
    grid = Grid(shape, (18.7, 13.3, 9.5))
    partition = CartesianSlabPartition.create(
        shape, context.rank, context.size, axis=args.axis)
    diffusion, removal, flux, active = fields(
        partition.local_shape, args.axis, partition.start)
    xp = context.xp
    local_diffusion = xp.asarray(diffusion)
    local_removal = xp.asarray(removal)
    local_flux = xp.asarray(flux)
    local_active = xp.asarray(active)
    bc = (("vacuum", "zero-flux"),
          ("reflective", 0.2),
          ("zero-flux", "vacuum"))

    operator = DistributedCartesianGroupOperator(
        xp, grid, local_diffusion, local_removal, partition, context,
        active=local_active, bc=bc, mask_bc="vacuum")
    local_applied = operator.apply(local_flux)
    gathered = context.gather_spatial(local_applied, partition, root=0)
    gathered_diag = context.gather_spatial(operator.diag, partition, root=0)

    if context.rank == 0:
        global_fields = fields(shape, args.axis)
        reference = GroupOperator(
            np, grid, global_fields[0], global_fields[1],
            active=global_fields[3], bc=bc, mask_bc="vacuum")
        expected = reference.apply(global_fields[2])
        np.testing.assert_allclose(
            gathered_diag, reference.diag, rtol=2e-15, atol=2e-15)
        np.testing.assert_allclose(
            gathered, expected, rtol=3e-15, atol=3e-15)
        summary = {
            "status": "passed",
            "mpi_size": context.size,
            "axis": args.axis,
            "device": args.device,
            "communication_mode": context.communication_mode,
            "max_operator_error": float(np.max(np.abs(gathered - expected))),
            "max_diagonal_error": float(
                np.max(np.abs(gathered_diag - reference.diag))),
            "global_shape": shape,
            "rank_local_shapes": communicator.gather(
                partition.local_shape, root=0),
        }
        print("NDGPU_PHASE2_OPERATOR_GATE " + json.dumps(summary), flush=True)
    else:
        communicator.gather(partition.local_shape, root=0)


if __name__ == "__main__":
    main()
