"""Exercise NDgpu's MPI reductions and neighbor exchange on CPU or GPU.

Run this under ``mpiexec`` or ``srun`` with at least two ranks.  GPU mode uses
explicit host staging by default; request ``--communication cuda-aware`` only
when validating a site MPI build that is expected to accept CUDA buffers.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from mpi4py import MPI

from ndgpu.backend import asnumpy, device_name, synchronize
from ndgpu.distributed import DistributedContext


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.elements < 1 or args.iterations < 1:
        raise ValueError("elements and iterations must be positive")

    communicator = MPI.COMM_WORLD
    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication)
    if context.size < 2:
        raise RuntimeError("the MPI environment probe requires at least two ranks")

    xp = context.xp
    source = (context.rank - 1) % context.size
    destination = (context.rank + 1) % context.size
    values = xp.arange(args.elements, dtype=xp.float64) + context.rank
    expected = np.arange(args.elements, dtype=np.float64) + source

    received = context.sendrecv(
        values, destination=destination, source=source, tag=31)
    np.testing.assert_array_equal(asnumpy(received), expected)

    rank_sum = context.reductions.sum(xp.asarray(float(context.rank)))
    expected_sum = context.size * (context.size - 1) / 2
    if float(asnumpy(rank_sum)) != expected_sum:
        raise RuntimeError(
            f"all-reduce returned {float(asnumpy(rank_sum))}, "
            f"expected {expected_sum}")
    vector_sum = context.allreduce_sum(
        xp.asarray([float(context.rank), 1.0], dtype=xp.float64))
    np.testing.assert_array_equal(
        asnumpy(vector_sum), np.asarray([expected_sum, float(context.size)]))

    communicator.Barrier()
    synchronize(xp)
    started = MPI.Wtime()
    for iteration in range(args.iterations):
        received = context.sendrecv(
            values, destination=destination, source=source, tag=100 + iteration)
    synchronize(xp)
    elapsed = MPI.Wtime() - started
    max_elapsed = communicator.allreduce(elapsed, op=MPI.MAX)
    bytes_transferred = values.nbytes * args.iterations

    placement = context.describe()
    placement["backend"] = device_name(xp)
    for item in communicator.gather(placement, root=0) or []:
        print("NDGPU_MPI_RANK " + json.dumps(item, sort_keys=True), flush=True)

    if context.rank == 0:
        summary = {
            "status": "passed",
            "ranks": context.size,
            "device": args.device,
            "communication_mode": context.communication_mode,
            "elements": args.elements,
            "iterations": args.iterations,
            "max_elapsed_s": max_elapsed,
            "one_way_bandwidth_gb_s": bytes_transferred / max_elapsed / 1e9,
        }
        print("NDGPU_MPI_PROBE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
