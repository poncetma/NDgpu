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
from ndgpu.distributed import DistributedContext, SpatialPartition


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--allreduce-iterations", type=int, default=1000)
    parser.add_argument("--batched-halos", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.elements < 1 or args.iterations < 1
            or args.allreduce_iterations < 1):
        raise ValueError("elements and iterations must be positive")

    communicator = MPI.COMM_WORLD
    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication,
        batched_halos=args.batched_halos)
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

    scalar = xp.asarray(float(context.rank), dtype=xp.float64)
    communicator.Barrier()
    synchronize(xp)
    started = MPI.Wtime()
    for _ in range(args.allreduce_iterations):
        reduced = context.allreduce_sum(scalar)
    synchronize(xp)
    allreduce_elapsed = MPI.Wtime() - started
    max_allreduce_elapsed = communicator.allreduce(
        allreduce_elapsed, op=MPI.MAX)
    if float(asnumpy(reduced)) != expected_sum:
        raise RuntimeError("timed all-reduce returned an incorrect value")

    halo_elapsed = None
    if args.batched_halos:
        start = 2 * context.rank
        partition = SpatialPartition(
            (2 * context.size, args.elements), 0, context.rank,
            context.size, start, start + 2)
        halo_values = xp.stack([
            xp.full(args.elements, 10.0 * context.rank, dtype=xp.float64),
            xp.full(args.elements, 10.0 * context.rank + 1.0,
                    dtype=xp.float64),
        ])
        lower, upper = context.exchange_halos(
            halo_values, partition, tag=700)
        if context.rank == 0:
            assert lower is None
        else:
            np.testing.assert_array_equal(
                asnumpy(lower),
                np.full(args.elements, 10.0 * (context.rank - 1) + 1.0))
        if context.rank == context.size - 1:
            assert upper is None
        else:
            np.testing.assert_array_equal(
                asnumpy(upper),
                np.full(args.elements, 10.0 * (context.rank + 1)))

        communicator.Barrier()
        synchronize(xp)
        started = MPI.Wtime()
        for iteration in range(args.iterations):
            lower, upper = context.exchange_halos(
                halo_values, partition, tag=800 + 2 * iteration)
        synchronize(xp)
        halo_elapsed = communicator.allreduce(
            MPI.Wtime() - started, op=MPI.MAX)

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
            "allreduce_iterations": args.allreduce_iterations,
            "allreduce_latency_us": (
                1e6 * max_allreduce_elapsed / args.allreduce_iterations),
            "batched_halos": args.batched_halos,
        }
        if halo_elapsed is not None:
            summary["batched_halo_latency_us"] = (
                1e6 * halo_elapsed / args.iterations)
        print("NDGPU_MPI_PROBE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
