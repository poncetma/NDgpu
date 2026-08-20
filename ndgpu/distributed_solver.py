"""Explicit distributed diffusion solver APIs.

Phase 1 intentionally supports a size-one communicator only.  The wrappers
exercise global reductions and distributed result ownership while reusing the
serial numerical implementation.  Multi-rank construction fails before field
allocation until the Cartesian and triangular halo operators land.
"""

from __future__ import annotations

import numpy as np

from .distributed import (CartesianSlabPartition, DistributedContext,
                          DistributedResult, TriRowPartition)
from .solver import DiffusionEigenSolver
from .tri import TriDiffusionEigenSolver


class _SizeOneDistributedMixin:
    partition_type = None
    decomposition_names = ()

    def __init__(self, *args, communicator=None, context=None,
                 decomposition="auto", communication="auto", device="auto",
                 **kwargs):
        if communicator is not None and context is not None:
            raise ValueError("pass communicator or context, not both")
        if context is None:
            if communicator is None:
                raise ValueError("distributed solver requires a communicator or context")
            context = DistributedContext.from_mpi(
                communicator, device=device, communication=communication)
        elif not isinstance(context, DistributedContext):
            raise TypeError("context must be a DistributedContext")

        if context.size != 1:
            raise NotImplementedError(
                "multi-rank diffusion requires the Phase 2/3 spatial operator; "
                "the Phase 1 wrapper accepts a size-one communicator only")
        if decomposition != "auto" and decomposition not in self.decomposition_names:
            expected = " or ".join(repr(name) for name in self.decomposition_names)
            raise ValueError(f"decomposition must be 'auto' or {expected}")

        context_device = "cpu" if context.xp is np else "gpu"
        requested = device.lower()
        if requested != "auto":
            requested = "gpu" if requested == "cuda" else requested
            if requested != context_device:
                raise ValueError(
                    f"device={device!r} does not match the context's "
                    f"{context_device} backend")

        self.distributed_context = context
        super().__init__(*args, device=context_device, **kwargs)
        self.partition = self.partition_type.create(
            self.grid.shape, context.rank, context.size)

    def solve(self, *args, **kwargs):
        if "reductions" in kwargs:
            raise TypeError("distributed solver owns its reduction provider")
        result = super().solve(
            *args, reductions=self.distributed_context.reductions, **kwargs)
        return DistributedResult.from_local_result(
            result, self.partition, self.distributed_context)


class DistributedDiffusionEigenSolver(
        _SizeOneDistributedMixin, DiffusionEigenSolver):
    """Size-one Cartesian entry point for the distributed diffusion path."""

    partition_type = CartesianSlabPartition
    decomposition_names = ("slab",)


class DistributedTriDiffusionEigenSolver(
        _SizeOneDistributedMixin, TriDiffusionEigenSolver):
    """Size-one triangular entry point for the distributed diffusion path."""

    partition_type = TriRowPartition
    decomposition_names = ("rows",)
