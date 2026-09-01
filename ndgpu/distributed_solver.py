"""Explicit spatially distributed diffusion eigenvalue solver APIs."""

from __future__ import annotations

import numpy as np

from .distributed import (CartesianSlabPartition, DistributedContext,
                          DistributedResult, TriRowPartition)
from .distributed_stencil import DistributedCartesianGroupOperator
from .grid import Grid
from .solver import DiffusionEigenSolver
from .tri import TriDiffusionEigenSolver


def _resolve_context(communicator, context, device, communication):
    if communicator is not None and context is not None:
        raise ValueError("pass communicator or context, not both")
    if context is None:
        if communicator is None:
            raise ValueError("distributed solver requires a communicator or context")
        context = DistributedContext.from_mpi(
            communicator, device=device, communication=communication)
    elif not isinstance(context, DistributedContext):
        raise TypeError("context must be a DistributedContext")

    context_device = "cpu" if context.xp is np else "gpu"
    requested = device.lower()
    if requested != "auto":
        requested = "gpu" if requested == "cuda" else requested
        if requested != context_device:
            raise ValueError(
                f"device={device!r} does not match the context's "
                f"{context_device} backend")
    return context, context_device


def _local_array(value, partition, name):
    if value is None:
        return None
    array = np.asarray(value)
    if tuple(array.shape) == partition.local_shape:
        return array
    if tuple(array.shape) == partition.global_shape:
        return np.array(array[partition.owned_slice], copy=True)
    raise ValueError(
        f"{name} shape {array.shape} must be local {partition.local_shape} "
        f"or global {partition.global_shape}")


class DistributedDiffusionEigenSolver(DiffusionEigenSolver):
    """Cartesian diffusion solve with one MPI-owned spatial slab per rank."""

    def __init__(self, grid, materials, material_map=None, *, communicator=None,
                 context=None, decomposition="auto", communication="auto",
                 device="auto", axis=None, active=None, mix_material=None,
                 mix_weight=None, **kwargs):
        context, context_device = _resolve_context(
            communicator, context, device, communication)
        if decomposition not in ("auto", "slab"):
            raise ValueError("decomposition must be 'auto' or 'slab'")
        if getattr(grid, "geometry", None) != "cartesian":
            raise ValueError("distributed Cartesian solver requires a Cartesian grid")

        partition = CartesianSlabPartition.create(
            grid.shape, context.rank, context.size, axis=axis)
        local_grid = Grid(
            partition.local_shape,
            tuple(d * n for d, n in zip(grid.spacing, partition.local_shape)))

        self.distributed_context = context
        self.partition = partition
        self.global_grid = grid
        self._distributed_active = _local_array(active, partition, "active")
        local_material_map = _local_array(
            material_map, partition, "material_map")
        local_mix_material = _local_array(
            mix_material, partition, "mix_material")
        local_mix_weight = _local_array(mix_weight, partition, "mix_weight")

        super().__init__(
            local_grid, materials, material_map=local_material_map,
            device=context_device, active=self._distributed_active,
            mix_material=local_mix_material, mix_weight=local_mix_weight,
            **kwargs)
        self._normalization_cell_count = grid.n_cells

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        del grid, sigma_t
        if self.hybrid_mask is not None:
            raise ValueError("hybrid_mask has no effect on the diffusion solver")
        if not self.symmetric_operator:
            raise ValueError(
                "distributed Cartesian diffusion requires symmetric_operator=True")
        self.ops = [
            DistributedCartesianGroupOperator(
                self.xp, self.global_grid, diffusion[g], removal[g],
                self.partition, self.distributed_context, bc=bc,
                active=self.active, mask_bc=self.mask_bc,
                communication_tag=100 + 10 * g)
            for g in range(self.n_groups)
        ]

    def solve(self, *args, **kwargs):
        if "reductions" in kwargs:
            raise TypeError("distributed solver owns its reduction provider")
        result = super().solve(
            *args, reductions=self.distributed_context.reductions, **kwargs)
        return DistributedResult.from_local_result(
            result, self.partition, self.distributed_context)


class DistributedTriDiffusionEigenSolver(TriDiffusionEigenSolver):
    """Size-one triangular entry point pending Phase 3 row decomposition."""

    def __init__(self, *args, communicator=None, context=None,
                 decomposition="auto", communication="auto", device="auto",
                 **kwargs):
        context, context_device = _resolve_context(
            communicator, context, device, communication)
        if context.size != 1:
            raise NotImplementedError(
                "multi-rank triangular diffusion requires the Phase 3 row operator")
        if decomposition not in ("auto", "rows"):
            raise ValueError("decomposition must be 'auto' or 'rows'")

        self.distributed_context = context
        super().__init__(*args, device=context_device, **kwargs)
        self.partition = TriRowPartition.create(
            self.grid.shape, context.rank, context.size)

    def solve(self, *args, **kwargs):
        if "reductions" in kwargs:
            raise TypeError("distributed solver owns its reduction provider")
        result = super().solve(
            *args, reductions=self.distributed_context.reductions, **kwargs)
        return DistributedResult.from_local_result(
            result, self.partition, self.distributed_context)
