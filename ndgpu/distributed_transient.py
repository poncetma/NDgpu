"""Spatially distributed transient diffusion solvers."""

from __future__ import annotations

from .distributed import (DistributedContext, DistributedResult,
                          DistributedTransientResult, TriRowPartition)
from .distributed_solver import (_local_array, _resolve_context,
                                 DistributedTriDiffusionEigenSolver)
from .distributed_stencil import DistributedTriGroupOperator
from .transient import TransientSolver
from .tri import TriGrid
from .solver import Result


class DistributedTriTransientSolver(TransientSolver):
    """Fixed-point transient diffusion over MPI-owned triangular row slabs."""

    def __init__(self, grid, problem_at, kinetics, *, communicator=None,
                 context=None, communication="auto", device="auto",
                 decomposition="auto", active=None, mix_material=None,
                 mix_weight=None, xs_update_at=None, feedback_iteration=None,
                 **kwargs):
        context, context_device = _resolve_context(
            communicator, context, device, communication)
        if decomposition not in ("auto", "rows"):
            raise ValueError("decomposition must be 'auto' or 'rows'")
        if xs_update_at is not None or feedback_iteration is not None:
            raise NotImplementedError(
                "distributed transient feedback callbacks are not implemented")

        partition = TriRowPartition.create(
            grid.shape, context.rank, context.size)
        local_grid = TriGrid(
            partition.local_shape, side=grid.side, height=grid.height)
        local_active = _local_array(active, partition, "active")
        local_mix_material = _local_array(
            mix_material, partition, "mix_material")
        local_mix_weight = _local_array(
            mix_weight, partition, "mix_weight")

        cache_key = None
        cache_value = None

        def local_problem_at(time):
            nonlocal cache_key, cache_value
            spec = tuple(problem_at(time))
            if len(spec) not in (2, 4):
                raise ValueError(
                    "problem_at must return two or four entries")
            key = tuple(id(value) for value in spec)
            if key == cache_key:
                return cache_value
            if len(spec) == 2:
                mats, material_map = spec
                cache_value = (
                    mats, _local_array(material_map, partition, "material_map"))
            else:
                mats, material_map, moving_mix, moving_weight = spec
                cache_value = (
                    mats,
                    _local_array(material_map, partition, "material_map"),
                    _local_array(moving_mix, partition, "mix_material"),
                    _local_array(moving_weight, partition, "mix_weight"),
                )
            cache_key = key
            return cache_value

        def group_operator(xp, ignored_grid, D, removal, **operator_kwargs):
            del ignored_grid
            operator_kwargs.pop("symmetric", None)
            return DistributedTriGroupOperator(
                xp, grid, D, removal, partition, context,
                communication_tag=900, **operator_kwargs)

        def eigen_solver(ignored_grid, materials, material_map, **solver_kwargs):
            del ignored_grid
            solver_kwargs.pop("device", None)
            return DistributedTriDiffusionEigenSolver(
                grid, materials, material_map, context=context,
                decomposition="rows", **solver_kwargs)

        self.distributed_context = context
        self.partition = partition
        self.global_grid = grid
        super().__init__(
            local_grid, local_problem_at, kinetics,
            device=context_device, active=local_active,
            mix_material=local_mix_material, mix_weight=local_mix_weight,
            group_operator=group_operator, eig_solver=eigen_solver,
            reductions=context.reductions, **kwargs)

    def solve(self, *args, **kwargs):
        if kwargs.get("step_solver", "fixed-point") != "fixed-point":
            raise NotImplementedError(
                "distributed transient currently supports fixed-point steps")
        if kwargs.get("adaptive_bdf") is not None:
            raise NotImplementedError(
                "distributed adaptive BDF is not implemented")
        initial_steady = kwargs.get("initial_steady")
        if isinstance(initial_steady, DistributedResult):
            if initial_steady.context is not self.distributed_context:
                raise ValueError(
                    "initial_steady must use the transient solver's context")
            if initial_steady.partition != self.partition:
                raise ValueError(
                    "initial_steady partition does not match the transient")
            kwargs = dict(kwargs)
            kwargs["initial_steady"] = Result(
                k_eff=initial_steady.k_eff,
                flux=initial_steady.local_flux,
                converged=initial_steady.converged,
                outer_iterations=initial_steady.outer_iterations,
                inner_iterations=initial_steady.inner_iterations,
                solve_seconds=initial_steady.solve_seconds,
                device=initial_steady.device,
                k_history=initial_steady.k_history,
                source_error_history=initial_steady.source_error_history,
            )
        elif initial_steady is not None and self.distributed_context.size > 1:
            raise TypeError(
                "multi-rank initial_steady must be a DistributedResult")
        result = super().solve(*args, **kwargs)
        return DistributedTransientResult(
            result, self.partition, self.distributed_context)
