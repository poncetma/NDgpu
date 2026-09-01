"""Partition-aware matrix-free stencils for distributed diffusion."""

from __future__ import annotations

import math

from .distributed import (CartesianSlabPartition, DistributedContext,
                          TriRowPartition)
from .grid import Grid
from .stencil import (BC_REFLECTIVE, BC_VACUUM, GroupOperator, face_alpha,
                      harmonic_mean, normalize_bc, robin_face_term)
from .tri import TriGrid, TriGroupOperator, _SQRT3


def _plane(value, axis, index):
    sl = [slice(None)] * value.ndim
    sl[axis] = index
    return value[tuple(sl)]


class DistributedCartesianGroupOperator:
    """Seven-point Cartesian operator over one MPI-owned slab.

    The wrapped :class:`GroupOperator` handles owned faces and true global
    boundaries. Faces on an MPI cut are reflective in that local operator;
    this wrapper restores their diagonal and off-rank neighbor contributions
    after a blocking halo exchange.
    """

    supports_out = True

    def __init__(self, xp, grid, D, removal, partition, context, *,
                 bc="zero-flux", active=None, mask_bc=BC_VACUUM,
                 communication_tag=100):
        if not isinstance(partition, CartesianSlabPartition):
            raise TypeError("partition must be a CartesianSlabPartition")
        if not isinstance(context, DistributedContext):
            raise TypeError("context must be a DistributedContext")
        if grid.geometry != "cartesian":
            raise ValueError("distributed Cartesian slabs require a Cartesian grid")
        if tuple(grid.shape) != partition.global_shape:
            raise ValueError("grid shape does not match the global partition shape")
        if tuple(D.shape) != partition.local_shape:
            raise ValueError(f"local D shape {D.shape} != {partition.local_shape}")
        if tuple(removal.shape) != partition.local_shape:
            raise ValueError(
                f"local removal shape {removal.shape} != {partition.local_shape}")
        if context.xp is not xp:
            raise ValueError("operator backend does not match the distributed context")

        self.xp = xp
        self.shape = partition.local_shape
        self.partition = partition
        self.context = context
        self.communication_tag = int(communication_tag)
        axis = partition.axis

        local_size = tuple(
            spacing * count
            for spacing, count in zip(grid.spacing, partition.local_shape))
        local_grid = Grid(partition.local_shape, local_size)

        local_bc = [list(pair) for pair in normalize_bc(bc)]
        if partition.lower_rank is not None:
            local_bc[axis][0] = BC_REFLECTIVE
        if partition.upper_rank is not None:
            local_bc[axis][1] = BC_REFLECTIVE

        local_active = None
        if active is not None:
            local_active = xp.asarray(active).astype(bool)
            if tuple(local_active.shape) != partition.local_shape:
                raise ValueError(
                    f"local active shape {local_active.shape} != "
                    f"{partition.local_shape}")

        self.local_operator = GroupOperator(
            xp, local_grid, D, removal, bc=local_bc, active=local_active,
            mask_bc=mask_bc)
        self.rhs_weight = self.local_operator.rhs_weight

        lower_D, upper_D = context.exchange_halos(
            D, partition, tag=self.communication_tag)
        if local_active is None:
            lower_active = upper_active = None
        else:
            lower_active, upper_active = context.exchange_halos(
                local_active, partition, tag=self.communication_tag + 2)

        spacing = grid.spacing[axis]
        mask_alpha = face_alpha(mask_bc)
        self._lower_coupling, self._lower_diagonal = self._interface_terms(
            _plane(D, axis, 0), lower_D,
            None if local_active is None else _plane(local_active, axis, 0),
            lower_active, spacing, mask_alpha)
        self._upper_coupling, self._upper_diagonal = self._interface_terms(
            _plane(D, axis, -1), upper_D,
            None if local_active is None else _plane(local_active, axis, -1),
            upper_active, spacing, mask_alpha)

        self.diag = xp.array(self.local_operator.diag, copy=True)
        if self._lower_diagonal is not None:
            _plane(self.diag, axis, 0)[...] += self._lower_diagonal
        if self._upper_diagonal is not None:
            _plane(self.diag, axis, -1)[...] += self._upper_diagonal
        self.inv_diag = 1.0 / self.diag

    def _interface_terms(self, local_D, remote_D, local_active,
                         remote_active, spacing, mask_alpha):
        if remote_D is None:
            return None, None
        coupling = harmonic_mean(local_D, remote_D) / spacing**2
        if local_active is None:
            return coupling, coupling

        connected = local_active & remote_active
        coupling = self.xp.where(connected, coupling, 0.0)
        boundary = self.xp.where(
            local_active & ~remote_active,
            robin_face_term(self.xp, local_D, spacing, mask_alpha), 0.0)
        return coupling, coupling + boundary

    def apply(self, phi, out=None):
        """Apply the local stencil after exchanging the current flux halos."""
        if tuple(phi.shape) != self.shape:
            raise ValueError(f"local flux shape {phi.shape} != {self.shape}")
        (lower_phi, upper_phi), out = self.context.exchange_halos_while(
            phi, self.partition,
            lambda: self.local_operator.apply(phi, out=out),
            tag=self.communication_tag + 4)
        axis = self.partition.axis

        if self._lower_diagonal is not None:
            local_out = _plane(out, axis, 0)
            local_phi = _plane(phi, axis, 0)
            local_out += self._lower_diagonal * local_phi
            local_out -= self._lower_coupling * lower_phi
        if self._upper_diagonal is not None:
            local_out = _plane(out, axis, -1)
            local_phi = _plane(phi, axis, -1)
            local_out += self._upper_diagonal * local_phi
            local_out -= self._upper_coupling * upper_phi
        return out


def _tri_component(value, row, orientation):
    sl = [slice(None)] * value.ndim
    sl[0] = row
    sl[2] = orientation
    return value[tuple(sl)]


class DistributedTriGroupOperator:
    """Triangular-FV operator over one MPI-owned contiguous row slab."""

    supports_out = True

    def __init__(self, xp, grid, D, removal, partition, context, *,
                 bc=BC_VACUUM, active=None, mask_bc=BC_VACUUM,
                 communication_tag=500):
        if not isinstance(partition, TriRowPartition):
            raise TypeError("partition must be a TriRowPartition")
        if not isinstance(context, DistributedContext):
            raise TypeError("context must be a DistributedContext")
        if tuple(grid.shape) != partition.global_shape:
            raise ValueError("grid shape does not match the global partition shape")
        if tuple(D.shape) != partition.local_shape:
            raise ValueError(f"local D shape {D.shape} != {partition.local_shape}")
        if tuple(removal.shape) != partition.local_shape:
            raise ValueError(
                f"local removal shape {removal.shape} != {partition.local_shape}")
        if context.xp is not xp:
            raise ValueError("operator backend does not match the distributed context")

        self.xp = xp
        self.shape = partition.local_shape
        self.partition = partition
        self.context = context
        self.communication_tag = int(communication_tag)
        local_grid = TriGrid(partition.local_shape, grid.side, grid.height)

        local_active = None
        if active is not None:
            local_active = xp.asarray(active).astype(bool)
            if tuple(local_active.shape) != partition.local_shape:
                raise ValueError(
                    f"local active shape {local_active.shape} != "
                    f"{partition.local_shape}")

        interfaces = (
            partition.lower_rank is not None,
            partition.upper_rank is not None)
        self.local_operator = TriGroupOperator(
            xp, local_grid, D, removal, bc=bc, active=local_active,
            mask_bc=mask_bc, partition_interfaces=interfaces)
        self.rhs_weight = None

        lower_D, upper_D = context.exchange_halos(
            D, partition, tag=self.communication_tag)
        if local_active is None:
            lower_active = upper_active = None
        else:
            lower_active, upper_active = context.exchange_halos(
                local_active, partition, tag=self.communication_tag + 2)

        alpha = face_alpha(mask_bc)
        self._lower_coupling, self._lower_diagonal = self._interface_terms(
            _tri_component(D, 0, 0),
            None if lower_D is None else lower_D[:, 1, ...],
            None if local_active is None else _tri_component(local_active, 0, 0),
            None if lower_active is None else lower_active[:, 1, ...],
            grid.side, alpha)
        self._upper_coupling, self._upper_diagonal = self._interface_terms(
            _tri_component(D, -1, 1),
            None if upper_D is None else upper_D[:, 0, ...],
            None if local_active is None else _tri_component(local_active, -1, 1),
            None if upper_active is None else upper_active[:, 0, ...],
            grid.side, alpha)

        self.diag = xp.array(self.local_operator.diag, copy=True)
        if self._lower_diagonal is not None:
            _tri_component(self.diag, 0, 0)[...] += self._lower_diagonal
        if self._upper_diagonal is not None:
            _tri_component(self.diag, -1, 1)[...] += self._upper_diagonal
        self.inv_diag = 1.0 / self.diag

    def _interface_terms(self, local_D, remote_D, local_active,
                         remote_active, side, alpha):
        if remote_D is None:
            return None, None
        coupling = harmonic_mean(local_D, remote_D) * (4.0 / side**2)
        if local_active is None:
            return coupling, coupling

        coupling = self.xp.where(
            local_active & remote_active, coupling, 0.0)
        if alpha == 0.0:
            boundary = self.xp.zeros_like(local_D)
        elif math.isinf(alpha):
            boundary = 8.0 * local_D / side**2
        else:
            boundary = (8.0 * local_D * alpha /
                        (side * (side * alpha + 2.0 * _SQRT3 * local_D)))
        boundary = self.xp.where(
            local_active & ~remote_active, boundary, 0.0)
        return coupling, coupling + boundary

    def apply(self, phi, out=None):
        """Apply the owned triangular stencil after exchanging row halos."""
        if tuple(phi.shape) != self.shape:
            raise ValueError(f"local flux shape {phi.shape} != {self.shape}")
        (lower_phi, upper_phi), out = self.context.exchange_halos_while(
            phi, self.partition,
            lambda: self.local_operator.apply(phi, out=out),
            tag=self.communication_tag + 4)

        if self._lower_diagonal is not None:
            local_out = _tri_component(out, 0, 0)
            local_flux = _tri_component(phi, 0, 0)
            remote_flux = lower_phi[:, 1, ...]
            local_out += self._lower_diagonal * local_flux
            local_out -= self._lower_coupling * remote_flux
        if self._upper_diagonal is not None:
            local_out = _tri_component(out, -1, 1)
            local_flux = _tri_component(phi, -1, 1)
            remote_flux = upper_phi[:, 0, ...]
            local_out += self._upper_diagonal * local_flux
            local_out -= self._upper_coupling * remote_flux
        return out
