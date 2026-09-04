"""Tensor-product extrusion of a compatible 2-D unstructured TPFA mesh.

This path keeps the validated 2-D face connectivity, including recursive
midpoint hanging interfaces, and adds uniform axial finite-volume couplings.
It deliberately does not assemble a general 3-D nonconforming mesh: every
axial layer has identical radial topology, which is both cheaper to store and
the natural decomposition for prismatic reactor transients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import kernels
from .mesh import Mesh, _build_ell
from .solver import DiffusionEigenSolver
from .stencil import (BC_VACUUM, face_alpha, harmonic_mean, normalize_bc)
from .transient import TransientSolver


@dataclass(frozen=True, eq=False)
class ExtrudedMeshGrid:
    """A 2-D finite-volume mesh repeated over uniform axial layers."""

    mesh: Mesh
    height: float
    nz: int

    def __post_init__(self):
        if self.mesh.coords.shape[1] != 2:
            raise ValueError("ExtrudedMeshGrid requires a 2-D base mesh")
        if not math.isfinite(self.height) or self.height <= 0.0:
            raise ValueError("height must be finite and positive")
        if not isinstance(self.nz, (int, np.integer)) or self.nz < 1:
            raise ValueError("nz must be a positive integer")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.mesh.n_cells, int(self.nz))

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells * int(self.nz)

    @property
    def dz(self) -> float:
        return float(self.height) / int(self.nz)

    @property
    def cell_volumes(self) -> np.ndarray:
        return np.broadcast_to(
            np.asarray(self.mesh.area)[:, None] * self.dz, self.shape).copy()


def _integrated_robin(xp, diffusion, face_measure, distance, alpha):
    """Volume-integrated Robin leakage for one boundary-face collection."""
    if alpha == 0.0:
        return xp.zeros_like(diffusion)
    if math.isinf(alpha):
        return diffusion * face_measure / distance
    return (alpha * diffusion * face_measure
            / (distance * alpha + diffusion))


class ExtrudedMeshGroupOperator:
    """TPFA group operator on ``base_mesh x uniform_z``.

    Radial faces reuse the 2-D ELLPACK gather connectivity independently in
    every layer. Axial faces are a regular two-point stencil. The equation is
    volume-integrated, so ``rhs_weight`` must multiply every volumetric source.
    """

    supports_out = True

    def __init__(self, xp, grid: ExtrudedMeshGrid, D, removal,
                 bc=BC_VACUUM, active=None, mask_bc=BC_VACUUM,
                 partition_interfaces=(False, False)):
        if not isinstance(grid, ExtrudedMeshGrid):
            raise TypeError("grid must be an ExtrudedMeshGrid")
        if tuple(D.shape) != grid.shape:
            raise ValueError(f"D shape {D.shape} != {grid.shape}")
        if tuple(removal.shape) != grid.shape:
            raise ValueError(f"removal shape {removal.shape} != {grid.shape}")
        if active is not None and not bool(xp.all(xp.asarray(active))):
            raise NotImplementedError(
                "inactive cells are not supported on an extruded mesh; "
                "remove void cells from the 2-D base mesh")

        self.xp = xp
        self.grid = grid
        self.shape = grid.shape
        mesh = grid.mesh
        n = mesh.n_cells
        dz = grid.dz

        area = xp.asarray(np.asarray(mesh.area), dtype=D.dtype)
        volume = area[:, None] * dz
        self.rhs_weight = volume

        face_i = np.asarray([face[0] for face in mesh.faces], dtype=np.int64)
        face_j = np.asarray([face[1] for face in mesh.faces], dtype=np.int64)
        face_measure = xp.asarray(
            np.asarray([face[2] for face in mesh.faces]), dtype=D.dtype)[:, None]
        face_distance = xp.asarray(
            np.asarray([face[3] for face in mesh.faces]), dtype=D.dtype)[:, None]
        nbr, face_map = _build_ell(face_i, face_j, n)
        safe_map = np.where(face_map >= 0, face_map, 0)
        self.nbr = xp.asarray(nbr.astype(np.int32))

        fi = xp.asarray(face_i)
        fj = xp.asarray(face_j)
        radial_face = (harmonic_mean(D[fi], D[fj])
                       * face_measure * dz / face_distance)
        self.radial_weight = xp.where(
            xp.asarray(face_map >= 0)[:, :, None],
            radial_face[xp.asarray(safe_map)], 0.0)

        diagonal = removal * volume + self.radial_weight.sum(axis=0)

        boundary_i = np.asarray(
            [face[0] for face in mesh.bfaces], dtype=np.int64)
        if boundary_i.size:
            bi = xp.asarray(boundary_i)
            boundary_measure = xp.asarray(
                np.asarray([face[1] for face in mesh.bfaces]),
                dtype=D.dtype)[:, None] * dz
            boundary_distance = xp.asarray(
                np.asarray([face[2] for face in mesh.bfaces]),
                dtype=D.dtype)[:, None]
            boundary_weight = _integrated_robin(
                xp, D[bi], boundary_measure, boundary_distance,
                face_alpha(mask_bc))
            xp.add.at(diagonal, bi, boundary_weight)

        if grid.nz > 1:
            self.axial_weight = (harmonic_mean(D[:, :-1], D[:, 1:])
                                 * area[:, None] / dz)
            diagonal[:, :-1] += self.axial_weight
            diagonal[:, 1:] += self.axial_weight
        else:
            self.axial_weight = None

        if len(partition_interfaces) != 2:
            raise ValueError("partition_interfaces must be (lower, upper)")
        z_faces = normalize_bc(bc)[2]
        for layer, spec, is_interface in (
                (0, z_faces[0], partition_interfaces[0]),
                (-1, z_faces[1], partition_interfaces[1])):
            if is_interface:
                continue
            weight = _integrated_robin(
                xp, D[:, layer], area, 0.5 * dz, face_alpha(spec))
            diagonal[:, layer] += weight

        self.diag = diagonal
        self.inv_diag = 1.0 / diagonal

    def apply(self, phi, out=None):
        if tuple(phi.shape) != self.shape:
            raise ValueError(f"flux shape {phi.shape} != {self.shape}")
        fused = kernels.extruded_ell_apply(
            self.xp, phi, self.diag, self.nbr, self.radial_weight,
            self.axial_weight, out=out)
        if fused is not None:
            return fused
        if out is None:
            out = self.diag * phi
        else:
            out[...] = self.diag * phi
        out -= (self.radial_weight * phi[self.nbr]).sum(axis=0)
        if self.axial_weight is not None:
            out[:, :-1] -= self.axial_weight * phi[:, 1:]
            out[:, 1:] -= self.axial_weight * phi[:, :-1]
        return out


class ExtrudedMeshDiffusionEigenSolver(DiffusionEigenSolver):
    """Steady diffusion k-eigenvalue solver on an ``ExtrudedMeshGrid``."""

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        del sigma_t
        if self.hybrid_mask is not None:
            raise ValueError("hybrid_mask has no effect on diffusion")
        if not self.symmetric_operator:
            raise ValueError("extruded mesh diffusion requires symmetric_operator=True")
        self.ops = [
            ExtrudedMeshGroupOperator(
                self.xp, grid, diffusion[group], removal[group], bc=bc,
                active=self.active, mask_bc=self.mask_bc)
            for group in range(self.n_groups)
        ]


class ExtrudedMeshTransientSolver(TransientSolver):
    """Transient diffusion solver on an ``ExtrudedMeshGrid``."""

    def __init__(self, grid, problem_at, kinetics, **kwargs):
        if not isinstance(grid, ExtrudedMeshGrid):
            raise TypeError("grid must be an ExtrudedMeshGrid")
        kwargs.pop("group_operator", None)
        kwargs.pop("eig_solver", None)
        super().__init__(
            grid, problem_at, kinetics,
            group_operator=ExtrudedMeshGroupOperator,
            eig_solver=ExtrudedMeshDiffusionEigenSolver, **kwargs)
