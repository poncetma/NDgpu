"""Structured hexagonal-lattice diffusion, matrix-free on the same backend.

A hex lattice is *structured* in axial (skewed) coordinates: every interior
cell has six neighbours at fixed index offsets, so the finite-volume diffusion
operator is again a small set of shifted multiply-adds -- no sparse assembly,
no indirection. This reuses the Cartesian solver's power-iteration / transient
machinery unchanged; only the within-group operator changes.

Cells are stored on an (rows, cols, nz) array in axial coordinates: the six
in-plane neighbours of (r, c) are (r, c+-1), (r+-1, c) and (r+1, c-1) /
(r-1, c+1). FEMFFUSION-style offset (row-staggered) maps convert in with
`offset_to_axial`. A regular hexagon of flat-to-flat pitch p has centre-to-
centre spacing p, face width p/sqrt(3) and area (sqrt(3)/2) p^2, giving an
interior face coupling  w = D_face * 2 / (3 p^2)  and a boundary (Robin) term
4 D alpha / (3 p (p alpha + 2 D)) for the law J_net = alpha * phi_surface
(alpha = 1/2 vacuum, 0 reflective, inf zero-flux) -- the hex analogues of the
Cartesian expressions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .stencil import (BC_VACUUM, face_alpha, harmonic_mean, normalize_bc,
                      robin_face_term)
from .solver import DiffusionEigenSolver


@dataclass(frozen=True)
class HexGrid:
    """Axial-coordinate hex lattice: shape (rows, cols, nz), flat-to-flat pitch."""

    shape: tuple
    pitch: float           # flat-to-flat = centre-to-centre spacing, cm
    height: float = 1.0    # total z extent, cm

    def __post_init__(self):
        if len(self.shape) != 3:
            raise ValueError("hex shape must be (rows, cols, nz)")
        if self.pitch <= 0 or self.height <= 0:
            raise ValueError("pitch and height must be positive")

    @property
    def n_cells(self) -> int:
        r, c, z = self.shape
        return r * c * z

    @property
    def dz(self) -> float:
        return self.height / self.shape[2]

    @property
    def cell_volume(self) -> float:
        return (math.sqrt(3.0) / 2.0) * self.pitch**2 * self.dz


def offset_to_axial(offset_map: np.ndarray) -> np.ndarray:
    """Convert a row-staggered (odd rows shifted right) offset map to axial.

    Cell (r, c) moves to axial column c - r//2; the array is widened and padded
    with 0 (void) so every cell keeps six well-defined neighbour slots.
    """
    offset_map = np.asarray(offset_map)
    nrow, ncol = offset_map.shape
    shift = (nrow - 1) // 2
    out = np.zeros((nrow, ncol + shift), dtype=offset_map.dtype)
    for r in range(nrow):
        s = shift - r // 2
        out[r, s:s + ncol] = offset_map[r]
    return out


class HexGroupOperator:
    """Within-group hex FV operator: A = -div(D grad .) + Sigma_r on a HexGrid.

    Same (apply, inv_diag) interface as the Cartesian GroupOperator, with an
    optional `active` mask for non-lattice-filling cores and a Robin/albedo
    `mask_bc` on every core-surface and array-edge face.
    """

    def __init__(self, xp, grid: HexGrid, D, removal, bc=BC_VACUUM, active=None,
                 mask_bc=BC_VACUUM):
        bc = normalize_bc(bc)  # only the z faces read from bc; in-plane uses mask_bc
        self.xp = xp
        self.shape = grid.shape
        p, dz = grid.pitch, grid.dz
        kf = 2.0 / (3.0 * p * p)          # in-plane geometric factor
        alpha_edge = face_alpha(mask_bc)  # in-plane core-surface / edge law

        # In-plane boundary term: same derivation as robin_face_term, with the
        # hex face area p/sqrt(3), cell area (sqrt(3)/2) p^2 and centre-to-face
        # distance p/2, giving 4 D alpha / (3 p (p alpha + 2 D)).
        def robin_plane(Dface, alpha):
            if alpha == 0.0:
                return xp.zeros_like(Dface)
            if math.isinf(alpha):
                return 4.0 * Dface / (3.0 * p * p)
            return 4.0 * Dface * alpha / (3.0 * p * (p * alpha + 2.0 * Dface))

        # In-plane couplings for the three axial directions.
        wA = harmonic_mean(D[:, :-1, :], D[:, 1:, :]) * kf     # (r,c)-(r,c+1)
        wB = harmonic_mean(D[:-1, :, :], D[1:, :, :]) * kf     # (r,c)-(r+1,c)
        wC = harmonic_mean(D[:-1, 1:, :], D[1:, :-1, :]) * kf  # (r,c+1)-(r+1,c)
        wz = (harmonic_mean(D[:, :, :-1], D[:, :, 1:]) / dz**2
              if grid.shape[2] > 1 else None)

        act = None
        if active is not None:
            act = xp.asarray(active).astype(bool)
            if act.shape != grid.shape:
                raise ValueError("active mask shape must match hex grid shape")
            wA = xp.where(act[:, :-1, :] & act[:, 1:, :], wA, 0.0)
            wB = xp.where(act[:-1, :, :] & act[1:, :, :], wB, 0.0)
            wC = xp.where(act[:-1, 1:, :] & act[1:, :-1, :], wC, 0.0)
            if wz is not None:
                wz = xp.where(act[:, :, :-1] & act[:, :, 1:], wz, 0.0)

        self.wA, self.wB, self.wC, self.wz = wA, wB, wC, wz

        diag = removal.copy()
        diag[:, :-1, :] += wA; diag[:, 1:, :] += wA
        diag[:-1, :, :] += wB; diag[1:, :, :] += wB
        diag[:-1, 1:, :] += wC; diag[1:, :-1, :] += wC
        if wz is not None:
            diag[:, :, :-1] += wz; diag[:, :, 1:] += wz

        # Core-surface Robin faces. A face is a boundary for an active cell when
        # its in-plane neighbour is inactive; add the Robin term to that cell.
        # The active mask must carry a one-cell void border (assert below), so
        # every boundary face is such an active/void interface and no off-array
        # face is missed.
        if alpha_edge != 0.0 and act is not None:
            if bool(act[0].any() or act[-1].any()
                    or act[:, 0].any() or act[:, -1].any()):
                raise ValueError("active mask must have a one-cell void border "
                                 "(use build helpers, which pad it)")

            def add_plane(diag_sl, self_act, nbr_act, D_sl):
                diag[diag_sl] += xp.where(
                    self_act & ~nbr_act, robin_plane(D_sl, alpha_edge), 0.0)

            # Both orientations of each of the three axial directions.
            add_plane((slice(None), slice(0, -1), slice(None)),
                      act[:, :-1, :], act[:, 1:, :], D[:, :-1, :])
            add_plane((slice(None), slice(1, None), slice(None)),
                      act[:, 1:, :], act[:, :-1, :], D[:, 1:, :])
            add_plane((slice(0, -1), slice(None), slice(None)),
                      act[:-1, :, :], act[1:, :, :], D[:-1, :, :])
            add_plane((slice(1, None), slice(None), slice(None)),
                      act[1:, :, :], act[:-1, :, :], D[1:, :, :])
            add_plane((slice(0, -1), slice(1, None), slice(None)),
                      act[:-1, 1:, :], act[1:, :-1, :], D[:-1, 1:, :])
            add_plane((slice(1, None), slice(0, -1), slice(None)),
                      act[1:, :-1, :], act[:-1, 1:, :], D[1:, :-1, :])

        # z boundary faces (Cartesian) use the bc spec on the z axis.
        if grid.shape[2] > 1:
            for alpha, sl in ((face_alpha(bc[2][0]), (slice(None), slice(None), 0)),
                              (face_alpha(bc[2][1]), (slice(None), slice(None), -1))):
                if alpha == 0.0:
                    continue
                term = robin_face_term(xp, D[sl], dz, alpha)
                if act is not None:
                    term = xp.where(act[sl], term, 0.0)
                diag[sl] += term

        if act is not None:
            diag = xp.where(act, diag, 1.0)

        self.diag = diag
        self.inv_diag = 1.0 / diag

    def apply(self, phi):
        out = self.diag * phi
        out[:, :-1, :] -= self.wA * phi[:, 1:, :]
        out[:, 1:, :] -= self.wA * phi[:, :-1, :]
        out[:-1, :, :] -= self.wB * phi[1:, :, :]
        out[1:, :, :] -= self.wB * phi[:-1, :, :]
        out[:-1, 1:, :] -= self.wC * phi[1:, :-1, :]
        out[1:, :-1, :] -= self.wC * phi[:-1, 1:, :]
        if self.wz is not None:
            out[:, :, :-1] -= self.wz * phi[:, :, 1:]
            out[:, :, 1:] -= self.wz * phi[:, :, :-1]
        return out


class HexDiffusionEigenSolver(DiffusionEigenSolver):
    """Multigroup hex-lattice diffusion k-eigenvalue solver."""

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        self.ops = [HexGroupOperator(self.xp, grid, diffusion[g], removal[g], bc=bc,
                                     active=self.active, mask_bc=self.mask_bc)
                    for g in range(self.n_groups)]
