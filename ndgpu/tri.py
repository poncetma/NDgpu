"""Body-fitted triangular finite-volume mesh for hexagonal cores.

A pointy-top hexagon subdivides *exactly* into 6 r^2 equilateral triangles, so
a triangular mesh refines a hex-assembly core with no staircase at assembly or
outer boundaries (unlike a fine hex lattice, whose nearest-cell boundary
inflates surface area and over-leaks). The triangular lattice is still
structured: cells are stored on an (nrows, ncols, 2) array where the last index
selects the down (0) and up (1) triangle of each rhombus, and every interior
triangle couples to three neighbours at fixed offsets -- so the operator is
again a handful of shifted multiply-adds and reuses the power-iteration solver.

For equilateral triangles of side h: shared-edge length h, centroid spacing
h/sqrt(3), area (sqrt(3)/4) h^2, giving an interior face coupling w = 4 D / h^2
and a boundary (Robin) term 8 D alpha / (h (h alpha + 2 sqrt(3) D)) for the law
J_net = alpha * phi_surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .operator import BC_VACUUM, face_alpha, normalize_bc
from .solver import DiffusionEigenSolver

_SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True)
class TriGrid:
    """Triangular lattice: shape (nrows, ncols, 2) (last axis = down/up), side h."""

    shape: tuple
    side: float            # triangle edge length h, cm
    height: float = 1.0    # slab thickness (2D), cm

    def __post_init__(self):
        if len(self.shape) != 3 or self.shape[2] != 2:
            raise ValueError("tri shape must be (nrows, ncols, 2)")
        if self.side <= 0:
            raise ValueError("side must be positive")

    @property
    def n_cells(self) -> int:
        r, c, _ = self.shape
        return r * c * 2

    @property
    def cell_volume(self) -> float:
        return (_SQRT3 / 4.0) * self.side**2 * self.height


class TriGroupOperator:
    """Within-group triangular-FV operator A = -div(D grad .) + Sigma_r.

    Cells: down = [..., 0], up = [..., 1]. A down triangle (i, j) couples to
    up(i, j) (shared hypotenuse), up(i-1, j) (bottom edge) and up(i, j-1) (left
    edge). Same (apply, inv_diag) interface and active-mask / Robin-boundary
    handling as the Cartesian and hex operators.
    """

    def __init__(self, xp, grid: TriGrid, D, removal, bc=BC_VACUUM, active=None,
                 mask_bc=BC_VACUUM):
        normalize_bc(bc)
        self.xp = xp
        self.shape = grid.shape
        h = grid.side
        kf = 4.0 / (h * h)
        alpha_edge = face_alpha(mask_bc)

        def hm(Da, Db):
            return 2.0 * Da * Db / (Da + Db)

        def robin(Dface, alpha):
            if alpha == 0.0:
                return xp.zeros_like(Dface)
            if math.isinf(alpha):
                return 8.0 * Dface / (h * h)
            return 8.0 * Dface * alpha / (h * (h * alpha + 2.0 * _SQRT3 * Dface))

        Dd, Du = D[:, :, 0], D[:, :, 1]                       # down, up sublattices
        w_hyp = hm(Dd, Du) * kf                               # down(i,j)-up(i,j)
        w_v = hm(Dd[1:, :], Du[:-1, :]) * kf                  # down(i,j)-up(i-1,j)
        w_h = hm(Dd[:, 1:], Du[:, :-1]) * kf                  # down(i,j)-up(i,j-1)

        act = None
        if active is not None:
            act = xp.asarray(active).astype(bool)
            if act.shape != grid.shape:
                raise ValueError("active mask shape must match tri grid shape")
            ad, au = act[:, :, 0], act[:, :, 1]
            w_hyp = xp.where(ad & au, w_hyp, 0.0)
            w_v = xp.where(ad[1:, :] & au[:-1, :], w_v, 0.0)
            w_h = xp.where(ad[:, 1:] & au[:, :-1], w_h, 0.0)
        self.w_hyp, self.w_v, self.w_h = w_hyp, w_v, w_h

        diag = removal.copy()
        diag[:, :, 0] += w_hyp; diag[:, :, 1] += w_hyp
        diag[1:, :, 0] += w_v;  diag[:-1, :, 1] += w_v
        diag[:, 1:, 0] += w_h;  diag[:, :-1, 1] += w_h

        # Robin faces: an active cell whose neighbour across an edge is inactive
        # (or off-array, via the required void border). Each triangle has three
        # edges; add the boundary term per exposed edge.
        if alpha_edge != 0.0 and act is not None:
            border = (act[0].any() or act[-1].any() or act[:, 0].any() or act[:, -1].any())
            if bool(border):
                raise ValueError("active mask must have a one-cell void border")
            ad, au = act[:, :, 0], act[:, :, 1]

            def add(diag_sl, self_act, nbr_act, D_sl):
                diag[diag_sl] += xp.where(self_act & ~nbr_act, robin(D_sl, alpha_edge), 0.0)

            # hypotenuse edge (down<->up, same rhombus)
            add((slice(None), slice(None), 0), ad, au, Dd)
            add((slice(None), slice(None), 1), au, ad, Du)
            # bottom edge  down(i,j)<->up(i-1,j)
            add((slice(1, None), slice(None), 0), ad[1:, :], au[:-1, :], Dd[1:, :])
            add((slice(0, -1), slice(None), 1), au[:-1, :], ad[1:, :], Du[:-1, :])
            # left edge    down(i,j)<->up(i,j-1)
            add((slice(None), slice(1, None), 0), ad[:, 1:], au[:, :-1], Dd[:, 1:])
            add((slice(None), slice(0, -1), 1), au[:, :-1], ad[:, 1:], Du[:, :-1])

            diag = xp.where(act, diag, 1.0)

        self.diag = diag
        self.inv_diag = 1.0 / diag

    def apply(self, phi):
        out = self.diag * phi
        w_hyp, w_v, w_h = self.w_hyp, self.w_v, self.w_h
        out[:, :, 0] -= w_hyp * phi[:, :, 1]
        out[:, :, 1] -= w_hyp * phi[:, :, 0]
        out[1:, :, 0] -= w_v * phi[:-1, :, 1]
        out[:-1, :, 1] -= w_v * phi[1:, :, 0]
        out[:, 1:, 0] -= w_h * phi[:, :-1, 1]
        out[:, :-1, 1] -= w_h * phi[:, 1:, 0]
        return out


class TriDiffusionEigenSolver(DiffusionEigenSolver):
    """Multigroup triangular-FV diffusion k-eigenvalue solver."""

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        self.ops = [TriGroupOperator(self.xp, grid, diffusion[g], removal[g], bc=bc,
                                     active=self.active, mask_bc=self.mask_bc)
                    for g in range(self.n_groups)]
