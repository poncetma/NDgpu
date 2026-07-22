"""Body-fitted triangular finite-volume mesh for hexagonal cores (2D or
extruded 3D prisms).

A pointy-top hexagon subdivides *exactly* into 6 r^2 equilateral triangles, so
a triangular mesh refines a hex-assembly core with no staircase at assembly or
outer boundaries (unlike a fine hex lattice, whose nearest-cell boundary
inflates surface area and over-leaks). The triangular lattice is still
structured: cells are stored on an (nrows, ncols, 2) array where the last index
selects the down (0) and up (1) triangle of each rhombus, and every interior
triangle couples to three neighbours at fixed offsets -- so the operator is
again a handful of shifted multiply-adds and reuses the power-iteration solver.

Extruded reactors (any 2D cross-section swept vertically -- the geometry of
prismatic microreactor cores) add a trailing z axis: shape (nrows, ncols, 2,
nz) with uniform layer height dz. The in-plane stencil broadcasts over z
unchanged; the z coupling is the Cartesian one (hm(D) / dz^2), and the z faces
take their boundary condition from the ``bc`` spec's z axis (in-plane
boundaries are governed by the active mask's ``mask_bc``, as in 2D).

For equilateral triangles of side h: shared-edge length h, centroid spacing
h/sqrt(3), area (sqrt(3)/4) h^2, giving an interior face coupling w = 4 D / h^2
and a boundary (Robin) term 8 D alpha / (h (h alpha + 2 sqrt(3) D)) for the law
J_net = alpha * phi_surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


from .sp3 import SP3GroupOperator
from .stencil import (BC_VACUUM, face_alpha, harmonic_mean, normalize_bc,
                      robin_face_term)
from .solver import (DiffusionEigenSolver, SDPNEigenSolver, SP3EigenSolver,
                     SPNEigenSolver)

_SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True)
class TriGrid:
    """Triangular lattice: shape (nrows, ncols, 2) (2D; axis 2 = down/up
    triangle) or (nrows, ncols, 2, nz) (extruded prisms), side h. ``height``
    is the slab thickness in 2D and the total z extent in 3D."""

    shape: tuple
    side: float            # triangle edge length h, cm
    height: float = 1.0    # slab thickness (2D) / total z extent (3D), cm

    def __post_init__(self):
        if len(self.shape) not in (3, 4) or self.shape[2] != 2:
            raise ValueError("tri shape must be (nrows, ncols, 2[, nz])")
        if self.side <= 0 or self.height <= 0:
            raise ValueError("side and height must be positive")

    @property
    def nz(self) -> int:
        return self.shape[3] if len(self.shape) == 4 else 1

    @property
    def dz(self) -> float:
        return self.height / self.nz

    @property
    def n_cells(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def cell_volume(self) -> float:
        return (_SQRT3 / 4.0) * self.side**2 * self.dz


class TriGroupOperator:
    """Within-group triangular-FV operator A = -div(D grad .) + Sigma_r.

    Cells: down = [..., 0], up = [..., 1]. A down triangle (i, j) couples to
    up(i, j) (shared hypotenuse), up(i-1, j) (bottom edge) and up(i, j-1) (left
    edge). Same (apply, inv_diag) interface and active-mask / Robin-boundary
    handling as the Cartesian and hex operators.
    """

    def __init__(self, xp, grid: TriGrid, D, removal, bc=BC_VACUUM, active=None,
                 mask_bc=BC_VACUUM):
        bc = normalize_bc(bc)  # only the z faces read bc; in-plane uses mask_bc
        self.xp = xp
        self.shape = grid.shape
        h, dz = grid.side, grid.dz
        kf = 4.0 / (h * h)
        alpha_edge = face_alpha(mask_bc)

        # In-plane boundary term: same derivation as robin_face_term, with the
        # triangle edge length h, cell area (sqrt(3)/4) h^2 and centre-to-edge
        # distance h/(2 sqrt(3)), giving 8 D alpha / (h (h alpha + 2 sqrt(3) D)).
        def robin(Dface, alpha):
            if alpha == 0.0:
                return xp.zeros_like(Dface)
            if math.isinf(alpha):
                return 8.0 * Dface / (h * h)
            return 8.0 * Dface * alpha / (h * (h * alpha + 2.0 * _SQRT3 * Dface))

        Dd, Du = D[:, :, 0], D[:, :, 1]                # down, up sublattices
        w_hyp = harmonic_mean(Dd, Du) * kf             # down(i,j)-up(i,j)
        w_v = harmonic_mean(Dd[1:, :], Du[:-1, :]) * kf   # down(i,j)-up(i-1,j)
        w_h = harmonic_mean(Dd[:, 1:], Du[:, :-1]) * kf   # down(i,j)-up(i,j-1)
        wz = (harmonic_mean(D[..., :-1], D[..., 1:]) / dz**2   # axial neighbour
              if len(grid.shape) == 4 and grid.shape[3] > 1 else None)

        act = None
        if active is not None:
            act = xp.asarray(active).astype(bool)
            if act.shape != grid.shape:
                raise ValueError("active mask shape must match tri grid shape")
            ad, au = act[:, :, 0], act[:, :, 1]
            w_hyp = xp.where(ad & au, w_hyp, 0.0)
            w_v = xp.where(ad[1:, :] & au[:-1, :], w_v, 0.0)
            w_h = xp.where(ad[:, 1:] & au[:, :-1], w_h, 0.0)
            if wz is not None:
                wz = xp.where(act[..., :-1] & act[..., 1:], wz, 0.0)
        self.w_hyp, self.w_v, self.w_h, self.wz = w_hyp, w_v, w_h, wz

        diag = removal.copy()
        diag[:, :, 0] += w_hyp; diag[:, :, 1] += w_hyp
        diag[1:, :, 0] += w_v;  diag[:-1, :, 1] += w_v
        diag[:, 1:, 0] += w_h;  diag[:, :-1, 1] += w_h
        if wz is not None:
            diag[..., :-1] += wz; diag[..., 1:] += wz

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

        # z boundary faces (extruded grids) use the bc spec's z axis.
        if len(grid.shape) == 4:
            for alpha, sl in ((face_alpha(bc[2][0]), (Ellipsis, 0)),
                              (face_alpha(bc[2][1]), (Ellipsis, -1))):
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
        w_hyp, w_v, w_h = self.w_hyp, self.w_v, self.w_h
        out[:, :, 0] -= w_hyp * phi[:, :, 1]
        out[:, :, 1] -= w_hyp * phi[:, :, 0]
        out[1:, :, 0] -= w_v * phi[:-1, :, 1]
        out[:-1, :, 1] -= w_v * phi[1:, :, 0]
        out[:, 1:, 0] -= w_h * phi[:, :-1, 1]
        out[:, :-1, 1] -= w_h * phi[:, 1:, 0]
        if self.wz is not None:
            out[..., :-1] -= self.wz * phi[..., 1:]
            out[..., 1:] -= self.wz * phi[..., :-1]
        return out


class TriDiffusionEigenSolver(DiffusionEigenSolver):
    """Multigroup triangular-FV diffusion k-eigenvalue solver."""

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        self.ops = [TriGroupOperator(self.xp, grid, diffusion[g], removal[g], bc=bc,
                                     active=self.active, mask_bc=self.mask_bc)
                    for g in range(self.n_groups)]


class TriSP3EigenSolver(SP3EigenSolver):
    """Multigroup simplified-P3 k-eigenvalue solver on the triangular mesh.

    SP3 on the body-fitted hex/triangular geometry: the same angular block as
    the Cartesian :class:`~ndgpu.SP3EigenSolver`, with the in-plane leakage of
    both SP3 moments discretized by :class:`TriGroupOperator`. It captures the
    transport effects (steep gradients at strong absorbers and small cores)
    that triangular diffusion misses -- e.g. it serves as the transport
    reference for SPH homogenization on the HP-MR core. Same interface and ~2x
    the per-group work of :class:`TriDiffusionEigenSolver`; the scalar flux is
    phi0 = Phi1 - 2 phi2 (moment state carried internally)."""

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        self.ops = [SP3GroupOperator(self.xp, grid, diffusion[g], sigma_t[g],
                                     removal[g], bc=bc, active=self.active,
                                     variant=self._sp_variant,
                                     mask_bc=self.mask_bc, op_cls=TriGroupOperator,
                                     hybrid_mask=self.hybrid_mask,
                                     hybrid_confine=self.hybrid_confine)
                    for g in range(self.n_groups)]


class TriSDP1EigenSolver(TriSP3EigenSolver):
    """Simplified double-P1 (SDP1) k-eigenvalue solver on the triangular mesh.

    The body-fitted hex/triangular counterpart of :class:`~ndgpu.SDP1EigenSolver`:
    same angular block as :class:`TriSP3EigenSolver` at identical cost, but with
    the double-P1 second-moment coefficient (see
    :class:`~ndgpu.operator.SP3GroupOperator`). Preferable to SP3 on the HP-MR
    core, where strong drum absorbers drive steep, near-discontinuous angular
    flux gradients.
    """

    _sp_variant = "sdp1"


class TriSDPNEigenSolver(SDPNEigenSolver):
    """Simplified double-PN (SDPN, N=2/3) solver on the triangular mesh: the
    body-fitted counterpart of :class:`~ndgpu.SDPNEigenSolver`, with every
    moment's in-plane leakage discretized by :class:`TriGroupOperator`."""

    _moment_op_cls = TriGroupOperator


class TriSDP2EigenSolver(TriSDPNEigenSolver):
    """SDP2 (3-moment) k-eigenvalue solver on the triangular mesh."""

    _order = 2


class TriSDP3EigenSolver(TriSDPNEigenSolver):
    """SDP3 (4-moment) k-eigenvalue solver on the triangular mesh."""

    _order = 3


class TriSPNEigenSolver(SPNEigenSolver):
    """Standard SPN (SP5/SP7) k-eigenvalue solver on the triangular mesh."""

    _moment_op_cls = TriGroupOperator


class TriSP1EigenSolver(TriSPNEigenSolver):
    """SP1 (1-moment) k-eigenvalue solver on the triangular mesh -- equivalent
    to triangular-mesh diffusion; see :class:`ndgpu.solver.SP1EigenSolver`."""

    _order = 0


class TriSP5EigenSolver(TriSPNEigenSolver):
    """SP5 (3-moment) k-eigenvalue solver on the triangular mesh."""

    _order = 2


class TriSP7EigenSolver(TriSPNEigenSolver):
    """SP7 (4-moment) k-eigenvalue solver on the triangular mesh."""

    _order = 3
