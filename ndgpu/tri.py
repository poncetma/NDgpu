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

import numpy as np
from dataclasses import dataclass


from .sp3 import SP3GroupOperator
from .backend import asnumpy
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

    #: apply() accepts out=, so block operators can write straight into a row
    #: of their state instead of allocating a temporary per moment.
    supports_out = True

    def __init__(self, xp, grid: TriGrid, D, removal, bc=BC_VACUUM, active=None,
                 mask_bc=BC_VACUUM, df=None, bcf=None):
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

        # Face coupling. Without discontinuity factors the two directions share
        # one symmetric weight (the harmonic-mean two-point current). With them,
        # continuity of the *heterogeneous* surface flux, f_L phi_s^L =
        # f_R phi_s^R, gives
        #     J = (f_L phi_L - f_R phi_R) / (h (f_L/2D_L + f_R/2D_R))
        # so the face carries a PAIR (a, b) = (f_L C, f_R C) with
        # C = kf / (f_L/2D_L + f_R/2D_R). At f_L = f_R = 1 this collapses to
        # C = harmonic_mean(D_L, D_R) * kf, i.e. exactly the symmetric weight --
        # which is the regression test for this path. The operator becomes
        # non-symmetric, hence linear_solver must be a non-symmetric one.
        def pair(DL, DR, fL, fR):
            if df is None:
                w = harmonic_mean(DL, DR) * kf
                return w, w
            C = kf / (fL / (2.0 * DL) + fR / (2.0 * DR))
            return fL * C, fR * C

        # df may be given two ways:
        #   per CELL, an array shaped like the grid -- one factor per cell, so a
        #     region presents the SAME factor to every neighbour; or
        #   per FACE, a 6-tuple (fL, fR) x (hyp, v, h) of arrays matching each
        #     face family -- one factor per ORDERED region pair, which is what
        #     classical GET actually specifies.
        # The per-cell form cannot satisfy per-surface reference currents: a
        # region touching n neighbours gets n conditions but one unknown, and
        # least squares drives it to its bound (measured on HP-MR: cost stalled
        # at 0.678 with every drum factor pegged at exp(+-1.2)).
        if isinstance(df, tuple):
            if len(df) != 6:
                raise ValueError("per-face df must be a 6-tuple "
                                 "(fL, fR) for the hyp, v and h face families")
            fL_hyp, fR_hyp, fL_v, fR_v, fL_h, fR_h = (xp.asarray(f) for f in df)
        elif df is not None:
            df = xp.asarray(df)
            fd, fu = df[:, :, 0], df[:, :, 1]
            fL_hyp, fR_hyp = fd, fu
            fL_v, fR_v = fd[1:, :], fu[:-1, :]
            fL_h, fR_h = fd[:, 1:], fu[:, :-1]
        else:
            fL_hyp = fR_hyp = fL_v = fR_v = fL_h = fR_h = None
        a_hyp, b_hyp = pair(Dd, Du, fL_hyp, fR_hyp)
        a_v, b_v = pair(Dd[1:, :], Du[:-1, :], fL_v, fR_v)
        a_h, b_h = pair(Dd[:, 1:], Du[:, :-1], fL_h, fR_h)
        # Half-face transmissibilities t = 2 D kf and the per-face factors, kept
        # so tri_partial_currents can reconstruct the GET surface flux
        # phi_s^L = phi_L - J/t_L and Phi_s = f_L phi_s^L. (Check: 1/t_L + 1/t_R
        # = 1/w at f = 1, and C = 1/(f_L/t_L + f_R/t_R) reproduces `pair`.)
        one = lambda a: xp.ones_like(a)
        # Geometry needed to turn operator-space face terms into true currents.
        # w (phi_L - phi_R) is current x area / volume, so the current itself is
        # that times V/A = (sqrt3/4) h^2 / h. Without this the phi_s/4 and J/2
        # halves of J_out = phi_s/4 + J/2 are in different units.
        self.h = h
        self.alpha_edge = alpha_edge
        self.j_scale = (_SQRT3 / 4.0) * h
        # Face area, so partial currents can be reported edge-INTEGRATED. The
        # S_N reference (aggregate_partial_currents) already folds the edge
        # length in, so a per-unit-area coarse current is inconsistent with it by
        # a factor h. That used to cancel in an all-row gauge; once the gauge is
        # read off the flux rows only it is a real offset, and it showed up as
        # the coarse/reference ratio splitting into distinct clusters (flux ~31,
        # interface ~10, boundary ~5) where one common value was required.
        self.face_area = h * dz
        self.D_cell = D
        if not hasattr(self, "bcf_arr"):
            self.bcf_arr = None if bcf is None else xp.asarray(bcf)
        self.face_fac = {
            "hyp": (fL_hyp if df is not None else one(Dd),
                    fR_hyp if df is not None else one(Du),
                    2.0 * Dd * kf, 2.0 * Du * kf),
            "v": (fL_v if df is not None else one(Dd[1:, :]),
                  fR_v if df is not None else one(Du[:-1, :]),
                  2.0 * Dd[1:, :] * kf, 2.0 * Du[:-1, :] * kf),
            "h": (fL_h if df is not None else one(Dd[:, 1:]),
                  fR_h if df is not None else one(Du[:, :-1]),
                  2.0 * Dd[:, 1:] * kf, 2.0 * Du[:, :-1] * kf)}
        w_hyp, w_v, w_h = a_hyp, a_v, a_h
        wz = (harmonic_mean(D[..., :-1], D[..., 1:]) / dz**2   # axial neighbour
              if len(grid.shape) == 4 and grid.shape[3] > 1 else None)

        act = None
        if active is not None:
            act = xp.asarray(active).astype(bool)
            if act.shape != grid.shape:
                raise ValueError("active mask shape must match tri grid shape")
            ad, au = act[:, :, 0], act[:, :, 1]
            m_hyp, m_v = ad & au, ad[1:, :] & au[:-1, :]
            m_h = ad[:, 1:] & au[:, :-1]
            a_hyp = xp.where(m_hyp, a_hyp, 0.0); b_hyp = xp.where(m_hyp, b_hyp, 0.0)
            a_v = xp.where(m_v, a_v, 0.0); b_v = xp.where(m_v, b_v, 0.0)
            a_h = xp.where(m_h, a_h, 0.0); b_h = xp.where(m_h, b_h, 0.0)
            w_hyp, w_v, w_h = a_hyp, a_v, a_h
            if wz is not None:
                wz = xp.where(act[..., :-1] & act[..., 1:], wz, 0.0)
        self.w_hyp, self.w_v, self.w_h, self.wz = w_hyp, w_v, w_h, wz
        self.a_hyp, self.b_hyp = a_hyp, b_hyp
        self.a_v, self.b_v = a_v, b_v
        self.a_h, self.b_h = a_h, b_h

        diag = removal.copy()
        diag[:, :, 0] += a_hyp; diag[:, :, 1] += b_hyp
        diag[1:, :, 0] += a_v;  diag[:-1, :, 1] += b_v
        diag[:, 1:, 0] += a_h;  diag[:, :-1, 1] += b_h
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

            # Boundary coefficient: a generalized-equivalence BCf scales the
            # Marshak albedo on masked/outer faces so the coarse solve can match
            # the reference PARTIAL OUTGOING current there. Discontinuity
            # factors act only on interior faces, so without this the vacuum
            # boundary -- where a leaky core's equivalence error actually lives
            # -- has no free parameter at all. bcf = 1 leaves it untouched.
            bcf_arr = None if bcf is None else xp.asarray(bcf)
            # Kept so tri_partial_currents can reproduce the operator's ACTUAL
            # boundary leakage. It previously used a bare phi/4, which carries no
            # dependence on D, alpha or bcf -- so the boundary condition it
            # matched was disconnected from the leakage the operator applies, and
            # the fit could drive the residual to machine zero (measured 1.2e-28)
            # with k still 350 pcm off.
            self.bcf_arr = bcf_arr

            def add(diag_sl, self_act, nbr_act, D_sl):
                if bcf_arr is None:
                    term = robin(D_sl, alpha_edge)
                else:
                    a_loc = alpha_edge * bcf_arr[diag_sl]
                    term = xp.where(
                        a_loc > 0.0,
                        8.0 * D_sl * a_loc / (h * (h * a_loc + 2.0 * _SQRT3 * D_sl)),
                        0.0)
                diag[diag_sl] += xp.where(self_act & ~nbr_act, term, 0.0)

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
        self._fused = {}            # dtype -> fused-kernel argument buffers
        self._op_dtype = None       # promoted dtype of the stencil arrays

    def _fusable(self, dtype):
        """True if A phi has dtype `dtype`, so the cached arrays can be cast to
        it losslessly. Refuses a real phi against the complex removal the noise
        solver builds, where casting down would drop the imaginary part."""
        if self._op_dtype is None:
            self._op_dtype = np.result_type(
                self.diag.dtype, self.a_hyp.dtype, self.a_v.dtype,
                self.a_h.dtype,
                *([] if self.wz is None else [self.wz.dtype]))
        return np.result_type(self._op_dtype, dtype) == dtype

    def _fused_arrays(self, dtype):
        """Contiguous, common-dtype copies of the coupling arrays for the kernel.

        Zero-size families (a grid one cell wide, so there are no v or h faces)
        become 1-element dummies: the kernel always needs a valid pointer, and
        the branches that would read them are dead in that case.
        """
        buf = self._fused.get(dtype)
        if buf is None:
            xp = self.xp

            def prep(a):
                a = xp.ascontiguousarray(a, dtype=dtype)
                return a if a.size else xp.zeros(1, dtype=dtype)

            nz = self.shape[3] if len(self.shape) == 4 else 1
            buf = (prep(self.diag), prep(self.a_hyp), prep(self.b_hyp),
                   prep(self.a_v), prep(self.b_v), prep(self.a_h),
                   prep(self.b_h),
                   None if self.wz is None else prep(self.wz),
                   self.shape[0], self.shape[1], nz)
            self._fused[dtype] = buf
        return buf

    def assemble(self):
        """Sparse CSR form of this operator.

        The solver is matrix-free, but an ILU preconditioner needs the matrix.
        Only the discontinuity-factor path really needs this: with df the face
        coupling is asymmetric (diag gets a on one side, b on the other), so the
        margin diag - sum|offdiag| is C*(f_L - f_R) and goes NEGATIVE as soon as
        a neighbour's factor is larger. The operator is then no longer an
        M-matrix, and the Jacobi (inv_diag) preconditioner the solvers default to
        has lost its premise -- measured: GMRES stalls at residual 2.7e4 against
        a 3.1e1 target. ILU does not assume diagonal dominance.
        """
        import numpy as _np
        import scipy.sparse as _sp
        shape = self.shape
        idx = _np.arange(int(_np.prod(shape))).reshape(shape)
        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(_np.asarray(r).ravel())
            cols.append(_np.asarray(c).ravel())
            vals.append(_np.asarray(asnumpy(v), dtype=float).ravel())

        add(idx, idx, asnumpy(self.diag))
        # Each face contributes -b to the L row and -a to the R row (a == b
        # without discontinuity factors, recovering the symmetric matrix).
        add(idx[:, :, 0], idx[:, :, 1], -asnumpy(self.b_hyp))
        add(idx[:, :, 1], idx[:, :, 0], -asnumpy(self.a_hyp))
        add(idx[1:, :, 0], idx[:-1, :, 1], -asnumpy(self.b_v))
        add(idx[:-1, :, 1], idx[1:, :, 0], -asnumpy(self.a_v))
        add(idx[:, 1:, 0], idx[:, :-1, 1], -asnumpy(self.b_h))
        add(idx[:, :-1, 1], idx[:, 1:, 0], -asnumpy(self.a_h))
        if self.wz is not None:
            wz = asnumpy(self.wz)
            add(idx[..., :-1], idx[..., 1:], -wz)
            add(idx[..., 1:], idx[..., :-1], -wz)
        n = idx.size
        return _sp.csr_matrix((_np.concatenate(vals),
                               (_np.concatenate(rows), _np.concatenate(cols))),
                              shape=(n, n))

    def ilu_preconditioner(self, drop_tol=1e-4, fill_factor=10.0):
        """ILU(0)-style preconditioner as a callable, for the df path.

        Returns f(x) ~ A^-1 x with x shaped like the flux, suitable for the
        ``precond=`` argument of ndgpu's gmres/bicgstab.
        """
        from scipy.sparse.linalg import spilu
        lu = spilu(self.assemble().tocsc(), drop_tol=drop_tol,
                   fill_factor=fill_factor)
        shape = self.shape

        def apply_prec(x):
            import numpy as _np
            flat = _np.asarray(asnumpy(x), dtype=float).ravel()
            return self.xp.asarray(lu.solve(flat).reshape(shape))

        return apply_prec

    def apply(self, phi, out=None):
        """Return A phi (allocating the output unless ``out`` is given).

        The slice form below is ~15 kernel launches; on GPU it collapses into
        one hand-written kernel (see :mod:`ndgpu.kernels`). The NumPy path is
        what runs on CPU, when phi would have to be promoted, or with fusion
        switched off.
        """
        from . import kernels

        if (kernels.use_fused(self.xp, "stencil") and phi.flags.c_contiguous
                and self._fusable(phi.dtype)):
            dg, ah, bh, av, bv, ahh, bhh, wz, nx, ny, nz = \
                self._fused_arrays(phi.dtype)
            return kernels.tri_stencil_apply(self.xp, phi, dg, ah, bh, av, bv,
                                             ahh, bhh, wz, nx, ny, nz, out=out)
        if out is None:
            out = self.diag * phi
        else:
            self.xp.multiply(self.diag, phi, out=out)
        out[:, :, 0] -= self.b_hyp * phi[:, :, 1]
        out[:, :, 1] -= self.a_hyp * phi[:, :, 0]
        out[1:, :, 0] -= self.b_v * phi[:-1, :, 1]
        out[:-1, :, 1] -= self.a_v * phi[1:, :, 0]
        out[:, 1:, 0] -= self.b_h * phi[:, :-1, 1]
        out[:, :-1, 1] -= self.a_h * phi[:, 1:, 0]
        if self.wz is not None:
            out[..., :-1] -= self.wz * phi[..., 1:]
            out[..., 1:] -= self.wz * phi[..., :-1]
        return out


def face_df_from_pairs(region_map, df_by_pair, shape=None):
    """Per-face discontinuity-factor arrays from per-ordered-region-pair factors.

    Returns the 6-tuple (fL, fR) x (hyp, v, h) that TriGroupOperator accepts as
    its per-face ``df``. ``df_by_pair`` maps an ORDERED pair (region_from,
    region_to) to that surface's factor -- the same keying as the reference
    interfaces from TriSNTransportSolver.aggregate_partial_currents -- so each
    surface carries its own unknown, which is what GET specifies and what the
    per-cell form cannot represent.

    Faces interior to one region get 1 on both sides: there is no discontinuity
    inside a homogeneous region. (They would cancel anyway -- f_L = f_R = f
    gives weights f C = kf D -- but making it explicit keeps the interior
    stencil bit-identical to the no-DF operator.)
    """
    r = np.asarray(region_map)
    if shape is not None and tuple(r.shape) != tuple(shape):
        raise ValueError("region_map shape must match the grid")

    def lookup(ra, rb):
        """Factor each side of a face presents, as arrays over that face family."""
        fL = np.ones(ra.shape, dtype=float)
        fR = np.ones(ra.shape, dtype=float)
        for (a, b), v in df_by_pair.items():
            if a == b:
                continue
            m = (ra == a) & (rb == b)
            if m.any():
                fL[m] = v
                fR[m] = df_by_pair.get((b, a), 1.0)
        return fL, fR

    rd, ru = r[:, :, 0], r[:, :, 1]
    return (*lookup(rd, ru),
            *lookup(rd[1:, :], ru[:-1, :]),
            *lookup(rd[:, 1:], ru[:, :-1]))


def tri_partial_currents(op, phi, region_map, active=None):
    """Coarse-side partial currents on equivalence-region surfaces.

    The reference side comes from the S_N angular flux
    (TriSNTransportSolver.partial_currents_from_psi + aggregate_partial_currents);
    this is the diffusion counterpart, so the two can be equated to generate
    discontinuity factors (interior interfaces) and boundary coefficients
    (vacuum boundaries).

    Diffusion has no angular flux, so the partial currents come from the P1
    relation on each face:

        J_out = phi_s / 4 + J / 2,   J_in = phi_s / 4 - J / 2

    with the net two-point current J = w (phi_L - phi_R) taken OUTWARD from the
    source cell and the surface flux phi_s approximated by the face average.
    On a vacuum boundary the Marshak condition gives J_in = 0 and phi_s = 2 J,
    so J_out = J -- the leakage itself.

    Returns (interfaces, boundaries) in the same form as
    TriSNTransportSolver.aggregate_partial_currents.
    """
    from .tri_sn import _EDGES
    nr, nc = op.shape[0], op.shape[1]
    reg = np.asarray(region_map).reshape(nr, nc, 2)
    phi = np.asarray(asnumpy(phi)).reshape(nr, nc, 2)
    act = None if active is None else np.asarray(active).reshape(nr, nc, 2)
    # Face weights, per _EDGES family: hyp, bottom, left (each seen from both
    # sublattices). a is the coefficient on the SOURCE cell, b on the neighbour,
    # so entries 3-5 -- where the source is the up triangle -- must present the
    # pair REVERSED. Entries 4 and 5 previously did not, which is invisible while
    # the weights are symmetric (no df, uniform D: a == b) and breaks reciprocity
    # as soon as they are not. Likewise (f, t) come from the source side.
    ff = op.face_fac
    W = {0: (op.a_hyp, op.b_hyp, (slice(None), slice(None)), ff["hyp"][0], ff["hyp"][2]),
         1: (op.a_v, op.b_v, (slice(1, None), slice(None)), ff["v"][0], ff["v"][2]),
         2: (op.a_h, op.b_h, (slice(None), slice(1, None)), ff["h"][0], ff["h"][2]),
         3: (op.b_hyp, op.a_hyp, (slice(None), slice(None)), ff["hyp"][1], ff["hyp"][3]),
         4: (op.b_v, op.a_v, (slice(0, -1), slice(None)), ff["v"][1], ff["v"][3]),
         5: (op.b_h, op.a_h, (slice(None), slice(0, -1)), ff["h"][1], ff["h"][3])}
    interfaces, boundaries = {}, {}
    for k, (t, di, dj, tn, _n) in enumerate(_EDGES):
        a_w, b_w, sl, f_s, t_s = W[k]
        a_w = np.asarray(asnumpy(a_w)); b_w = np.asarray(asnumpy(b_w))
        f_s = np.asarray(asnumpy(f_s)); t_s = np.asarray(asnumpy(t_s))
        for i in range(nr):
            ii = i + di
            for j in range(nc):
                jj = j + dj
                if act is not None and not act[i, j, t]:
                    continue
                rs = int(reg[i, j, t])
                inside = (0 <= ii < nr) and (0 <= jj < nc)
                nbr_ok = inside and (act is None or bool(act[ii, jj, tn]))
                pL = float(phi[i, j, t])
                if not nbr_ok:                       # vacuum boundary (Marshak)
                    # The operator's own Robin term, so BCf actually moves this
                    # quantity: leak = 8 D a / (h (h a + 2 sqrt3 D)) * phi, with
                    # a = alpha_edge * bcf, converted to a true current by
                    # j_scale. J_in = 0 there, so J_out is the leakage itself.
                    a_loc = op.alpha_edge * (1.0 if op.bcf_arr is None
                                             else float(asnumpy(op.bcf_arr)[i, j, t]))
                    Dc = float(asnumpy(op.D_cell)[i, j, t])
                    if a_loc > 0.0:
                        term = (8.0 * Dc * a_loc
                                / (op.h * (op.h * a_loc + 2.0 * _SQRT3 * Dc)))
                    else:
                        term = 0.0
                    b = boundaries.setdefault(rs, [0.0, 0.0])
                    b[0] += op.face_area * op.j_scale * term * pL
                    continue
                rn = int(reg[ii, jj, tn])
                if rn == rs:
                    continue
                # index the face-weight array for this (i, j) within its slice
                ri = i - (sl[0].start or 0)
                cj = j - (sl[1].start or 0)
                try:
                    wa = float(a_w[ri, cj]); wb = float(b_w[ri, cj])
                    fs = float(f_s[ri, cj]); ts = float(t_s[ri, cj])
                except (IndexError, ValueError):
                    continue
                pR = float(phi[ii, jj, tn])
                J_op = wa * pL - wb * pR             # outward, operator space
                # GET surface flux, seen from the SOURCE side. The old face
                # average 0.5 (phi_L + phi_R) is the D_L = D_R, f = 1 limit of
                # this and is wrong exactly where DFs matter -- it made the fitted
                # functional differ from the reference one, so the least squares
                # drove k away (measured: cost 1.05e-2 at k = 0.868 vs a
                # reference 0.9939).
                # phi_s^L = phi_L - J/t_L uses the OPERATOR-space current, since
                # t = 2 D kf lives in that space too; only the current entering
                # the partial-current combination below is converted.
                phi_s = fs * (pL - J_op / ts) if ts > 0.0 else 0.5 * (pL + pR)
                Jnet = op.j_scale * J_op
                e = interfaces.setdefault((rs, rn), [0.0, 0.0])
                e[0] += op.face_area * (0.25 * phi_s + 0.5 * Jnet)
                e[1] += op.face_area * (0.25 * phi_s - 0.5 * Jnet)
    return interfaces, boundaries


class TriDiffusionEigenSolver(DiffusionEigenSolver):
    """Multigroup triangular-FV diffusion k-eigenvalue solver.

    ``df`` optionally supplies per-group discontinuity factors, shape
    (G, *grid.shape), for generalized-equivalence (GET) homogenization: the
    face coupling then enforces continuity of the *heterogeneous* surface flux
    rather than the homogeneous one, which is the degree of freedom SPH lacks
    (SPH preserves region reaction rates but not interface leakage). Omitting it
    -- or passing all ones -- reproduces the symmetric operator exactly.

    NOTE the operator is non-symmetric once df varies across a face, so a
    non-symmetric ``linear_solver`` (gmres/bicgstab) is required; CG assumes
    symmetry and will not converge to the right answer.
    """

    def __init__(self, *args, df=None, bcf=None, **kwargs):
        # Set before super().__init__, which calls _build_operators.
        self.df = df
        self.bcf = bcf
        super().__init__(*args, **kwargs)
        if df is not None:
            # With discontinuity factors the face coupling is asymmetric, the
            # operator is no longer an M-matrix (diag - sum|offdiag| goes
            # negative wherever a neighbour's factor is larger), and the default
            # Jacobi/Neumann preconditioner loses its premise -- GMRES stalled
            # at residual 2.7e4 against a 3.1e1 target. ILU makes no diagonal
            # dominance assumption.
            self.preconds = [op.ilu_preconditioner() for op in self.ops]

    def _build_operators(self, grid, diffusion, sigma_t, removal, bc):
        df = getattr(self, "df", None)
        bcf = getattr(self, "bcf", None)
        self.ops = [TriGroupOperator(self.xp, grid, diffusion[g], removal[g], bc=bc,
                                     active=self.active, mask_bc=self.mask_bc,
                                     df=None if df is None else df[g],
                                     bcf=None if bcf is None else bcf[g])
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
