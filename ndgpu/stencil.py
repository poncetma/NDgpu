"""Matrix-free finite-volume diffusion stencil and boundary conditions.

Discretizes  A phi = -div(D grad phi) + Sigma_r phi  with a cell-centered
7-point stencil. Face diffusion coefficients use the harmonic mean, which is
exact for piecewise-constant D (so heterogeneous cores are handled correctly).

The operator is deliberately matrix-free: instead of assembling a sparse
matrix, we precompute one face-coupling array per axis plus the diagonal, and
`apply()` is six shifted multiply-adds. On the GPU backend each of these is a
fused elementwise CUDA kernel over contiguous memory -- this is both faster and
lighter than sparse CSR SpMV, and it keeps the entire operator on-device.

The resulting operator is symmetric positive definite as long as removal is
positive somewhere (zero-flux BC) which is what lets us use CG. On
cylindrical grids the SPD property comes from volume-weighting the equations;
``symmetric=False`` opts out of that and builds the natural divergence form
instead, which is non-symmetric and needs GMRES/BiCGStab (see ndgpu.linalg).

This module also owns the boundary-condition vocabulary (``normalize_bc``,
``face_alpha``, ``robin_face_term``) shared by every spatial operator -- the
hex and triangular stencils and the SP3/SDPN angular blocks all reuse it.
"""

from __future__ import annotations

import math

import numpy as np

BC_ZERO_FLUX = "zero-flux"
BC_REFLECTIVE = "reflective"
BC_VACUUM = "vacuum"

# Robin coefficient alpha in the boundary law  J_net = alpha * phi_surface
# (the leakage current out of the domain, per unit surface flux). The diffusion
# Marshak vacuum condition gives alpha = 1/2 (FEMFFUSION's bc_factor); a custom
# albedo boundary passes alpha directly. Reflective is alpha = 0 (zero current)
# and zero-flux is the alpha -> infinity (Dirichlet) limit.
_VACUUM_ALPHA = 0.5


def _face_valid(f):
    """Validate and canonicalize one face spec: a known string or an albedo."""
    if f in (BC_ZERO_FLUX, BC_REFLECTIVE, BC_VACUUM):
        return f
    try:
        alpha = float(f)
    except (TypeError, ValueError):
        raise ValueError(
            f"unknown bc {f!r}; use {BC_ZERO_FLUX!r}, {BC_REFLECTIVE!r}, "
            f"{BC_VACUUM!r}, or a non-negative albedo coefficient")
    if alpha < 0:
        raise ValueError(f"albedo coefficient must be non-negative, got {alpha}")
    return alpha


def face_alpha(spec):
    """Robin coefficient alpha for one face spec (see _VACUUM_ALPHA note)."""
    if spec == BC_REFLECTIVE:
        return 0.0
    if spec == BC_ZERO_FLUX:
        return math.inf
    if spec == BC_VACUUM:
        return _VACUUM_ALPHA
    return float(spec)


def normalize_bc(bc):
    """Expand a boundary-condition spec to ((xlo, xhi), (ylo, yhi), (zlo, zhi)).

    Accepts a single face spec (applied to all six faces), a length-3 sequence
    of per-axis entries (each a face spec or a (lo, hi) pair), or a flat
    length-6 sequence ordered x_lo, x_hi, y_lo, y_hi, z_lo, z_hi. A face spec is
    "zero-flux", "reflective", "vacuum", or a non-negative float albedo
    coefficient alpha (the boundary law J_net = alpha * phi_surface).
    """
    def is_scalar(x):
        return isinstance(x, (str, int, float))

    if is_scalar(bc):
        faces = [bc] * 6
    else:
        bc = list(bc)
        if len(bc) == 3:
            faces = []
            for axis in bc:
                pair = [axis, axis] if is_scalar(axis) else list(axis)
                if len(pair) != 2:
                    raise ValueError(f"per-axis bc must be a face spec or (lo, hi) pair, got {axis!r}")
                faces += pair
        elif len(bc) == 6:
            faces = bc
        else:
            raise ValueError(f"bc must be a face spec, 3 per-axis entries, or 6 faces, got {bc!r}")
    faces = [_face_valid(f) for f in faces]
    return ((faces[0], faces[1]), (faces[2], faces[3]), (faces[4], faces[5]))


def harmonic_mean(Da, Db):
    """Face diffusion coefficient between two cells with piecewise-constant D.

    Requiring the two-point flux to be continuous across the shared face gives
    the harmonic mean 2*Da*Db/(Da + Db): exact for piecewise-constant D, and
    it correctly chokes transport at an interface where either D vanishes
    (the arithmetic mean would leak through strong absorbers).
    """
    return 2.0 * Da * Db / (Da + Db)


def robin_face_term(xp, Dface, d, alpha):
    """Boundary-face contribution to the operator diagonal, per unit volume,
    for a face perpendicular to a cell axis with centre-to-face distance d/2.

    The Robin law J_net = alpha * phi_s relates the outward current to the
    surface flux. Coupling the cell centre to the surface with the two-point
    current J = (2D/d)(phi_c - phi_s) and eliminating phi_s gives

        leakage / volume = 2 D alpha / (d (d alpha + 2 D)).

    alpha -> inf recovers the zero-flux Dirichlet term 2D/d^2; alpha = 0
    (reflective) contributes nothing; alpha = 1/2 is the Marshak vacuum.
    """
    if alpha == 0.0:
        return xp.zeros_like(Dface)
    if math.isinf(alpha):
        return 2.0 * Dface / d**2
    return 2.0 * Dface * alpha / (d * (d * alpha + 2.0 * Dface))


class GroupOperator:
    """A_g = -div(D_g grad .) + Sigma_r,g  on a uniform structured grid.

    Cartesian by default; a grid with geometry="cylindrical" yields the (r-z)
    revolution-body stencil instead (volume-weighted form; pair the operator
    with a source scaled by .rhs_weight, which is None on Cartesian grids).

    Parameters
    ----------
    xp        : array module (numpy or cupy)
    grid      : Grid
    D         : per-cell diffusion coefficient, shape grid.shape (device array)
    removal   : per-cell removal cross section, shape grid.shape (device array)
    bc        : boundary conditions, anything accepted by normalize_bc():
                "zero-flux" (phi = 0 on the outer boundary surface) or
                "reflective" (zero net current), globally, per axis, or per face.
    symmetric : cylindrical grids only. True (default) builds the
                volume-weighted form, which is SPD and pairs with CG plus a
                source scaled by .rhs_weight. False builds the natural
                divergence (per-unit-volume) form instead: the same equations
                with each row divided by its cell volume, so the source is
                used as-is (.rhs_weight is None) -- but the radial coupling
                seen from cell i (r_face/r_i) no longer equals its transpose
                (r_face/r_{i+1}), so the matrix is non-symmetric and must be
                paired with linear_solver="gmres" or "bicgstab", not CG.
                Ignored on Cartesian grids (already symmetric either way).
    """

    def __init__(self, xp, grid, D, removal, bc=BC_ZERO_FLUX, active=None,
                 mask_bc=BC_VACUUM, symmetric=True):
        bc = normalize_bc(bc)
        self.xp = xp
        self.shape = grid.shape
        dx, dy, dz = grid.spacing

        # Cylindrical (r-z) grids: every term of the per-unit-volume Cartesian
        # stencil is scaled by its radial metric factor -- cell terms (removal,
        # y/z couplings, y/z boundary faces) by the cell-center radius cw, and
        # radial face terms by the face radius. This is the equation multiplied
        # by the cell volume, so the radial two-point stencil stays symmetric
        # (both neighbours see the shared face radius) and CG still applies;
        # the matching source-side weight is exposed as self.rhs_weight. On a
        # Cartesian grid all factors are 1 and the stencil is unchanged.
        met = getattr(grid, "cylindrical_metrics", lambda: None)()
        if met is None:
            cw = fwx = 1.0
            bwx_lo = bwx_hi = 1.0
            self.rhs_weight = None
        else:
            rc, rf, r_lo, r_hi = met
            cw = xp.asarray(rc, dtype=D.dtype)
            fwx = xp.asarray(rf, dtype=D.dtype)
            bwx_lo, bwx_hi = float(r_lo), float(r_hi)
            self.rhs_weight = cw

        def robin(Dface, d, alpha):
            return robin_face_term(xp, Dface, d, alpha)

        # Interior face couplings (harmonic mean of the two adjacent cells).
        # wx[i] couples cells i and i+1 along x; shape (nx-1, ny, nz).
        def face(Da, Db, d):
            return harmonic_mean(Da, Db) / (d * d)

        self.wx = fwx * face(D[:-1, :, :], D[1:, :, :], dx)
        self.wy = cw * face(D[:, :-1, :], D[:, 1:, :], dy)
        self.wz = cw * face(D[:, :, :-1], D[:, :, 1:], dz)

        # Non-rectangular cores: an "active" mask marks in-domain cells. Faces
        # touching an inactive (excised) cell carry no diffusion; the active
        # cell instead sees the core-surface boundary law mask_bc on that face.
        act = None
        if active is not None:
            act = xp.asarray(active).astype(bool)
            if act.shape != grid.shape:
                raise ValueError(f"active mask shape {act.shape} != grid shape {grid.shape}")
            self.wx = xp.where(act[:-1, :, :] & act[1:, :, :], self.wx, 0.0)
            self.wy = xp.where(act[:, :-1, :] & act[:, 1:, :], self.wy, 0.0)
            self.wz = xp.where(act[:, :, :-1] & act[:, :, 1:], self.wz, 0.0)

        # Diagonal: removal + sum of face couplings incident on each cell.
        diag = removal * cw
        diag[1:, :, :] += self.wx
        diag[:-1, :, :] += self.wx
        diag[:, 1:, :] += self.wy
        diag[:, :-1, :] += self.wy
        diag[:, :, 1:] += self.wz
        diag[:, :, :-1] += self.wz

        # Outer box-boundary Robin terms (only where the boundary cell is
        # active), each scaled by its face's metric weight: the boundary radius
        # for the two radial faces (0 at the symmetry axis, so any bc there is
        # inert), the cell radius for y/z faces.
        cw_at = (lambda sl: 1.0) if met is None else (lambda sl: cw[sl])
        boundary_faces = [
            (face_alpha(bc[0][0]), (0, slice(None), slice(None)), dx, lambda sl: bwx_lo),
            (face_alpha(bc[0][1]), (-1, slice(None), slice(None)), dx, lambda sl: bwx_hi),
            (face_alpha(bc[1][0]), (slice(None), 0, slice(None)), dy, cw_at),
            (face_alpha(bc[1][1]), (slice(None), -1, slice(None)), dy, cw_at),
            (face_alpha(bc[2][0]), (slice(None), slice(None), 0), dz, cw_at),
            (face_alpha(bc[2][1]), (slice(None), slice(None), -1), dz, cw_at),
        ]
        for alpha, sl, d, wgt in boundary_faces:
            if alpha == 0.0:
                continue
            term = wgt(sl) * robin(D[sl], d, alpha)
            if act is not None:
                term = xp.where(act[sl], term, 0.0)
            diag[sl] += term

        # Internal core-surface faces (active cell facing an excised cell).
        if act is not None:
            ma = face_alpha(_face_valid(mask_bc))
            if ma != 0.0:
                lo_a, hi_a = act[:-1, :, :], act[1:, :, :]
                diag[:-1, :, :] += xp.where(lo_a & ~hi_a, fwx * robin(D[:-1, :, :], dx, ma), 0.0)
                diag[1:, :, :] += xp.where(hi_a & ~lo_a, fwx * robin(D[1:, :, :], dx, ma), 0.0)
                lo_a, hi_a = act[:, :-1, :], act[:, 1:, :]
                diag[:, :-1, :] += xp.where(lo_a & ~hi_a, cw * robin(D[:, :-1, :], dy, ma), 0.0)
                diag[:, 1:, :] += xp.where(hi_a & ~lo_a, cw * robin(D[:, 1:, :], dy, ma), 0.0)
                lo_a, hi_a = act[:, :, :-1], act[:, :, 1:]
                diag[:, :, :-1] += xp.where(lo_a & ~hi_a, cw * robin(D[:, :, :-1], dz, ma), 0.0)
                diag[:, :, 1:] += xp.where(hi_a & ~lo_a, cw * robin(D[:, :, 1:], dz, ma), 0.0)
            # Excised cells: decouple (all face weights into them are already 0)
            # and give them a unit diagonal so Jacobi is well defined; with a
            # void material (no fission/scatter) their flux stays exactly zero.
            diag = xp.where(act, diag, 1.0)

        # Divergence form (symmetric=False, cylindrical only): apply() runs
        # the weighted stencil and then divides each row by its cell weight,
        # which is exactly the matrix W^{-1} A_w -- the classic non-symmetric
        # finite-volume form. The Jacobi diagonal scales along.
        self._stencil_diag = diag
        self.row_scale = None
        if not symmetric and met is not None:
            self.row_scale = 1.0 / cw
            self.rhs_weight = None
            diag = diag * self.row_scale
        self.diag = diag
        self.inv_diag = 1.0 / diag  # Jacobi preconditioner

    def apply(self, phi):
        """Return A phi (allocates the output array)."""
        out = self._stencil_diag * phi
        out[1:, :, :] -= self.wx * phi[:-1, :, :]
        out[:-1, :, :] -= self.wx * phi[1:, :, :]
        out[:, 1:, :] -= self.wy * phi[:, :-1, :]
        out[:, :-1, :] -= self.wy * phi[:, 1:, :]
        out[:, :, 1:] -= self.wz * phi[:, :, :-1]
        out[:, :, :-1] -= self.wz * phi[:, :, 1:]
        if self.row_scale is not None:
            out *= self.row_scale
        return out
