"""Matrix-free finite-volume diffusion operator for one energy group.

Discretizes  A phi = -div(D grad phi) + Sigma_r phi  with a cell-centered
7-point stencil. Face diffusion coefficients use the harmonic mean, which is
exact for piecewise-constant D (so heterogeneous cores are handled correctly).

The operator is deliberately matrix-free: instead of assembling a sparse
matrix, we precompute one face-coupling array per axis plus the diagonal, and
`apply()` is six shifted multiply-adds. On the GPU backend each of these is a
fused elementwise CUDA kernel over contiguous memory — this is both faster and
lighter than sparse CSR SpMV, and it keeps the entire operator on-device.

The resulting operator is symmetric positive definite as long as removal is
positive somewhere (zero-flux BC) which is what lets us use CG.
"""

from __future__ import annotations

import math

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


class GroupOperator:
    """A_g = -div(D_g grad .) + Sigma_r,g  on a uniform structured grid.

    Parameters
    ----------
    xp        : array module (numpy or cupy)
    grid      : Grid
    D         : per-cell diffusion coefficient, shape grid.shape (device array)
    removal   : per-cell removal cross section, shape grid.shape (device array)
    bc        : boundary conditions, anything accepted by normalize_bc():
                "zero-flux" (phi = 0 on the outer boundary surface) or
                "reflective" (zero net current), globally, per axis, or per face.
    """

    def __init__(self, xp, grid, D, removal, bc=BC_ZERO_FLUX, active=None,
                 mask_bc=BC_VACUUM):
        bc = normalize_bc(bc)
        self.xp = xp
        self.shape = grid.shape
        dx, dy, dz = grid.spacing

        # Robin boundary diagonal per unit volume for the law J_net = alpha*phi.
        # Couple the cell center to the surface flux by the two-point diffusion
        # current J = (2D/d)(phi_c - phi_s) and eliminate phi_s via J = alpha*phi_s:
        #   leakage/volume = (1/d) * alpha*(2D/d) / (alpha + 2D/d)
        #                  = 2 D alpha / (d (d alpha + 2 D)).
        # alpha -> inf recovers the zero-flux Dirichlet term 2D/d^2; alpha = 0
        # (reflective) contributes nothing.
        def robin(Dface, d, alpha):
            if alpha == 0.0:
                return xp.zeros_like(Dface)
            if math.isinf(alpha):
                return 2.0 * Dface / d**2
            return 2.0 * Dface * alpha / (d * (d * alpha + 2.0 * Dface))

        # Interior face couplings (harmonic mean of the two adjacent cells).
        # wx[i] couples cells i and i+1 along x; shape (nx-1, ny, nz).
        def face(Da, Db, d):
            return 2.0 * Da * Db / ((Da + Db) * d * d)

        self.wx = face(D[:-1, :, :], D[1:, :, :], dx)
        self.wy = face(D[:, :-1, :], D[:, 1:, :], dy)
        self.wz = face(D[:, :, :-1], D[:, :, 1:], dz)

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
        diag = removal.copy()
        diag[1:, :, :] += self.wx
        diag[:-1, :, :] += self.wx
        diag[:, 1:, :] += self.wy
        diag[:, :-1, :] += self.wy
        diag[:, :, 1:] += self.wz
        diag[:, :, :-1] += self.wz

        # Outer box-boundary Robin terms (only where the boundary cell is active).
        boundary_faces = [
            (face_alpha(bc[0][0]), (0, slice(None), slice(None)), dx),
            (face_alpha(bc[0][1]), (-1, slice(None), slice(None)), dx),
            (face_alpha(bc[1][0]), (slice(None), 0, slice(None)), dy),
            (face_alpha(bc[1][1]), (slice(None), -1, slice(None)), dy),
            (face_alpha(bc[2][0]), (slice(None), slice(None), 0), dz),
            (face_alpha(bc[2][1]), (slice(None), slice(None), -1), dz),
        ]
        for alpha, sl, d in boundary_faces:
            if alpha == 0.0:
                continue
            term = robin(D[sl], d, alpha)
            if act is not None:
                term = xp.where(act[sl], term, 0.0)
            diag[sl] += term

        # Internal core-surface faces (active cell facing an excised cell).
        if act is not None:
            ma = face_alpha(_face_valid(mask_bc))
            if ma != 0.0:
                lo_a, hi_a = act[:-1, :, :], act[1:, :, :]
                diag[:-1, :, :] += xp.where(lo_a & ~hi_a, robin(D[:-1, :, :], dx, ma), 0.0)
                diag[1:, :, :] += xp.where(hi_a & ~lo_a, robin(D[1:, :, :], dx, ma), 0.0)
                lo_a, hi_a = act[:, :-1, :], act[:, 1:, :]
                diag[:, :-1, :] += xp.where(lo_a & ~hi_a, robin(D[:, :-1, :], dy, ma), 0.0)
                diag[:, 1:, :] += xp.where(hi_a & ~lo_a, robin(D[:, 1:, :], dy, ma), 0.0)
                lo_a, hi_a = act[:, :, :-1], act[:, :, 1:]
                diag[:, :, :-1] += xp.where(lo_a & ~hi_a, robin(D[:, :, :-1], dz, ma), 0.0)
                diag[:, :, 1:] += xp.where(hi_a & ~lo_a, robin(D[:, :, 1:], dz, ma), 0.0)
            # Excised cells: decouple (all face weights into them are already 0)
            # and give them a unit diagonal so Jacobi is well defined; with a
            # void material (no fission/scatter) their flux stays exactly zero.
            diag = xp.where(act, diag, 1.0)

        self.diag = diag
        self.inv_diag = 1.0 / diag  # Jacobi preconditioner

    def apply(self, phi):
        """Return A phi (allocates the output array)."""
        out = self.diag * phi
        out[1:, :, :] -= self.wx * phi[:-1, :, :]
        out[:-1, :, :] -= self.wx * phi[1:, :, :]
        out[:, 1:, :] -= self.wy * phi[:, :-1, :]
        out[:, :-1, :] -= self.wy * phi[:, 1:, :]
        out[:, :, 1:] -= self.wz * phi[:, :, :-1]
        out[:, :, :-1] -= self.wz * phi[:, :, 1:]
        return out


class SP3GroupOperator:
    """Symmetrized within-group SP3 block operator.

    The SP3 equations for one group, in the moments Phi1 = phi0 + 2*phi2 and
    phi2 (Brantley & Larsen form, isotropic sources):

        -div(D1 grad Phi1) + Sig0 * Phi1 - 2*Sig0 * phi2            = q0
        -div(D2 grad phi2) + (Sig2 + 4/5*Sig0) * phi2
                           - 2/5*Sig0 * Phi1                        = -2/5 q0

    with D1 = 1/(3*Sig1), D2 = 9/(35*Sig3), Sig0 the group removal cross
    section and Sig_l = Sigma_t - Sigma_s,l (here Sig1 uses the
    transport-corrected total via the material's D, and Sig2 = Sig3 = Sigma_t
    since l >= 2 scattering moments are not part of the data model).

    Multiplying the second equation by 5 makes the 2x2 block system symmetric
    (off-diagonal blocks both -2*Sig0) and positive definite
    (5*Sig0*Sig2 > 0), so the coupled system is solved directly by CG on
    states of shape (2, nx, ny, nz). The scalar flux is phi0 = Phi1 - 2*phi2.

    Boundary conditions: "reflective" is exact for SPN; "zero-flux" imposes
    phi = 0 on the surface for both moments (a good approximation whenever the
    physical vacuum boundary sits behind a reflector; Marshak vacuum
    conditions, which couple the moments at the boundary, are not implemented).
    """

    def __init__(self, xp, grid, D1, sigma_t, removal, bc=BC_ZERO_FLUX,
                 active=None, mask_bc=BC_VACUUM):
        self.xp = xp
        self.moment1 = GroupOperator(xp, grid, D1, removal, bc=bc,
                                     active=active, mask_bc=mask_bc)
        D2 = 9.0 / (35.0 * sigma_t)
        self.moment2 = GroupOperator(xp, grid, D2, sigma_t + 0.8 * removal, bc=bc,
                                     active=active, mask_bc=mask_bc)
        self.coupling = 2.0 * removal
        self.inv_diag = xp.stack([self.moment1.inv_diag, self.moment2.inv_diag / 5.0])

    def apply(self, u):
        """Return the symmetrized block operator applied to u = (Phi1, phi2)."""
        out = self.xp.empty_like(u)
        out[0] = self.moment1.apply(u[0]) - self.coupling * u[1]
        out[1] = 5.0 * self.moment2.apply(u[1]) - self.coupling * u[0]
        return out
