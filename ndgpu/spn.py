"""Order-N simplified-PN / simplified-double-PN block operators.

The general M = N+1 moment machinery for the SDPN family (Carreno et al., Ann.
Nucl. Energy 207 (2024) 110675) and the standard SPN family that shares its
diffusive U-form: the coefficient tables (``_SDPN_C`` / ``_SPN_C`` and the
Marshak boundary matrices ``_SDPN_G`` / ``_SPN_G``), the plain per-moment block
(:class:`SDPNGroupOperator`), and the symmetrizing-congruence CG path for tables
that admit no diagonal similarity (:class:`CongruentSDPNOperator`).

The dedicated 2x2 SP3/SDP1 block lives in ``sp3.py``; both reuse the structured
stencil and boundary vocabulary from ``stencil.py``.
"""

from __future__ import annotations

import numpy as np

from .stencil import (BC_REFLECTIVE, BC_VACUUM, BC_ZERO_FLUX, GroupOperator,
                      normalize_bc)


class CongruentSDPNOperator:
    """Order-N SDPN block in a symmetrizing congruence basis (V = S^-1 U) --
    the CG path for tables with no diagonal similarity (SDP3).

    The transformed gradient block couples moments, but it is a sum of two
    low-rank constant matrices times scalar fields (D_1 and 1/Sigma_t), so its
    eigen-decomposition reduces the apply to one scalar stencil per retained
    eigenpair -- M stencil applies in total, the same count as the plain
    per-moment form. Reactions are M x M symmetric cell couplings. All pieces
    are PSD (verified at transform construction), so the block is SPD.

    Only reflective / zero-flux faces are supported: they impose the same
    condition on every moment and therefore commute with the moment mixing
    (the projected variables inherit the identical scalar boundary term). The
    per-moment vacuum Robin term is moment-dependent and does not, so vacuum
    problems keep the plain non-symmetric form. Cartesian/tri grids only (no
    cylindrical metric weighting).
    """

    _ALLOWED = (BC_REFLECTIVE, BC_ZERO_FLUX)

    def __init__(self, xp, grid, D1, sigma_t, removal, order, bc=BC_ZERO_FLUX,
                 active=None, mask_bc=BC_VACUUM, op_cls=None, coeffs=None,
                 theta=None):
        tr = _congruence_transform(order, coeffs)
        if tr is None:
            raise ValueError(f"order-{order} tables admit no symmetrizing "
                             "congruence")
        if getattr(grid, "cylindrical_metrics", lambda: None)() is not None:
            raise NotImplementedError("congruence path is not implemented for "
                                      "cylindrical grids")
        bc_n = normalize_bc(bc)
        faces = [f for axis in bc_n for f in axis]
        if any(f not in self._ALLOWED for f in faces):
            raise ValueError("the symmetrized SDP3 block supports only "
                             "reflective/zero-flux faces (vacuum Robin is "
                             "moment-dependent and breaks the symmetry); got "
                             f"{bc_n!r}")
        if active is not None and mask_bc not in self._ALLOWED:
            raise ValueError("active masks need mask_bc reflective/zero-flux "
                             "on the symmetrized SDP3 block")
        self.xp = xp
        op_cls = op_cls or GroupOperator
        M = order + 1
        self.M = M
        self.order = order
        self.symmetric = True
        self.marshak = None
        self.rhs_weight = None
        self.src_weights = [float(w) for w in tr["src"]]
        self.phi0_weights = [float(w) for w in tr["phi0"]]
        self.time_weights = [[float(tr["tau"][i][j]) for j in range(M)]
                             for i in range(M)]
        zero = xp.zeros(grid.shape, dtype=D1.dtype)
        # Projected scalar leakage operators: (weights w, op on lam*field).
        self._proj = []
        for field, pairs in ((D1, tr["grad_tr"]),
                             (1.0 / sigma_t, tr["grad_t"])):
            for lam, w in pairs:
                L = op_cls(xp, grid, lam * field, zero, bc=bc_n,
                           active=active, mask_bc=mask_bc)
                self._proj.append(([float(x) for x in w], L))
        c1, ct = tr["c1"], tr["ct"]
        # Pre-combined per-pair reaction fields: apply adds R_ij * u_j with
        # R_ij = c1_ij removal + ct_ij sigma_t (+ theta tau_ij) -- one fused
        # multiply-add per (i, j) instead of two or three (measurably cheaper:
        # the reactions are dense M x M after the congruence).
        self._react = {}
        for i in range(M):
            for j in range(M):
                a, b = c1[i][j], ct[i][j]
                t = 0.0 if theta is None else theta * self.time_weights[i][j]
                if a == 0.0 and b == 0.0 and t == 0.0:
                    continue
                self._react[(i, j)] = a * removal + b * sigma_t + t
        diag = xp.zeros((M,) + grid.shape, dtype=D1.dtype)
        for w, L in self._proj:
            for i in range(M):
                if w[i] != 0.0:
                    diag[i] += (w[i] * w[i]) * L.diag
        for i in range(M):
            if (i, i) in self._react:
                diag[i] += self._react[(i, i)]
        self.inv_diag = 1.0 / diag

    def apply(self, u):
        """Return the symmetrized block applied to u of shape (M, *grid)."""
        xp = self.xp
        M = self.M
        out = xp.zeros_like(u)
        for w, L in self._proj:
            s = None
            for i in range(M):
                if w[i] != 0.0:
                    s = w[i] * u[i] if s is None else s + w[i] * u[i]
            Ls = L.apply(s)
            for i in range(M):
                if w[i] != 0.0:
                    out[i] += w[i] * Ls
        for (i, j), f in self._react.items():
            out[i] += f * u[j]
        return out


# Simplified double-PN (SDPN) coefficient matrices c^(m), from Carreno,
# Vidal-Ferrandiz, Ginestar & Verdu, Ann. Nucl. Energy 207 (2024) 110675,
# Eqs. (43) [SDP1] and (A.3)/(A.6) [SDP2/SDP3]. For angular order N the block
# carries M = N + 1 even-moment pseudo-fluxes U_1..U_M and the diffusive system
#     -div(D grad U) + A U = (1/k) F U,
# with D_i = 1/((4i-1) Sigma_{2i-1}),  A_ij = sum_{m=1..M} c^(m)_ij Sigma_{2m-2},
# and F_ij = c^(1)_ij F. Only the last (closure) row of each c^(m) differs from
# the standard SPN coefficients -- that row is the half-range (double-PN)
# correction that lets SDPN track discontinuous angular flux better than SPN at
# equal cost. c^(m)[N] indexes the m-th matrix (m = 1..M) for order N.
_SDPN_C = {
    1: [
        [[1.0, -2.0 / 3.0], [-6.0 / 7.0, 4.0 / 7.0]],
        [[0.0, 0.0], [0.0, 5.0 / 7.0]],
    ],
    2: [
        # c^(1) must be rank-1 (fission depends only on phi0): row i = c1[i][0]
        # * row0. The paper's printed (A.3) third row [-32/33, -64/99, 256/495]
        # violates this: the first entry's sign is a typo -- with +32/33 the row
        # is exactly (32/33)*row0 = [32/33, -64/99, 256/495], rank-1, and
        # reproduces the paper's SDP2 k_eff (Table 6). Verified on the Fig. 3
        # Brantley-Larsen problem (examples/sdpn_brantley_larsen_2d.py).
        [[1.0, -2.0 / 3.0, 8.0 / 15.0],
         [-2.0 / 3.0, 4.0 / 9.0, -16.0 / 45.0],
         [32.0 / 33.0, -64.0 / 99.0, 256.0 / 495.0]],
        [[0.0, 0.0, 0.0],
         [0.0, 5.0 / 9.0, -4.0 / 9.0],
         [0.0, -80.0 / 99.0, 64.0 / 99.0]],
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 36.0 / 55.0]],
    ],
    # SDP3: rows 0-1 match the paper's (A.6); rows 2-3 are re-derived from
    # scratch (sympy, from the paper's own Eq. (34) DPN closure -- half-range
    # moments of the PN system truncated at l = 2N+1, odd moments eliminated,
    # U-transform of Eq. (13), D = diag(1/3, 1/7, 1/11, 1/15) Sigma^-1). The
    # derivation machinery reproduces the paper's SDP1 and SDP2 tables (and the
    # SDP3 Marshak g of (A.7)) exactly, but NOT its printed SDP3 rows 2-3 --
    # which also appear verbatim in the authors' FEMFFUSION code and look like a
    # transcription chimera (c^(1)/c^(2) row 2 are the SP7 rows, c^(3) row 2
    # carries SDP2's closure value 36/55). Note the true c^(4) has a nonzero
    # (2,3) entry, absent from the paper's sparsity pattern. On the paper's
    # Fig. 3 benchmark these tables give k = 0.80394 vs the published 0.80402
    # (-9 pcm); the paper's own tables give +24 pcm.
    3: [
        [[1.0, -2.0 / 3.0, 8.0 / 15.0, -16.0 / 35.0],
         [-2.0 / 3.0, 4.0 / 9.0, -16.0 / 45.0, 32.0 / 105.0],
         [200.0 / 429.0, -400.0 / 1287.0, 320.0 / 1287.0, -640.0 / 3003.0],
         [-16.0 / 13.0, 32.0 / 39.0, -128.0 / 195.0, 256.0 / 455.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 5.0 / 9.0, -4.0 / 9.0, 8.0 / 21.0],
         [0.0, -500.0 / 1287.0, 400.0 / 1287.0, -800.0 / 3003.0],
         [0.0, 40.0 / 39.0, -32.0 / 39.0, 64.0 / 91.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 45.0 / 143.0, -270.0 / 1001.0],
         [0.0, 0.0, -54.0 / 65.0, 324.0 / 455.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 3.0 / 77.0],
         [0.0, 0.0, 0.0, 5.0 / 7.0]],
    ],
}

# Standard SPN coefficient matrices c^(m), from the same paper's Eqs. (16)-(17)
# (order 1/2/3 = SP3/SP5/SP7, obtained by truncating the SP7 matrices). Same
# U-form and D block as SDPN above -- the two families differ *only* in these
# coefficients (and SDPN only in each matrix's last, closure, row). Unlike SDPN,
# every c^(m) here is symmetric, so the SPN block operator is symmetric.
_SPN_C = {
    # Order 0 = SP1: one moment, U = phi0, A = removal, D = 1/(3 Sigma_1) --
    # the plain P1/diffusion equation run through the same block machinery.
    0: [[[1.0]]],
    1: [
        [[1.0, -2.0 / 3.0], [-2.0 / 3.0, 4.0 / 9.0]],
        [[0.0, 0.0], [0.0, 5.0 / 9.0]],
    ],
    2: [
        [[1.0, -2.0 / 3.0, 8.0 / 15.0],
         [-2.0 / 3.0, 4.0 / 9.0, -16.0 / 45.0],
         [8.0 / 15.0, -16.0 / 45.0, 64.0 / 225.0]],
        [[0.0, 0.0, 0.0],
         [0.0, 5.0 / 9.0, -4.0 / 9.0],
         [0.0, -4.0 / 9.0, 16.0 / 45.0]],
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, 9.0 / 25.0]],
    ],
    3: [
        [[1.0, -2.0 / 3.0, 8.0 / 15.0, -16.0 / 35.0],
         [-2.0 / 3.0, 4.0 / 9.0, -16.0 / 45.0, 32.0 / 105.0],
         [8.0 / 15.0, -16.0 / 45.0, 64.0 / 225.0, -128.0 / 525.0],
         [-16.0 / 35.0, 32.0 / 105.0, -128.0 / 525.0, 256.0 / 1225.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 5.0 / 9.0, -4.0 / 9.0, 8.0 / 21.0],
         [0.0, -4.0 / 9.0, 16.0 / 45.0, -32.0 / 105.0],
         [0.0, 8.0 / 21.0, -32.0 / 105.0, 64.0 / 245.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 9.0 / 25.0, -54.0 / 175.0],
         [0.0, 0.0, -54.0 / 175.0, 324.0 / 1225.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 13.0 / 49.0]],
    ],
}

# Marshak vacuum boundary matrices g (B = g (x) I over groups) for the coupled
# condition -n . D grad U = B U, from the same paper: SPN in Eq. (23) [SP7, with
# the lower orders its leading submatrices], SDPN in Eqs. (47)/(A.4)/(A.7). g_11
# = 1/2 is the standard 0th-moment Marshak value; the off-diagonals couple the
# moments at the boundary (the piece a per-moment albedo omits). SPN g is
# symmetric, SDPN g is not.
_SPN_G = {
    3: [[1.0 / 2, -1.0 / 8, 1.0 / 16, -5.0 / 128],
        [-1.0 / 8, 7.0 / 24, -41.0 / 384, 1.0 / 16],
        [1.0 / 16, -41.0 / 384, 407.0 / 1920, -233.0 / 2560],
        [-5.0 / 128, 1.0 / 16, -233.0 / 2560, 3023.0 / 17920]],
}
_SPN_G[0] = [row[:1] for row in _SPN_G[3][:1]]   # SP1: the classic alpha = 1/2
_SPN_G[1] = [row[:2] for row in _SPN_G[3][:2]]
_SPN_G[2] = [row[:3] for row in _SPN_G[3][:3]]
_SDPN_G = {
    1: [[1.0 / 2, -1.0 / 8], [-2.0 / 7, 23.0 / 42]],
    2: [[1.0 / 2, -1.0 / 8, 1.0 / 16],
        [-2.0 / 21, 31.0 / 126, -59.0 / 1260],
        [8.0 / 33, -38.0 / 99, 287.0 / 495]],
    3: [[1.0 / 2, -1.0 / 8, 1.0 / 16, -5.0 / 128],
        [-1.0 / 8, 7.0 / 24, -41.0 / 384, 1.0 / 16],
        [-2.0 / 143, 4.0 / 429, 1087.0 / 17160, 3611.0 / 40040],
        [-10.0 / 39, 46.0 / 117, -2417.0 / 4680, 1891.0 / 2730]],
}


def _diag_similarity(c):
    """Diagonal similarity r with r_i c^(m)_ij / r_j symmetric for EVERY m, or
    None when the tables admit none. Determined by the rank-1 c^(1) = u v^T:
    the symmetry condition (r_i/r_j)^2 = c_ji/c_ij gives r_i = sqrt(v_i/u_i);
    the remaining c^(m) are then verified. Already-symmetric tables (SPN)
    yield the identity. SDP1/SDP2 admit one; SDP3 does not (its c^(4) has
    (2,3) = 3/77 against (3,2) = 0, which no scaling can bridge)."""
    M = len(c[0])
    u = [c[0][i][0] for i in range(M)]
    v = [c[0][0][j] for j in range(M)]
    if any(ui * vi <= 0.0 for ui, vi in zip(u, v)):
        return None
    r = [(vi / ui) ** 0.5 for ui, vi in zip(u, v)]
    for m in range(M):
        for i in range(M):
            for j in range(M):
                a = r[i] * c[m][i][j] / r[j]
                b = r[j] * c[m][j][i] / r[i]
                if abs(a - b) > 1e-12 * (abs(a) + abs(b) + 1e-300):
                    return None
    return r


_CONGRUENCE_CACHE = {}


def _congruence_transform(order, coeffs=None):
    """Constant congruence (R, S) symmetrizing the order-N SDPN block in
    ndgpu's collapsed data model (Sigma_2 = Sigma_3 = ... = Sigma_t), for
    tables that admit NO diagonal similarity (SDP3).

    The block is built from four constant matrices paired with two spatial
    fields: gradients D_1(x)*e1e1^T + (1/Sigma_t)(x)*diag(0, kappa_2..) and
    reactions Sigma_0(x)*c^(1) + Sigma_t(x)*sum_{m>=2} c^(m). A transform
    R P S making all four symmetric exists iff the linear system
    H T_k = T_k^T H (T_k = P_k Q^-1, Q the invertible gradient anchor) has an
    SPD solution H; then R = H^(1/2), S = Q^-1 H^(-1/2). For SDP3 the solution
    space is one-dimensional and definite, and every transformed piece comes
    out PSD, so the block is SPD and CG applies. Returns a dict of the
    transformed pieces (or None): projection eigenpairs of the two gradient
    matrices, reaction matrices, time matrix, and source/flux weights.
    """
    key = (order, id(coeffs) if coeffs is not None else None)
    if key in _CONGRUENCE_CACHE:
        return _CONGRUENCE_CACHE[key]
    c = (coeffs if coeffs is not None else _SDPN_C)[order]
    M = order + 1
    c1 = np.array(c[0], float)
    Ct = sum(np.array(cm, float) for cm in c[1:])
    kap = np.diag([0.0] + [1.0 / (4 * (i + 1) - 1) for i in range(1, M)])
    e11 = np.zeros((M, M)); e11[0, 0] = 1.0
    Q = e11 + kap
    pieces = [e11, kap, c1, Ct]
    Ts = [P @ np.linalg.inv(Q) for P in pieces]
    # solve H T = T^T H over symmetric H (linear; null space by SVD)
    idx = [(i, j) for i in range(M) for j in range(i, M)]
    A = np.zeros((len(Ts) * (M * (M - 1) // 2), len(idx)))
    for a, (i, j) in enumerate(idx):
        Hb = np.zeros((M, M)); Hb[i, j] = Hb[j, i] = 1.0
        r = 0
        for T in Ts:
            C = Hb @ T - T.T @ Hb
            A[r:r + M * (M - 1) // 2, a] = C[np.triu_indices(M, 1)]
            r += M * (M - 1) // 2
    _, sv, vt = np.linalg.svd(A)
    nnull = len(idx) - int(np.sum(sv > 1e-12 * max(sv[0], 1.0)))
    result = None
    for v in (vt[-nnull:] if nnull else []):
        H = np.zeros((M, M))
        for a, (i, j) in enumerate(idx):
            H[i, j] = H[j, i] = v[a]
        ev = np.linalg.eigvalsh(H)
        if ev[-1] < 0:
            H, ev = -H, -ev[::-1]
        if ev[0] <= 1e-10 * ev[-1]:
            continue                                  # not definite
        lam, W = np.linalg.eigh(H)
        R = W @ np.diag(np.sqrt(lam)) @ W.T
        S = np.linalg.inv(Q) @ (W @ np.diag(1.0 / np.sqrt(lam)) @ W.T)
        sym = lambda P: 0.5 * ((R @ P @ S) + (R @ P @ S).T)
        M1, M2, C1p, Ctp = sym(e11), sym(kap), sym(c1), sym(Ct)
        if min(np.linalg.eigvalsh(P)[0] for P in (M1, M2, C1p, Ctp)) < -1e-10:
            continue                                  # needs PSD pieces
        def eigpairs(P):
            lam_p, W_p = np.linalg.eigh(P)
            keep = lam_p > 1e-12 * max(lam_p.max(), 1.0)
            return [(float(l), W_p[:, k].copy())
                    for k, l in enumerate(lam_p) if keep[k]]
        result = {
            "grad_tr": eigpairs(M1),      # pairs (lambda, w) on the D1 field
            "grad_t": eigpairs(M2),       # pairs on the 1/Sigma_t field
            "c1": C1p, "ct": Ctp, "tau": C1p + Ctp,
            "src": R @ c1[:, 0], "phi0": S.T @ c1[0],
            "R": R, "S": S,
        }
        break
    _CONGRUENCE_CACHE[key] = result
    return result


def _strip_vacuum(bc):
    """Replace vacuum faces with reflective (their leakage is taken over by the
    coupled Marshak boundary); other faces are unchanged. bc is normalized."""
    return tuple(tuple(BC_REFLECTIVE if f == BC_VACUUM else f for f in axis)
                 for axis in bc)


def _congruence_available(order, grid, bc, active, mask_bc, coeffs=None):
    """True when :class:`CongruentSDPNOperator` can serve this problem: the
    order's tables admit the symmetrizing congruence AND every boundary the
    operator will see -- the six faces plus the active-mask boundary, if any --
    is reflective or zero-flux (the class's restrictions; see its docstring).
    The solvers use this to auto-select the CG path per problem."""
    if _congruence_transform(order, coeffs) is None:
        return False
    if grid is None or getattr(grid, "cylindrical_metrics", lambda: None)() is not None:
        return False
    try:
        faces = [f for axis in normalize_bc(bc) for f in axis]
    except ValueError:
        return False
    allowed = CongruentSDPNOperator._ALLOWED
    if any(f not in allowed for f in faces):
        return False
    if active is not None and mask_bc not in allowed:
        return False
    return True


def _marshak_faces(xp, grid, bc, D_list, g):
    """Precompute, for each vacuum face, the per-boundary-cell coupling matrix
    K (shape (*slab, M, M)) of the Marshak condition -n.D grad U = (g (x) I) U.

    With the two-point boundary current J_i = (2 D_i/d)(U_c,i - U_s,i) and
    J_i = sum_j g_ij U_s,j, eliminating the surface value U_s gives the leakage
    per unit volume K = (1/d) g (diag(a) + g)^-1 diag(a), a_i = 2 D_i/d. For one
    moment this is exactly robin_face_term with alpha = g_00. Cartesian only.
    """
    if getattr(grid, "cylindrical_metrics", lambda: None)() is not None:
        raise NotImplementedError("Marshak vacuum not implemented for cylindrical grids")
    M = len(D_list)
    gm = xp.asarray(g, dtype=D_list[0].dtype)                # (M, M)
    dx, dy, dz = grid.spacing
    spacing = (dx, dy, dz)
    faces = [(0, (0, slice(None), slice(None))), (0, (-1, slice(None), slice(None))),
             (1, (slice(None), 0, slice(None))), (1, (slice(None), -1, slice(None))),
             (2, (slice(None), slice(None), 0)), (2, (slice(None), slice(None), -1))]
    specs = [bc[0][0], bc[0][1], bc[1][0], bc[1][1], bc[2][0], bc[2][1]]
    out = []
    eye = xp.eye(M, dtype=gm.dtype)
    for (axis, sl), spec in zip(faces, specs):
        if spec != BC_VACUUM:
            continue
        d = spacing[axis]
        # a_i on the boundary layer, moved to a trailing moment axis: (*slab, M).
        a = xp.stack([2.0 * D_list[i][sl] / d for i in range(M)], axis=-1)
        Mmat = gm + a[..., None] * eye                       # diag(a) + g, (*slab,M,M)
        Minv = xp.linalg.inv(Mmat)
        gMinv = xp.matmul(gm, Minv)                          # (*slab,M,M)
        K = (1.0 / d) * gMinv * a[..., None, :]              # K_ij = (1/d) a_j (gMinv)_ij
        out.append((sl, K))
    return out


class SDPNGroupOperator:
    """Within-group simplified double-PN (SDPN) block operator, order N in
    {1, 2, 3} (M = N+1 moments), in the paper's diffusive U-form.

    Row i of the block is a diffusion-like operator on pseudo-moment U_i coupled
    to the others only through cell (reaction) terms:

        [op(U)]_i = -div(D_i grad U_i) + sum_j A_ij U_j,

    with D_i = 1/((4i-1) Sigma_{2i-1}) (D_1 is the transport-corrected group
    diffusion coefficient; Sigma_3 = Sigma_5 = Sigma_7 = Sigma_t in the isotropic
    data model) and A_ij = sum_m c^(m)_ij Sigma_{2m-2} (Sigma_0 = removal, the
    higher even removals = Sigma_t). The coefficient matrices c^(m) are
    :data:`_SDPN_C`.

    The same class serves the standard SPN family (``coeffs=_SPN_C``): the block
    structure, D and source/flux wiring are identical -- only the c^(m) matrices
    change. For SDPN the coupling matrix A is non-symmetric (the double-PN
    closure breaks the SPN symmetry), so the block is solved with BiCGStab/GMRES
    rather than CG; the SPN coefficients are symmetric. Group coupling
    (in-scatter, fission) is isotropic and enters only the phi0 moment; the
    driver distributes the scalar source over rows by the first column of c^(1)
    (``src_weights``), and the scalar flux is recovered as phi0 = c^(1)[0] . U
    (``phi0_weights``). Any ``op_cls`` exposing GroupOperator's (apply, inv_diag)
    interface supplies the per-moment spatial stencil, so both families run
    unchanged on the triangular mesh.
    """

    def __init__(self, xp, grid, D1, sigma_t, removal, order, bc=BC_ZERO_FLUX,
                 active=None, mask_bc=BC_VACUUM, op_cls=None, coeffs=None,
                 boundary_g=None, theta=None, symmetrize=False,
                 hybrid_mask=None, hybrid_mask_bc=BC_REFLECTIVE,
                 hybrid_confine=False):
        coeffs = coeffs if coeffs is not None else _SDPN_C
        if order not in coeffs:
            raise ValueError(f"order must be one of {sorted(coeffs)}, got {order!r}")
        self.xp = xp
        op_cls = op_cls or GroupOperator
        c = coeffs[order]
        M = order + 1
        self.M = M
        self.order = order
        # symmetrize: conjugate the c tables by their diagonal similarity
        # (state V = diag(r) U, rows scaled by r) so the block becomes
        # symmetric and CG applies. The per-moment D and the (diagonal) Robin
        # boundary terms are similarity-invariant, so only the cell couplings
        # and the source/flux weight vectors change; the r_0 gauge cancels
        # between src_weights and phi0_weights. Not available with the coupled
        # Marshak boundary (its g matrix is not symmetrized by the scaling).
        if symmetrize:
            if boundary_g is not None:
                raise ValueError("symmetrize is incompatible with the coupled "
                                 "Marshak boundary")
            r = _diag_similarity(c)
            if r is None:
                raise ValueError(f"the order-{order} coefficient tables admit "
                                 "no diagonal symmetrizing similarity (use the "
                                 "non-symmetric form with BiCGStab)")
            c = [[[r[i] * c[m][i][j] / r[j] for j in range(M)]
                  for i in range(M)] for m in range(M)]

        def _is_sym(mat):
            return all(abs(mat[i][j] - mat[j][i])
                       <= 1e-12 * (abs(mat[i][j]) + abs(mat[j][i]) + 1e-300)
                       for i in range(M) for j in range(M))

        # True when the assembled block is symmetric (SPD up to physics):
        # symmetric tables and, if Marshak is on, a symmetric g (SPN's is;
        # K = g (diag(a)+g)^-1 diag(a) is then symmetric as well).
        self.symmetric = all(_is_sym(c[m]) for m in range(M)) and (
            boundary_g is None or _is_sym(boundary_g))
        # phi0 = c^(1)[0] . U ; isotropic source distributes by c^(1)[:, 0].
        self.phi0_weights = [c[0][0][j] for j in range(M)]
        self.src_weights = [c[0][i][0] for i in range(M)]
        # Backward-Euler time term theta = 1/(v*dt): a time derivative is
        # structurally a cross section that is equal in every even moment, so
        # in the U-form it transforms exactly like A with all Sigma_{2m-2}
        # replaced by theta -- the time matrix is theta * sum_m c^(m)
        # (``time_weights``; odd-moment time derivatives neglected, the
        # standard quasi-static closure of time-SPN kinetics). The driver adds
        # the matching source theta * time_weights . U_old per row.
        self.time_weights = [[sum(c[m][i][j] for m in range(M))
                              for j in range(M)] for i in range(M)]
        # Sigma_{2m-2}: m=1 -> removal (Sigma_0), m>=2 -> Sigma_t.
        sig_even = [removal] + [sigma_t] * (M - 1)

        def A(i, j):
            a = c[0][i][j] * sig_even[0]
            for m in range(1, M):
                cm = c[m][i][j]
                if cm != 0.0:
                    a = a + cm * sig_even[m]
            if theta is not None and self.time_weights[i][j] != 0.0:
                a = a + theta * self.time_weights[i][j]
            return a

        # Marshak vacuum (coupled boundary): the per-moment operators must NOT
        # carry their own vacuum leakage on those faces -- it is replaced by the
        # coupled -n.D grad U = (g (x) I) U term below -- so build them with the
        # vacuum faces made reflective.
        D_list = [D1 if i == 0 else 1.0 / ((4 * (i + 1) - 1) * sigma_t)
                  for i in range(M)]
        bc_mom = normalize_bc(bc)
        self.marshak = None
        if boundary_g is not None:
            bc_mom = _strip_vacuum(bc_mom)
        # Hybrid transport/diffusion: generate the higher moments (i >= 1) only
        # in a masked subdomain -- the mask zeroes their source rows (driver
        # _rhs) and the inter-moment coupling (below) outside itself -- so there
        # the block collapses to -div(D1 grad U0) + Sigma_0 U0 = q0, i.e. exactly
        # the diffusion solver (moment 0 stays one global operator, so the net
        # current is continuous). The masked subdomain keeps the full SDPN
        # equations. hybrid_confine=False (default) keeps every moment's
        # spatial operator global, so the unsourced higher moments decay
        # smoothly out of the region (the SDPN boundary layer, no interface
        # closure, well-posed on the tri mesh); True excises them outside the
        # mask (pinned to zero, bit-exact diffusion there) with hybrid_mask_bc
        # on the interface -- see SP3GroupOperator for the trade-off.
        # Incompatible with the coupled Marshak boundary (its g couples moments
        # on the domain surface).
        self.hybrid_mask = None
        self.hybrid_confine = bool(hybrid_confine)
        if hybrid_mask is not None:
            if boundary_g is not None:
                raise ValueError("hybrid_mask is incompatible with the coupled "
                                 "Marshak boundary")
            hm = xp.asarray(hybrid_mask).astype(bool)
            if hm.shape != grid.shape:
                raise ValueError(f"hybrid_mask shape {hm.shape} != grid shape "
                                 f"{grid.shape}")
            self.hybrid_mask = hm.astype(D1.dtype)
        confined = hybrid_mask is not None and self.hybrid_confine
        hi_active = (active if not confined else
                     (hm if active is None else xp.asarray(active).astype(bool) & hm))
        self.moments = [
            op_cls(xp, grid, D_list[i], A(i, i), bc=bc_mom,
                   active=(active if i == 0 or not confined else hi_active),
                   mask_bc=(mask_bc if i == 0 or not confined else hybrid_mask_bc))
            for i in range(M)]
        # Off-diagonal reaction coupling A_ij (i != j) is a cell term, so it
        # carries the same metric weight as the moment operators' removal.
        w = getattr(self.moments[0], "rhs_weight", None)
        self.coupling = {}
        for i in range(M):
            for j in range(M):
                if i != j:
                    a = A(i, j)
                    a = a if w is None else a * w
                    # Every off-diagonal coupling touches a higher moment, so it
                    # is confined to the hybrid subdomain (zero elsewhere).
                    if self.hybrid_mask is not None:
                        a = a * self.hybrid_mask
                    self.coupling[(i, j)] = a
        self.rhs_weight = w
        diag = xp.stack([m.diag for m in self.moments])   # (M, *grid)
        if boundary_g is not None:
            self.marshak = _marshak_faces(xp, grid, normalize_bc(bc), D_list,
                                          boundary_g)
            # Fold the coupled boundary's diagonal (i,i) into the Jacobi diagonal.
            for sl, K in self.marshak:
                for i in range(M):
                    diag[i][sl] = diag[i][sl] + K[..., i, i]
        self.inv_diag = 1.0 / diag

    def apply(self, u):
        """Return the SDPN block operator applied to u of shape (M, *grid)."""
        xp = self.xp
        out = xp.empty_like(u)
        for i in range(self.M):
            out[i] = self.moments[i].apply(u[i])
            for j in range(self.M):
                if i != j:
                    out[i] = out[i] + self.coupling[(i, j)] * u[j]
        if self.marshak is not None:
            # Coupled Marshak vacuum: row i += sum_j K_ij U_j on each vacuum-face
            # boundary layer (K = leakage-per-volume, reduces to the per-moment
            # Robin term when the moments decouple).
            for sl, K in self.marshak:
                us = xp.stack([u[j][sl] for j in range(self.M)], axis=-1)  # (*slab,M)
                corr = xp.matmul(K, us[..., None])[..., 0]                 # (*slab,M)
                for i in range(self.M):
                    out[i][sl] = out[i][sl] + corr[..., i]
        return out
