"""LDFE (linear discontinuous finite element) angular quadrature for 2D XY S_N.

The product Gauss-Legendre x uniform-azimuthal set in :mod:`ndgpu.sn` is cheap
and exact for high-order moments, but it is a *product* set: ordinates cluster
near the poles, and refining anywhere refines everywhere. LDFE instead tiles the
sphere with locally supported cells, so the ordinate set can be refined where
the angular flux is peaked -- the case that matters for a control-drum absorber
or a voided streaming channel, where product sets show ray effects.

Construction (Jarrell & Adams; this follows OpenSN's ``SLDFEsqQuadrature``):

1. Inscribe a cube of half-diagonal 1 in the unit sphere, so each face sits at
   ``a = 1/sqrt(3)``.  A face point ``(x~, y~, a)``, normalized, lands on the
   sphere; one face covers exactly one octant-pair of directions.
2. Subdivide the face into ``(level+1)^2`` squares in the face ("tilde")
   coordinates.  Each maps to a *spherical quadrilateral* (SQ).
3. Inside each SQ place four directions at the centroids of its four sub-squares
   and solve for four weights such that the rule integrates every linear
   function ``c0 + c1 x + c2 y + c3 z`` over that SQ exactly.  With the four
   nodal shape functions ``f_k`` (``f_k(omega_i) = delta_ki``) the weights are
   simply ``w_k = integral of f_k over the SQ``, evaluated with a Gauss-Legendre
   rule in tilde coordinates against the exact Jacobian ``a / r^3``.
4. Copy the face construction to all six faces (all eight octants).

That locality is the point: step 3 only ever couples the four directions of one
cell, so a cell may be refined without touching the rest of the sphere.

The equal-solid-angle diagonal spacing of the reference implementation (its
``alpha``/``beta`` tweak) is not reproduced here; this module subdivides the
face uniformly in tilde coordinates.  Cell solid angles therefore vary by ~1.6x
between the face centre and its corner at low level.  That costs some
uniformity but nothing in *correctness*: the weights are still exact for linear
functions per cell, which is what makes the rule second-order accurate.
"""

from __future__ import annotations

import numpy as np

_A = 1.0 / np.sqrt(3.0)          # cube face offset: |(x~, y~, a)| = 1 at corner

# The six cube faces, as (right, up, normal) frames in xyz.
_FACES = (
    ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
)


def _to_sphere(xt, yt, frame):
    """Map face ("tilde") coordinates onto the unit sphere."""
    right, up, normal = (np.asarray(v, float) for v in frame)
    v = (np.multiply.outer(np.asarray(xt, float), right)
         + np.multiply.outer(np.asarray(yt, float), up)
         + _A * normal)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _cell_weights(x0, x1, y0, y1, frame, n_gl=16):
    """Four directions and their weights for one spherical quadrilateral.

    Directions sit at the centroids of the cell's four sub-squares (the
    "centroid" point placement).  Weights integrate the nodal linear shape
    functions over the cell, so the rule is exact for any linear function there.
    """
    xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    subs = ((x0, xm, y0, ym), (xm, x1, y0, ym), (x0, xm, ym, y1), (xm, x1, ym, y1))
    cx = np.array([0.5 * (a + b) for a, b, _, _ in subs])
    cy = np.array([0.5 * (c + d) for _, _, c, d in subs])
    pts = np.array([_to_sphere(x, y, frame) for x, y in zip(cx, cy)])   # (4, 3)

    # Nodal shape functions are bilinear in the *face* coordinates, {1, x~, y~,
    # x~y~}: the natural basis on a quadrilateral. A basis linear in xyz would
    # be rank-deficient here -- on a symmetric cell all four sub-centroids map
    # to directions with the same normal component, so the z column collapses
    # onto the constant one.
    V = np.column_stack([np.ones(4), cx, cy, cx * cy])
    try:
        coeffs = np.linalg.solve(V, np.eye(4))         # column k -> f_k
    except np.linalg.LinAlgError:                      # degenerate cell
        return None, None

    # Integrate each f_k over the cell: Gauss-Legendre in tilde coords with the
    # exact Jacobian a / r^3, r = |(x~, y~, a)|.
    g, wg = np.polynomial.legendre.leggauss(n_gl)
    xq = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * g
    yq = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * g
    XQ, YQ = np.meshgrid(xq, yq, indexing="ij")
    WQ = np.outer(wg, wg) * 0.25 * (x1 - x0) * (y1 - y0)
    r = np.sqrt(XQ ** 2 + YQ ** 2 + _A ** 2)
    detJ = _A / r ** 3                                  # dOmega = detJ dx~ dy~
    xq_f, yq_f = XQ.ravel(), YQ.ravel()

    basis = np.column_stack([np.ones(xq_f.size), xq_f, yq_f, xq_f * yq_f])
    integ = basis.T @ (WQ.ravel() * detJ.ravel())       # ∫{1, x~, y~, x~y~} dOmega
    w = coeffs.T @ integ                                # w_k = ∫ f_k dOmega
    return pts, w


def ldfe_sphere(level: int = 0, n_gl: int = 16):
    """Full-sphere LDFE ordinate set.

    Returns (omega, w) with omega (M, 3) unit vectors and w (M,) solid-angle
    weights summing to 4*pi.  M = 6 * (level + 1)^2 * 4.
    """
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    ns = level + 1
    edges = np.linspace(-_A, _A, ns + 1)
    omegas, weights = [], []
    for frame in _FACES:
        for i in range(ns):
            for j in range(ns):
                pts, w = _cell_weights(edges[i], edges[i + 1],
                                       edges[j], edges[j + 1], frame, n_gl)
                if pts is None:
                    raise RuntimeError(f"degenerate LDFE cell at level {level}")
                omegas.append(pts)
                weights.append(w)
    return np.concatenate(omegas), np.concatenate(weights)


def ldfe_quadrature_2d(level: int = 0, n_gl: int = 16):
    """LDFE ordinate set reduced to 2D XY, matching ``sn.quadrature_2d``.

    Returns (mu, eta, w): the in-plane direction cosines and weights summing to
    1.  The z > 0 hemisphere is kept and its weight doubled, exactly as the
    product set folds its lower-hemisphere mirror in -- the 2D solve is
    invariant under z -> -z.
    """
    omega, w = ldfe_sphere(level, n_gl)
    keep = omega[:, 2] > 0.0
    mu, eta = omega[keep, 0], omega[keep, 1]
    ww = 2.0 * w[keep]
    ww = ww / ww.sum()
    return mu, eta, ww


def mirror_maps(mu, eta, tol=1e-9):
    """Index maps m -> (x-mirror, y-mirror) for reflective boundaries.

    The product set gets these analytically from its layout; a general ordinate
    set has to be matched. Raises if the set is not closed under mu -> -mu and
    eta -> -eta, which would silently break reflective boundaries.
    """
    mu = np.asarray(mu, float)
    eta = np.asarray(eta, float)
    pts = np.column_stack([mu, eta])

    def match(target, label):
        d = np.linalg.norm(pts[:, None, :] - target[None, :, :], axis=2)
        idx = np.argmin(d, axis=0)
        worst = float(d[idx, np.arange(d.shape[1])].max())
        if worst > tol:
            raise ValueError(
                f"ordinate set is not closed under {label} (worst mismatch "
                f"{worst:.2e} > {tol:.1e}); reflective boundaries need a "
                f"symmetric set")
        return idx

    xmir = match(np.column_stack([-mu, eta]), "mu -> -mu")
    ymir = match(np.column_stack([mu, -eta]), "eta -> -eta")
    return xmir, ymir
