"""2D discrete-ordinates (S_N) transport k-eigenvalue solver.

The 2D extension of the 1D Gauss-Legendre S_N reference that validates the SDPN
family (``examples/sdpn_benchmark_1d.py``). Same ingredients, one dimension up:

* **Angle** -- a product quadrature over the unit sphere for XY geometry:
  Gauss-Legendre in the polar cosine xi = cos(theta) on (0, 1] (the upper
  hemisphere; the lower one mirrors it, weight doubled) times a uniform
  azimuthal set on [0, 2pi). Direction m has in-plane cosines
  (mu, eta) = sin(theta)(cos(phi), sin(phi)); weights are normalized to sum to
  1, so the scalar flux is phi = sum_m w_m psi_m and the isotropic within-group
  source is simply Sigma_s phi + q (no 1/4pi factor).

* **Space** -- diamond differencing on a Cartesian grid, swept one direction at
  a time. A 2D sweep factorizes into a loop over rows in the direction's y sense
  (each row's incoming bottom-edge flux comes from the previously swept row) with
  a 1D diamond-difference bidiagonal solve along x inside it -- the exact 1D
  recurrence (``_row_solve``), so the whole sweep is O(cells) with only ``ny``
  Python steps per direction.

* **Within-group** -- the scattering fixed point (spectral radius = the
  scattering ratio, near 1 in thermal media) is collapsed by GMRES on the scalar
  system (I - T) phi = b, T = one sweep of the scattering source, with the
  reflective boundary frozen and iterated in an outer pass (as in 1D).

* **Outer** -- power iteration on the fission source with a Gauss-Seidel group
  sweep, identical in structure to ``ndgpu.solver``.

Cross sections come from ndgpu ``Material`` objects, reconstructed into a
transport-consistent problem whose P1/diffusion limit is exactly the diffusion
data the other solvers use: Sigma_t = material.sigma_t (= 1/(3D) when no total is
given, i.e. isotropic scattering / no transport correction), within-group
scatter Sigma_s,gg = Sigma_t - Sigma_a - Sigma_out, group transfer from
material.sigma_s. So S_N, diffusion and SDPN all approximate the *same* physical
problem and their k-eigenvalues are directly comparable.

This is a CPU/numpy reference solver (scipy banded solves + GMRES); it is not the
GPU-native production path. Vacuum and reflective boundaries are supported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse.linalg import LinearOperator, gmres

from .grid import Grid
from .materials import Material

BC_VACUUM = "vacuum"
BC_REFLECTIVE = "reflective"


def quadrature_2d(n_polar: int, n_azi: int):
    """Product Gauss-Legendre(polar) x uniform(azimuthal) ordinate set for 2D XY.

    Returns (mu, eta, w): in-plane direction cosines and weights (sum to 1),
    each of length M = n_polar * n_azi. n_azi must be a multiple of 4 so the
    azimuthal set is closed under mu->-mu and eta->-eta (needed for exact
    reflective boundaries); the polar set uses the upper hemisphere with the
    lower-hemisphere mirror folded in (weight x2, then normalized).
    """
    if n_azi % 4 != 0:
        raise ValueError(f"n_azi must be a multiple of 4, got {n_azi}")
    if n_polar < 1:
        raise ValueError("n_polar must be >= 1")
    # Gauss-Legendre on xi = cos(theta) in [0, 1] (map the standard [-1, 1] set).
    x, wgl = np.polynomial.legendre.leggauss(2 * n_polar)
    pos = x > 0
    xi = x[pos]                                  # cos(theta) in (0, 1)
    wpol = wgl[pos]                              # sum(wpol) = 1 over [0, 1]
    sint = np.sqrt(1.0 - xi * xi)
    a = np.arange(n_azi)
    phi = 2.0 * np.pi * (a + 0.5) / n_azi
    mu = np.outer(sint, np.cos(phi)).ravel()
    eta = np.outer(sint, np.sin(phi)).ravel()
    w = np.outer(wpol, np.full(n_azi, 1.0 / n_azi)).ravel()
    return mu, eta, w


def _azimuth_mirrors(n_polar: int, n_azi: int):
    """Index maps m -> (x-mirror, y-mirror): the ordinate with mu -> -mu and the
    one with eta -> -eta, for the product set laid out polar-major."""
    a = np.arange(n_azi)
    xmir_a = (n_azi // 2 - 1 - a) % n_azi        # phi -> pi - phi  (mu -> -mu)
    ymir_a = (n_azi - 1 - a) % n_azi             # phi -> -phi      (eta -> -eta)
    xmir = np.concatenate([p * n_azi + xmir_a for p in range(n_polar)])
    ymir = np.concatenate([p * n_azi + ymir_a for p in range(n_polar)])
    return xmir, ymir


def _anderson(hist):
    """Anderson-accelerated next iterate from (input, raw_output) pairs (latest
    last): the residual-minimizing affine combination that collapses the slow
    fixed-point modes. Falls back to the plain iterate with < 2 pairs."""
    raw = hist[-1][1]
    if len(hist) < 2:
        return raw
    res = [gj - sj for sj, gj in hist]
    dres = [res[i] - res[-1] for i in range(len(res) - 1)]
    m = len(dres)
    A = np.array([[float(np.dot(dres[i], dres[j])) for j in range(m)]
                  for i in range(m)])
    b = np.array([-float(np.dot(dres[i], res[-1])) for i in range(m)])
    A[np.diag_indices(m)] += 1e-12 * (np.trace(A) + 1e-300)
    try:
        gamma = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return raw
    if not np.all(np.abs(gamma) < 1e4):
        return raw
    out = raw.copy()
    for j in range(m):
        out += gamma[j] * (hist[j][1] - hist[-1][1])
    return out


def _row_solve(q_ang, st_eff, c, binc, forward):
    """One row's 1D diamond-difference sweep along x (the 1D reference recurrence).

    st_eff already folds the y-leakage and Sigma_t; c = 2|mu|/hx; q_ang the row's
    angular source (scatter/fission + y-incoming); binc the upstream x-edge flux.
    Returns the row's cell fluxes (physical order) and the far-wall outgoing edge.
    """
    idx = slice(None) if forward else slice(None, None, -1)
    st = st_eff[idx]
    q = q_ang[idx]
    denom = st + c
    alpha = (c - st) / denom
    beta = 2.0 * q / denom
    n = st.size
    ab = np.zeros((2, n))
    ab[0] = 1.0
    ab[1, :-1] = -alpha[1:]
    b = beta.copy()
    b[0] += alpha[0] * binc
    e = solve_banded((1, 0), ab, b)              # outgoing edge of each cell
    e_in = np.empty(n)
    e_in[0] = binc
    e_in[1:] = e[:-1]
    psi = (q + c * e_in) / denom
    return psi[idx], e[-1]


@dataclass
class SNResult:
    k_eff: float
    flux: np.ndarray                             # (G, nx, ny) scalar flux
    converged: bool
    outer_iterations: int
    solve_seconds: float
    n_ordinates: int
    k_history: list = field(default_factory=list)

    def __repr__(self):
        status = "converged" if self.converged else "NOT CONVERGED"
        return (f"SNResult(k_eff={self.k_eff:.6f}, {status}, "
                f"{self.outer_iterations} outers, S{self.n_ordinates}, "
                f"{self.solve_seconds:.2f} s)")


class SNTransportSolver:
    """2D discrete-ordinates k-eigenvalue solver on a Cartesian grid.

    Parameters
    ----------
    grid          : ndgpu Grid with nz == 1 (the solve is 2D in x, y).
    materials     : a Material or list indexed by material_map.
    material_map  : int array of shape (nx, ny) or (nx, ny, 1); omit for a
                    homogeneous medium.
    n_polar, n_azi: product-quadrature sizes (n_azi a multiple of 4). Total
                    ordinates M = n_polar * n_azi.
    bc            : "vacuum" or "reflective" (all four in-plane faces).
    """

    def __init__(self, grid: Grid, materials, material_map=None,
                 n_polar: int = 3, n_azi: int = 12, bc: str = BC_VACUUM,
                 require_fissile: bool = True):
        if grid.shape[2] != 1:
            raise ValueError("SNTransportSolver is 2D: grid must have nz == 1")
        if bc not in (BC_VACUUM, BC_REFLECTIVE):
            raise ValueError(f"bc must be {BC_VACUUM!r} or {BC_REFLECTIVE!r}")
        self.grid = grid
        self.nx, self.ny = grid.shape[0], grid.shape[1]
        self.hx, self.hy = grid.spacing[0], grid.spacing[1]
        self.bc = bc

        mats = [materials] if isinstance(materials, Material) else list(materials)
        G = mats[0].n_groups
        if any(m.n_groups != G for m in mats):
            raise ValueError("all materials must share the group count")
        self.G = G
        if material_map is None:
            if len(mats) > 1:
                raise ValueError("material_map required with multiple materials")
            mmap = np.zeros((self.nx, self.ny), dtype=int)
        else:
            mmap = np.asarray(material_map).reshape(self.nx, self.ny)

        # Transport-consistent per-cell fields (see module docstring).
        def per_group(fn):
            table = np.array([fn(m) for m in mats])          # (nmat, G)
            return np.stack([table[mmap, g] for g in range(G)])  # (G, nx, ny)

        self.st = per_group(lambda m: np.asarray(m.sigma_t, float))
        removal = per_group(lambda m: np.asarray(m.removal, float))
        self.ss_self = np.maximum(self.st - removal, 0.0)        # within-group scatter
        self.nsf = per_group(lambda m: np.asarray(m.nu_sigma_f, float))
        self.chi = per_group(lambda m: np.asarray(m.chi, float))
        # Group transfer Sigma_s[g' -> g] (off-diagonal only), per cell.
        sig_s = np.array([m.sigma_s for m in mats])             # (nmat, G, G)
        self.scatter = [[None] * G for _ in range(G)]           # [g_from][g_to]
        for gf in range(G):
            for gt in range(G):
                if gf != gt and np.any(sig_s[:, gf, gt]):
                    self.scatter[gf][gt] = sig_s[mmap, gf, gt]  # (nx, ny)
        if require_fissile and not np.any(self.nsf):
            raise ValueError("no fissile material: k-eigenvalue is undefined")

        self.mu, self.eta, self.w = quadrature_2d(n_polar, n_azi)
        self.M = self.mu.size
        self.xmir, self.ymir = _azimuth_mirrors(n_polar, n_azi)
        # Precompute per-direction sweep constants.
        self._a = 2.0 * np.abs(self.mu) / self.hx
        self._b = 2.0 * np.abs(self.eta) / self.hy
        self._fx = self.mu > 0.0
        self._fy = self.eta > 0.0

    # ---- one full transport sweep -----------------------------------------
    def _sweep(self, q, st_g, inc):
        """Transport the isotropic source q (nx, ny) through every ordinate.

        inc holds per-face incoming edge fluxes {'x0','x1'}: (M, ny), {'y0','y1'}:
        (M, nx) (all zero for vacuum). Returns the scalar flux (nx, ny) and the
        outgoing edge fluxes in the same dict layout (for the reflective update).
        """
        nx, ny = self.nx, self.ny
        phi = np.zeros((nx, ny))
        out = {"x0": np.zeros((self.M, ny)), "x1": np.zeros((self.M, ny)),
               "y0": np.zeros((self.M, nx)), "y1": np.zeros((self.M, nx))}
        for m in range(self.M):
            a, b, fx, fy, w = self._a[m], self._b[m], self._fx[m], self._fy[m], self.w[m]
            xin = inc["x0"][m] if fx else inc["x1"][m]         # (ny,) per row
            yin = (inc["y0"][m] if fy else inc["y1"][m]).copy()  # (nx,)
            rows = range(ny) if fy else range(ny - 1, -1, -1)
            x_far = np.empty(ny)
            y_far = None
            for j in rows:
                st_eff = st_g[:, j] + b
                q_ang = b * yin + q[:, j]
                psi_row, e_far = _row_solve(q_ang, st_eff, a, xin[j], fx)
                phi[:, j] += w * psi_row
                yout = 2.0 * psi_row - yin
                yin = yout
                x_far[j] = e_far
                y_far = yout                                   # last swept row wins
            out["x1" if fx else "x0"][m] = x_far
            out["y1" if fy else "y0"][m] = y_far
        return phi, out

    def _reflect(self, out):
        """Incoming edge fluxes from a sweep's outgoing fluxes under reflection
        (vacuum -> all zero)."""
        z_ny = np.zeros((self.M, self.ny))
        z_nx = np.zeros((self.M, self.nx))
        if self.bc == BC_VACUUM:
            return {"x0": z_ny, "x1": z_ny, "y0": z_nx, "y1": z_nx}
        return {"x0": out["x0"][self.xmir], "x1": out["x1"][self.xmir],
                "y0": out["y0"][self.ymir], "y1": out["y1"][self.ymir]}

    def _zero_inc(self):
        return {"x0": np.zeros((self.M, self.ny)), "x1": np.zeros((self.M, self.ny)),
                "y0": np.zeros((self.M, self.nx)), "y1": np.zeros((self.M, self.nx))}

    def _scatter_solve(self, qext, ss_g, st_g, phi, frozen, tol):
        """GMRES the within-group scattering fixed point (I - T) phi = b with the
        boundary incoming fluxes ``frozen``; T = one sweep of the scatter source."""
        nx, ny = self.nx, self.ny

        def sweep_src(src):
            p, _ = self._sweep(src + qext, st_g, frozen)
            return p

        b = sweep_src(np.zeros((nx, ny)))                      # source-only response

        def op(x):
            p = sweep_src(ss_g * x.reshape(nx, ny))
            return x - (p - b).ravel()

        A = LinearOperator((nx * ny, nx * ny), matvec=op, dtype=float)
        phi_v, _ = gmres(A, b.ravel(), x0=phi.ravel(),
                         rtol=min(tol, 1e-4), atol=0.0, maxiter=400)
        return phi_v.reshape(nx, ny)

    # ---- within-group solve ------------------------------------------------
    def _solve_group(self, qext, ss_g, st_g, phi0, inc, tol):
        """Scattering solve nested in the reflective-boundary fixed point.

        For each frozen boundary the scattering system is GMRES-solved to
        convergence; the boundary incoming fluxes are then refreshed from the
        solution's outgoing fluxes under reflection. That outer fixed point,
        inc <- reflect(sweep(phi(inc))), converges only at the boundary
        scattering rate (near 1 in thermal media), so it is Anderson-accelerated
        over the flattened face fluxes. Vacuum has no incoming flux -- one solve.
        """
        phi = self._scatter_solve(qext, ss_g, st_g, phi0, inc, tol)
        if self.bc == BC_VACUUM:
            return phi, inc

        faces = ("x0", "x1", "y0", "y1")
        flat = lambda d: np.concatenate([d[f].ravel() for f in faces])
        sizes = [inc[f].size for f in faces]

        def unflat(v):
            out, o = {}, 0
            for f, s in zip(faces, sizes):
                out[f] = v[o:o + s].reshape(inc[f].shape)
                o += s
            return out

        def g(v):                                              # one fixed-point step
            inc_v = unflat(v)
            ph = self._scatter_solve(qext, ss_g, st_g, phi, inc_v, tol)
            _, out = self._sweep(ss_g * ph + qext, st_g, inc_v)
            return flat(self._reflect(out)), ph

        v = flat(inc)
        hist = []                                              # (v_in, g(v_in)) pairs
        for _ in range(200):
            gv, phi = g(v)
            hist.append((v, gv))
            if len(hist) > 6:
                hist.pop(0)
            v_next = _anderson(hist)
            d = np.max(np.abs(gv - v))
            v = v_next
            if d < tol * max(1.0, np.max(phi)):
                break
        return phi, unflat(v)

    def solve(self, tol_k: float = 1e-7, tol_source: float = 1e-7,
              max_outer: int = 500, verbose: bool = False) -> SNResult:
        """Power iteration on the fission source until |dk| < tol_k and the
        scalar-flux change < tol_source."""
        t0 = time.perf_counter()
        G, nx, ny = self.G, self.nx, self.ny
        phi = np.ones((G, nx, ny))
        inc = [self._zero_inc() for _ in range(G)]
        fiss = sum(self.nsf[g] * phi[g] for g in range(G))
        k = 1.0
        k_hist = []
        converged = False
        outer = 0
        prev_rel = 1.0
        for outer in range(1, max_outer + 1):
            fs = fiss / k
            # Inner tolerance tracks the outer convergence: loose early, tight
            # late (the reflective boundary fixed point in particular must be
            # converged well below the target k accuracy or it leaks).
            rtol = min(1e-3, max(0.05 * prev_rel, 0.01 * tol_k, 1e-11))
            phi_new = np.zeros_like(phi)
            for g in range(G):
                qext = self.chi[g] * fs
                for gf in range(G):
                    s = self.scatter[gf][g]
                    if gf != g and s is not None:
                        src = phi_new[gf] if gf < g else phi[gf]   # Gauss-Seidel
                        qext = qext + s * src
                phi_new[g], inc[g] = self._solve_group(
                    qext, self.ss_self[g], self.st[g], phi[g], inc[g], rtol)
            fiss_new = sum(self.nsf[g] * phi_new[g] for g in range(G))
            k_new = k * fiss_new.sum() / fiss.sum()
            dk = abs(k_new - k)
            denom = max(np.max(phi_new), 1e-30)
            rel = np.max(np.abs(phi_new - phi)) / denom
            phi, fiss, k = phi_new, fiss_new, k_new
            prev_rel = max(rel, dk)
            k_hist.append(k)
            if verbose:
                print(f"  outer {outer:3d}  k = {k:.7f}  dk = {dk:.2e}  "
                      f"rel = {rel:.2e}  rtol = {rtol:.1e}")
            # Only accept convergence once the inner solve is tight enough that
            # its residual cannot masquerade as a converged k (else a trivial
            # fission shape breaks the outer before the boundary is resolved).
            if dk < tol_k and rel < tol_source and rtol < max(1e-8, tol_k):
                converged = True
                break
        return SNResult(k_eff=k, flux=phi, converged=converged,
                        outer_iterations=outer,
                        solve_seconds=time.perf_counter() - t0,
                        n_ordinates=self.M, k_history=k_hist)
