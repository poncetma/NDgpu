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

* **Space** -- diamond differencing on a Cartesian grid, swept one quadrant at
  a time. The default sweep is a vectorized *wavefront* (KBA-style): in swept
  coordinates every direction of a quadrant marches +x, +y, and all cells on
  the anti-diagonal i + j = d depend only on diagonal d - 1, so each diagonal
  is one batched update over (all four quadrants) x (M/4 directions) x (cells
  on the diagonal). That leaves only nx + ny - 1 sequential steps per sweep --
  the natural structure for both vectorized CPU execution (numpy) and GPU
  execution (cupy, one kernel batch per diagonal); the sweep itself is written
  against the backend-agnostic array API (``device=``). ``sweep="rows"`` keeps
  the original per-direction scipy banded row solves as a CPU cross-check.

* **Within-group** -- the scattering fixed point (spectral radius = the
  scattering ratio c, near 1 in thermal media) is collapsed by one of
  (``acceleration=``):

  - ``"dsa"`` (default): diffusion synthetic acceleration -- after each
    transport sweep the iteration error is estimated by a cell-centered
    finite-volume diffusion solve, -div(1/(3 Sigma_t) grad) d + Sigma_r d =
    Sigma_s (phi^(l+1/2) - phi^l), Marshak-vacuum on all faces (the frozen
    incoming flux makes the *error* equation zero-incoming even under
    reflective outer boundaries), and added back. One sparse LU per group,
    factorized lazily and reused; the spectral radius drops from c to
    <~ 0.23 c. The FV diffusion operator is not strictly consistent with
    diamond differencing, so for optically thick cells (Sigma_t h >> 1)
    effectiveness can degrade; the loop watches the residual and drops the
    acceleration if it stops contracting.
  - ``"dsa-gmres"``: GMRES on (I - T) phi = b, left-preconditioned with the
    DSA operator I + F^-1 Sigma_s -- the robust choice when plain DSA's
    consistency limits bite.
  - ``"gmres"``: unpreconditioned GMRES on the scalar system (the original
    scheme).
  - ``"si"``: plain source iteration (reference / worst case).

  The reflective boundary condition stays a frozen-incoming outer fixed point,
  Anderson-accelerated over the face fluxes (as before); DSA accelerates the
  scattering iteration inside it.

* **Outer** -- power iteration on the fission source with a Gauss-Seidel group
  sweep, identical in structure to ``ndgpu.solver``. With
  ``outer_acceleration="cmfd"`` (the default) each outer is followed by a
  CMFD/NDA step: one extra current-accumulating sweep per group yields the
  transport scalar flux and face net currents (exact, since diamond
  differencing is conservative cell-by-cell), a drift-corrected finite-volume
  diffusion eigenproblem is assembled from them (face current model
  J = -beta (phi_R - phi_L) + gamma (phi_R + phi_L), gamma chosen to reproduce
  the transport current), solved by an Anderson-accelerated power iteration
  (cheap -- it is a small sparse system), and its flux and k replace the
  transport iterate. At transport convergence the CMFD problem reproduces the
  transport balance identically, so the fixed point is unchanged; the outers
  now converge at the CMFD rate instead of the transport dominance ratio.
  Negative or non-finite CMFD fluxes (drift terms make the matrix non-M) fall
  back to the unaccelerated outer. ``outer_acceleration="power"`` is the plain
  power iteration.

Cross sections come from ndgpu ``Material`` objects, reconstructed into a
transport-consistent problem whose P1/diffusion limit is exactly the diffusion
data the other solvers use: Sigma_t = material.sigma_t (= 1/(3D) when no total is
given, i.e. isotropic scattering / no transport correction), within-group
scatter Sigma_s,gg = Sigma_t - Sigma_a - Sigma_out, group transfer from
material.sigma_s. So S_N, diffusion and SDPN all approximate the *same* physical
problem and their k-eigenvalues are directly comparable.

The wavefront sweep runs on numpy or cupy (``device=``); the DSA correction and
GMRES stay host-side (scipy LU / Krylov) on the small scalar system. Vacuum and
reflective boundaries are supported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded
from scipy.sparse.linalg import LinearOperator, factorized, gmres

from .backend import asnumpy, get_backend
from .grid import Grid
from .materials import Material
from .stencil import harmonic_mean

BC_VACUUM = "vacuum"
BC_REFLECTIVE = "reflective"

_VACUUM_ALPHA = 0.5

# Sweep quadrants (mu > 0, eta > 0); the wavefront sweep batches all four.
_QUADS = ((True, True), (False, True), (True, False), (False, False))


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


def quadrature_3d(n_polar: int, n_azi: int):
    """Product Gauss-Legendre(polar) x uniform(azimuthal) ordinate set for 3D.

    Returns (mu, eta, xi, w): the x/y in-plane cosines, the z (axial) cosine
    xi = cos(theta), and weights (sum to 1), each of length M = n_polar * n_azi.
    Unlike :func:`quadrature_2d` the polar set spans the FULL sphere (both
    hemispheres, xi in (-1, 1)) -- in 3D the axial direction streams, so xi is a
    real transport cosine and cannot be folded. n_azi must be a multiple of 4 so
    the azimuthal set is closed under mu->-mu and eta->-eta; n_polar (the
    Gauss-Legendre order over [-1, 1]) should be even so no ordinate lands on
    xi = 0 (a grazing, non-streaming axial direction).
    """
    if n_azi % 4 != 0:
        raise ValueError(f"n_azi must be a multiple of 4, got {n_azi}")
    if n_polar < 2:
        raise ValueError("n_polar must be >= 2 for 3D")
    xi1, wgl = np.polynomial.legendre.leggauss(n_polar)   # xi in (-1, 1)
    wpol = wgl / 2.0                                       # sum(wpol) = 1
    sint = np.sqrt(1.0 - xi1 * xi1)
    a = np.arange(n_azi)
    phi = 2.0 * np.pi * (a + 0.5) / n_azi
    mu = np.outer(sint, np.cos(phi)).ravel()
    eta = np.outer(sint, np.sin(phi)).ravel()
    xi = np.repeat(xi1, n_azi)
    w = np.outer(wpol, np.full(n_azi, 1.0 / n_azi)).ravel()
    return mu, eta, xi, w


def _azimuth_mirrors(n_polar: int, n_azi: int):
    """Index maps m -> (x-mirror, y-mirror): the ordinate with mu -> -mu and the
    one with eta -> -eta, for the product set laid out polar-major."""
    a = np.arange(n_azi)
    xmir_a = (n_azi // 2 - 1 - a) % n_azi        # phi -> pi - phi  (mu -> -mu)
    ymir_a = (n_azi - 1 - a) % n_azi             # phi -> -phi      (eta -> -eta)
    xmir = np.concatenate([p * n_azi + xmir_a for p in range(n_polar)])
    ymir = np.concatenate([p * n_azi + ymir_a for p in range(n_polar)])
    return xmir, ymir


def diffusion_matrix(D, removal, hx, hy, bc, active=None):
    """Assemble the 2D finite-volume diffusion operator -div(D grad) + removal
    on an (nx, ny) grid: harmonic face D, Marshak vacuum (alpha=1/2) or
    reflective outer faces, matching ndgpu's stencil. ``bc`` is
    ((x0, x1), (y0, y1)) face specs ("vacuum", "reflective", or an albedo alpha).

    ``active`` (bool mask) excises cells: faces between an active and an inactive
    cell carry no diffusion coupling (used by the hybrid solver, which couples
    the transport subdomain there by an imposed interface current instead), and
    inactive cells get a unit diagonal (decoupled). Returns a CSC matrix ready
    to factorize."""
    nx, ny = D.shape
    N = nx * ny
    idx = np.arange(N).reshape(nx, ny)
    rows, cols, vals = [], [], []
    diag = removal.copy().astype(float)
    act = np.ones((nx, ny), bool) if active is None else active

    def add(r, c, v):
        rows.append(np.atleast_1d(r)); cols.append(np.atleast_1d(c))
        vals.append(np.atleast_1d(v))

    # x faces (only where both cells are active)
    wx = harmonic_mean(D[:-1, :], D[1:, :]) / hx**2          # (nx-1, ny)
    wx = np.where(act[:-1, :] & act[1:, :], wx, 0.0)
    add(idx[:-1, :].ravel(), idx[1:, :].ravel(), -wx.ravel())
    add(idx[1:, :].ravel(), idx[:-1, :].ravel(), -wx.ravel())
    diag[:-1, :] += wx; diag[1:, :] += wx
    # y faces
    wy = harmonic_mean(D[:, :-1], D[:, 1:]) / hy**2
    wy = np.where(act[:, :-1] & act[:, 1:], wy, 0.0)
    add(idx[:, :-1].ravel(), idx[:, 1:].ravel(), -wy.ravel())
    add(idx[:, 1:].ravel(), idx[:, :-1].ravel(), -wy.ravel())
    diag[:, :-1] += wy; diag[:, 1:] += wy
    # outer boundary Robin terms
    faces = [(bc[0][0], (0, slice(None)), hx), (bc[0][1], (-1, slice(None)), hx),
             (bc[1][0], (slice(None), 0), hy), (bc[1][1], (slice(None), -1), hy)]
    for spec, sl, d in faces:
        if spec == "reflective":
            continue
        alpha = _VACUUM_ALPHA if spec == "vacuum" else float(spec)
        Db = D[sl]
        term = 2.0 * Db * alpha / (d * (d * alpha + 2.0 * Db))
        diag[sl] += np.where(act[sl], term, 0.0)
    diag = np.where(act, diag, 1.0)                          # excised: unit diagonal
    add(idx.ravel(), idx.ravel(), diag.ravel())
    A = sp.csr_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N))
    return A.tocsc()


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


def _cmfd_power(facs, nsf, chi, scatter, phi0, k0):
    """Anderson-accelerated power iteration on an assembled CMFD eigenproblem,
    shared by the Cartesian and triangular solvers (everything flat, (N,)).

    facs[g] solves the group-g drift-corrected diffusion system; scatter[gf][g]
    is the flat group-transfer field or None. Returns (phi, k, ok): ok=False
    (caller keeps the transport iterate) on breakdown or a negative flux (the
    drift matrix is not an M-matrix). The returned flux is scaled so the total
    fission source matches the input's, keeping the outer iterate continuous."""
    G = len(facs)
    phi = [np.asarray(p, float).ravel() for p in phi0]
    fiss = sum(nsf[g] * phi[g] for g in range(G))
    total0 = float(fiss.sum())
    if not np.isfinite(total0) or total0 <= 0.0:
        return phi0, k0, False
    k = k0
    tn = total0
    hist = []
    prev_err = np.inf
    for _ in range(500):
        fiss_in = fiss
        fs = fiss / k
        phi_new = [None] * G
        for g in range(G):
            q = chi[g] * fs
            for gf in range(G):
                s = scatter[gf][g]
                if gf != g and s is not None:
                    q = q + s * (phi_new[gf] if gf < g else phi[gf])
            phi_new[g] = facs[g](q)
        fiss_new = sum(nsf[g] * phi_new[g] for g in range(G))
        tn = float(fiss_new.sum())
        if not np.isfinite(tn) or tn <= 0.0:
            return phi0, k0, False
        k_new = k * tn / float(fiss.sum())
        dk = abs(k_new - k)
        raw = fiss_new * (total0 / tn)
        rel = (float(np.max(np.abs(raw - fiss_in)))
               / max(float(np.max(np.abs(fiss_in))), 1e-30))
        phi, k = phi_new, k_new
        if dk < 1e-11 and rel < 1e-9:
            break
        if rel > 1.1 * prev_err:
            hist = []
        hist.append((fiss_in, raw))
        if len(hist) > 6:
            hist.pop(0)
        fiss = _anderson(hist)
        fiss = fiss * (total0 / fiss.sum())
        prev_err = rel
    scale = total0 / tn
    phi = [p * scale for p in phi]
    if not all(np.isfinite(p).all() for p in phi):
        return phi0, k0, False
    if min(float(p.min()) for p in phi) < 0.0:
        return phi0, k0, False
    return phi, k, True


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
    n_sweeps: int = 0                            # full transport sweeps performed

    def __repr__(self):
        status = "converged" if self.converged else "NOT CONVERGED"
        return (f"SNResult(k_eff={self.k_eff:.6f}, {status}, "
                f"{self.outer_iterations} outers, {self.n_sweeps} sweeps, "
                f"S{self.n_ordinates}, {self.solve_seconds:.2f} s)")


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
    acceleration  : within-group scattering acceleration -- "dsa" (default,
                    DSA-accelerated source iteration), "dsa-gmres" (DSA-
                    preconditioned GMRES), "gmres" (plain GMRES), "si" (plain
                    source iteration). See the module docstring.
    outer_acceleration : "cmfd" (drift-corrected diffusion eigensolve after
                    each outer; the default with the wavefront sweep) or
                    "power" (plain power iteration; the default -- and only
                    option -- with sweep="rows").
    sweep         : "wavefront" (default; vectorized diagonal sweep, CPU/GPU)
                    or "rows" (per-direction scipy banded row solves, CPU only).
    device        : "cpu" (default), "gpu"/"cuda", or "auto" -- array backend
                    for the wavefront sweep (numpy or cupy).
    max_inner     : cap on source iterations per within-group solve.
    """

    ACCELERATIONS = ("dsa", "dsa-gmres", "gmres", "si")
    OUTER_ACCELERATIONS = ("cmfd", "power")

    def __init__(self, grid: Grid, materials, material_map=None,
                 n_polar: int = 3, n_azi: int = 12, bc: str = BC_VACUUM,
                 require_fissile: bool = True, acceleration: str = "dsa",
                 outer_acceleration: str | None = None,
                 sweep: str = "wavefront", device: str = "cpu",
                 max_inner: int = 800):
        if grid.shape[2] != 1:
            raise ValueError("SNTransportSolver is 2D: grid must have nz == 1")
        if bc not in (BC_VACUUM, BC_REFLECTIVE):
            raise ValueError(f"bc must be {BC_VACUUM!r} or {BC_REFLECTIVE!r}")
        if acceleration not in self.ACCELERATIONS:
            raise ValueError(f"acceleration must be one of {self.ACCELERATIONS}")
        if sweep not in ("wavefront", "rows"):
            raise ValueError("sweep must be 'wavefront' or 'rows'")
        if outer_acceleration is None:                       # cmfd needs currents
            outer_acceleration = "cmfd" if sweep == "wavefront" else "power"
        if outer_acceleration not in self.OUTER_ACCELERATIONS:
            raise ValueError(
                f"outer_acceleration must be one of {self.OUTER_ACCELERATIONS}")
        if sweep == "rows" and outer_acceleration == "cmfd":
            raise ValueError("outer_acceleration='cmfd' needs the wavefront "
                             "sweep (face-current accumulation)")
        self.grid = grid
        self.nx, self.ny = grid.shape[0], grid.shape[1]
        self.hx, self.hy = grid.spacing[0], grid.spacing[1]
        self.bc = bc
        self.acceleration = acceleration
        self.outer_acceleration = outer_acceleration
        self.sweep_mode = sweep
        self.max_inner = max_inner
        self.xp = get_backend(device)
        if sweep == "rows" and self.xp is not np:
            raise ValueError("sweep='rows' is CPU-only; use sweep='wavefront'")

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
        self._setup_wavefront()
        # Lazily built DSA diffusion factorization per group.
        self._dsa_fac = [None] * G
        self._sweep_count = 0

    def _setup_wavefront(self):
        """Precompute the wavefront sweep's index tables: the ordinate split into
        the four sweep quadrants and, per anti-diagonal d = i + j (in swept
        coordinates, where every quadrant marches +x, +y), the swept cell
        indices plus each quadrant's flattened *physical* cell indices."""
        xp, nx, ny = self.xp, self.nx, self.ny
        self._qsel = [np.where((self._fx == fx) & (self._fy == fy))[0]
                      for fx, fy in _QUADS]
        # Product quadrature with n_azi % 4 == 0 puts M/4 ordinates per quadrant.
        self._aq = xp.asarray(np.stack([self._a[s] for s in self._qsel])[:, :, None])
        self._bq = xp.asarray(np.stack([self._b[s] for s in self._qsel])[:, :, None])
        self._wq = xp.asarray(np.stack([self.w[s] for s in self._qsel])[:, :, None])
        wmu, weta = self.w * self.mu, self.w * self.eta      # signed: net currents
        self._wmuq = xp.asarray(np.stack([wmu[s] for s in self._qsel])[:, :, None])
        self._wetaq = xp.asarray(np.stack([weta[s] for s in self._qsel])[:, :, None])
        self._diags = []
        for d in range(nx + ny - 1):
            isw = np.arange(max(0, d - ny + 1), min(d, nx - 1) + 1)
            jsw = d - isw
            fph = np.stack([(isw if fx else nx - 1 - isw) * ny
                            + (jsw if fy else ny - 1 - jsw)
                            for fx, fy in _QUADS])
            self._diags.append((xp.asarray(isw), xp.asarray(jsw),
                                xp.asarray(isw * ny + jsw), xp.asarray(fph)))

    # ---- one full transport sweep -----------------------------------------
    def _sweep(self, q, st_g, inc, currents=False):
        """Transport the isotropic source q (nx, ny) through every ordinate.

        inc holds per-face incoming edge fluxes {'x0','x1'}: (M, ny), {'y0','y1'}:
        (M, nx) (all zero for vacuum). Returns the scalar flux (nx, ny; a backend
        array) and the outgoing edge fluxes as host (numpy) arrays in the same
        dict layout (for the reflective update and the hybrid coupling). With
        ``currents`` a third element holds the face net currents
        (Jx (nx+1, ny), Jy (nx, ny+1); backend arrays) for CMFD.
        """
        self._sweep_count += 1
        if self.sweep_mode == "rows":
            if currents:
                raise ValueError("currents accumulation needs sweep='wavefront'")
            return self._sweep_rows(q, st_g, inc)
        return self._sweep_wavefront(q, st_g, inc, currents)

    def _sweep_wavefront(self, q, st_g, inc, currents=False):
        """Vectorized wavefront sweep: one batched diamond-difference update per
        anti-diagonal over (4 quadrants) x (M/4 directions) x (diagonal cells).
        psi = (q + a psi_x_in + b psi_y_in) / (Sigma_t + a + b) with
        a = 2|mu|/hx, b = 2|eta|/hy; outgoing edges psi_out = 2 psi - psi_in.
        Each cell writes its outgoing edge exactly once (and it is the next
        cell's incoming), so accumulating w*mu-weighted edge fluxes as they are
        produced gives every face's net current with no extra passes."""
        xp, nx, ny = self.xp, self.nx, self.ny
        Mq = self.M // 4
        qf = xp.asarray(q).ravel()
        stf = xp.asarray(st_g).ravel()
        phi_sw = xp.zeros((4, nx * ny))
        ex = xp.empty((4, Mq, ny))                # x-edge flux at the wavefront, per row
        ey = xp.empty((4, Mq, nx))                # y-edge flux at the wavefront, per column
        for qd, (fx, fy) in enumerate(_QUADS):
            sel = self._qsel[qd]
            xin = np.asarray(inc["x0" if fx else "x1"])[sel]
            yin = np.asarray(inc["y0" if fy else "y1"])[sel]
            ex[qd] = xp.asarray(np.ascontiguousarray(xin[:, ::1 if fy else -1]))
            ey[qd] = xp.asarray(np.ascontiguousarray(yin[:, ::1 if fx else -1]))
        a, b, w = self._aq, self._bq, self._wq
        if currents:
            wmu, weta = self._wmuq, self._wetaq
            jx_sw = xp.zeros((4, nx + 1, ny))     # per-quadrant faces, swept coords
            jy_sw = xp.zeros((4, nx, ny + 1))
            jx_sw[:, 0, :] = (wmu * ex).sum(axis=1)   # incoming boundary faces
            jy_sw[:, :, 0] = (weta * ey).sum(axis=1)
        for isw, jsw, fsw, fph in self._diags:
            qv = qf[fph][:, None, :]              # (4, 1, nd)
            stv = stf[fph][:, None, :]
            exv = ex[:, :, jsw]                   # (4, Mq, nd)
            eyv = ey[:, :, isw]
            psi = (qv + a * exv + b * eyv) / (stv + a + b)
            exn = 2.0 * psi - exv
            eyn = 2.0 * psi - eyv
            ex[:, :, jsw] = exn
            ey[:, :, isw] = eyn
            phi_sw[:, fsw] = (w * psi).sum(axis=1)
            if currents:
                jx_sw[:, isw + 1, jsw] = (wmu * exn).sum(axis=1)
                jy_sw[:, isw, jsw + 1] = (weta * eyn).sum(axis=1)
        phi = xp.zeros((nx, ny))
        pq = phi_sw.reshape(4, nx, ny)
        out = {"x0": np.zeros((self.M, ny)), "x1": np.zeros((self.M, ny)),
               "y0": np.zeros((self.M, nx)), "y1": np.zeros((self.M, nx))}
        exh, eyh = asnumpy(ex), asnumpy(ey)       # final wavefront = far-wall edges
        if currents:
            Jx = xp.zeros((nx + 1, ny))
            Jy = xp.zeros((nx, ny + 1))
        for qd, (fx, fy) in enumerate(_QUADS):
            sx, sy = (1 if fx else -1), (1 if fy else -1)
            phi = phi + pq[qd][::sx, ::sy]
            sel = self._qsel[qd]
            out["x1" if fx else "x0"][sel] = exh[qd][:, ::sy]
            out["y1" if fy else "y0"][sel] = eyh[qd][:, ::sx]
            if currents:
                Jx = Jx + jx_sw[qd][::sx, ::sy]   # swept face s -> physical nx - s
                Jy = Jy + jy_sw[qd][::sx, ::sy]
        if currents:
            return phi, out, (Jx, Jy)
        return phi, out

    def _sweep_rows(self, q, st_g, inc):
        """Reference sweep: per-direction row loop with scipy banded solves."""
        nx, ny = self.nx, self.ny
        q = asnumpy(q)
        st_g = asnumpy(st_g)
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
        if self.bc == BC_VACUUM:
            return self._zero_inc()
        return {"x0": out["x0"][self.xmir], "x1": out["x1"][self.xmir],
                "y0": out["y0"][self.ymir], "y1": out["y1"][self.ymir]}

    def _zero_inc(self):
        return {"x0": np.zeros((self.M, self.ny)), "x1": np.zeros((self.M, self.ny)),
                "y0": np.zeros((self.M, self.nx)), "y1": np.zeros((self.M, self.nx))}

    # ---- diffusion synthetic acceleration ---------------------------------
    def _dsa_factor(self, g):
        """Sparse LU of the group-g DSA diffusion operator, built on first use.

        D = 1/(3 Sigma_t), removal = Sigma_t - Sigma_s,gg, Marshak vacuum on
        every face: with the incoming boundary flux frozen, the within-group
        *error* equation has zero incoming flux regardless of self.bc."""
        if self._dsa_fac[g] is None:
            st = np.maximum(asnumpy(self.st[g]), 1e-12)
            ss = asnumpy(self.ss_self[g])
            A = diffusion_matrix(1.0 / (3.0 * st), np.maximum(st - ss, 0.0),
                                 self.hx, self.hy,
                                 ((BC_VACUUM, BC_VACUUM), (BC_VACUUM, BC_VACUUM)))
            self._dsa_fac[g] = factorized(A)
        return self._dsa_fac[g]

    def _dsa_apply(self, g, r):
        """Diffusion estimate of the scattering-iteration error from the
        scatter-weighted residual r = Sigma_s (phi^(l+1/2) - phi^l)."""
        d = self._dsa_factor(g)(asnumpy(r).ravel())
        return self.xp.asarray(d.reshape(self.nx, self.ny))

    # ---- CMFD outer acceleration ------------------------------------------
    def _cmfd_factor(self, g, p, Jx, Jy):
        """LU of the group-g drift-corrected diffusion operator built from the
        transport flux p and face net currents (Jx, Jy): interior face current
        model J = -beta (phi_R - phi_L) + gamma (phi_R + phi_L) with beta the
        harmonic-D coupling and gamma fitted so the model reproduces the
        transport current at the transport flux; boundary faces carry the pure
        transport leakage ratio J / phi_cell. At the transport fixed point the
        operator's balance is the transport balance, so CMFD leaves the
        converged answer unchanged.

        Optically thick cells (tau = Sigma_t h > 1) destabilize plain CMFD, so
        the face coupling is enlarged odCMFD-style, beta += theta with
        theta = 0.25 (1 - 1/tau)_+ (0 for thin cells, -> 1/4 for thick): gamma
        refits the transport current for any beta, so the damping changes the
        outer convergence rate only, never the converged answer."""
        nx, ny, hx, hy = self.nx, self.ny, self.hx, self.hy
        st = asnumpy(self.st[g])
        D = 1.0 / (3.0 * np.maximum(st, 1e-12))
        rem = st - asnumpy(self.ss_self[g])

        def theta(tau_L, tau_R):
            tau = np.maximum(np.maximum(tau_L, tau_R), 1e-30)
            return 0.25 * np.maximum(0.0, 1.0 - 1.0 / tau)
        idx = np.arange(nx * ny).reshape(nx, ny)
        tiny = 1e-30
        rows, cols, vals = [], [], []

        def add(r, c, v):
            rows.append(np.asarray(r).ravel()); cols.append(np.asarray(c).ravel())
            vals.append(np.asarray(v).ravel())

        for axis, (J, h) in enumerate(((Jx, hx), (Jy, hy))):
            if axis == 0:
                beta = (harmonic_mean(D[:-1, :], D[1:, :]) / h
                        + theta(st[:-1, :] * h, st[1:, :] * h))
                pL, pR = p[:-1, :], p[1:, :]
                L, R = idx[:-1, :], idx[1:, :]
                Jin = J[1:nx, :]
                b0, b1 = J[0, :], J[nx, :]
                c0, c1 = idx[0, :], idx[-1, :]
                p0, p1 = p[0, :], p[-1, :]
            else:
                beta = (harmonic_mean(D[:, :-1], D[:, 1:]) / h
                        + theta(st[:, :-1] * h, st[:, 1:] * h))
                pL, pR = p[:, :-1], p[:, 1:]
                L, R = idx[:, :-1], idx[:, 1:]
                Jin = J[:, 1:ny]
                b0, b1 = J[:, 0], J[:, ny]
                c0, c1 = idx[:, 0], idx[:, -1]
                p0, p1 = p[:, 0], p[:, -1]
            gh = (Jin + beta * (pR - pL)) / np.maximum(pL + pR, tiny)
            add(L, L, (beta + gh) / h); add(L, R, (-beta + gh) / h)
            add(R, R, (beta - gh) / h); add(R, L, (-beta - gh) / h)
            add(c0, c0, -(b0 / np.maximum(p0, tiny)) / h)      # boundary leakage
            add(c1, c1, (b1 / np.maximum(p1, tiny)) / h)
        add(idx, idx, rem)
        A = sp.csr_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nx * ny, nx * ny))
        return factorized(A.tocsc())

    def _cmfd_eigen(self, facs, phi0, k0):
        """CMFD eigensolve on the (nx, ny) grid: flattens into the shared
        ``_cmfd_power`` power iteration and reshapes the result."""
        G, nx, ny = self.G, self.nx, self.ny
        nsf = [asnumpy(self.nsf[g]).ravel() for g in range(G)]
        chi = [asnumpy(self.chi[g]).ravel() for g in range(G)]
        scatter = [[None if s is None else np.asarray(s).ravel() for s in row]
                   for row in self.scatter]
        phi, k, ok = _cmfd_power(facs, nsf, chi, scatter,
                                 [np.asarray(p, float).ravel() for p in phi0], k0)
        if not ok:
            return phi0, k0, False
        return [p.reshape(nx, ny) for p in phi], k, True

    def _cmfd_update(self, phi_new, inc, fs, st, ss_self, chi, scatter, k):
        """One CMFD step after a transport outer: for each group, one
        current-accumulating sweep of the converged within-group source yields
        a consistent (flux, face-current) pair; the drift-corrected diffusion
        eigenproblem built from them is solved and its (flux, k) returned as
        the new outer iterate (transport iterate on fallback)."""
        xp, G = self.xp, self.G
        phi_h, facs = [None] * G, [None] * G
        for g in range(G):
            q = chi[g] * fs
            for gf in range(G):
                s = scatter[gf][g]
                if gf != g and s is not None:
                    q = q + s * phi_new[gf]
            p, _, (Jx, Jy) = self._sweep(ss_self[g] * phi_new[g] + q, st[g],
                                         inc[g], currents=True)
            phi_h[g] = asnumpy(p)
            facs[g] = self._cmfd_factor(g, phi_h[g], asnumpy(Jx), asnumpy(Jy))
        phi_c, k_c, ok = self._cmfd_eigen(facs, phi_h, k)
        if not ok:
            return phi_new, k, False
        return [xp.asarray(p) for p in phi_c], k_c, True

    # ---- within-group scattering solvers ----------------------------------
    def _si_solve(self, qext, ss_g, st_g, phi, frozen, tol, g, accelerate):
        """(DSA-accelerated) source iteration on the within-group scattering
        fixed point with the boundary incoming fluxes ``frozen``. With
        acceleration, each sweep is followed by the diffusion error correction;
        if the update norm stops contracting (inconsistent-discretization
        regime, optically thick cells) the acceleration is dropped for the
        remainder of the solve."""
        xp = self.xp
        phi = xp.asarray(asnumpy(phi).copy())
        q0 = xp.asarray(qext)
        prev = None
        bad = 0
        for _ in range(self.max_inner):
            half, _ = self._sweep(ss_g * phi + q0, st_g, frozen)
            if accelerate:
                new = half + self._dsa_apply(g, ss_g * (half - phi))
            else:
                new = half
            d = float(xp.max(xp.abs(new - phi)))
            scale = max(float(xp.max(xp.abs(new))), 1e-300)
            phi = new
            if d <= tol * scale:
                break
            if accelerate:
                if prev is not None and d > prev:
                    bad += 1
                    if bad >= 3:
                        accelerate = False
                else:
                    bad = 0
                prev = d
        return phi

    def _gmres_solve(self, qext, ss_g, st_g, phi, frozen, tol, g, precondition):
        """GMRES on the scalar within-group system (I - T) phi = b, T = one
        sweep of the scattering source, boundary incoming fluxes frozen.
        With ``precondition`` the DSA operator M = I + F^-1 Sigma_s is applied
        as a left preconditioner."""
        xp, nx, ny = self.xp, self.nx, self.ny
        q0 = xp.asarray(qext)

        def sweep_src(src):
            p, _ = self._sweep(src + q0, st_g, frozen)
            return p

        b = asnumpy(sweep_src(xp.zeros((nx, ny)))).ravel()     # source-only response

        def op(x):
            p = sweep_src(ss_g * xp.asarray(x.reshape(nx, ny)))
            return x - (asnumpy(p).ravel() - b)

        n = nx * ny
        A = LinearOperator((n, n), matvec=op, dtype=float)
        M = None
        if precondition:
            def prec(x):
                r = ss_g * xp.asarray(x.reshape(nx, ny))
                return x + asnumpy(self._dsa_apply(g, r)).ravel()
            M = LinearOperator((n, n), matvec=prec, dtype=float)
        phi_v, _ = gmres(A, b, x0=asnumpy(phi).ravel(), M=M,
                         rtol=min(tol, 1e-4), atol=0.0, maxiter=400)
        return xp.asarray(phi_v.reshape(nx, ny))

    def _scatter_solve(self, qext, ss_g, st_g, phi, frozen, tol, g):
        """Solve the within-group scattering fixed point with the configured
        acceleration (see class docstring)."""
        tol = min(tol, 1e-4)
        if self.acceleration in ("gmres", "dsa-gmres"):
            return self._gmres_solve(qext, ss_g, st_g, phi, frozen, tol, g,
                                     precondition=self.acceleration == "dsa-gmres")
        return self._si_solve(qext, ss_g, st_g, phi, frozen, tol, g,
                              accelerate=self.acceleration == "dsa")

    # ---- within-group solve ------------------------------------------------
    def _solve_group(self, qext, ss_g, st_g, phi0, inc, tol, g):
        """Scattering solve nested in the reflective-boundary fixed point.

        For each frozen boundary the scattering system is solved to
        convergence; the boundary incoming fluxes are then refreshed from the
        solution's outgoing fluxes under reflection. That outer fixed point,
        inc <- reflect(sweep(phi(inc))), converges only at the boundary
        scattering rate (near 1 in thermal media), so it is Anderson-accelerated
        over the flattened face fluxes. Vacuum has no incoming flux -- one solve.
        """
        phi = self._scatter_solve(qext, ss_g, st_g, phi0, inc, tol, g)
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

        def gstep(v):                                          # one fixed-point step
            inc_v = unflat(v)
            ph = self._scatter_solve(qext, ss_g, st_g, phi, inc_v, tol, g)
            _, out = self._sweep(ss_g * ph + self.xp.asarray(qext), st_g, inc_v)
            return flat(self._reflect(out)), ph

        v = flat(inc)
        hist = []                                              # (v_in, g(v_in)) pairs
        for _ in range(200):
            gv, phi = gstep(v)
            hist.append((v, gv))
            if len(hist) > 6:
                hist.pop(0)
            v_next = _anderson(hist)
            d = np.max(np.abs(gv - v))
            v = v_next
            if d < tol * max(1.0, float(self.xp.max(phi))):
                break
        return phi, unflat(v)

    def solve(self, tol_k: float = 1e-7, tol_source: float = 1e-7,
              max_outer: int = 500, verbose: bool = False) -> SNResult:
        """Power iteration on the fission source until |dk| < tol_k and the
        scalar-flux change < tol_source."""
        t0 = time.perf_counter()
        xp = self.xp
        G, nx, ny = self.G, self.nx, self.ny
        sweeps0 = self._sweep_count
        st = [xp.asarray(self.st[g]) for g in range(G)]
        ss_self = [xp.asarray(self.ss_self[g]) for g in range(G)]
        nsf = [xp.asarray(self.nsf[g]) for g in range(G)]
        chi = [xp.asarray(self.chi[g]) for g in range(G)]
        scatter = [[None if s is None else xp.asarray(s) for s in row]
                   for row in self.scatter]
        phi = [xp.ones((nx, ny)) for _ in range(G)]
        inc = [self._zero_inc() for _ in range(G)]
        fiss = sum(nsf[g] * phi[g] for g in range(G))
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
            phi_new = [None] * G
            for g in range(G):
                qext = chi[g] * fs
                for gf in range(G):
                    s = scatter[gf][g]
                    if gf != g and s is not None:
                        src = phi_new[gf] if gf < g else phi[gf]   # Gauss-Seidel
                        qext = qext + s * src
                phi_new[g], inc[g] = self._solve_group(
                    qext, ss_self[g], st[g], phi[g], inc[g], rtol, g)
            cmfd_ok = False
            if self.outer_acceleration == "cmfd":
                phi_new, k_cmfd, cmfd_ok = self._cmfd_update(
                    phi_new, inc, fs, st, ss_self, chi, scatter, k)
            fiss_new = sum(nsf[g] * phi_new[g] for g in range(G))
            k_new = (k_cmfd if cmfd_ok
                     else k * float(fiss_new.sum()) / float(fiss.sum()))
            dk = abs(k_new - k)
            denom = max(max(float(xp.max(p)) for p in phi_new), 1e-30)
            rel = max(float(xp.max(xp.abs(phi_new[g] - phi[g])))
                      for g in range(G)) / denom
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
        flux = np.stack([asnumpy(p) for p in phi])
        return SNResult(k_eff=k, flux=flux, converged=converged,
                        outer_iterations=outer,
                        solve_seconds=time.perf_counter() - t0,
                        n_ordinates=self.M, k_history=k_hist,
                        n_sweeps=self._sweep_count - sweeps0)
