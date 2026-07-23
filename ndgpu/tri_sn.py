"""Discrete-ordinates (S_N) transport on the body-fitted triangular mesh.

The triangular counterpart of ``ndgpu.sn`` (2D Cartesian S_N), so discrete-
ordinates transport -- and the hybrid S_N/diffusion drum treatment -- run on the
actual HP-MR hex/triangular core, not a Cartesian stand-in.

The mesh is the same structured equilateral-triangle lattice the diffusion/SP3
solvers use (:class:`ndgpu.tri.TriGrid`): cells stored on (nrows, ncols, 2) with
the last index the down/up triangle of each rhombus, every interior cell coupled
to three neighbours at fixed offsets. That structure is what makes S_N tractable
here: rather than order the transport sweep by hand (and break cycles), we
assemble, for each ordinate, the sparse streaming+collision operator
L_Omega = Omega.grad + Sigma_t with upwind (step) differencing and factorize it
once. A "sweep" is then a triangular solve; the within-group scattering fixed
point is GMRES'd exactly as in the Cartesian solver, inside a fission power
iteration.

Upwind (step) differencing keeps every angular flux non-negative -- robust
through the near-black B4C drums where diamond differencing would ring -- at the
cost of being first-order in space (more numerically diffusive than the diamond
scheme of ``ndgpu.sn``). The equilateral triangle has edge length h, area
sqrt(3)/4 h^2, and the six (type, edge) outward normals below.

Boundaries: ``vacuum`` (the HP-MR core surface; incoming flux zero on edges
facing an excised/void cell or off-mesh) and ``periodic`` (wrap the lattice to a
torus -- an infinite medium whose flat-flux k is exactly k_inf, used to validate
the operator). CPU/numpy reference, not the GPU path.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, factorized, gmres

from .materials import Material
from .sn import SNResult, _anderson, quadrature_2d
from .solver import Fields
from .tri import TriGrid

_SQRT3_2 = np.sqrt(3.0) / 2.0

# (source type, neighbour rhombus offset di, dj, neighbour type, outward normal).
# Down triangle (t=0): hypotenuse->up(i,j), bottom->up(i-1,j), left->up(i,j-1).
# Up   triangle (t=1): hypotenuse->down(i,j), top->down(i+1,j), right->down(i,j+1).
_EDGES = [
    (0, 0, 0, 1, (_SQRT3_2, 0.5)),
    (0, -1, 0, 1, (0.0, -1.0)),
    (0, 0, -1, 1, (-_SQRT3_2, 0.5)),
    (1, 0, 0, 0, (-_SQRT3_2, -0.5)),
    (1, 1, 0, 0, (0.0, 1.0)),
    (1, 0, 1, 0, (_SQRT3_2, -0.5)),
]


class TriSNTransportSolver:
    """S_N transport k-eigenvalue solver on a triangular mesh.

    Parameters
    ----------
    grid          : TriGrid (2D; shape (nrows, ncols, 2)).
    materials     : a Material or list indexed by material_map.
    material_map  : int array of shape grid.shape; omit for a homogeneous medium.
    active        : optional bool mask of in-core cells (shape grid.shape). Cells
                    outside it are excised; active cells facing them see the
                    ``bc`` boundary law. Defaults to all-active.
    n_polar, n_azi: product-quadrature sizes (n_azi a multiple of 4).
    bc            : "vacuum" (default) or "periodic" (torus / infinite medium).
    scheme        : spatial differencing. "step" (default) -- upwind, robustly
                    non-negative, first-order. "scb" -- simple corner balance, a
                    second-order finite-volume scheme: each triangle is split into
                    three corner sub-volumes (3 unknowns per cell), with the cell
                    boundary half-edges upwinded to the neighbour's corner at the
                    shared vertex and the interior corner faces carrying the
                    average of the two corner fluxes. It is a genuine finite-volume
                    balance (not a difference stencil), stays linear so it
                    factorizes once, is exact for a flat flux (k_inf exact), and --
                    unlike the earlier edge-average scheme -- reaches second-order
                    convergence, resolving the HP-MR drum worth at far coarser mesh
                    than step. Costs ~3x the unknowns of step.
    """

    def __init__(self, grid: TriGrid, materials, material_map=None, active=None,
                 n_polar: int = 3, n_azi: int = 12, bc: str = "vacuum",
                 scheme: str = "step", require_fissile: bool = True,
                 mix_material=None, mix_weight=None):
        if len(grid.shape) != 3 or grid.shape[2] != 2:
            raise ValueError("TriSNTransportSolver is 2D: grid shape (nr, nc, 2)")
        if bc not in ("vacuum", "periodic"):
            raise ValueError("bc must be 'vacuum' or 'periodic'")
        if scheme not in ("step", "scb"):
            raise ValueError("scheme must be 'step' or 'scb'")
        self.scheme = scheme
        self.grid = grid
        self.nr, self.nc = grid.shape[0], grid.shape[1]
        self.bc = bc
        self.h = grid.side
        self.area = (np.sqrt(3.0) / 4.0) * self.h ** 2
        self.N = self.nr * self.nc * 2

        mats = [materials] if isinstance(materials, Material) else list(materials)
        self.G = mats[0].n_groups
        mmap = (np.zeros(grid.shape, int) if material_map is None
                else np.asarray(material_map).reshape(grid.shape))
        self.active = (np.ones(grid.shape, bool) if active is None
                       else np.asarray(active).reshape(grid.shape))

        # Per-cell fields via the validated Fields blend, so the optional polar
        # volume-mixing (mix_material/mix_weight) that dilutes the thin B4C arc
        # into the drum cells applies to S_N exactly as it does to diffusion:
        # cross sections mix linearly, D harmonically, chi by fission share. With
        # Sigma_t = 1/(3D) (no explicit total) the linear Sigma_t mix equals the
        # harmonic-D mix, so S_N stays P1-consistent with the diffusion reference.
        f = Fields(np, grid, mats, mmap, np.float64,
                   mix_material=mix_material, mix_weight=mix_weight)
        self.st = np.stack(f.sigma_t)                        # (G, nr, nc, 2)
        removal = np.stack(f.removal)
        self.ss_self = np.maximum(self.st - removal, 0.0)
        self.nsf = np.stack(f.nu_sigma_f)
        self.chi = np.stack(f.chi)
        self.scatter = [[None] * self.G for _ in range(self.G)]
        for gf in range(self.G):
            for gt in range(self.G):
                if gf != gt and f.sigma_s[gf][gt] is not None:
                    self.scatter[gf][gt] = np.asarray(f.sigma_s[gf][gt])
        if require_fissile and not np.any(self.nsf):
            raise ValueError("no fissile material: k-eigenvalue is undefined")

        self.mu, self.eta, self.w = quadrature_2d(n_polar, n_azi)
        self.M = self.mu.size
        self._cell = np.arange(self.N).reshape(self.nr, self.nc, 2)
        self._act_flat = self.active.reshape(-1)
        self._prefactor()

    def _prefactor(self):
        if self.scheme == "scb":
            self._build_corners()
            self._prefactor_scb()
        else:
            self._prefactor_step()

    def _prefactor_step(self):
        """Assemble and LU-factorize L_Omega = Omega.grad + Sigma_t (upwind) for
        every ordinate and group, once."""
        h, area = self.h, self.area
        nr, nc = self.nr, self.nc
        cell = self._cell
        act = self.active
        # base diagonal per group = Sigma_t * area (collision); streaming adds to it.
        self._solvers = [[None] * self.M for _ in range(self.G)]
        for g in range(self.G):
            st_area = (self.st[g] * area).reshape(-1)
            for m in range(self.M):
                mu, eta = self.mu[m], self.eta[m]
                diag = st_area.copy()
                rows, cols, vals = [], [], []
                for (t, di, dj, tn, (nx, ny)) in _EDGES:
                    On = (mu * nx + eta * ny) * h            # (Omega.n) * edge length
                    src = cell[:, :, t]                      # (nr, nc) source cell ids
                    src_act = act[:, :, t]
                    ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
                    ni, nj = ii + di, jj + dj
                    if self.bc == "periodic":
                        ni %= nr; nj %= nc
                        inb = np.ones_like(ni, bool)
                    else:
                        inb = (ni >= 0) & (ni < nr) & (nj >= 0) & (nj < nc)
                    nbr = np.full((nr, nc), -1)
                    nbr[inb] = cell[ni[inb], nj[inb], tn]
                    nbr_act = np.zeros((nr, nc), bool)
                    nbr_act[inb] = act[ni[inb], nj[inb], tn]
                    valid = src_act & nbr_act                # interior coupled edge
                    if On > 0:                               # outflow: psi_face = psi_c
                        # leakage out across every edge of an active cell (to an
                        # active neighbour or across the vacuum/void boundary).
                        d = np.where(src_act, On, 0.0)
                        np.add.at(diag, src.reshape(-1), d.reshape(-1))
                    else:                                    # inflow: psi_face = psi_nbr
                        s = src[valid]; n = nbr[valid]
                        rows.append(s); cols.append(n)
                        vals.append(np.full(s.size, On))
                        # inflow across a vacuum/void boundary contributes 0.
                # inactive cells: identity rows (psi = 0)
                inact = ~self._act_flat
                diag = np.where(inact, 1.0, diag)
                rows.append(np.arange(self.N)); cols.append(np.arange(self.N))
                vals.append(diag)
                A = sp.csr_matrix((np.concatenate(vals),
                                   (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(self.N, self.N))
                # zero any stray couplings out of inactive rows
                self._solvers[g][m] = factorized(A.tocsc())

    def _sweep(self, g, src_flat, iface_in=None):
        """phi = Sum_m w_m L_Omega^-1 (src * area) for an isotropic source.
        iface_in (SCB only) injects a hybrid incoming flux on interface edges."""
        if self.scheme == "scb":
            return self._sweep_scb(g, src_flat, iface_in)
        rhs = src_flat * self.area
        rhs = np.where(self._act_flat, rhs, 0.0)
        phi = np.zeros(self.N)
        for m in range(self.M):
            phi += self.w[m] * self._solvers[g][m](rhs)
        return phi

    # ---- simple corner balance (SCB), second-order ------------------------
    def _build_corners(self):
        """Corner connectivity for SCB (direction-independent). Each active cell
        is split into three corner sub-volumes (one per vertex); each corner has
        two external half-edges (upwind-coupled to the neighbour cell's corner at
        the shared vertex) and two internal faces (to the cell's other corners).
        Builds, per corner, the external half-edge normals + neighbour corner ids
        and the internal face normals + same-cell corner ids."""
        s = _SQRT3_2
        # per cell type: for each edge -> (corner set, outward normal, neighbour
        # rhombus offset (di,dj,type), and this->neighbour local-corner map).
        edge_spec = {
            0: [({1, 2}, (s, 0.5), (0, 0, 1), {1: 0, 2: 1}),      # down hyp -> up(i,j)
                ({0, 1}, (0.0, -1.0), (-1, 0, 1), {0: 1, 1: 2}),  # down bot -> up(i-1,j)
                ({0, 2}, (-s, 0.5), (0, -1, 1), {0: 0, 2: 2})],   # down left -> up(i,j-1)
            1: [({0, 1}, (-s, -0.5), (0, 0, 0), {0: 1, 1: 2}),    # up hyp -> down(i,j)
                ({1, 2}, (0.0, 1.0), (1, 0, 0), {1: 0, 2: 1}),    # up top -> down(i+1,j)
                ({0, 2}, (s, -0.5), (0, 1, 0), {0: 0, 2: 2})],    # up right -> down(i,j+1)
        }
        # internal face normal from corner v toward corner w = unit(w - v).
        int_dir = {
            0: {(0, 1): (1.0, 0.0), (0, 2): (0.5, s), (1, 0): (-1.0, 0.0),
                (1, 2): (-0.5, s), (2, 0): (-0.5, -s), (2, 1): (0.5, -s)},
            1: {(0, 1): (-0.5, s), (0, 2): (0.5, s), (1, 0): (0.5, -s),
                (1, 2): (1.0, 0.0), (2, 0): (-0.5, -s), (2, 1): (-1.0, 0.0)},
        }
        ext_tab, int_tab = {}, {}
        for t in (0, 1):
            for lc in (0, 1, 2):
                ext_tab[(t, lc)] = [(nrm, off, cmap[lc]) for (cs, nrm, off, cmap)
                                    in edge_spec[t] if lc in cs]
                others = [w for w in (0, 1, 2) if w != lc]
                int_tab[(t, lc)] = [(w, int_dir[t][(lc, w)]) for w in others]

        nr, nc = self.nr, self.nc
        cell = self._cell
        act = self.active
        ac = np.where(self._act_flat)[0]
        K = ac.size
        aidx = np.full(self.N, -1)
        aidx[ac] = np.arange(K)
        ext_n = np.zeros((K, 3, 2, 2))                       # [corner-cell, lc, face, xy]
        ext_nbr = np.full((K, 3, 2), -1)                    # neighbour corner global id
        ext_cell = np.full((K, 3, 2), -1)                   # full-mesh neighbour cell id
        int_n = np.zeros((K, 3, 2, 2))
        int_w = np.zeros((K, 3, 2), int)                    # same-cell corner global id
        for r in range(K):
            c = ac[r]
            i, j, t = c // (nc * 2), (c // 2) % nc, c % 2
            for lc in range(3):
                for f, (nrm, (di, dj, tn), nbr_lc) in enumerate(ext_tab[(t, lc)]):
                    ext_n[r, lc, f] = nrm
                    ni, nj = i + di, j + dj
                    if self.bc == "periodic":
                        ni, nj = ni % nr, nj % nc
                        ok = True
                    else:
                        ok = 0 <= ni < nr and 0 <= nj < nc
                    if ok:
                        nc_cell = cell[ni, nj, tn]
                        ext_cell[r, lc, f] = nc_cell
                        if act.reshape(-1)[nc_cell]:
                            ext_nbr[r, lc, f] = aidx[nc_cell] * 3 + nbr_lc
                for f, (w, nrm) in enumerate(int_tab[(t, lc)]):
                    int_n[r, lc, f] = nrm
                    int_w[r, lc, f] = r * 3 + w
        self._scb = {"ac": ac, "K": K, "ext_n": ext_n, "ext_nbr": ext_nbr,
                     "ext_cell": ext_cell, "int_n": int_n, "int_w": int_w}

    def _prefactor_scb(self):
        """Assemble and factorize the 3*K corner system per ordinate and group."""
        d = self._scb
        K = d["K"]
        h2 = self.h / 2.0                                    # external half-edge length
        hi = self.h / (2.0 * np.sqrt(3.0))                  # internal face length
        A3 = self.area / 3.0                                 # corner volume
        row = (np.arange(K)[:, None, None] * 3 + np.arange(3)[None, :, None])
        row = np.broadcast_to(row, (K, 3, 2))
        self._solvers = [[None] * self.M for _ in range(self.G)]
        for g in range(self.G):
            st_c = self.st[g].reshape(-1)[d["ac"]]           # (K,)
            for m in range(self.M):
                oe = self.mu[m] * d["ext_n"][..., 0] + self.eta[m] * d["ext_n"][..., 1]
                oi = self.mu[m] * d["int_n"][..., 0] + self.eta[m] * d["int_n"][..., 1]
                # diagonal: collision + outflow external + internal self-share
                diag = st_c[:, None] * A3                    # (K, 3)
                diag = diag + (np.where(oe > 0, oe, 0.0) * h2).sum(2)
                diag = diag + (oi * hi * 0.5).sum(2)
                rid = (np.arange(K)[:, None] * 3 + np.arange(3)[None, :]).ravel()
                rows = [rid]; cols = [rid]; vals = [diag.ravel()]
                # external inflow -> neighbour corner (skip vacuum/void boundary)
                inflow = (oe < 0) & (d["ext_nbr"] >= 0)
                rows.append(row[inflow]); cols.append(d["ext_nbr"][inflow])
                vals.append((oe * h2)[inflow])
                # internal faces -> other-corner share
                rows.append(row.ravel()); cols.append(d["int_w"].ravel())
                vals.append((oi * hi * 0.5).ravel())
                Amat = sp.csr_matrix((np.concatenate(vals),
                                      (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(3 * K, 3 * K))
                self._solvers[g][m] = factorized(Amat.tocsc())

    def _iface_rhs(self, m, iface_in):
        """Per-ordinate corner RHS contribution from a prescribed incoming flux on
        interface half-edges (hybrid coupling): an inflow interface face moves its
        known incoming to the RHS, -(Omega.n)(h/2) psi_in. Returns (K,3)."""
        d = self._scb
        psi_in, is_iface = iface_in
        oe = self.mu[m] * d["ext_n"][..., 0] + self.eta[m] * d["ext_n"][..., 1]
        contrib = np.where(is_iface & (oe < 0), -oe * (self.h / 2.0) * psi_in, 0.0)
        return contrib.sum(2), oe

    def _sweep_scb(self, g, src_flat, iface_in=None):
        d = self._scb
        ac, K = d["ac"], d["K"]
        base = np.repeat(src_flat[ac] * (self.area / 3.0), 3)  # source into each corner
        phi = np.zeros(self.N)
        for m in range(self.M):
            rhs = base if iface_in is None else base + self._iface_rhs(m, iface_in)[0].ravel()
            psi = self._solvers[g][m](rhs).reshape(K, 3)
            phi[ac] += self.w[m] * psi.mean(1)                # cell flux = mean of corners
        return phi

    def interface_currents(self, g, cell_source, iface_in):
        """Net current (drum -> bulk, outward-normal positive) on each interface
        half-edge, given the converged within-group source per cell (scatter +
        external) and the incoming from the bulk. Returns (K, 3, 2)."""
        d = self._scb
        ac, K = d["ac"], d["K"]
        base = np.repeat(cell_source[ac] * (self.area / 3.0), 3)
        is_iface = iface_in[1]
        J = np.zeros((K, 3, 2))
        for m in range(self.M):
            add, oe = self._iface_rhs(m, iface_in)
            psi = self._solvers[g][m](base + add.ravel()).reshape(K, 3)
            # outflow face uses this corner's flux; inflow uses the incoming.
            face_flux = np.where(oe > 0, psi[:, :, None], iface_in[0])
            J += self.w[m] * np.where(is_iface, oe * (self.h / 2.0) * face_flux, 0.0)
        return J

    def _solve_group(self, g, qext_flat, phi0, tol, iface_in=None):
        """Within-group GMRES on (I - T) phi = b, T = one sweep of the scatter
        source; the boundary is vacuum/periodic (folded into L_Omega), so no
        boundary fixed point is needed. iface_in injects a hybrid incoming flux
        on interface half-edges (a fixed source, so only b carries it)."""
        ss = self.ss_self[g].reshape(-1)
        b = self._sweep(g, qext_flat, iface_in)              # source-only response

        def op(x):                                           # (I - T) x, T = scatter sweep
            return x - self._sweep(g, ss * x)

        A = LinearOperator((self.N, self.N), matvec=op, dtype=float)
        phi, _ = gmres(A, b, x0=phi0, rtol=min(tol, 1e-4), atol=0.0, maxiter=400)
        return phi

    def solve(self, tol_k: float = 1e-7, tol_source: float = 1e-6,
              max_outer: int = 500, verbose: bool = False) -> SNResult:
        t0 = time.perf_counter()
        G, N = self.G, self.N
        phi = [np.where(self._act_flat, 1.0, 0.0) for _ in range(G)]
        nsf = [self.nsf[g].reshape(-1) for g in range(G)]
        chi = [self.chi[g].reshape(-1) for g in range(G)]
        scat = [[None if self.scatter[gf][g] is None
                 else self.scatter[gf][g].reshape(-1) for g in range(G)]
                for gf in range(G)]
        fiss = sum(nsf[g] * phi[g] for g in range(G))
        n_act = float(self._act_flat.sum())
        total = fiss.sum()
        k = 1.0
        prev_rel = prev_err = 1.0
        k_hist = []
        converged = False
        outer = 0
        hist = []                                            # Anderson (fsrc_in, raw)
        for outer in range(1, max_outer + 1):
            fs = fiss / k
            tol = min(1e-3, max(0.05 * prev_rel, 0.01 * tol_k, 1e-10))
            fiss_in = fiss
            phi_new = [None] * G
            for g in range(G):
                q = chi[g] * fs
                for gf in range(G):
                    if gf != g and scat[gf][g] is not None:
                        src = phi_new[gf] if gf < g else phi[gf]
                        q = q + scat[gf][g] * src
                phi_new[g] = self._solve_group(g, q, phi[g], tol)
            fiss_new = sum(nsf[g] * phi_new[g] for g in range(G))
            total_new = fiss_new.sum()
            k_new = k * total_new / total
            dk = abs(k_new - k)
            rel = max(np.max(np.abs(phi_new[g] - phi[g])) /
                      max(np.max(np.abs(phi_new[g])), 1e-30) for g in range(G))
            phi, k = phi_new, k_new
            k_hist.append(k)
            if verbose:
                print(f"  outer {outer:3d}  k = {k:.7f}  dk = {dk:.2e}  rel = {rel:.2e}")
            if dk < tol_k and rel < tol_source and tol < max(1e-8, tol_k):
                converged = True
                break
            # Anderson-accelerate the fission source (the loosely-coupled core has
            # a dominance ratio near 1, so plain power iteration crawls).
            raw = fiss_new * (n_act / total_new)             # mean active source = 1
            if rel > 1.1 * prev_err:
                hist = []
            hist.append((fiss_in, raw))
            if len(hist) > 6:
                hist.pop(0)
            fiss = _anderson(hist)
            fiss *= n_act / fiss.sum()
            total = fiss.sum()
            prev_rel = max(rel, dk)
            prev_err = rel
        flux = np.stack([phi[g].reshape(self.grid.shape) for g in range(G)])
        return SNResult(k_eff=k, flux=flux, converged=converged,
                        outer_iterations=outer,
                        solve_seconds=time.perf_counter() - t0,
                        n_ordinates=self.M, k_history=k_hist)
