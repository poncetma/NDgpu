"""Hybrid discrete-ordinates (S_N) / diffusion on the HP-MR triangular mesh.

Runs full transport (S_N with the second-order SCB scheme) only in the
control-drum cells and diffusion in the bulk, on the body-fitted hex/triangular
core -- the triangular counterpart of the Cartesian ``HybridSNDiffusionSolver``
and the payoff of the tri-S_N work. The two regions are coupled by the interface
net current (the drums are excised from the diffusion domain; each drum's
outgoing current is a source on the adjacent bulk cell), the same coupling the
Cartesian version established (a Dirichlet-flux coupling double-counts the drum
absorption).

Both ingredients are validated on their own: ``ndgpu.tri`` diffusion with active
masking, and ``TriSNTransportSolver`` (SCB). This solver adds only the interface
coupling: an S_N drum solve with the incoming flux reconstructed from the
neighbouring bulk scalar flux, its outgoing interface current fed back as a
source to a bulk finite-volume diffusion solve, alternated to a Schwarz fixed
point inside the fission power iteration. Limits are exact: an empty drum mask is
the tri diffusion solver, a full mask is tri-S_N. CPU/numpy reference.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, factorized, gmres

from .materials import Material
from .sn import SNResult, _anderson
from .solver import Fields
from .stencil import face_alpha, harmonic_mean
from .tri import TriGrid
from .tri_sn import _EDGES, TriSNTransportSolver


def _tri_diffusion_matrix(grid, D, removal, bulk, active_full, sn_mask, mask_bc):
    """Sparse triangular finite-volume diffusion operator over the bulk (the
    drums and the void are excised). Matches ndgpu.tri.TriGroupOperator: interior
    bulk-bulk faces use harmonic-mean D with coupling 4 D / h^2; a bulk face onto
    the void (core surface, or off-mesh) carries the Robin ``mask_bc`` term; a
    bulk face onto a drum carries nothing -- the transport interface current is
    added as a source instead. Non-bulk cells get a unit diagonal."""
    nr, nc = grid.shape[0], grid.shape[1]
    N = nr * nc * 2
    h = grid.side
    kf = 4.0 / (h * h)
    s3 = np.sqrt(3.0)
    alpha = face_alpha(mask_bc)
    cell = np.arange(N).reshape(nr, nc, 2)
    Dv = D.reshape(-1)
    diag = removal.reshape(-1).astype(float).copy()
    bulk3 = bulk
    rows, cols, vals = [], [], []
    ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    for k in range(6):
        t, di, dj, tn, _ = _EDGES[k]
        src = cell[:, :, t]
        src_bulk = bulk3[:, :, t]
        ni, nj = ii + di, jj + dj
        inb = (ni >= 0) & (ni < nr) & (nj >= 0) & (nj < nc)
        nbr = np.where(inb, cell[np.clip(ni, 0, nr - 1), np.clip(nj, 0, nc - 1), tn], -1)
        nbr_bulk = np.zeros((nr, nc), bool)
        nbr_drum = np.zeros((nr, nc), bool)
        nbr_bulk[inb] = bulk3[ni[inb], nj[inb], tn]
        nbr_drum[inb] = sn_mask[ni[inb], nj[inb], tn]
        # bulk-bulk interior face: harmonic coupling
        bb = src_bulk & nbr_bulk
        if bb.any():
            w = harmonic_mean(Dv[src[bb]], Dv[nbr[bb]]) * kf
            rows.append(src[bb]); cols.append(nbr[bb]); vals.append(-w)
            np.add.at(diag, src[bb], w)
        # bulk face onto the void / off-mesh (not a drum): Robin mask_bc
        void_face = src_bulk & ~nbr_bulk & ~nbr_drum
        if alpha != 0.0 and void_face.any():
            Dc = Dv[src[void_face]]
            if np.isinf(alpha):
                term = 8.0 * Dc / (h * h)
            else:
                term = 8.0 * Dc * alpha / (h * (h * alpha + 2.0 * s3 * Dc))
            np.add.at(diag, src[void_face], term)
        # bulk face onto a drum: excised (interface current handled as a source)
    diag = np.where(bulk3.reshape(-1), diag, 1.0)
    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    A = sp.csr_matrix((np.concatenate([np.atleast_1d(v) for v in vals]),
                       (np.concatenate([np.atleast_1d(r) for r in rows]),
                        np.concatenate([np.atleast_1d(c) for c in cols]))),
                      shape=(N, N))
    return A.tocsc()


class HybridTriSNDiffusionSolver:
    """Hybrid S_N (drum cells) / diffusion (bulk) k-eigenvalue solver on a
    triangular mesh.

    Parameters mirror the tri solvers plus ``sn_mask`` (bool, grid.shape) marking
    the transport cells. An empty mask reduces to tri diffusion, a full one to
    tri-S_N.
    """

    def __init__(self, grid: TriGrid, materials, material_map=None, sn_mask=None,
                 active=None, mask_bc="vacuum", n_polar: int = 3, n_azi: int = 12,
                 mix_material=None, mix_weight=None,
                 acceleration: str = "dsa-gmres", coupling: str = "krylov"):
        # acceleration applies to the coupling="schwarz" path only (the nested
        # drum-box within-group solves); default "dsa-gmres" there because the
        # Schwarz loop re-solves each warm-started box against a changing
        # interface source, which suits a DSA-preconditioned Krylov far better
        # than DSA source iteration (HP-MR refine=4: 15 s vs 50 s; plain gmres
        # 25 s). coupling="krylov" (default) has no nested solves at all.
        if coupling not in ("krylov", "schwarz"):
            raise ValueError("coupling must be 'krylov' or 'schwarz'")
        self.coupling = coupling
        self.grid = grid
        self.nr, self.nc = grid.shape[0], grid.shape[1]
        self.N = self.nr * self.nc * 2
        self.area = (np.sqrt(3.0) / 4.0) * grid.side ** 2
        mats = [materials] if isinstance(materials, Material) else list(materials)
        self.G = mats[0].n_groups
        mmap = (np.zeros(grid.shape, int) if material_map is None
                else np.asarray(material_map).reshape(grid.shape))
        self.active = (np.ones(grid.shape, bool) if active is None
                       else np.asarray(active).reshape(grid.shape))
        self.sn_mask = (np.zeros(grid.shape, bool) if sn_mask is None
                        else np.asarray(sn_mask).reshape(grid.shape)) & self.active
        self.bulk = self.active & ~self.sn_mask

        # Per-cell fields via the validated Fields blend so the polar B4C
        # volume-mixing (mix_material/mix_weight) applies to the bulk diffusion
        # exactly as in TriDiffusionEigenSolver (and to the drum S_N below).
        f = Fields(np, grid, mats, mmap, np.float64,
                   mix_material=mix_material, mix_weight=mix_weight)
        self.D = np.stack(f.diffusion)
        self.removal = np.stack(f.removal)
        self.nsf = np.stack(f.nu_sigma_f)
        self.chi = np.stack(f.chi)
        self.scatter = [[None] * self.G for _ in range(self.G)]
        for gf in range(self.G):
            for gt in range(self.G):
                if gf != gt and f.sigma_s[gf][gt] is not None:
                    self.scatter[gf][gt] = np.asarray(f.sigma_s[gf][gt]).reshape(-1)
        if not np.any(self.nsf):
            raise ValueError("no fissile material: k-eigenvalue is undefined")

        # bulk diffusion, factorized once per group
        self._dfac = [factorized(_tri_diffusion_matrix(
            grid, self.D[g], self.removal[g], self.bulk, self.active,
            self.sn_mask, mask_bc)) for g in range(self.G)]

        # drum S_N (SCB) and the interface map. coupling="krylov" solves the
        # coupled within-group system monolithically (one fused drum sweep +
        # one bulk diffusion backsolve per GMRES matvec, drum-DSA
        # preconditioned); "schwarz" is the original alternating fixed point.
        self._has_drum = bool(self.sn_mask.any())
        if self._has_drum:
            self.sn = TriSNTransportSolver(grid, mats, material_map=mmap,
                                           active=self.sn_mask, n_polar=n_polar,
                                           n_azi=n_azi, bc="vacuum", scheme="scb",
                                           require_fissile=False,
                                           mix_material=mix_material,
                                           mix_weight=mix_weight,
                                           acceleration=acceleration)
            d = self.sn._scb
            bulk_flat = self.bulk.reshape(-1)
            ec = d["ext_cell"]                               # (K,3,2) full-mesh nbr cell
            safe = np.clip(ec, 0, self.N - 1)
            self._is_iface = (d["ext_nbr"] < 0) & (ec >= 0) & bulk_flat[safe]
            self._iface_cell = np.where(self._is_iface, ec, 0)  # bulk cell id per iface face
            self._ac = d["ac"]

    def _bulk_incoming(self, phi_bulk):
        """Per interface half-edge, the isotropic incoming flux = neighbouring
        bulk scalar flux (0 on non-interface boundary faces)."""
        return np.where(self._is_iface, phi_bulk[self._iface_cell], 0.0)

    def _apply_step(self, g, phi, qext):
        """One coupled interface step, jointly affine in (phi, qext): a single
        fused drum sweep (incoming reconstructed from the bulk flux, interface
        half-edge currents accumulated from the same per-ordinate solves)
        followed by one bulk diffusion backsolve with those currents as
        sources. Its fixed point is exactly the Schwarz limit."""
        ss = self.sn.ss_self[g].reshape(-1)
        drum_flat = self.sn_mask.reshape(-1)
        iface_in = (self._bulk_incoming(phi), self._is_iface)
        src = ss * np.where(drum_flat, phi, 0.0) + qext
        drum_phi, J = self.sn._sweep_iface(g, src, iface_in)
        bulk_src = qext * self.bulk.reshape(-1)
        np.add.at(bulk_src, self._iface_cell[self._is_iface],
                  J[self._is_iface] / self.area)
        out = self._dfac[g](bulk_src).reshape(self.N)
        out[drum_flat] = drum_phi[drum_flat]
        return out

    def _solve_group_krylov(self, g, qext, phi, tol):
        """Monolithic within-group solve: GMRES on the fixed point of
        ``_apply_step``, (I - L) phi = c -- one fused drum sweep + one bulk
        diffusion backsolve per matvec, no nested drum solves. Left-
        preconditioned with the drum DSA operator (the interface error
        equation is zero-incoming on the drum faces, so the drum solver's
        vacuum-Robin diffusion LU is the right correction)."""
        ss = self.sn.ss_self[g].reshape(-1)
        drum_flat = self.sn_mask.reshape(-1)
        N = self.N
        c = self._apply_step(g, np.zeros(N), qext)
        zero = np.zeros(N)

        def op(x):
            return x - self._apply_step(g, x, zero)

        fac = self.sn._dsa_factor(g)

        def prec(x):
            return x + fac(ss * np.where(drum_flat, x, 0.0))

        A = LinearOperator((N, N), matvec=op, dtype=float)
        M = LinearOperator((N, N), matvec=prec, dtype=float)
        phi_v, _ = gmres(A, c, x0=phi, M=M, rtol=min(tol, 1e-4), atol=0.0,
                         maxiter=400)
        return phi_v

    def _solve_group(self, g, qext, phi, tol):
        """Coupled within-group solve: the drum S_N (incoming from the bulk)
        against the bulk diffusion (drum interface current as a source) -- the
        monolithic Krylov solve (coupling="krylov") or the alternating Schwarz
        fixed point (coupling="schwarz")."""
        if not self._has_drum:
            return self._dfac[g](qext).reshape(self.N)
        if self.coupling == "krylov":
            return self._solve_group_krylov(g, qext, phi, tol)
        ss = self.sn.ss_self[g].reshape(-1)
        drum_flat = self.sn_mask.reshape(-1)
        bulk_flat = self.bulk.reshape(-1)

        def step(phi_in):
            # (1) transport on the drum, incoming from the current bulk flux
            iface_in = (self._bulk_incoming(phi_in), self._is_iface)
            drum_phi = self.sn._solve_group(g, qext, phi_in, tol, iface_in=iface_in)
            # (2) net interface current -> source on the adjacent bulk cells
            src = ss * drum_phi + qext                        # full within-group source
            J = self.sn.interface_currents(g, src, iface_in)  # (K,3,2)
            bulk_src = qext * bulk_flat
            np.add.at(bulk_src, self._iface_cell[self._is_iface],
                      J[self._is_iface] / self.area)
            out = self._dfac[g](bulk_src).reshape(self.N)
            out[drum_flat] = drum_phi[drum_flat]
            return out

        # The Schwarz fixed point converges only at the interface coupling rate
        # (~0.6 per sweep on the HP-MR drums), so Anderson-accelerate it.
        hist = []
        for _ in range(60):
            new = step(phi)
            d = np.max(np.abs(new - phi)) / max(np.max(np.abs(new)), 1e-30)
            hist.append((phi, new))
            if len(hist) > 5:
                hist.pop(0)
            phi = _anderson(hist)
            if not np.all(np.isfinite(phi)):
                phi = new                                     # fall back on breakdown
                break
            if d < tol:
                break
        return phi

    def solve(self, tol_k: float = 1e-7, tol_source: float = 1e-6,
              max_outer: int = 500, verbose: bool = False) -> SNResult:
        t0 = time.perf_counter()
        G, N = self.G, self.N
        act = self.active.reshape(-1)
        phi = [np.where(act, 1.0, 0.0) for _ in range(G)]
        nsf = [self.nsf[g].reshape(-1) for g in range(G)]
        chi = [self.chi[g].reshape(-1) for g in range(G)]
        fiss = sum(nsf[g] * phi[g] for g in range(G))
        n_act = float(act.sum())
        total = fiss.sum()
        k = 1.0
        prev = prev_err = 1.0
        converged = False
        outer = 0
        hist = []                                            # Anderson (fsrc_in, raw)
        for outer in range(1, max_outer + 1):
            fs = fiss / k
            tol = min(1e-3, max(0.05 * prev, 0.01 * tol_k, 1e-10))
            fiss_in = fiss
            phi_new = [None] * G
            for g in range(G):
                qext = chi[g] * fs
                for gf in range(G):
                    if gf != g and self.scatter[gf][g] is not None:
                        src = phi_new[gf] if gf < g else phi[gf]
                        qext = qext + self.scatter[gf][g] * src
                phi_new[g] = self._solve_group(g, qext, phi[g], tol)
            fiss_new = sum(nsf[g] * phi_new[g] for g in range(G))
            total_new = fiss_new.sum()
            k_new = k * total_new / total
            dk = abs(k_new - k)
            rel = max(np.max(np.abs(phi_new[g] - phi[g])) /
                      max(np.max(np.abs(phi_new[g])), 1e-30) for g in range(G))
            phi, k = phi_new, k_new
            if verbose:
                print(f"  outer {outer:3d}  k = {k:.7f}  dk = {dk:.2e}  rel = {rel:.2e}")
            if dk < tol_k and rel < tol_source and tol < max(1e-8, tol_k):
                converged = True
                break
            # Anderson-accelerate the fission source (dominance ratio near 1).
            raw = fiss_new * (n_act / total_new)
            if rel > 1.1 * prev_err:
                hist = []
            hist.append((fiss_in, raw))
            if len(hist) > 6:
                hist.pop(0)
            fiss = _anderson(hist)
            fiss *= n_act / fiss.sum()
            total = fiss.sum()
            prev, prev_err = max(rel, dk), rel
        flux = np.stack([phi[g].reshape(self.grid.shape) for g in range(G)])
        M = self.sn.M if self._has_drum else 0
        return SNResult(k_eff=k, flux=flux, converged=converged,
                        outer_iterations=outer,
                        solve_seconds=time.perf_counter() - t0, n_ordinates=M)
