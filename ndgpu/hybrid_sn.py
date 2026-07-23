"""Hybrid discrete-ordinates (S_N) / diffusion k-eigenvalue solver.

Runs full transport (S_N) only where a mask marks it -- the strongly absorbing,
steep-gradient control-drum cells -- and plain diffusion everywhere else, the
S_N counterpart of the hybrid SP3/diffusion solver (``hybrid_mask=`` on the SPN
solvers). Where SP3 is a moment method that shares diffusion's stencil (so the
hybrid was a masking), S_N has its own angular unknowns and spatial sweep, so the
two discretizations are genuinely different and must be coupled at the interface.

Coupling -- non-overlapping domain decomposition, Dirichlet--Dirichlet Schwarz
with a one-cell overlap:

  * the transport subdomain is the mask's bounding box; an S_N fixed-source solve
    runs there with the incoming angular flux on each box face reconstructed
    (isotropically) from the neighbouring diffusion scalar flux;
  * the diffusion problem is solved on the whole grid by a finite-volume stencil
    that matches ndgpu's (harmonic face D, Marshak alpha=1/2 vacuum), with the
    masked cells pinned (Dirichlet) to the transport scalar flux;
  * the two alternate to a fixed point each within-group solve, inside the
    fission power iteration.

Limits are exact by construction: an empty mask is pure diffusion (no transport
subdomain); a full mask makes every diffusion cell Dirichlet to the transport
flux, i.e. plain S_N. In between, transport self-shielding is captured in the
drums while the bulk stays diffusion.

This coupling layer is a CPU/numpy reference (the S_N subdomain solves use
``ndgpu.sn``'s wavefront sweep and DSA on their default CPU backend); it is not
the GPU production path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import factorized

from .grid import Grid
from .materials import Material
from .sn import SNResult, SNTransportSolver, diffusion_matrix


@dataclass
class HybridSNResult(SNResult):
    schwarz_iterations: int = 0


class HybridSNDiffusionSolver:
    """Hybrid S_N (masked subdomain) / diffusion k-eigenvalue solver.

    Parameters mirror :class:`~ndgpu.sn.SNTransportSolver` plus ``sn_mask``, a
    bool array of shape (nx, ny) marking the transport cells (e.g. the control
    drums). ``bc`` is the outer boundary ("vacuum" or "reflective", all four
    in-plane faces). An empty mask reduces to diffusion, a full mask to S_N.
    """

    def __init__(self, grid: Grid, materials, material_map=None, sn_mask=None,
                 n_polar: int = 3, n_azi: int = 12, bc: str = "vacuum",
                 acceleration: str = "dsa"):
        if grid.shape[2] != 1:
            raise ValueError("hybrid solver is 2D: grid must have nz == 1")
        self.grid = grid
        self.nx, self.ny = grid.shape[0], grid.shape[1]
        self.hx, self.hy = grid.spacing[0], grid.spacing[1]
        self.bc_spec = ((bc, bc), (bc, bc))
        self.bc = bc

        mats = [materials] if isinstance(materials, Material) else list(materials)
        self.G = mats[0].n_groups
        mmap = (np.zeros((self.nx, self.ny), int) if material_map is None
                else np.asarray(material_map).reshape(self.nx, self.ny))
        self.mmap = mmap

        def per_group(fn):
            table = np.array([fn(m) for m in mats])
            return np.stack([table[mmap, g] for g in range(self.G)])

        self.D = per_group(lambda m: np.asarray(m.diffusion, float))
        self.removal = per_group(lambda m: np.asarray(m.removal, float))
        self.st = per_group(lambda m: np.asarray(m.sigma_t, float))
        self.nsf = per_group(lambda m: np.asarray(m.nu_sigma_f, float))
        self.chi = per_group(lambda m: np.asarray(m.chi, float))
        sig_s = np.array([m.sigma_s for m in mats])
        self.scatter = [[None] * self.G for _ in range(self.G)]
        for gf in range(self.G):
            for gt in range(self.G):
                if gf != gt and np.any(sig_s[:, gf, gt]):
                    self.scatter[gf][gt] = sig_s[mmap, gf, gt]
        if not np.any(self.nsf):
            raise ValueError("no fissile material: k-eigenvalue is undefined")

        self.sn_mask = (np.zeros((self.nx, self.ny), bool) if sn_mask is None
                        else np.asarray(sn_mask).reshape(self.nx, self.ny))
        self.acceleration = acceleration
        self._prefactor_diffusion()
        self._setup_box(mats, n_polar, n_azi)

    def _prefactor_diffusion(self):
        """LU-factorize the diffusion operator over the bulk (transport cells
        excised) -- one factorization per group, reused every iteration."""
        self._active = ~self.sn_mask
        self._dfac = []
        for g in range(self.G):
            A = diffusion_matrix(self.D[g], self.removal[g], self.hx, self.hy,
                                 self.bc_spec, active=self._active)
            self._dfac.append(factorized(A))

    def _setup_box(self, mats, n_polar, n_azi):
        """One S_N solver per connected transport region (its bounding box). Each
        region must be rectangular (box == region) so the diffusion domain
        excises exactly the transport cells and the interface is the box faces."""
        from scipy.ndimage import label
        self.boxes = []
        if not self.sn_mask.any():
            return
        lab, nlab = label(self.sn_mask)
        for c in range(1, nlab + 1):
            ii, jj = np.where(lab == c)
            i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
            if (i1 - i0) * (j1 - j0) != ii.size:
                raise ValueError("each transport (sn_mask) region must be "
                                 "rectangular for the S_N/diffusion coupling")
            bnx, bny = i1 - i0, j1 - j0
            bgrid = Grid(shape=(bnx, bny, 1),
                         size=(bnx * self.hx, bny * self.hy, 1.0))
            bmap = self.mmap[i0:i1, j0:j1].reshape(bnx, bny, 1)
            sn = SNTransportSolver(bgrid, mats, material_map=bmap, n_polar=n_polar,
                                   n_azi=n_azi, bc="vacuum", require_fissile=False,
                                   acceleration=self.acceleration)
            self.boxes.append({"span": (i0, i1, j0, j1), "sn": sn,
                               "phi": [None] * self.G,
                               "J": [None] * self.G})       # face net currents

    def _box_incoming(self, box, g, phi_g):
        """Isotropic incoming edge fluxes for a box's faces, from the bulk
        diffusion scalar flux in the cells just outside it (0 at the domain
        edge). The net current the box exchanges with the bulk is carried by the
        interface-current source on the diffusion side; the incoming angular
        flux level is the neighbouring scalar flux."""
        i0, i1, j0, j1 = box["span"]
        sn = box["sn"]
        inc = sn._zero_inc()
        if i0 > 0:
            inc["x0"][:] = phi_g[i0 - 1, j0:j1][None, :]
        if i1 < self.nx:
            inc["x1"][:] = phi_g[i1, j0:j1][None, :]
        if j0 > 0:
            inc["y0"][:] = phi_g[i0:i1, j0 - 1][None, :]
        if j1 < self.ny:
            inc["y1"][:] = phi_g[i0:i1, j1][None, :]
        return inc

    def _box_net_currents(self, box, g, src, box_inc):
        """Net current (Sum_m w_m Omega.n psi) crossing each box face."""
        sn = box["sn"]
        _, out = sn._sweep(src, sn.st[g], box_inc)
        mu, eta, w = sn.mu, sn.eta, sn.w
        # face flux per direction = outgoing where the direction exits, incoming
        # where it enters; net current is the w*(Omega.n)-weighted sum.
        px, py = (mu > 0)[:, None], (eta > 0)[:, None]
        psi_x0 = np.where(px, box_inc["x0"], out["x0"])        # x0 normal -x: exits if mu<0
        psi_x1 = np.where(px, out["x1"], box_inc["x1"])
        psi_y0 = np.where(py, box_inc["y0"], out["y0"])
        psi_y1 = np.where(py, out["y1"], box_inc["y1"])
        wm, wme = (w * mu)[:, None], (w * eta)[:, None]
        return (np.sum(wm * psi_x0, 0), np.sum(wm * psi_x1, 0),
                np.sum(wme * psi_y0, 0), np.sum(wme * psi_y1, 0))

    def _solve_group(self, g, qext, phi_g, tol):
        """Coupled within-group solve for group g: Schwarz between each box's S_N
        transport and the shared bulk diffusion, coupled by the interface net
        current (the transport regions are excised from the diffusion domain)."""
        if not self.boxes:                                  # pure diffusion
            return self._dfac[g](qext.ravel()).reshape(self.nx, self.ny), 0
        n_schwarz = 0
        for _ in range(80):
            n_schwarz += 1
            rhs = (qext * self._active).copy()
            for box in self.boxes:
                i0, i1, j0, j1 = box["span"]
                sn = box["sn"]
                box_q = qext[i0:i1, j0:j1]
                inc = self._box_incoming(box, g, phi_g)
                bp0 = box["phi"][g]
                bp0 = phi_g[i0:i1, j0:j1].copy() if bp0 is None else bp0
                # (1) transport on the box, incoming from the current bulk flux
                box_phi, box_inc = sn._solve_group(box_q, sn.ss_self[g],
                                                   sn.st[g], bp0, inc, tol, g)
                box["phi"][g] = box_phi
                # (2) net interface current -> source on the ring of bulk cells
                src = sn.ss_self[g] * box_phi + box_q
                Jx0, Jx1, Jy0, Jy1 = self._box_net_currents(box, g, src, box_inc)
                box["J"][g] = (Jx0, Jx1, Jy0, Jy1)
                if i0 > 0:
                    rhs[i0 - 1, j0:j1] += -Jx0 / self.hx
                if i1 < self.nx:
                    rhs[i1, j0:j1] += Jx1 / self.hx
                if j0 > 0:
                    rhs[i0:i1, j0 - 1] += -Jy0 / self.hy
                if j1 < self.ny:
                    rhs[i0:i1, j1] += Jy1 / self.hy
            new = self._dfac[g](rhs.ravel()).reshape(self.nx, self.ny)
            for box in self.boxes:
                i0, i1, j0, j1 = box["span"]
                new[i0:i1, j0:j1] = box["phi"][g]            # report S_N flux in drums
            d = np.max(np.abs(new - phi_g)) / max(np.max(np.abs(new)), 1e-30)
            phi_g = new
            if d < tol:
                break
        return phi_g, n_schwarz

    def solve(self, tol_k: float = 1e-7, tol_source: float = 1e-6,
              max_outer: int = 500, verbose: bool = False) -> HybridSNResult:
        t0 = time.perf_counter()
        G, nx, ny = self.G, self.nx, self.ny
        phi = np.ones((G, nx, ny))
        fiss = sum(self.nsf[g] * phi[g] for g in range(G))
        k = 1.0
        prev_rel = 1.0
        schwarz_total = 0
        converged = False
        outer = 0
        for outer in range(1, max_outer + 1):
            fs = fiss / k
            tol = min(1e-3, max(0.05 * prev_rel, 0.01 * tol_k, 1e-10))
            phi_new = np.zeros_like(phi)
            for g in range(G):
                qext = self.chi[g] * fs
                for gf in range(G):
                    s = self.scatter[gf][g]
                    if gf != g and s is not None:
                        src = phi_new[gf] if gf < g else phi[gf]
                        qext = qext + s * src
                phi_new[g], ns = self._solve_group(g, qext, phi[g], tol)
                schwarz_total += ns
            fiss_new = sum(self.nsf[g] * phi_new[g] for g in range(G))
            k_new = k * fiss_new.sum() / fiss.sum()
            dk = abs(k_new - k)
            rel = np.max(np.abs(phi_new - phi)) / max(np.max(phi_new), 1e-30)
            phi, fiss, k = phi_new, fiss_new, k_new
            prev_rel = max(rel, dk)
            if verbose:
                print(f"  outer {outer:3d}  k = {k:.7f}  dk = {dk:.2e}  rel = {rel:.2e}")
            if dk < tol_k and rel < tol_source and tol < max(1e-8, tol_k):
                converged = True
                break
        M = self.boxes[0]["sn"].M if self.boxes else 0
        n_sweeps = sum(box["sn"]._sweep_count for box in self.boxes)
        return HybridSNResult(k_eff=k, flux=phi, converged=converged,
                              outer_iterations=outer,
                              solve_seconds=time.perf_counter() - t0,
                              n_ordinates=M, schwarz_iterations=schwarz_total,
                              n_sweeps=n_sweeps)
