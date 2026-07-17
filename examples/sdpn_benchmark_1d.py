"""1D benchmark: diffusion vs SDP1/SDP2/SDP3 against an S16 transport reference.

A strongly heterogeneous two-group slab -- fuel assemblies separated by water
gaps, with a central gadolinia-poisoned assembly (Fuel IIg, thermal Sigma_a =
0.49 cm^-1) that carves a deep, near-discontinuous thermal-flux notch. This is
exactly the regime the simplified double-PN paper (Carreno et al., Ann. Nucl.
Energy 207, 2024) targets: steep angular-flux gradients where diffusion and even
SPN struggle.

Cross sections are from Table 1 of that paper (Rahnema & Nichita, 1997). The
reference is a fine-mesh S16 (Gauss-Legendre) discrete-ordinates k-eigenvalue
solve implemented below, so every ndgpu approximation is scored against a true
transport solution: k error (pcm), thermal-flux shape error (RMSE %), and wall
time.
"""

import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse.linalg import LinearOperator, gmres

from ndgpu import (Grid, Material, DiffusionEigenSolver, SDP1EigenSolver,
                   SDP2EigenSolver, SDP3EigenSolver)

# --- materials (Table 1) -------------------------------------------------------

def _mat(name, D, sa, nsf, s12):
    return Material(name=name, diffusion=D, sigma_a=sa, nu_sigma_f=nsf,
                    sigma_s=[[0.0, s12], [0.0, 0.0]], chi=[1.0, 0.0])

WATER   = _mat("water",    [1.7639, 0.2278], [0.0003, 0.0097], [0.0, 0.0],       0.0380)
FUEL_I  = _mat("fuel_I",   [1.4730, 0.3294], [0.0096, 0.0764], [0.0067, 0.1241], 0.0161)
FUEL_IIg = _mat("fuel_IIg", [1.5342, 0.3143], [0.0135, 0.4873], [0.0056, 0.0187], 0.0136)

# Symmetric 7-region core, gadolinia assembly in the middle. Vacuum boundaries.
REGIONS = [(WATER, 1.158), (FUEL_I, 3.231), (WATER, 1.158), (FUEL_IIg, 3.231),
           (WATER, 1.158), (FUEL_I, 3.231), (WATER, 1.158)]
EDGES = np.concatenate([[0.0], np.cumsum([t for _, t in REGIONS])])
LENGTH = float(EDGES[-1])


def _sigma_t(m):
    return np.asarray(m.sigma_t, float)


def build_cellwise(ncell):
    """Per-cell (nu_sigma_f, sigma_a, sigma_s12, sigma_t) two-group arrays."""
    h = LENGTH / ncell
    xc = (np.arange(ncell) + 0.5) * h
    reg = np.searchsorted(EDGES, xc, side="right") - 1
    mats = [m for m, _ in REGIONS]
    def field(fn):
        return np.array([fn(mats[r]) for r in reg])          # (ncell, 2)
    nsf = field(lambda m: np.asarray(m.nu_sigma_f, float))
    sa = field(lambda m: np.asarray(m.sigma_a, float))
    s12 = field(lambda m: np.array([m.sigma_s[0, 1], 0.0]))
    st = field(_sigma_t)
    return h, reg, nsf, sa, s12, st


# --- S_N discrete-ordinates reference (diamond difference, reflective BC) ------
# The Table-1 data is diffusion data (D, Sigma_a, nu Sigma_f, down-scatter). Its
# consistent transport problem -- whose isotropic/P1 limit is exactly this data,
# and which the diffusion/SDPN solvers all approximate -- takes Sigma_t = 1/(3D)
# (isotropic scattering, so no transport correction) with within-group scatter
# Sigma_s,gg = Sigma_t - Sigma_a - Sigma_out. We solve that transport problem
# with S_N discrete ordinates (Gauss-Legendre), diamond differencing, reflective
# (zero-current) boundaries, source iteration accelerated by Anderson mixing and
# an outer power iteration on fission.

def _transport_xs(ncell):
    """Per-cell Sigma_t, within-group scatter Sigma_s,gg, down-scatter, nu Sf."""
    _, reg, nsf, sa, s12, st = build_cellwise(ncell)
    out = s12.copy(); out[:, 1] = 0.0                       # only 1->2 out-scatter
    ss_self = st - sa - out                                 # within-group scatter
    return st, ss_self, s12[:, 0], nsf


def _sweep_dir(q_ang, st_g, c, binc, forward):
    """Diamond-difference transport of one direction as a bidiagonal solve
    (edge flux e_i = alpha_i e_{i-1} + beta_i), done in C by solve_banded.
    Returns cell fluxes (physical order) and the far-wall outgoing edge flux."""
    idx = slice(None) if forward else slice(None, None, -1)
    st = st_g[idx]; q = q_ang[idx]
    denom = st + c
    alpha = (c - st) / denom                 # edge attenuation per cell
    beta = 2.0 * q / denom
    n = st.size
    ab = np.zeros((2, n)); ab[0] = 1.0; ab[1, :-1] = -alpha[1:]
    b = beta.copy(); b[0] += alpha[0] * binc
    e = solve_banded((1, 0), ab, b)          # outgoing edge of each cell
    e_in = np.empty(n); e_in[0] = binc; e_in[1:] = e[:-1]
    psi = np.maximum((q + c * e_in) / denom, 0.0)
    return psi[idx], e[-1]


def _sweep(q_ang, st_g, c_pos, c_neg, mu_pos_idx, mu_neg_idx, w, ncell, bL, bR):
    """One transported sweep of an isotropic angular source q_ang; loops over the
    few half-range directions (each an O(ncell) C solve). Returns scalar flux and
    the raw outgoing boundary fluxes (mirror reflection applied by the caller)."""
    phig = np.zeros(ncell)
    right_out = np.empty(len(c_pos)); left_out = np.empty(len(c_neg))
    for j in range(len(c_pos)):              # mu > 0, left -> right
        psi, right_out[j] = _sweep_dir(q_ang, st_g, c_pos[j], bL[j], True)
        phig += w[mu_pos_idx[j]] * psi
    for j in range(len(c_neg)):              # mu < 0, right -> left
        psi, left_out[j] = _sweep_dir(q_ang, st_g, c_neg[j], bR[j], False)
        phig += w[mu_neg_idx[j]] * psi
    return phig, left_out, right_out


def _solve_group(qext, ss_self, st_g, geom, phi0, bL, bR, tol=1e-9):
    """Within-group solve for a fixed external source qext. The scattering fixed
    point is linear with spectral radius = the scattering ratio (near 1 in the
    thermal water), so plain source iteration crawls; instead we GMRES the
    scalar-flux system (I - T) phi = b with the reflective boundary frozen (T =
    one sweep's response to the scattering source), then update the boundary and
    repeat -- a fast outer fixed point on the wall fluxes."""
    c_pos, c_neg, ipos, ineg, w, ncell = geom
    half_ss = 0.5 * ss_self
    q_fixed = 0.5 * qext
    phi = phi0.copy()
    for _ in range(60):
        b0L, b0R = bL, bR                                    # frozen this pass

        def sweep_fixed(src):
            return _sweep(src + q_fixed, st_g, c_pos, c_neg, ipos, ineg, w,
                          ncell, b0L, b0R)

        b, _, _ = sweep_fixed(np.zeros(ncell))              # source-only response

        def op(x):
            p, _, _ = sweep_fixed(half_ss * x)
            return x - (p - b)

        A = LinearOperator((ncell, ncell), matvec=op, dtype=float)
        phi, _ = gmres(A, b, x0=phi, rtol=min(tol, 1e-4), atol=0.0, maxiter=400)
        np.maximum(phi, 0.0, out=phi)
        _, lo, ro = sweep_fixed(half_ss * phi)              # boundary from soln
        nbL, nbR = lo[::-1], ro[::-1]
        d = max(np.max(np.abs(nbL - bL)), np.max(np.abs(nbR - bR)))
        bL, bR = nbL, nbR
        if d < tol * max(1.0, np.max(phi)):
            break
    return phi, bL, bR


def solve_sn(ncell=2000, n_ord=16, tol=1e-9):
    st, ss_self, s12, nsf = _transport_xs(ncell)
    h = LENGTH / ncell
    mu, w = np.polynomial.legendre.leggauss(n_ord)          # sum(w) = 2
    ipos = np.where(mu > 0)[0]; ineg = np.where(mu < 0)[0]
    c_pos = 2.0 * mu[ipos] / h; c_neg = 2.0 * np.abs(mu[ineg]) / h
    npos = len(ipos)
    geom = (c_pos, c_neg, ipos, ineg, w, ncell)

    G = 2
    phi = np.ones((ncell, G))
    k = 1.0
    fiss = nsf[:, 0] * phi[:, 0] + nsf[:, 1] * phi[:, 1]
    bL = [np.zeros(npos) for _ in range(G)]
    bR = [np.zeros(npos) for _ in range(G)]
    for _ in range(500):
        fs = fiss / k                                       # chi = (1,0)
        phi_new = np.zeros_like(phi)
        for g in range(G):
            qext = s12 * phi[:, 0] if g == 1 else fs
            pg, bL[g], bR[g] = _solve_group(qext, ss_self[:, g], st[:, g], geom,
                                            phi[:, g], bL[g], bR[g])
            phi_new[:, g] = pg
        fiss_new = nsf[:, 0] * phi_new[:, 0] + nsf[:, 1] * phi_new[:, 1]
        k_new = k * fiss_new.sum() / fiss.sum()
        dk = abs(k_new - k)
        rel = np.max(np.abs(phi_new - phi)) / np.max(phi_new)
        phi, fiss, k = phi_new, fiss_new, k_new
        if dk < tol and rel < 1e-8:
            break
    return k, xc_of(ncell), phi


def xc_of(ncell):
    h = LENGTH / ncell
    return (np.arange(ncell) + 0.5) * h


# --- ndgpu diffusive solves ----------------------------------------------------

def solve_ndgpu(solver_cls, ncell):
    h, reg, *_ = build_cellwise(ncell)
    mmap = reg.reshape(ncell, 1, 1)
    grid = Grid(shape=(ncell, 1, 1), size=(LENGTH, 1.0, 1.0))
    mats = [m for m, _ in REGIONS]
    t0 = time.perf_counter()
    r = solver_cls(grid, mats, material_map=mmap, bc="reflective",
                   device="cpu").solve(tol_k=1e-9, tol_source=1e-8)
    dt = time.perf_counter() - t0
    flux = np.asarray(r.flux_numpy).reshape(2, ncell)       # (group, x)
    return r.k_eff, flux, dt, r.outer_iterations, r.inner_iterations


def resample(x_from, y_from, x_to):
    return np.interp(x_to, x_from, y_from)


def norm_shape(y):
    return y / np.trapezoid(y, dx=1.0)


def main():
    print(f"Core: {LENGTH:.2f} cm, 7 regions (gadolinia in centre), reflective BC\n")
    print("Computing S16 transport reference (1500 cells)...", flush=True)
    t0 = time.perf_counter()
    k_ref, x_ref, phi_ref = solve_sn(ncell=1500, n_ord=16)
    t_ref = time.perf_counter() - t0
    print(f"  S16  k = {k_ref:.6f}   ({t_ref:.1f} s)\n")

    ncell = 600                                             # spatially converged
    x = xc_of(ncell)
    ref_th = resample(x_ref, phi_ref[:, 1], x)             # thermal flux ref
    ref_th_n = norm_shape(ref_th)

    methods = [("diffusion", DiffusionEigenSolver), ("SDP1", SDP1EigenSolver),
               ("SDP2", SDP2EigenSolver), ("SDP3", SDP3EigenSolver)]
    hdr = f"{'method':10s}{'k':>12s}{'dk (pcm)':>11s}{'th-flux RMSE':>14s}{'time (s)':>10s}{'in-iter':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for name, cls in methods:
        k, flux, dt, no, ni = solve_ndgpu(cls, ncell)
        dk = (k - k_ref) / k_ref * 1e5
        th_n = norm_shape(flux[1])
        rmse = float(np.sqrt(np.mean(((th_n - ref_th_n) / ref_th_n.max()) ** 2))) * 100
        print(f"{name:10s}{k:12.6f}{dk:11.0f}{rmse:13.3f}%{dt:10.3f}{ni:9d}")

    print("\n(th-flux RMSE = normalized thermal-flux shape error vs S16, "
          "in % of peak; dk in pcm vs S16.)")


if __name__ == "__main__":
    main()
