"""Re-derivation of the ANL-7416 8-A1 reference codes' spatial scheme.

All three published solutions of Problem 8-A1 (TWODTA, TWODQD, ADEP; ANL-7416
Suppl. 2, pp. 187-227) use mesh-point (vertex-centered) 5-point finite
differences on the Delta_r = 8 cm x Delta_z = 18.75 cm mesh, with zero flux
at the outer boundary points. This standalone script (scipy only, never
imported by ndgpu) rebuilds that scheme by box integration around the mesh
points and reruns the benchmark transient with it. It is the evidence behind
two claims recorded in ndgpu/benchmarks/anl_bss8.py:

1. Data transcription + erratum. With region 16's fast-group D taken as
   1.2997 cm (the book prints "1.2997+1"), this scheme reproduces the
   published initial eigenvalues to ~1 pcm (0.866849 here vs ADEP's 0.866861
   and TWODTA's eigensolve 0.867053) and Exhibit A's power trace to ~3%.
   With 12.997 cm as printed, k is ~1200 pcm off every published value.

2. Discretization band. The scheme's ramp worth on the benchmark mesh is
   0.398 $, *below* the mesh-converged worth (0.4195 $, from refining
   ndgpu's cell-centered FV, which approaches it from above: 0.454 $ on the
   coarse mesh). The published excursion tail (P(4s) = 2.66-2.68) therefore
   carries the coarse-mesh vertex-scheme bias; the mesh-converged trace ends
   ~8.5% higher (P(4s) ~ 2.89).

Run:  python dev-refs/anl8a1_vertex_scheme.py
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ndgpu.benchmarks.anl_bss8 import (_LAYOUT, _RAMP, _XS, ANL8A1_KINETICS,
                                       P_REFERENCE)

DR, DZ = 8.0, 18.75
NR, NZ = 30, 28  # intervals
R = np.arange(NR + 1) * DR
Z = np.arange(NZ + 1) * DZ

# Region id per interval cell (ir, jz).
region = np.zeros((NR, NZ), dtype=int)
rc = (R[:-1] + R[1:]) / 2
zc = (Z[:-1] + Z[1:]) / 2
for reg, r0, r1, z0, z1 in _LAYOUT:
    m = (rc[:, None] >= r0) & (rc[:, None] < r1) & (zc[None, :] >= z0) & (zc[None, :] < z1)
    region[m] = reg


def xs_arrays(t):
    """Two-group XS arrays indexed by region id (1..16) at time t."""
    fac = {reg: 1.0 + rate * min(t, 1.0) for reg, rate in _RAMP.items()}
    D = np.zeros((17, 2)); rem = np.zeros((17, 2)); nsf = np.zeros((17, 2)); s12 = np.zeros(17)
    for reg, (D1, D2, s1, s2, nf1, nf2, s12_) in _XS.items():
        f = fac.get(reg, 1.0)
        D[reg] = (D1, D2)
        rem[reg] = (s1, s2 * f)     # Sigma_1 already includes the downscatter
        nsf[reg] = (nf1, nf2)
        s12[reg] = s12_
    return D, rem, nsf, s12


# Unknowns: mesh points with i = 0..NR-1 (the r = 0 axis point is kept -- the
# axis is a natural boundary), j = 1..NZ-1; all other boundary points are
# zero-flux Dirichlet.
II = np.arange(0, NR)
JJ = np.arange(1, NZ)
idx = {(i, j): n for n, (i, j) in enumerate((i, j) for j in JJ for i in II)}
N = len(idx)


def quadrants(i, j):
    """The up-to-four interval cells around point (i, j), with the radial
    moment integral r dr over the half-interval and the axial half-height
    (box integration weights in cylindrical geometry)."""
    out = []
    for di in (0, 1):
        ic = i - di
        if ic < 0 or ic >= NR:
            continue
        ra, rb = (R[i], R[i] + DR / 2) if di == 0 else (R[i] - DR / 2, R[i])
        wr = (rb ** 2 - ra ** 2) / 2.0
        for dj in (0, 1):
            jc = j - dj
            if 0 <= jc < NZ:
                out.append((ic, jc, wr, DZ / 2))
    return out


def build(t):
    """Volume-integrated two-group operators at time t: per-group
    leakage+removal matrices, downscatter and fission point fields."""
    D, rem, nsf, s12t = xs_arrays(t)
    A = [sp.lil_matrix((N, N)) for _ in range(2)]
    S12 = np.zeros(N)
    F = np.zeros((2, N))
    VOL = np.zeros(N)
    for (i, j), n in idx.items():
        for ic, jc, wr, hz in quadrants(i, j):
            r_ = region[ic, jc]
            v = wr * hz
            VOL[n] += v
            for g in range(2):
                A[g][n, n] += rem[r_, g] * v
                F[g, n] += nsf[r_, g] * v
            S12[n] += s12t[r_] * v
        # radial coupling to (i+1, j): face at r_{i+1/2}, split axially.
        rf = R[i] + DR / 2
        for g in range(2):
            w = sum(D[region[i, jc], g] * rf * (DZ / 2) / DR
                    for jc in (j - 1, j) if 0 <= jc < NZ)
            if (i + 1, j) in idx:
                m = idx[(i + 1, j)]
                A[g][n, n] += w; A[g][m, m] += w
                A[g][n, m] -= w; A[g][m, n] -= w
            elif i + 1 == NR:          # Dirichlet neighbour
                A[g][n, n] += w
        # axial couplings to (i, j +/- 1): face split radially.
        rlo, rhi = max(R[i] - DR / 2, 0.0), min(R[i] + DR / 2, R[-1])
        halves = [(i, (R[i], rhi)), (i - 1, (rlo, R[i]))]
        for jn, jc in (((i, j + 1), j), ((i, j - 1), j - 1)):
            for g in range(2):
                w = sum(D[region[ic, jc], g] * (rb ** 2 - ra ** 2) / 2.0 / DZ
                        for ic, (ra, rb) in halves if 0 <= ic < NR)
                if jn in idx:
                    m = idx[jn]
                    if jn > (i, j):    # add each interior pair once
                        A[g][n, n] += w; A[g][m, m] += w
                        A[g][n, m] -= w; A[g][m, n] -= w
                elif jn[1] in (0, NZ):  # Dirichlet neighbour
                    A[g][n, n] += w
    return [a.tocsr() for a in A], S12, F, VOL


def solve_k(t, tol=1e-10):
    A, S12, F, VOL = build(t)
    lu = [spla.splu(a.tocsc()) for a in A]
    phi = [np.ones(N), np.ones(N)]
    k = 1.0
    src = F[0] * phi[0] + F[1] * phi[1]
    for _ in range(2000):
        phi[0] = lu[0].solve(src / k)
        phi[1] = lu[1].solve(S12 * phi[0])
        new = F[0] * phi[0] + F[1] * phi[1]
        knew = k * new.sum() / src.sum()
        err = abs(knew - k)
        k, src = knew, new
        if err < tol:
            break
    return k, phi, (A, S12, F, VOL)


def transient(k0, phi, dt=0.01, t_end=4.0):
    """Backward-Euler march of the 8-A1 transient (mirrors ndgpu's scheme)."""
    kin = ANL8A1_KINETICS
    beta, lam, v = kin.beta, kin.decay, kin.velocities
    _, _, (A, S12, F, VOL) = solve_k(0.0)
    S = (F[0] * phi[0] + F[1] * phi[1]) / k0
    scale = 1.0 / S.sum()
    phi = [phi[0] * scale, phi[1] * scale]
    S = S * scale
    C = [(beta[i] / lam[i]) * S for i in range(6)]
    omega = float(np.sum(lam * dt * beta / (1 + lam * dt)))
    fis_w = (1 - beta.sum()) + omega
    mass = [VOL / (v[g] * dt) for g in range(2)]
    lu, tprev, out = None, None, {}
    for n in range(1, int(round(t_end / dt)) + 1):
        t = n * dt
        tc = min(t, 1.0)
        if tc != tprev:
            A, S12, F, VOL = build(t)
            lu = [spla.splu((A[g] + sp.diags(mass[g])).tocsc()) for g in range(2)]
            tprev = tc
        dsrc = sum((lam[i] / (1 + lam[i] * dt)) * C[i] for i in range(6))
        for _ in range(200):
            phi0 = lu[0].solve(mass[0] * phi[0] + fis_w * S + dsrc)  # chi = [1, 0]
            phi1 = lu[1].solve(mass[1] * phi[1] + S12 * phi0)
            Sn = (F[0] * phi0 + F[1] * phi1) / k0
            change = np.linalg.norm(Sn - S) / np.linalg.norm(Sn)
            phi, S = [phi0, phi1], Sn
            if change < 1e-8:
                break
        C = [(C[i] + dt * beta[i] * S) / (1 + lam[i] * dt) for i in range(6)]
        if round(t, 10) in P_REFERENCE:
            out[round(t, 10)] = S.sum()
    return out


if __name__ == "__main__":
    k0, phi0, _ = solve_k(0.0)
    k1, _, _ = solve_k(1.0)
    rho = 1 - k0 / k1
    print(f"vertex-centered 30x28: k0 = {k0:.6f}"
          "  (book: 0.867053 eigensolve / 0.866861 ADEP)")
    print(f"frozen-ramp worth: rho = {rho:.6f} = {rho / 6.499e-3:.4f} $"
          "  (mesh-converged FV: 0.4195 $)")
    print("\ntransient vs Exhibit A (dt = 0.01):")
    out = transient(k0, phi0)
    for t, pref in P_REFERENCE.items():
        print(f"  t={t:3.1f}  P={out[t]:.3f}  book={pref:.3f}"
              f"  diff={100 * (out[t] / pref - 1):+.1f}%")
