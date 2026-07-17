"""Pin the SDPN coefficient tables to first principles.

The SDPN c^(m)/g tables cannot be validated by internal-consistency tests (k_inf,
dense eigensolve) -- those pass for *any* tables. Both real bugs found so far
were coefficient errors: the paper's SDP2 c^(1) sign typo (A.3), and the paper's
SDP3 c rows 2-3 (wrong in the paper AND in the authors' FEMFFUSION code; ndgpu
carries independently re-derived values, see _SDPN_C in operator.py).

Two tiers:
  * fast structural invariants -- rank-1 c^(1), and the signature entries that
    distinguish the corrected tables from the paper's printed ones;
  * a full from-scratch sympy re-derivation (marked slow) following the paper's
    own recipe: half-range moments of the slab PN system (Eq. 34), odd-moment
    elimination, U-transform (Eq. 13) with D = diag(1/(4j-1)) Sigma^-1, and the
    Marshak condition (Eq. 44). The machinery reproduces the paper's SDP1/SDP2
    tables and every Marshak g exactly, so a mismatch at SDP3 is a paper error,
    not a derivation-convention difference.
"""

import numpy as np
import pytest

from ndgpu.operator import _SDPN_C, _SDPN_G

sympy = pytest.importorskip("sympy")
sp = sympy
R = sp.Rational


# ------------------------------------------------------------- fast invariants

@pytest.mark.parametrize("N", [1, 2, 3])
def test_c1_is_rank_one(N):
    """Fission depends only on phi0, so c^(1) = src (x) (V^-1 row 0)."""
    c1 = np.array(_SDPN_C[N][0], dtype=float)
    outer = np.outer(c1[:, 0], c1[0] / c1[0, 0])
    assert np.allclose(c1, outer, atol=1e-14)


def test_signature_entries_of_corrected_tables():
    """The entries where the corrected tables differ from the paper's printed
    appendix -- guards against 'fixing them back' to match the paper."""
    # SDP2 (A.3 typo): closure source weight is +32/33, not -32/33.
    assert _SDPN_C[2][0][2][0] == pytest.approx(32.0 / 33.0)
    # SDP3 (A.6 rows 2-3 wrong in paper and FEMFFUSION): true closure weights.
    assert _SDPN_C[3][0][2][0] == pytest.approx(200.0 / 429.0)   # not 8/15
    assert _SDPN_C[3][0][3][0] == pytest.approx(-16.0 / 13.0)    # not -176/125
    # True SDP3 c^(4) has a nonzero (2,3) coupling the paper's pattern lacks.
    assert _SDPN_C[3][3][2][3] == pytest.approx(3.0 / 77.0)
    assert _SDPN_C[3][3][3][3] == pytest.approx(5.0 / 7.0)       # not 143/175


# ------------------------------------------------- full re-derivation (sympy)

def _derive(N):
    """Re-derive (c^(m) list, g) for SDPN order N from the paper's Eq. (34)."""
    x = sp.Symbol('x')
    M, L = N + 1, 2 * N + 1
    C = {(n, l): sp.integrate(sp.legendre(n, 2 * x - 1) * sp.legendre(l, x),
                              (x, 0, 1))
         for n in range(M) for l in range(L + 2)}

    def Cpm(n, l, s):
        return C[(n, l)] if s > 0 else (-1) ** (n + l) * C[(n, l)]

    dphi = [sp.Symbol(f'dphi{l}') for l in range(L + 1)]
    phi = [sp.Symbol(f'phi{l}') for l in range(L + 1)]
    Sig = [sp.Symbol(f'S{l}') for l in range(L + 1)]
    q = sp.Symbol('q')

    def eq(n, s):
        e = -R(1, 2) * Cpm(n, 0, s) * q
        for l in range(L + 1):
            dc = R(1, 2) * ((l + 1) * Cpm(n, l + 1, s)
                            + (l * Cpm(n, l - 1, s) if l > 0 else 0))
            e += dc * dphi[l] + R(2 * l + 1, 2) * Cpm(n, l, s) * Sig[l] * phi[l]
        return sp.expand(e)

    sums = [sp.expand(eq(n, 1) + eq(n, -1)) for n in range(M)]
    difs = [sp.expand(eq(n, 1) - eq(n, -1)) for n in range(M)]
    even_eqs = [sums[n] if n % 2 == 0 else difs[n] for n in range(M)]
    odd_eqs = [difs[n] if n % 2 == 0 else sums[n] for n in range(M)]

    # odd-moment closure y_j = Sigma_{2j+1} phi_{2j+1} = -K_j . grad(phi_even)
    y = [sp.Symbol(f'y{j}') for j in range(M)]
    oe = [sp.expand(e.subs({phi[2 * j + 1]: y[j] / Sig[2 * j + 1]
                            for j in range(M)})) for e in odd_eqs]
    sol = sp.solve(oe, y, dict=True)[0]
    K = sp.Matrix(M, M, lambda j, i: -sp.expand(sol[y[j]]).coeff(dphi[2 * i]))

    V = sp.zeros(M, M)
    for m in range(1, M):
        V[m - 1, m - 1] = 2 * m - 1
        V[m - 1, m] = 2 * m
    V[M - 1, M - 1] = 2 * M - 1
    Vinv = V.inv()

    # closure rows are proportional to U rows: K V^-1 = diag(kappa), kappa the
    # paper's diffusion constants 1/(4j+3)
    KV = K * Vinv
    for j in range(M):
        for m in range(M):
            expected = R(1, 4 * (j + 1) - 1) if m == j else 0
            assert sp.simplify(KV[j, m] - expected) == 0
    kappa = [KV[j, j] for j in range(M)]

    Ep = sp.Matrix(M, M, lambda n, j: even_eqs[n].coeff(dphi[2 * j + 1]))
    Rm = sp.Matrix(M, M, lambda n, l: sp.simplify(
        even_eqs[n].coeff(phi[2 * l]) / Sig[2 * l]))
    s = sp.Matrix(M, 1, lambda n, _: -even_eqs[n].coeff(q))

    W = sp.Matrix(M, M, lambda n, j: Ep[n, j] * kappa[j])
    T = sp.diag(*[R(1, 4 * (j + 1) - 1) for j in range(M)]) * W.inv()
    TR = T * Rm
    c = [sp.Matrix(M, M, lambda i, j: TR[i, m] * Vinv[m, j]) for m in range(M)]

    # Marshak: P phi_even + Q phi_odd = 0, phi_odd = -(D dU/dx) -> g = Q^-1 P V^-1
    P = sp.Matrix(M, M, lambda n, i: R(4 * i + 1, 2) * C[(n, 2 * i)])
    Q = sp.Matrix(M, M, lambda n, j: R(4 * j + 3, 2) * C[(n, 2 * j + 1)])
    g = Q.inv() * P * Vinv
    return c, g


@pytest.mark.slow
@pytest.mark.parametrize("N", [1, 2, 3])
def test_tables_match_first_principles(N):
    c, g = _derive(N)
    M = N + 1
    for m in range(M):
        derived = np.array(c[m].tolist(), dtype=float)
        assert np.allclose(derived, np.array(_SDPN_C[N][m], dtype=float),
                           atol=1e-14), f"SDP{N} c^({m+1}) mismatch"
    derived_g = np.array(g.tolist(), dtype=float)
    assert np.allclose(derived_g, np.array(_SDPN_G[N], dtype=float),
                       atol=1e-14), f"SDP{N} Marshak g mismatch"
