"""Real spherical harmonics and the moment/discrete operators for anisotropic
scattering in 2D XY S_N.

The P_L scattering kernel expands as

    Sigma_s(Omega' -> Omega) = sum_l (2l+1)/(4pi) Sigma_s,l P_l(Omega'.Omega)

and the Legendre addition theorem turns that into a sum over real harmonics, so
the scattering source needs only the flux *moments*

    phi_lm = sum_n w_n R_lm(Omega_n) psi_n            (discrete-to-moment)
    q(Omega_n) = sum_lm (2l+1) Sigma_s,l phi_lm R_lm(Omega_n)   (moment-to-discrete)

matching OpenSN's ``d2m = w Ylm`` / ``m2d = ((2l+1)/sum(w)) Ylm`` with ndgpu's
weights normalized to 1.

Normalization here is the transport convention, *not* the quantum-mechanical
one: R_00 = 1, and R_1m are the direction cosines, so that

    sum_n w_n R_lm R_l'm' = delta_ll' delta_mm' / (2l+1)

and the l = 0 term reproduces the isotropic source Sigma_s,0 * phi exactly --
which is what makes anisotropic support a strict superset of the current
isotropic path rather than a reinterpretation of it.

2D XY keeps only the harmonics even in Omega_z (m = -l, -l+2, ..., l, i.e.
l + m even), because the geometry is invariant under Omega_z -> -Omega_z. That
is exactly OpenSN's 2D rule, and it is also what lets ndgpu's quadrature fold
the two hemispheres: an Omega_z-even harmonic takes the same value at +-Omega_z,
so the folded set integrates it correctly.
"""

from __future__ import annotations

import math

import numpy as np


def moment_indices_2d(order: int):
    """(l, m) pairs for 2D XY up to scattering order L: m = -l, -l+2, ..., l."""
    if order < 0:
        raise ValueError(f"scattering order must be >= 0, got {order}")
    return [(l, m) for l in range(order + 1) for m in range(-l, l + 1, 2)]


def _assoc_legendre(l, m, x):
    """P_l^m(x) WITHOUT the Condon-Shortley phase (transport convention)."""
    m = abs(m)
    if m > l:
        return np.zeros_like(x)
    # P_m^m = (2m-1)!! (1-x^2)^{m/2}
    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        fact = 1.0
        for _ in range(m):
            pmm = pmm * fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = np.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm, pmmp1 = pmmp1, pll
    return pll


def real_harmonic(l, m, mu, eta, xi):
    """R_lm(Omega) in the transport normalization (R_00 = 1, R_1m = cosines).

    mu, eta, xi are the x, y, z direction cosines.
    """
    absm = abs(m)
    norm = math.sqrt((2.0 - (1.0 if m == 0 else 0.0))
                     * math.factorial(l - absm) / math.factorial(l + absm))
    plm = _assoc_legendre(l, absm, xi)
    phi = np.arctan2(eta, mu)
    trig = np.cos(absm * phi) if m >= 0 else np.sin(absm * phi)
    return norm * plm * trig


def harmonic_matrix(mu, eta, order, xi=None):
    """R of shape (n_moments, M): R[k, n] = R_{l_k m_k}(Omega_n).

    xi defaults to the positive branch sqrt(1 - mu^2 - eta^2), which is the
    hemisphere ndgpu's 2D quadrature retains; only Omega_z-even harmonics are
    used, so the sign of xi does not matter.
    """
    mu = np.asarray(mu, float)
    eta = np.asarray(eta, float)
    if xi is None:
        xi = np.sqrt(np.maximum(1.0 - mu * mu - eta * eta, 0.0))
    idx = moment_indices_2d(order)
    return np.array([real_harmonic(l, m, mu, eta, xi) for l, m in idx]), idx


def d2m_m2d(mu, eta, w, order, xi=None):
    """Discrete-to-moment and moment-to-discrete operators.

    Returns (D2M, M2D, indices) with

        D2M (n_moments, M)  phi_lm  = D2M @ psi
        M2D (M, n_moments)  q_n     = M2D @ (Sigma_s,l * phi_lm)

    so that ``M2D @ (sigma_l * (D2M @ psi))`` is the full anisotropic
    scattering source, and at order 0 it collapses to ``sigma_0 * phi``.
    """
    R, idx = harmonic_matrix(mu, eta, order, xi)
    w = np.asarray(w, float)
    wsum = w.sum()
    d2m = R * w[None, :]
    m2d = np.array([(2.0 * l + 1.0) / wsum for l, _ in idx])[None, :] * R.T
    return d2m, m2d, idx


def check_orthogonality(mu, eta, w, order, xi=None):
    """Max deviation of sum_n w_n R_lm R_l'm' from delta/(2l+1).

    A quadrature that cannot integrate the harmonics it is asked to expand in
    will silently corrupt the scattering source, so this is worth asserting
    before trusting an ordinate set at a given scattering order.
    """
    R, idx = harmonic_matrix(mu, eta, order, xi)
    w = np.asarray(w, float)
    G = (R * w[None, :]) @ R.T
    T = np.diag([1.0 / (2.0 * l + 1.0) for l, _ in idx])
    return float(np.max(np.abs(G - T)))
