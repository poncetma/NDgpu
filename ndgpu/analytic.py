"""Analytic references for bare homogeneous reactors (code validation).

For a bare homogeneous reactor with zero flux on the surface, the flux
separates as phi_g(r) = psi_g * f(r) with -lap(f) = B^2 f, where B^2 is the
geometric buckling of the shape. The multigroup balance then collapses to a
G x G problem; because the fission operator is rank one, k_eff has the closed
form  k = nuSigma_f . M^{-1} chi  with
M = diag(D_g B^2 + Sigma_r,g) - (in-scatter)^T.
"""

from __future__ import annotations

import numpy as np

from .materials import Material


def geometric_buckling_box(size) -> float:
    """B^2 of a rectangular parallelepiped with zero flux on the surface."""
    return float(sum((np.pi / L) ** 2 for L in size))


def k_from_buckling(material: Material, buckling: float) -> float:
    """Exact multigroup k_eff of a bare homogeneous reactor with buckling B^2."""
    G = material.n_groups
    M = np.diag(material.diffusion * buckling + material.removal)
    off = material.sigma_s.T.copy()  # in-scatter: M[g, g'] -= sigma_s[g'->g]
    np.fill_diagonal(off, 0.0)
    M -= off
    return float(material.nu_sigma_f @ np.linalg.solve(M, material.chi))


def k_infinite(material: Material) -> float:
    return k_from_buckling(material, 0.0)


def k_bare_box(material: Material, size) -> float:
    return k_from_buckling(material, geometric_buckling_box(size))


def k_from_buckling_sp3(material: Material, buckling: float) -> float:
    """Exact multigroup SP3 k_eff of a bare homogeneous reactor.

    Both moments satisfy zero-flux boundary conditions on the surface, so they
    share the fundamental buckling mode and the balance collapses to a
    2G x 2G system in (Phi1_g, phi2_g); the fission operator is again rank one.
    Matches the SP3EigenSolver discretization in the fine-mesh limit.
    """
    G = material.n_groups
    D1 = material.diffusion
    st = material.sigma_t
    s0 = material.removal
    D2 = 9.0 / (35.0 * st)
    S = material.sigma_s.T.copy()  # in-scatter matrix, acts on phi0
    np.fill_diagonal(S, 0.0)

    M = np.zeros((2 * G, 2 * G))
    M[:G, :G] = np.diag(D1 * buckling + s0) - S
    M[:G, G:] = -2.0 * np.diag(s0) + 2.0 * S
    M[G:, :G] = -0.4 * np.diag(s0) + 0.4 * S
    M[G:, G:] = np.diag(D2 * buckling + st + 0.8 * s0) - 0.8 * S

    emission = np.concatenate([material.chi, -0.4 * material.chi])
    production = np.concatenate([material.nu_sigma_f, -2.0 * material.nu_sigma_f])
    return float(production @ np.linalg.solve(M, emission))


def k_bare_box_sp3(material: Material, size) -> float:
    return k_from_buckling_sp3(material, geometric_buckling_box(size))
