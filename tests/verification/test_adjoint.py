"""Adjoint (importance) k-eigenproblem: DiffusionEigenSolver.solve(adjoint=True).

The adjoint shares the forward spectrum, so k_adjoint == k_forward is a
necessary check -- but a weak one, since a solver that ignored the transpose
entirely could still land the eigenvalue. The decisive test is that the adjoint
*flux shape* is right, which we pin two ways: its physical signature (thermal
importance exceeds fast in a thermal reactor) and, quantitatively, that it
weights first-order perturbation theory to match a directly re-solved reactivity
change. The PWR two-group set has downscatter only, so its scattering matrix is
triangular -- genuinely nonsymmetric -- and the transpose actually matters.
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, Material, PWR_TWO_GROUP

GRID = Grid(shape=(24, 24, 24), size=(80.0, 80.0, 80.0))
TIGHT = dict(tol_k=1e-10, tol_source=1e-9)


def _solve(material=PWR_TWO_GROUP, **kw):
    res = DiffusionEigenSolver(GRID, material, device="cpu").solve(**TIGHT, **kw)
    assert res.converged, res
    return res


def test_adjoint_eigenvalue_equals_forward():
    fwd = _solve()
    adj = _solve(adjoint=True)
    assert adj.k_eff == pytest.approx(fwd.k_eff, abs=1e-7), (fwd.k_eff, adj.k_eff)


def test_adjoint_flux_is_importance_weighted_not_the_forward_flux():
    # Forward flux is fast-peaked (neutrons born fast, slow down, leak/absorb).
    # Adjoint flux is the *importance*: a thermal neutron is worth more in a
    # thermal reactor, so the thermal (g=1) adjoint exceeds the fast (g=0) one
    # -- the opposite ordering to the forward flux. This only holds if the
    # transpose was actually applied.
    fwd = _solve().flux_numpy
    adj = _solve(adjoint=True).flux_numpy
    assert fwd[0].mean() > fwd[1].mean()          # forward: fast-peaked
    assert adj[1].mean() > adj[0].mean()          # adjoint: thermal-peaked


def test_adjoint_weights_first_order_perturbation_theory():
    # Bump thermal absorption by a small eps and compare the exact reactivity
    # change (re-solve) with the first-order estimate
    #     d_rho = -<phi*, dSigma_a phi> / <phi*, F phi>.
    # Agreement to O(eps) confirms the adjoint flux magnitude *and* shape.
    eps = 1e-3
    fwd = _solve()
    adj = _solve(adjoint=True)
    phi, star = fwd.flux_numpy, adj.flux_numpy

    perturbed = Material(
        name="thermal-poison", diffusion=PWR_TWO_GROUP.diffusion,
        sigma_a=PWR_TWO_GROUP.sigma_a + np.array([0.0, eps]),
        nu_sigma_f=PWR_TWO_GROUP.nu_sigma_f, sigma_s=PWR_TWO_GROUP.sigma_s,
        chi=PWR_TWO_GROUP.chi)
    k_pert = _solve(perturbed).k_eff
    drho_exact = 1.0 / fwd.k_eff - 1.0 / k_pert    # rho' - rho

    dSa = np.array([0.0, eps])
    num = sum((star[g] * dSa[g] * phi[g]).sum() for g in range(2))
    fission = sum(PWR_TWO_GROUP.nu_sigma_f[g] * phi[g] for g in range(2))
    denom = sum((star[g] * PWR_TWO_GROUP.chi[g] * fission).sum() for g in range(2))
    drho_pt = -num / denom

    assert drho_exact < 0                          # adding poison drops reactivity
    rel = abs(drho_pt - drho_exact) / abs(drho_exact)
    assert rel < 0.02, f"PT estimate {drho_pt:.3e} vs exact {drho_exact:.3e} (rel {rel:.1e})"
