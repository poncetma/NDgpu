"""Global neutron balance of the converged eigenpair: production/k = absorption
+ leakage.

This is the physics invariant every k-eigenvalue solution must satisfy, and it
is *independent* of how the solver got there: production and absorption come
straight from the cross sections and the flux, while leakage is read out of the
discrete operator (sum of the operator apply minus the removal diagonal is the
net surface current, since the interior -div D grad telescopes to the boundary).
Scattering cancels globally -- every neutron scattered out of a group is
scattered into another -- so only absorption and leakage balance production. A
sign error, a mis-scaled face coefficient, or a dropped boundary term would
break this even though the power iteration still "converged".
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, Material, PWR_TWO_GROUP


def _balance(solver, res):
    """Return (production/k, absorption + leakage) for a converged solve."""
    xp, G, f = solver.xp, solver.n_groups, solver.fields
    phi = [xp.asarray(res.flux[g]) for g in range(G)]

    production = float(sum(xp.sum(f.nu_sigma_f[g] * phi[g]) for g in range(G)))
    absorption = leakage = 0.0
    for g in range(G):
        out_scatter = xp.zeros_like(phi[g])
        for gt in range(G):
            s = f.sigma_s[g][gt]
            if gt != g and s is not None:
                out_scatter = out_scatter + s
        sigma_a = f.removal[g] - out_scatter               # removal minus out-scatter
        absorption += float(xp.sum(sigma_a * phi[g]))
        # operator apply = leakage + removal*phi; strip removal to get leakage
        leakage += float(xp.sum(solver.ops[g].apply(phi[g]) - f.removal[g] * phi[g]))
    return production / res.k_eff, absorption + leakage


def _assert_balanced(solver, res):
    prod_over_k, loss = _balance(solver, res)
    rel = abs(prod_over_k - loss) / prod_over_k
    assert rel < 1e-8, f"production/k={prod_over_k:.6g} vs absorption+leakage={loss:.6g} (rel {rel:.1e})"
    return prod_over_k, loss


def test_balance_homogeneous_bare_box():
    grid = Grid(shape=(24, 24, 24), size=(80.0, 80.0, 80.0))
    solver = DiffusionEigenSolver(grid, PWR_TWO_GROUP, device="cpu")
    res = solver.solve(tol_k=1e-10, tol_source=1e-9)
    assert res.converged
    _assert_balanced(solver, res)


def test_balance_reflected_core_has_small_leakage():
    # Fissile cube in a scattering reflector: balance still holds, and the
    # reflector should hand most neutrons back, so leakage << absorption.
    reflector = Material(name="reflector", diffusion=[1.13, 0.16],
                         sigma_a=[0.0004, 0.0197], nu_sigma_f=[0.0, 0.0],
                         sigma_s=[[0.0, 0.0494], [0.0, 0.0]])
    n = 24
    grid = Grid(shape=(n, n, n), size=(120.0, 120.0, 120.0))
    mmap = np.ones(grid.shape, dtype=np.int64)
    lo, hi = n // 4, 3 * n // 4
    mmap[lo:hi, lo:hi, lo:hi] = 0                          # fuel cube in reflector
    solver = DiffusionEigenSolver(grid, [PWR_TWO_GROUP, reflector],
                                  material_map=mmap, device="cpu")
    res = solver.solve(tol_k=1e-10, tol_source=1e-9)
    assert res.converged
    prod_over_k, _ = _assert_balanced(solver, res)

    # leakage is the operator's boundary current; for a well-reflected core it
    # is a small fraction of the production.
    _, loss = _balance(solver, res)
    leakage = 0.0
    xp, f = solver.xp, solver.fields
    for g in range(solver.n_groups):
        phi_g = xp.asarray(res.flux[g])
        leakage += float(xp.sum(solver.ops[g].apply(phi_g) - f.removal[g] * phi_g))
    assert 0.0 < leakage < 0.05 * prod_over_k
