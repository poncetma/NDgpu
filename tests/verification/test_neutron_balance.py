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
from ndgpu.mesh import assemble_mesh_3d, UnstructuredDiffusionSolver


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


def _hex_grid(n, L):
    dx = L / n
    nid, coords = {}, []

    def gid(i, j, k):
        if (i, j, k) not in nid:
            nid[(i, j, k)] = len(coords); coords.append((i * dx, j * dx, k * dx))
        return nid[(i, j, k)]

    cells = [(gid(i, j, k), gid(i + 1, j, k), gid(i + 1, j + 1, k), gid(i, j + 1, k),
              gid(i, j, k + 1), gid(i + 1, j, k + 1), gid(i + 1, j + 1, k + 1), gid(i, j + 1, k + 1))
             for i in range(n) for j in range(n) for k in range(n)]
    return assemble_mesh_3d(coords, cells, [0] * len(cells))


def test_mesh_solver_neutron_balance_3d():
    # The same global invariant for the unstructured 3D mesh solver, computed
    # independently of the structured solver it is usually cross-checked against:
    # production and absorption from the cross sections and the volume-integrated
    # flux, leakage read out of the mesh operator (apply minus the removal*volume
    # diagonal telescopes to the boundary current). This is the mesh solver's own
    # conservation check -- it does not lean on any other solver being correct.
    mesh = _hex_grid(16, 80.0)
    mats, cm = [PWR_TWO_GROUP], np.zeros(mesh.n_cells, int)
    solver = UnstructuredDiffusionSolver(mesh, mats, cm, alpha_boundary=0.5, device="cpu")
    res = solver.solve(tol_k=1e-10, tol_source=1e-9)
    assert res.converged

    G, vol = solver.G, solver.area
    phi = [res.flux[g] for g in range(G)]
    nsf = [np.array([mats[m].nu_sigma_f[g] for m in cm]) for g in range(G)]
    sigma_a = [np.array([mats[m].sigma_a[g] for m in cm]) for g in range(G)]
    removal = [np.array([mats[m].removal[g] for m in cm]) for g in range(G)]

    production = float(sum((nsf[g] * phi[g] * vol).sum() for g in range(G)))
    absorption = float(sum((sigma_a[g] * phi[g] * vol).sum() for g in range(G)))
    leakage = float(sum((solver.ops[g].apply(phi[g]) - removal[g] * vol * phi[g]).sum()
                        for g in range(G)))

    prod_over_k = production / res.k_eff
    rel = abs(prod_over_k - (absorption + leakage)) / prod_over_k
    assert rel < 1e-8, f"production/k={prod_over_k:.6g} vs abs+leak={absorption+leakage:.6g} (rel {rel:.1e})"
    assert 0.0 < leakage < prod_over_k               # a bare box leaks, but not everything


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
