"""Neumann-polynomial (POLYN) preconditioner for the inner CG.

The preconditioner must change the *path* to the solution, never the
solution: k must match plain Jacobi to solver tolerance while the CG
iteration count drops materially (cf. E et al., NED 320 (2017), where
degree-3 Neumann-PCG was the fastest GPU solver for 2e4-3e6 unknowns).
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, PWR_TWO_GROUP
from ndgpu.backend import get_backend
from ndgpu.linalg import neumann_preconditioner, pcg
from ndgpu.operator import GroupOperator
from ndgpu.solver import Fields

GRID = Grid(shape=(32, 32, 32), size=(120.0, 120.0, 120.0))


def test_polyn_cuts_cold_cg_iterations():
    # Single fixed-source solve, cold start, tight tolerance: the regime the
    # preconditioner is designed for. Degree 3 must cut iterations >= 2x and
    # reproduce the same solution.
    xp = get_backend("cpu")
    f = Fields(xp, GRID, PWR_TWO_GROUP, None, np.float64)
    op = GroupOperator(xp, GRID, f.diffusion[0], f.removal[0])
    b = xp.ones(GRID.shape)
    x0 = xp.zeros(GRID.shape)
    x_j, it_j = pcg(op.apply, b, x0, op.inv_diag, xp, rtol=1e-10)
    M = neumann_preconditioner(op.apply, op.inv_diag, 3)
    x_p, it_p = pcg(op.apply, b, x0, op.inv_diag, xp, rtol=1e-10, precond=M)
    assert it_p * 2 <= it_j, (it_p, it_j)
    assert float(xp.max(xp.abs(x_p - x_j))) < 1e-7 * float(xp.max(xp.abs(x_j)))


def test_polyn_eigenvalue_unchanged_fewer_inners():
    ks, inners = {}, {}
    for deg in (0, 3):
        res = DiffusionEigenSolver(GRID, PWR_TWO_GROUP, device="cpu",
                                   precond_degree=deg).solve(
            tol_k=1e-9, tol_source=1e-8)
        assert res.converged
        ks[deg], inners[deg] = res.k_eff, res.inner_iterations
    assert ks[3] == pytest.approx(ks[0], abs=1e-7)
    assert inners[3] < inners[0]


# --- Anderson acceleration of the outer power iteration ----------------------
def test_anderson_matches_power_iteration_and_cuts_outers_structured():
    # Anderson must change only the path, never the eigenpair: the converged k
    # matches plain power iteration to solver tolerance while the outer-iteration
    # count drops on this leakage-dominated core.
    g = Grid(shape=(24, 24, 24), size=(100.0, 100.0, 100.0))
    tol = dict(tol_k=1e-9, tol_source=1e-8)
    plain = DiffusionEigenSolver(g, PWR_TWO_GROUP, bc="vacuum", device="cpu").solve(
        anderson_depth=1, **tol)
    accel = DiffusionEigenSolver(g, PWR_TWO_GROUP, bc="vacuum", device="cpu").solve(
        anderson_depth=8, **tol)
    assert accel.k_eff == pytest.approx(plain.k_eff, abs=1e-7)
    assert accel.outer_iterations < 0.7 * plain.outer_iterations


def test_anderson_matches_power_iteration_on_the_mesh_solver():
    from ndgpu.mesh import assemble_mesh, UnstructuredDiffusionSolver
    # A bare rectangular reactor on a quad mesh (self-contained, loosely coupled).
    n, L = 30, 90.0
    dx = L / n
    nid, coords = {}, []

    def gid(i, j):
        if (i, j) not in nid:
            nid[(i, j)] = len(coords); coords.append((i * dx, j * dx))
        return nid[(i, j)]

    cells = [(gid(i, j), gid(i + 1, j), gid(i + 1, j + 1), gid(i, j + 1))
             for i in range(n) for j in range(n)]
    mesh = assemble_mesh(coords, cells, [0] * len(cells))
    cm = np.zeros(mesh.n_cells, int)
    tol = dict(tol_k=1e-8, tol_source=1e-7)
    plain = UnstructuredDiffusionSolver(mesh, [PWR_TWO_GROUP], cm, alpha_boundary=0.5,
                                        device="cpu").solve(anderson_depth=1, **tol)
    accel = UnstructuredDiffusionSolver(mesh, [PWR_TWO_GROUP], cm, alpha_boundary=0.5,
                                        device="cpu").solve(anderson_depth=8, **tol)
    assert accel.k_eff == pytest.approx(plain.k_eff, abs=1e-7)
    assert accel.outer_iterations < 0.7 * plain.outer_iterations
