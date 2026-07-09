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
