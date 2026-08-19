"""Non-symmetric Krylov options (GMRES, BiCGStab) next to the default CG.

CG is the default everywhere because the discretized operators are kept SPD
by construction. GMRES and BiCGStab (ndgpu.linalg) are the escape hatches for
future operators that cannot be symmetrized; these tests pin down that they
(a) solve what CG solves, bit-tight, through every solver entry point that
exposes ``linear_solver=``, and (b) solve a genuinely non-symmetric system,
which is the case CG cannot handle at all.
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, Material, TransientSolver
from ndgpu.benchmarks import build_twigl
from ndgpu.linalg import (PCGWorkspace, bicgstab, fgmres, get_linear_solver, gmres,
                          mixed_precision_preconditioner,
                          neumann_preconditioner, pcg)
from ndgpu.operator import GroupOperator

NONSYM = [gmres, bicgstab]


def _spd_stencil_system(n=12):
    rng = np.random.default_rng(7)
    grid = Grid(shape=(n, n, n), size=(60.0, 60.0, 60.0))
    D = np.full(grid.shape, 1.3)
    removal = np.full(grid.shape, 0.04)
    op = GroupOperator(np, grid, D, removal)
    b = rng.random(grid.shape)
    return op, b


def _nonsymmetric_system(n=200, seed=3):
    # Diagonally dominant but clearly non-symmetric (an upwinded
    # convection-like matrix) -- the case CG cannot handle.
    rng = np.random.default_rng(seed)
    A = rng.random((n, n)) - 0.5
    A += np.diag(np.abs(A).sum(axis=1) + 1.0)
    assert np.linalg.norm(A - A.T) > 1.0
    return A, rng.random(n)


@pytest.mark.parametrize("solver", NONSYM)
def test_matches_cg_on_spd_stencil(solver):
    op, b = _spd_stencil_system()
    x_cg, it_cg = pcg(op.apply, b, np.zeros_like(b), op.inv_diag, np, rtol=1e-10)
    x, it = solver(op.apply, b, np.zeros_like(b), op.inv_diag, np, rtol=1e-10)
    assert it_cg > 0 and it > 0
    assert np.linalg.norm(x - x_cg) <= 1e-8 * np.linalg.norm(x_cg)


def test_pcg_workspace_preserves_iterations_and_reuses_operator_output():
    op, b = _spd_stencil_system(n=9)
    x0 = np.zeros_like(b)
    reference, reference_it = pcg(
        op.apply, b, x0, op.inv_diag, np, rtol=1e-11)

    calls = {"out": 0}

    def apply_out(value, out=None):
        if out is not None:
            calls["out"] += 1
        return op.apply(value, out=out)

    workspace = PCGWorkspace.like(x0, operator_out=True)
    solved, iterations = pcg(
        apply_out, b, x0, op.inv_diag, np, rtol=1e-11,
        workspace=workspace)
    assert solved is workspace.x
    assert iterations == reference_it
    assert calls["out"] == iterations + 1
    np.testing.assert_array_equal(solved, reference)
    np.testing.assert_array_equal(x0, np.zeros_like(x0))

    # A nearby second right-hand side uses the exact same five work arrays.
    identities = tuple(id(getattr(workspace, name))
                       for name in ("x", "r", "z", "p", "ap"))
    b2 = 1.001 * b
    solved2, iterations2 = pcg(
        apply_out, b2, solved, op.inv_diag, np, rtol=1e-11,
        workspace=workspace)
    assert solved2 is workspace.x
    assert identities == tuple(id(getattr(workspace, name))
                               for name in ("x", "r", "z", "p", "ap"))
    assert iterations2 > 0
    residual = np.linalg.norm(op.apply(solved2) - b2)
    assert residual <= 1.1e-11 * np.linalg.norm(b2)


def test_pcg_workspace_validates_shape_and_dtype():
    workspace = PCGWorkspace.like(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="shape/dtype"):
        pcg(lambda x: x, np.ones((4, 3)), np.zeros((4, 3)),
            np.ones((4, 3)), np, workspace=workspace)


def test_pcg_graph_block_must_match_residual_cadence():
    op, b = _spd_stencil_system(n=5)
    with pytest.raises(ValueError, match="must equal check_every"):
        pcg(op.apply, b, np.zeros_like(b), op.inv_diag, np,
            check_every=2, graph_block=3)


def test_pcg_exact_convergence_before_delayed_check_stays_finite():
    A = np.array([[2.0, -1.0], [-1.0, 2.0]])
    b = np.array([1.0, 0.0])
    solved, iterations = pcg(
        lambda x: A @ x, b, np.zeros_like(b), 1.0 / np.diag(A), np,
        rtol=1e-12, maxiter=25, check_every=25)
    assert iterations == 25
    assert np.all(np.isfinite(solved))
    np.testing.assert_allclose(solved, np.linalg.solve(A, b), rtol=1e-12)


def test_pcg_graph_request_falls_back_on_cpu_without_changing_answer():
    op, b = _spd_stencil_system(n=6)
    workspace = PCGWorkspace.like(b, operator_out=True)
    reference, reference_it = pcg(
        op.apply, b, np.zeros_like(b), op.inv_diag, np,
        rtol=1e-10, check_every=2)
    solved, iterations = pcg(
        op.apply, b, np.zeros_like(b), op.inv_diag, np,
        rtol=1e-10, check_every=2, graph_block=2,
        workspace=workspace)
    np.testing.assert_array_equal(solved, reference)
    assert iterations == reference_it
    assert workspace.graph_captures == workspace.graph_replays == 0
    assert "capture needs CuPy" in workspace.graph_error


def test_neumann_pcg_workspace_reuses_preconditioner_scratch():
    op, b = _spd_stencil_system(n=8)
    calls = {"out": 0}

    def apply_out(value, out=None):
        if out is not None:
            calls["out"] += 1
        return op.apply(value, out=out)

    precond = neumann_preconditioner(apply_out, op.inv_diag, 2)
    workspace = PCGWorkspace.like(b, operator_out=True)
    solved, iterations = pcg(
        apply_out, b, np.zeros_like(b), op.inv_diag, np, rtol=1e-10,
        precond=precond, workspace=workspace)
    # One initial A*x, one A*p per iteration, and two polynomial stencil
    # applications per initial/iterated preconditioner evaluation all use the
    # same scratch array.
    assert calls["out"] == 1 + iterations + 2 * iterations
    assert np.linalg.norm(op.apply(solved) - b) <= 1.1e-10 * np.linalg.norm(b)


def test_fp32_preconditioner_keeps_fp64_true_residual_accuracy():
    op, b = _spd_stencil_system(n=11)
    low_grid = Grid(shape=b.shape, size=(60.0, 60.0, 60.0))
    low = GroupOperator(
        np, low_grid,
        np.full(b.shape, 1.3, dtype=np.float32),
        np.full(b.shape, 0.04, dtype=np.float32))
    precond = mixed_precision_preconditioner(
        low.apply, low.inv_diag, degree=2, outer_dtype=np.float64)
    workspace = PCGWorkspace.like(b, operator_out=True, fallback=True)
    solved, iterations = pcg(
        op.apply, b, np.zeros_like(b), op.inv_diag, np, rtol=1e-11,
        precond=precond, fallback_precond=neumann_preconditioner(
            op.apply, op.inv_diag, 2),
        residual_replace_every=8, workspace=workspace)
    assert solved.dtype == np.float64
    assert iterations > 0
    assert workspace.fallback_count == 0
    assert np.linalg.norm(op.apply(solved) - b) <= 1.1e-11 * np.linalg.norm(b)


def test_mixed_preconditioner_failure_restarts_with_fp64():
    op, b = _spd_stencil_system(n=7)

    def broken(residual):
        return np.full_like(residual, np.nan)

    workspace = PCGWorkspace.like(b, operator_out=True, fallback=True)
    solved, iterations = pcg(
        op.apply, b, np.zeros_like(b), op.inv_diag, np, rtol=1e-10,
        precond=broken,
        fallback_precond=neumann_preconditioner(op.apply, op.inv_diag, 1),
        workspace=workspace)
    assert workspace.fallback_count == 1
    assert iterations > 1
    assert np.linalg.norm(op.apply(solved) - b) <= 1.1e-10 * np.linalg.norm(b)


def test_mixed_preconditioner_requires_a_narrower_dtype():
    with pytest.raises(ValueError, match="narrower"):
        mixed_precision_preconditioner(
            lambda x: x, np.ones(4, dtype=np.float64), 0, np.float64)


@pytest.mark.parametrize("solver", NONSYM)
def test_solves_nonsymmetric_system(solver):
    A, b = _nonsymmetric_system()
    inv_diag = 1.0 / np.diag(A)
    x, it = solver(lambda v: A @ v, b, np.zeros_like(b), inv_diag, np, rtol=1e-12)
    assert np.linalg.norm(A @ x - b) <= 1e-11 * np.linalg.norm(b)
    assert it > 0
    # Warm start from the solution: zero iterations, like pcg.
    _, it0 = solver(lambda v: A @ v, b, x, inv_diag, np, rtol=1e-10)
    assert it0 == 0


def _convection_dominated_system(n=64, D=1.0, v=40.0, sigma=0.1, h=1.0):
    """1D upwinded convection-diffusion  -D u'' + v u' + sigma u = q,
    cell Peclet number v h / D = 40: the operator a flowing-fuel drift term
    (e.g. MSR precursor advection) would produce. Strongly non-symmetric --
    SPD only at v = 0."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2 * D / h**2 + v / h + sigma
        if i > 0:
            A[i, i - 1] = -D / h**2 - v / h
        if i < n - 1:
            A[i, i + 1] = -D / h**2
    return A, np.ones(n)


def test_convection_dominated_needs_nonsymmetric_solver():
    # The case that *requires* the new solvers: CG's recurrence is only
    # valid for SPD operators, and on this one it diverges outright (the
    # residual grows without bound), while GMRES and BiCGStab solve it.
    A, b = _convection_dominated_system()
    inv_diag = 1.0 / np.diag(A)
    apply_A = lambda u: A @ u

    with pytest.raises(RuntimeError, match="PCG failed to converge"):
        pcg(apply_A, b, np.zeros_like(b), inv_diag, np, rtol=1e-10, maxiter=2000)

    x, it = bicgstab(apply_A, b, np.zeros_like(b), inv_diag, np, rtol=1e-10)
    assert np.linalg.norm(A @ x - b) <= 1e-9 * np.linalg.norm(b)
    assert it > 0
    # GMRES needs its subspace to span the downwind sweep: the default
    # restart=30 stagnates on this strongly non-normal operator, full-memory
    # GMRES (restart >= n) finishes in at most n applies.
    x, it = gmres(apply_A, b, np.zeros_like(b), inv_diag, np, rtol=1e-10,
                  restart=len(b))
    assert np.linalg.norm(A @ x - b) <= 1e-9 * np.linalg.norm(b)
    assert it <= len(b)


def test_gmres_restart_cycles():
    A, b = _nonsymmetric_system()
    inv_diag = 1.0 / np.diag(A)
    x_full, it_full = gmres(lambda v: A @ v, b, np.zeros_like(b), inv_diag, np,
                            rtol=1e-12)
    x5, it5 = gmres(lambda v: A @ v, b, np.zeros_like(b), inv_diag, np,
                    rtol=1e-12, restart=5)
    assert np.linalg.norm(A @ x5 - b) <= 1e-11 * np.linalg.norm(b)
    assert it5 >= it_full   # discarding the Krylov space costs iterations


def test_fgmres_accepts_a_variable_right_preconditioner():
    A, b = _nonsymmetric_system(n=36)
    inv_diag = 1.0 / np.diag(A)
    calls = 0

    def variable_preconditioner(r):
        nonlocal calls
        calls += 1
        # Alternating strength makes this a variable map, for which ordinary
        # GMRES's M(sum V*y) reconstruction is invalid but stored-Z FGMRES is
        # mathematically well defined.
        scale = 1.0 if calls % 2 else 0.75
        return scale * inv_diag * r

    x, iterations = fgmres(
        lambda v: A @ v, b, np.zeros_like(b), inv_diag, np,
        rtol=1e-12, restart=len(b), precond=variable_preconditioner)
    assert iterations == calls > 0
    assert np.linalg.norm(A @ x - b) <= 1e-11 * np.linalg.norm(b)
    assert get_linear_solver("fgmres") is fgmres


@pytest.mark.parametrize("solver", NONSYM)
def test_neumann_preconditioner_helps(solver):
    op, b = _spd_stencil_system()
    M = neumann_preconditioner(op.apply, op.inv_diag, 2)
    x_pl, it_plain = solver(op.apply, b, np.zeros_like(b), op.inv_diag, np, rtol=1e-10)
    x_pc, it_prec = solver(op.apply, b, np.zeros_like(b), op.inv_diag, np,
                           rtol=1e-10, precond=M)
    assert it_prec < it_plain
    assert np.linalg.norm(x_pc - x_pl) <= 1e-8 * np.linalg.norm(x_pl)


def test_divergence_form_operator_is_nonsymmetric():
    # GroupOperator(symmetric=False) on a cylindrical grid: the natural
    # per-unit-volume form W^{-1} A_w. Exactly the weighted stencil with each
    # row divided by its cell weight, genuinely non-symmetric, and its plain-
    # source solve equals the weighted-source SPD solve.
    rng = np.random.default_rng(11)
    grid = Grid(shape=(12, 1, 10), size=(60.0, 1.0, 50.0), geometry="cylindrical")
    D = 1.0 + rng.random(grid.shape)
    removal = 0.05 + 0.1 * rng.random(grid.shape)
    bc = (("reflective", "zero-flux"), "reflective", "zero-flux")
    A_sym = GroupOperator(np, grid, D.copy(), removal.copy(), bc=bc)
    A_div = GroupOperator(np, grid, D.copy(), removal.copy(), bc=bc,
                          symmetric=False)
    w = A_sym.rhs_weight
    assert A_div.rhs_weight is None

    u, v = rng.random(grid.shape), rng.random(grid.shape)
    assert np.abs(A_div.apply(u) - A_sym.apply(u) / w).max() < 1e-14
    skew = float(np.sum(A_div.apply(u) * v) - np.sum(u * A_div.apply(v)))
    assert abs(skew) > 1e-3   # non-symmetric for real

    q = rng.random(grid.shape)
    x_sym, _ = pcg(A_sym.apply, w * q, np.zeros_like(q), A_sym.inv_diag, np,
                   rtol=1e-12)
    x_div, _ = bicgstab(A_div.apply, q, np.zeros_like(q), A_div.inv_diag, np,
                        rtol=1e-12)
    assert np.linalg.norm(x_div - x_sym) <= 1e-9 * np.linalg.norm(x_sym)


def test_divergence_form_rejects_cg():
    grid = Grid(shape=(8, 1, 8), size=(40.0, 1.0, 40.0), geometry="cylindrical")
    fuel = Material(diffusion=[1.3], sigma_a=[0.05], nu_sigma_f=[0.06])
    with pytest.raises(ValueError, match="gmres.*bicgstab"):
        DiffusionEigenSolver(grid, fuel, device="cpu", symmetric_operator=False)


@pytest.mark.parametrize("name", ["gmres", "bicgstab"])
def test_eigensolver_matches_cg(name):
    # Heterogeneous cylindrical core: exercises the volume-weighted stencil,
    # the Neumann preconditioner, and the source-weighting path.
    fuel = Material(diffusion=[1.3, 0.4], sigma_a=[0.011, 0.12],
                    nu_sigma_f=[0.008, 0.17], sigma_s=[[0.0, 0.025], [0.0, 0.0]])
    refl = Material(diffusion=[1.1, 0.35], sigma_a=[0.002, 0.03],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.028], [0.0, 0.0]])
    grid = Grid(shape=(16, 1, 20), size=(80.0, 1.0, 100.0), geometry="cylindrical")
    mmap = np.zeros(grid.shape, dtype=np.int64)
    mmap[10:, :, :] = 1
    bc = (("reflective", "zero-flux"), "reflective", "zero-flux")
    ks = {}
    for solver_name in ("cg", name):
        s = DiffusionEigenSolver(grid, [fuel, refl], mmap, bc=bc, device="cpu",
                                 precond_degree=2, linear_solver=solver_name)
        res = s.solve(tol_k=1e-9, tol_source=1e-8)
        assert res.converged
        ks[solver_name] = res.k_eff
    assert ks[name] == pytest.approx(ks["cg"], abs=1e-8)


@pytest.mark.parametrize("name", ["gmres", "bicgstab"])
def test_transient_matches_cg(name):
    prob = build_twigl(perturbation="step", cells_per_8cm=1)
    power = {}
    for solver_name in ("cg", name):
        s = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                            bc=prob.bc, device="cpu", linear_solver=solver_name)
        power[solver_name] = s.solve(t_end=0.1, dt=5e-3).power
    # Different Krylov methods stop at different points inside the same
    # residual tolerance; over 20 steps that accumulates to ~1e-5 in power.
    assert np.allclose(power[name], power["cg"], rtol=1e-4)


def test_unknown_linear_solver_rejected():
    with pytest.raises(ValueError, match="unknown linear solver"):
        get_linear_solver("sor")
    assert get_linear_solver("gmres") is gmres
    assert get_linear_solver("bicgstab") is bicgstab
    assert get_linear_solver(pcg) is pcg   # callables pass through
