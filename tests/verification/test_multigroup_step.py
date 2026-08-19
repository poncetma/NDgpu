"""CPU gates for the experimental monolithic transient block solve."""

import numpy as np

from ndgpu import Grid
from ndgpu.linalg import FGMRESWorkspace, PCGWorkspace, fgmres, fixed_pcg
from ndgpu.multigroup import (AdjointCoarseCorrectionPreconditioner,
                              BlockJacobiPreconditioner,
                              EnergyModeCoarseCorrectionPreconditioner,
                              EnergyGroupGaussSeidelPreconditioner,
                              GalerkinCMFDPreconditioner,
                              MultigroupStepOperator,
                              SourceShapeCoarseCorrectionPreconditioner,
                              SpatialCoarseSpace)
from ndgpu.operator import GroupOperator


def _matrix_from_apply(apply, shape):
    n = int(np.prod(shape))
    matrix = np.empty((n, n))
    for j in range(n):
        basis = np.zeros(n)
        basis[j] = 1.0
        matrix[:, j] = apply(basis.reshape(shape)).reshape(-1)
    return matrix


def _two_group_block(upscatter=True, fission=True, cells=4):
    grid = Grid(shape=(cells, 1, 1), size=(8.0, 1.0, 1.0))
    shape = grid.shape
    D = [np.full(shape, 1.1), np.full(shape, 0.7)]
    removal = [np.full(shape, 0.45), np.full(shape, 0.30)]
    ops = [GroupOperator(np, grid, D[g], removal[g]) for g in range(2)]
    sigma_s = [[None, np.full(shape, 0.08)],
               [np.full(shape, 0.025) if upscatter else None, None]]
    nsf = [np.full(shape, 0.10 if fission else 0.0),
           np.full(shape, 0.06 if fission else 0.0)]
    weights = [np.full(shape, 0.72), np.full(shape, 0.28)]
    return MultigroupStepOperator(
        ops, sigma_s, nsf, weights, k_eff=1.04)


def _energy_chain_block(groups=11):
    """One-cell system isolating bidirectional energy propagation."""
    grid = Grid(shape=(1, 1, 1), size=(1.0, 1.0, 1.0))
    shape = grid.shape
    ops = [GroupOperator(
        np, grid, np.zeros(shape), np.full(shape, 0.30))
        for _ in range(groups)]
    sigma_s = [[None for _ in range(groups)] for _ in range(groups)]
    for g in range(groups - 1):
        sigma_s[g][g + 1] = np.full(shape, 0.08)
        sigma_s[g + 1][g] = np.full(shape, 0.06)
    zero = [np.zeros(shape) for _ in range(groups)]
    return MultigroupStepOperator(
        ops, sigma_s, zero, zero, k_eff=1.0)


def test_matrix_free_multigroup_operator_matches_explicit_blocks():
    block = _two_group_block()
    n = int(np.prod(block.cell_shape))
    loss = [_matrix_from_apply(op.apply, block.cell_shape)
            for op in block.operators]
    eye = np.eye(n)
    nsf = [np.asarray(v).reshape(-1) for v in block.nu_sigma_f]
    weight = [np.asarray(v).reshape(-1) for v in block.emission_weights]

    expected = np.zeros((2 * n, 2 * n))
    for g in range(2):
        for gf in range(2):
            row = loss[g].copy() if g == gf else np.zeros((n, n))
            row -= np.diag(weight[g] * nsf[gf] / block.k_eff)
            scatter = block.sigma_s[gf][g]
            if gf != g and scatter is not None:
                row -= np.diag(np.asarray(scatter).reshape(-1)) @ eye
            expected[g*n:(g+1)*n, gf*n:(gf+1)*n] = row

    assembled = _matrix_from_apply(block.apply, block.shape)
    np.testing.assert_allclose(assembled, expected, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(
        block.assemble().toarray(), expected, rtol=0.0, atol=2e-16)


def test_conservative_coarse_operator_is_exact_rap():
    block = _two_group_block(cells=8)
    space = SpatialCoarseSpace(block.cell_shape, factors=(2, 1, 1))

    class Zero:
        ndgpu_out = True

        def __call__(self, residual, out=None):
            if out is None:
                return np.zeros_like(residual)
            out.fill(0)
            return out

    cmfd = GalerkinCMFDPreconditioner(block, Zero(), space)
    rng = np.random.default_rng(81)
    coarse = rng.standard_normal(cmfd.coarse_unknowns)
    fine = space.prolong(coarse, block.groups)
    restricted_apply = space.restrict(block.apply(fine), block.groups)
    np.testing.assert_allclose(
        cmfd.coarse_matrix @ coarse, restricted_apply,
        rtol=2e-14, atol=2e-14)


def test_cmfd_is_exact_on_piecewise_constant_coarse_subspace():
    block = _two_group_block(cells=8)
    space = SpatialCoarseSpace(block.cell_shape, factors=(2, 1, 1))

    class Zero:
        ndgpu_out = True

        def __call__(self, residual, out=None):
            if out is None:
                return np.zeros_like(residual)
            out.fill(0)
            return out

    cmfd = GalerkinCMFDPreconditioner(block, Zero(), space)
    rng = np.random.default_rng(82)
    expected = space.prolong(
        rng.standard_normal(cmfd.coarse_unknowns), block.groups)
    corrected = cmfd(block.apply(expected))
    np.testing.assert_allclose(corrected, expected, rtol=2e-13, atol=2e-13)
    assert cmfd.stats.applications == 1
    assert cmfd.stats.fine_residual_applies == 1


def test_cmfd_removes_slow_spatial_modes_on_refined_line_problem():
    """The prototype must help when spatial low-frequency error is present."""
    block = _two_group_block(cells=64)
    rng = np.random.default_rng(83)
    rhs = rng.random(block.shape)
    initial = np.zeros_like(rhs)
    plain = BlockJacobiPreconditioner(block.inv_diag)
    plain_solution, plain_iterations = fgmres(
        block.apply, rhs, initial, block.inv_diag, np, precond=plain,
        rtol=1e-10, restart=30, maxiter=300)

    space = SpatialCoarseSpace(block.cell_shape, factors=(4, 1, 1))
    cmfd = GalerkinCMFDPreconditioner(
        block, BlockJacobiPreconditioner(block.inv_diag), space,
        mode="additive")
    coarse_solution, coarse_iterations = fgmres(
        block.apply, rhs, initial, block.inv_diag, np, precond=cmfd,
        rtol=1e-10, restart=30, maxiter=300)

    assert plain_iterations >= 60
    assert coarse_iterations <= 32
    assert coarse_iterations < 0.55 * plain_iterations
    np.testing.assert_allclose(
        coarse_solution, plain_solution, rtol=2e-9, atol=2e-10)


def test_fixed_rhs_applies_cylindrical_row_weight():
    block = _two_group_block()
    weight = np.arange(1.0, 5.0).reshape(block.cell_shape)
    block.rhs_weight = weight
    theta = [2.0, 3.0]
    carried = [np.ones(block.cell_shape), 2.0*np.ones(block.cell_shape)]
    delayed = [0.25*np.ones(block.cell_shape), 0.5*np.ones(block.cell_shape)]
    rhs = block.fixed_rhs(theta, carried, delayed)
    np.testing.assert_array_equal(rhs[0], weight * 2.25)
    np.testing.assert_array_equal(rhs[1], weight * 6.5)


def test_one_group_sweep_exactly_inverts_lower_scatter_block():
    # With no upscatter/fission the energy matrix is block lower triangular;
    # one ordered group sweep is its exact inverse when inner PCG is tight.
    block = _two_group_block(upscatter=False, fission=False)
    preconditioner = EnergyGroupGaussSeidelPreconditioner(
        block, scatter_sweeps=1, inner_rtol=1e-13, inner_maxiter=200)
    rng = np.random.default_rng(12)
    residual = rng.random(block.shape)
    correction = preconditioner(residual)
    np.testing.assert_allclose(
        block.apply(correction), residual, rtol=2e-12, atol=2e-12)
    assert preconditioner.stats.applications == 1
    assert preconditioner.stats.group_solves == block.groups
    assert preconditioner.stats.inner_iterations > 0


def test_fixed_pcg_zero_residual_is_a_finite_no_op():
    block = _two_group_block(fission=False)
    op = block.operators[0]
    expected = np.random.default_rng(16).random(block.cell_shape)
    rhs = op.apply(expected)
    work = PCGWorkspace.like(expected, operator_out=True)
    got, iterations = fixed_pcg(
        op.apply, rhs, expected, op.inv_diag, np, iterations=2,
        workspace=work)
    assert iterations == 2
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=2e-15)


def test_reverse_group_sweep_exactly_inverts_upper_scatter_block():
    block = _two_group_block(upscatter=True, fission=False)
    # Remove downscatter, leaving an upper block-triangular system.
    block.sigma_s = ((None, None), (block.sigma_s[1][0], None))
    preconditioner = EnergyGroupGaussSeidelPreconditioner(
        block, scatter_sweeps=1, ordering="reverse",
        inner_rtol=1e-13, inner_maxiter=200)
    rng = np.random.default_rng(13)
    residual = rng.random(block.shape)
    correction = preconditioner(residual)
    np.testing.assert_allclose(
        block.apply(correction), residual, rtol=2e-12, atol=2e-12)


def test_energy_sweep_anderson_preserves_homogeneity():
    block = _two_group_block()
    rng = np.random.default_rng(14)
    residual = rng.random(block.shape)
    for depth in (1, 2):
        preconditioner = EnergyGroupGaussSeidelPreconditioner(
            block, scatter_sweeps=3, anderson_depth=depth,
            inner_rtol=1e-12, inner_maxiter=200)
        got = preconditioner(residual)
        scaled = preconditioner(-2.5 * residual)
        np.testing.assert_allclose(
            scaled, -2.5 * got, rtol=2e-10, atol=2e-11)


def test_energy_sweep_anderson_removes_bidirectional_chain_mode():
    block = _energy_chain_block()
    residual = np.random.default_rng(15).random(block.shape)
    defects = []
    for depth in (0, 1):
        preconditioner = EnergyGroupGaussSeidelPreconditioner(
            block, scatter_sweeps=3, anderson_depth=depth,
            inner_rtol=1e-13, inner_maxiter=50)
        correction = preconditioner(residual)
        defects.append(np.linalg.norm(block.apply(correction) - residual))
    assert defects[1] < 0.55 * defects[0]


def test_fgmres_monolithic_solution_matches_dense_reference():
    block = _two_group_block()
    dense = _matrix_from_apply(block.apply, block.shape)
    rng = np.random.default_rng(22)
    rhs = rng.random(block.shape)
    reference = np.linalg.solve(dense, rhs.reshape(-1)).reshape(block.shape)
    sweep = EnergyGroupGaussSeidelPreconditioner(
        block, scatter_sweeps=2, inner_rtol=1e-10, inner_maxiter=200)
    solved, iterations = fgmres(
        block.apply, rhs, np.zeros_like(rhs), block.inv_diag, np,
        precond=sweep, rtol=1e-11, restart=20, maxiter=100)
    assert 0 < iterations < 100
    np.testing.assert_allclose(solved, reference, rtol=2e-10, atol=2e-10)
    assert np.linalg.norm(block.apply(solved) - rhs) <= 1.1e-11 * np.linalg.norm(rhs)


def test_fgmres_workspace_matches_dense_reference_and_is_reusable():
    block = _two_group_block()
    dense = _matrix_from_apply(block.apply, block.shape)
    rng = np.random.default_rng(23)
    workspace = FGMRESWorkspace.like(
        np.zeros(block.shape), restart=12, operator_out=True)
    for _ in range(2):
        rhs = rng.random(block.shape)
        reference = np.linalg.solve(dense, rhs.reshape(-1)).reshape(block.shape)
        sweep = EnergyGroupGaussSeidelPreconditioner(
            block, scatter_sweeps=2, inner_rtol=1e-10, inner_maxiter=200)
        solved, iterations = fgmres(
            block.apply, rhs, np.zeros_like(rhs), block.inv_diag, np,
            precond=sweep, rtol=1e-11, restart=12, maxiter=100,
            workspace=workspace)
        assert 0 < iterations < 100
        np.testing.assert_allclose(
            solved, reference, rtol=2e-10, atol=2e-10)


def test_adjoint_rank_one_correction_is_exact_on_forward_mode():
    block = _two_group_block()
    base = EnergyGroupGaussSeidelPreconditioner(
        block, scatter_sweeps=1, inner_rtol=1e-13, inner_maxiter=200)
    rng = np.random.default_rng(31)
    forward = rng.random(block.shape) + 0.5
    adjoint = rng.random(block.shape) + 0.5
    coarse = AdjointCoarseCorrectionPreconditioner(
        block, base, forward, adjoint)
    got = coarse(block.apply(forward))
    np.testing.assert_allclose(got, forward, rtol=3e-12, atol=3e-12)
    assert coarse.applications == 1


def test_adjoint_rank_one_correction_rejects_singular_coordinate():
    block = _two_group_block()
    base = EnergyGroupGaussSeidelPreconditioner(block)
    forward = np.ones(block.shape)
    applied = block.apply(forward)
    # Construct q perpendicular to A*p exactly in Euclidean arithmetic.
    adjoint = np.zeros_like(applied)
    flat = adjoint.reshape(-1)
    av = applied.reshape(-1)
    flat[0], flat[1] = av[1], -av[0]
    with np.testing.assert_raises_regex(ValueError, "denominator"):
        AdjointCoarseCorrectionPreconditioner(
            block, base, forward, adjoint)


def test_energy_mode_correction_is_exact_on_group_amplitude_space():
    block = _two_group_block(cells=8)
    base = EnergyGroupGaussSeidelPreconditioner(
        block, scatter_sweeps=1, inner_rtol=1e-11, inner_maxiter=200)
    rng = np.random.default_rng(32)
    forward = rng.random(block.shape) + 0.5
    adjoint = rng.random(block.shape) + 0.5
    coarse = EnergyModeCoarseCorrectionPreconditioner(
        block, base, forward, adjoint)
    expected = np.empty_like(forward)
    amplitudes = np.array([0.7, -0.25])
    for g in range(block.groups):
        expected[g] = amplitudes[g] * coarse.forward_modes[g]
    got = coarse(block.apply(expected))
    np.testing.assert_allclose(got, expected, rtol=8e-11, atol=8e-11)
    assert coarse.coarse_unknowns == block.groups
    assert coarse.stats.setup_base_applications == block.groups
    assert coarse.stats.applications == 1


def test_energy_mode_correction_supports_output_buffer():
    block = _two_group_block()
    base = BlockJacobiPreconditioner(block.inv_diag)
    rng = np.random.default_rng(33)
    forward = rng.random(block.shape) + 0.5
    adjoint = rng.random(block.shape) + 0.5
    coarse = EnergyModeCoarseCorrectionPreconditioner(
        block, base, forward, adjoint)
    residual = rng.random(block.shape)
    expected = coarse(residual)
    out = np.empty_like(residual)
    got = coarse(residual, out=out)
    assert got is out
    np.testing.assert_allclose(got, expected, rtol=2e-15, atol=2e-15)


def test_source_shape_correction_is_exact_on_regional_amplitudes():
    block = _two_group_block(cells=8)
    base = BlockJacobiPreconditioner(block.inv_diag)
    rng = np.random.default_rng(34)
    forward = rng.random(block.shape) + 0.5
    adjoint = rng.random(block.shape) + 0.5
    masks = np.zeros((2,) + block.cell_shape)
    masks[0, :4] = 1.0
    masks[1, 4:] = 1.0
    coarse = SourceShapeCoarseCorrectionPreconditioner(
        block, base, forward, adjoint, masks)
    expected = (0.6 * coarse.trial_modes[0]
                - 0.35 * coarse.trial_modes[1])
    got = coarse(block.apply(expected))
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)
    assert coarse.coarse_unknowns == 2


def test_11_group_hpmr_driver_path_matches_fixed_point():
    """End-to-end gate with real upscatter, delayed spectra and tri geometry."""
    from ndgpu import TransientSolver, TriDiffusionEigenSolver
    from ndgpu.benchmarks.hpmr_transient_bench import (build_case,
                                                       scale_absorption)
    from ndgpu.tri import TriGroupOperator

    p, kinetics = build_case(refine=1, nz=0, groups="11")
    perturbed = scale_absorption(p.materials, 0.997065)
    dt = 0.02
    problem_at = lambda t: ((perturbed if t >= 0.5*dt else p.materials),
                            p.material_map, p.mix_material, p.mix_weight)
    steady = TriDiffusionEigenSolver(
        p.grid, p.materials, p.material_map, bc=p.bc, active=p.active,
        mask_bc=p.mask_bc, mix_material=p.mix_material,
        mix_weight=p.mix_weight, device="cpu").solve(
            tol_k=1e-9, tol_source=1e-8)

    def run(method):
        solver = TransientSolver(
            p.grid, problem_at, kinetics, bc=p.bc, active=p.active,
            mask_bc=p.mask_bc, mix_material=p.mix_material,
            mix_weight=p.mix_weight, group_operator=TriGroupOperator,
            eig_solver=TriDiffusionEigenSolver, precond_degree=1,
            device="cpu")
        kwargs = ({"multigroup_kwargs": {"rtol": 1e-10}}
                  if method == "monolithic" else
                  {"rebalance": True, "anderson_depth": 1,
                   "scatter_subsweeps": 6})
        return solver.solve(
            t_end=dt, dt=dt, tol_step=1e-9, initial_steady=steady,
            step_solver=method, **kwargs)

    fixed = run("fixed-point")
    direct = run("monolithic")
    np.testing.assert_allclose(direct.power, fixed.power,
                               rtol=1.5e-7, atol=0.0)
    np.testing.assert_allclose(direct.flux_numpy, fixed.flux_numpy,
                               rtol=2e-6, atol=2e-10)
    assert direct.total_inner_iterations < fixed.total_inner_iterations


def test_11_group_monolithic_path_survives_moving_drum_rebuilds():
    """Every control frame replaces blend fields/operators, not equations."""
    from ndgpu import TransientSolver, TriDiffusionEigenSolver
    from ndgpu.benchmarks.hpmr import build_hpmr2d
    from ndgpu.benchmarks.hpmr_thermal import (hpmr_drum_ramp,
                                               hpmr_endfb8_builtin,
                                               hpmr_kinetics_11g)
    from ndgpu.tri import TriGroupOperator

    materials = hpmr_endfb8_builtin()
    p = build_hpmr2d(refine=1, drum_angle_deg=150.0, absorber="polar",
                     materials=materials)
    problem_at = hpmr_drum_ramp(
        p, angle_from=150.0, angle_to=154.0, t_start=0.0, t_ramp=0.2,
        n_angles=5, refine=1, materials=materials)
    steady = TriDiffusionEigenSolver(
        p.grid, p.materials, p.material_map, bc=p.bc, active=p.active,
        mask_bc=p.mask_bc, mix_material=p.mix_material,
        mix_weight=p.mix_weight, device="cpu", precond_degree=1).solve(
            tol_k=1e-11, tol_source=1e-10)

    def run(method):
        solver = TransientSolver(
            p.grid, problem_at, hpmr_kinetics_11g(), bc=p.bc,
            active=p.active, mask_bc=p.mask_bc,
            mix_material=p.mix_material, mix_weight=p.mix_weight,
            group_operator=TriGroupOperator, eig_solver=TriDiffusionEigenSolver,
            precond_degree=1, device="cpu")
        controls = ({"rebalance": True, "anderson_depth": 1,
                     "scatter_subsweeps": 6} if method == "fixed-point" else
                    {"step_solver": "monolithic",
                     "multigroup_kwargs": {"scatter_sweeps": 3,
                                            "energy_anderson": 1,
                                            "inner_rtol": 0.1,
                                            "rtol": 1e-10}})
        return solver.solve(
            t_end=0.2, dt=0.05, tol_step=1e-9, max_sweeps=5000,
            initial_steady=steady, **controls)

    fixed, direct = run("fixed-point"), run("monolithic")
    np.testing.assert_allclose(direct.power, fixed.power,
                               rtol=2e-7, atol=6e-8)
    np.testing.assert_allclose(direct.flux_numpy, fixed.flux_numpy,
                               rtol=2e-6, atol=2e-10)
    assert direct.total_inner_iterations < fixed.total_inner_iterations
