"""The batched multigroup source assembly against the sparse loop it replaces.

The GPU transient assembles each group's in-scatter row with one
``kernels.group_accumulate`` over a dense (G, G, \\*grid) stack instead of a
Python loop over the sparse couplings. That rewrite has exactly one way to fail
silently: getting the scattering **transpose** backwards. ``sigma_s`` is indexed
``[g_from][g_to]``, so the in-scatter row for group g gathers ``sigma_s[gf][g]``
-- and a stack built the other way round is still a plausible-looking, still
symmetric-in-shape array that produces a wrong but converged answer.

These run on NumPy, against the real 11-group HP-MR scattering matrix (which has
upscatter, so the transpose is not a symmetry and the check has teeth). They
cover the arithmetic, not the dispatch: the batch itself is GPU-only, and the
CPU solver path is unchanged by construction because ``use_fused`` is False on
NumPy.
"""

import numpy as np
import pytest

from ndgpu import kernels
from ndgpu.benchmarks.hpmr import build_hpmr2d
from ndgpu.benchmarks.hpmr_thermal import hpmr_endfb8_builtin
from ndgpu.solver import Fields, scatter_stack


@pytest.fixture(scope="module")
def fields():
    p = build_hpmr2d(refine=2, drum_angle_deg=150.0, absorber="polar",
                     materials=hpmr_endfb8_builtin())
    return Fields(np, p.grid, p.materials, p.material_map, np.float64,
                  mix_material=p.mix_material, mix_weight=p.mix_weight)


def _flux(fields, seed=0):
    rng = np.random.default_rng(seed)
    G = fields.n_groups
    return [rng.random(fields.chi[0].shape) + 0.5 for _ in range(G)]


def test_the_library_actually_has_upscatter(fields):
    """Guards the premise: with downscatter only, the transpose check below
    would pass on a matrix that is wrong above the diagonal."""
    G = fields.n_groups
    assert any(fields.sigma_s[gf][gt] is not None
               for gt in range(G) for gf in range(gt + 1, G))


def test_batched_in_scatter_row_matches_the_sparse_loop(fields):
    G = fields.n_groups
    phi = _flux(fields)
    stack = np.stack(phi)
    S = scatter_stack(np, fields.sigma_s, G, False, phi[0].shape, phi[0].dtype)

    for g in range(G):
        want = np.zeros_like(phi[0])
        for gf in range(G):
            s = fields.sigma_s[gf][g]
            if gf != g and s is not None:
                want += s * phi[gf]
        got = kernels.group_accumulate(np, np.zeros_like(phi[0]), S[g], stack)
        np.testing.assert_allclose(got, want, rtol=0, atol=0)


def test_a_transposed_stack_would_be_caught(fields):
    """The negative control: build the stack the wrong way round and assert the
    comparison above fails. Without this, the test could be passing because both
    sides are computing the same wrong thing."""
    G = fields.n_groups
    phi = _flux(fields)
    stack = np.stack(phi)
    wrong = scatter_stack(np, fields.sigma_s, G, True, phi[0].shape,
                          phi[0].dtype)   # adjoint = the transpose
    diffs = []
    for g in range(G):
        want = np.zeros_like(phi[0])
        for gf in range(G):
            s = fields.sigma_s[gf][g]
            if gf != g and s is not None:
                want += s * phi[gf]
        got = kernels.group_accumulate(np, np.zeros_like(phi[0]), wrong[g],
                                       stack)
        diffs.append(float(np.abs(got - want).max()))
    assert max(diffs) > 1e-6, "transposing the scatter stack changed nothing"


def test_batched_fission_source_matches_fields(fields):
    phi = _flux(fields, seed=1)
    W = np.stack([fields.nu_sigma_f[g] for g in range(fields.n_groups)])
    got = kernels.group_accumulate(np, np.zeros_like(phi[0]), W, np.stack(phi))
    np.testing.assert_allclose(got, fields.fission_source(phi),
                               rtol=1e-14, atol=0)


def test_batched_transient_reproduces_the_sparse_one():
    """End to end, with the batch *forced on* under NumPy.

    The unit checks above cover the arithmetic; this covers the plumbing around
    it, which has its own failure mode: with the batch active the per-group
    fluxes are **views** into one (G, \\*grid) array, so every place the solver
    writes a group flux has to write *through* the view. Rebinding the list entry
    instead would leave the batched kernel reading a stale stack -- and the run
    would still converge, to a slightly wrong answer.

    Forcing the batch on is legitimate here precisely because
    ``kernels.group_accumulate`` has a NumPy path; in production
    ``_group_batch`` gates on ``use_fused``, which is False on NumPy, so this
    configuration is unreachable outside the test.
    """
    from ndgpu import TransientSolver, TriDiffusionEigenSolver
    from ndgpu.benchmarks.hpmr_transient_bench import build_case, scale_absorption
    from ndgpu.tri import TriGroupOperator

    # 11 groups (G >= 3, as production requires for the batch to engage) on the
    # smallest mesh, and a NEGATIVE insertion: extra absorption makes the
    # within-step fixed point contract quickly, so the test costs seconds rather
    # than the hundreds of sweeps a positive step near criticality needs.
    p, kin = build_case(refine=1, nz=0, groups="11")
    dt = 0.05
    mats1 = scale_absorption(p.materials, 1.01)
    problem_at = (lambda t: ((mats1 if t >= 0.5 * dt else p.materials),
                             p.material_map, p.mix_material, p.mix_weight))

    def run(force_batch):
        s = TransientSolver(
            p.grid, problem_at, kin, bc=p.bc, active=p.active,
            mask_bc=p.mask_bc, mix_material=p.mix_material,
            mix_weight=p.mix_weight, group_operator=TriGroupOperator,
            eig_solver=TriDiffusionEigenSolver, device="cpu")
        if force_batch:
            s._group_batch = lambda fields, phi, G: {
                "phi": np.stack(phi), "fsrc": np.zeros_like(phi[0]),
                **s._batch_fields(fields, G, phi[0])}
        return s.solve(t_end=4 * dt, dt=dt, rebalance=True, anderson_depth=1,
                       max_sweeps=4000)

    sparse = run(False)
    batched = run(True)
    np.testing.assert_allclose(batched.power, sparse.power, rtol=1e-10, atol=0)
    assert batched.step_iterations == sparse.step_iterations, (
        "the batched path took a different number of sweeps -- it is not "
        "computing the same fixed point")


def test_accumulate_adds_rather_than_overwrites(fields):
    """The transient seeds the buffer with the time/delayed/fission source and
    lets the kernel add the in-scatter row on top; an overwriting kernel would
    silently drop that seed."""
    phi = _flux(fields, seed=2)
    W = np.stack([fields.nu_sigma_f[g] for g in range(fields.n_groups)])
    seed = np.full_like(phi[0], 3.25)
    got = kernels.group_accumulate(np, seed.copy(), W, np.stack(phi))
    # Compared as the sum, not by subtracting the seed back out: cancelling a
    # seed of order 1 against fission sources many orders smaller is a
    # precision question, not a correctness one.
    np.testing.assert_allclose(got, seed + fields.fission_source(phi),
                               rtol=1e-14, atol=0)
