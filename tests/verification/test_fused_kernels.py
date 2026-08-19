"""The fused CUDA kernels of ndgpu.kernels, checked without a GPU.

The kernels themselves only run under CuPy, but the part that is easy to get
wrong is not the CUDA -- it is the flat-index arithmetic reaching coupling
arrays that are each one cell short on a different axis. So this module
re-implements the kernel bodies in NumPy, index for index, and pins them against
the slice forms that are the reference. A stride bug shows up here on CPU rather
than in a Colab session.

- `emulate_stencil7` mirrors ``ndgpu.kernels._STENCIL7_SRC`` (Cartesian).
- `emulate_tri` mirrors ``_TRI_SRC`` / ``_TRI_Z_SRC`` (triangular: two
  interleaved sublattices, three face families with ordered (a, b) weight
  pairs, 2D as nz = 1).

Both must be updated with the kernel source they transcribe.

The order-N block operators are checked differently -- against a transcription
of the per-moment expression their fused apply replaced -- since there the risk
is the restructuring, not the indexing.

The parts that *are* backend-independent -- the fusion toggles, the CPU
fallbacks, and the `out=` variants -- are tested directly. A GPU run
additionally exercises the real kernels through the notebook's gates.
"""

import numpy as np
import pytest

from ndgpu import Grid
from ndgpu import kernels
from ndgpu.stencil import GroupOperator


def emulate_stencil7(phi, diag, wx, wy, wz, row_scale=None):
    """NumPy transcription of the fused kernel, using its exact flat indices."""
    nx, ny, nz = phi.shape
    ph, dg = phi.ravel(), diag.ravel()
    WX, WY, WZ = wx.ravel(), wy.ravel(), wz.ravel()
    rs = None if row_scale is None else np.broadcast_to(row_scale, phi.shape).ravel()
    out = np.empty(ph.size, dtype=np.result_type(phi, diag))
    for i in range(ph.size):
        k = i % nz
        j = (i // nz) % ny
        ii = i // (ny * nz)
        yz = ny * nz
        v = dg[i] * ph[i]
        if ii > 0:
            v -= WX[i - yz] * ph[i - yz]
        if ii < nx - 1:
            v -= WX[i] * ph[i + yz]
        wyb = ii * (ny - 1) * nz + k
        if j > 0:
            v -= WY[wyb + (j - 1) * nz] * ph[i - nz]
        if j < ny - 1:
            v -= WY[wyb + j * nz] * ph[i + nz]
        wzb = ii * ny * (nz - 1) + j * (nz - 1)
        if k > 0:
            v -= WZ[wzb + k - 1] * ph[i - 1]
        if k < nz - 1:
            v -= WZ[wzb + k] * ph[i + 1]
        out[i] = v if rs is None else v * rs[i]
    return out.reshape(phi.shape)


def _operator(shape=(5, 4, 3), size=(10.0, 8.0, 6.0), seed=0, **kw):
    """A GroupOperator on randomized, strongly heterogeneous data.

    Random D and removal (rather than a uniform material) is deliberate: with
    constant coefficients a transposed stride would still give the right answer
    in the interior.
    """
    rng = np.random.default_rng(seed)
    grid = Grid(shape=shape, size=size, **{k: v for k, v in kw.items()
                                           if k == "geometry"})
    D = 0.5 + rng.random(shape)
    removal = 0.1 + rng.random(shape)
    op_kw = {k: v for k, v in kw.items() if k != "geometry"}
    return op_kw, GroupOperator(np, grid, D, removal, **op_kw), rng


def _fused_args(op, dtype=np.float64):
    """The arrays the kernel would receive, via the operator's own preparation."""
    return op._fused_arrays(np.dtype(dtype))


@pytest.mark.parametrize("shape", [(5, 4, 3), (7, 1, 3), (1, 1, 9), (4, 4, 1),
                                   (2, 2, 2)])
@pytest.mark.parametrize("bc", ["zero-flux", "reflective", "vacuum"])
def test_index_arithmetic_matches_slice_form(shape, bc):
    """The kernel's flat indices reproduce apply() on every grid thickness."""
    _, op, rng = _operator(shape=shape, bc=bc)
    diag, wx, wy, wz, rs = _fused_args(op)
    phi = rng.standard_normal(shape)
    assert rs is None
    got = emulate_stencil7(phi, diag, wx, wy, wz)
    np.testing.assert_allclose(got, op.apply(phi), rtol=1e-13, atol=0)


def test_index_arithmetic_with_active_mask():
    """The excised-cell decoupling lives in the coupling arrays, not in apply."""
    shape = (6, 5, 4)
    rng = np.random.default_rng(3)
    active = rng.random(shape) > 0.3
    active[0, 0, 0] = True
    _, op, rng = _operator(shape=shape, active=active, mask_bc="vacuum")
    diag, wx, wy, wz, rs = _fused_args(op)
    phi = rng.standard_normal(shape)
    np.testing.assert_allclose(emulate_stencil7(phi, diag, wx, wy, wz),
                               op.apply(phi), rtol=1e-13, atol=0)


def test_index_arithmetic_cylindrical_divergence_form():
    """row_scale (cylindrical, symmetric=False) folds into the kernel's store."""
    shape = (6, 1, 4)                          # cylindrical (r-z) grids are 2D
    rng = np.random.default_rng(5)
    grid = Grid(shape=shape, size=(10.0, 8.0, 6.0), geometry="cylindrical")
    op = GroupOperator(np, grid, 0.5 + rng.random(shape),
                       0.1 + rng.random(shape), symmetric=False)
    diag, wx, wy, wz, rs = _fused_args(op)
    assert rs is not None
    phi = rng.standard_normal(shape)
    np.testing.assert_allclose(emulate_stencil7(phi, diag, wx, wy, wz, rs),
                               op.apply(phi), rtol=1e-13, atol=0)


def test_complex_operator_is_not_fused_against_a_real_flux():
    """A complex removal (the noise solver) must not be cast down to phi's dtype."""
    shape = (4, 4, 4)
    rng = np.random.default_rng(7)
    grid = Grid(shape=shape, size=(8.0, 8.0, 8.0))
    removal = (0.1 + rng.random(shape)) + 1j * rng.random(shape)
    op = GroupOperator(np, grid, 0.5 + rng.random(shape), removal)
    assert not op._fusable(np.dtype(np.float64))
    assert op._fusable(np.dtype(np.complex128))
    # and the generic path still promotes correctly
    assert op.apply(np.ones(shape)).dtype == np.complex128


def test_apply_into_preallocated_output():
    """apply(out=...) is what lets the Krylov loop stop allocating."""
    _, op, rng = _operator()
    phi = rng.standard_normal(op.shape)
    out = np.empty_like(phi)
    ret = op.apply(phi, out=out)
    assert ret is out
    np.testing.assert_allclose(out, op.apply(phi), rtol=0, atol=0)


def test_fusion_toggle_round_trips():
    prev = kernels.set_fused(False)
    try:
        assert not kernels.fused_enabled()
        assert not kernels.use_fused(np)          # never fused on NumPy anyway
        kernels.set_fused(True)
        assert kernels.fused_enabled()
    finally:
        kernels.set_fused(prev)


def test_fusion_groups_toggle_independently():
    """Per-group switches are what let a benchmark attribute a slowdown."""
    prev = kernels.set_fused_group("krylov", False)
    try:
        assert kernels._GROUPS["krylov"] is False
        assert kernels._GROUPS["stencil"] is True
    finally:
        kernels.set_fused_group("krylov", prev)
    with pytest.raises(ValueError):
        kernels.set_fused_group("nonesuch", True)


def test_cpu_fallbacks_match_the_expressions_they_replace():
    """The NumPy branches of the kernel helpers are the plain expressions."""
    rng = np.random.default_rng(11)
    n = 32
    x, r, p, ap, z = (rng.standard_normal(n) for _ in range(5))
    inv_diag = 1.0 / (2.0 + rng.random(n))
    alpha, beta = 0.37, -1.25

    x_ref, r_ref = x + alpha * p, r - alpha * ap
    xf, rf = x.copy(), r.copy()
    kernels.cg_update(np, xf, rf, p, ap, alpha)
    np.testing.assert_allclose(xf, x_ref, rtol=1e-14)
    np.testing.assert_allclose(rf, r_ref, rtol=1e-14)

    p_ref = z + beta * p
    pf = p.copy()
    kernels.cg_direction(np, pf, z, beta)
    np.testing.assert_allclose(pf, p_ref, rtol=1e-14)

    out_ref = x + alpha * z
    out = x.copy()
    kernels.axpy_inplace(np, out, z, alpha)
    np.testing.assert_allclose(out, out_ref, rtol=1e-14)

    weights = rng.standard_normal((3, n))
    vectors = rng.standard_normal((3, n))
    out = x.copy()
    kernels.group_accumulate(np, out, weights, vectors, alpha=-0.4)
    np.testing.assert_allclose(
        out, x - 0.4 * np.sum(weights * vectors, axis=0), rtol=1e-14)

    out = x.copy()
    kernels.product_accumulate(np, out, r, z, alpha=-0.4)
    np.testing.assert_allclose(out, x - 0.4 * r * z, rtol=1e-14)

    zf, az = z.copy(), rng.standard_normal(n)
    kernels.neumann_step(np, zf, r, az, inv_diag)
    np.testing.assert_allclose(zf, z + inv_diag * (r - az), rtol=1e-14)

    residual_low = np.empty(n, dtype=np.float32)
    z_low = np.empty(n, dtype=np.float32)
    inv_low = inv_diag.astype(np.float32)
    kernels.mixed_jacobi_start(
        np, r, inv_low, residual_low, z_low)
    np.testing.assert_array_equal(residual_low, r.astype(np.float32))
    np.testing.assert_array_equal(z_low, inv_low * residual_low)

    numerator = np.asarray(7.0)
    denominator = np.asarray(4.0)
    quotient = np.empty(())
    kernels.scalar_divide(np, quotient, numerator, denominator)
    assert quotient == 1.75
    copied = np.empty(())
    kernels.scalar_copy(np, copied, quotient)
    assert copied == quotient

    dot_out = np.empty(())
    assert kernels.dot(np, x, r, out=dot_out) is dot_out
    assert dot_out == np.sum(x * r)

    np.testing.assert_allclose(kernels.dot(np, x, r), np.sum(x * r), rtol=1e-14)


def test_sp3_block_matches_the_expression_it_replaces():
    """SP3's apply now writes each moment into its row and couples in one pass."""
    from ndgpu.sp3 import SP3GroupOperator

    shape = (6, 5, 4)
    rng = np.random.default_rng(13)
    grid = Grid(shape=shape, size=(10.0, 8.0, 6.0))
    D1 = 0.5 + rng.random(shape)
    sigma_t = 1.0 + rng.random(shape)
    removal = 0.1 + rng.random(shape)
    op = SP3GroupOperator(np, grid, D1, sigma_t, removal)

    u = rng.standard_normal((2,) + shape)
    ref = np.empty_like(u)
    ref[0] = op.moment1.apply(u[0]) - op.coupling * u[1]
    ref[1] = 5.0 * op.moment2.apply(u[1]) - op.coupling * u[0]
    np.testing.assert_allclose(op.apply(u), ref, rtol=1e-13, atol=0)


def emulate_tri(phi, diag, a_hyp, b_hyp, a_v, b_v, a_h, b_h, wz, nx, ny, nz):
    """NumPy transcription of the fused tri kernel, using its exact flat indices.

    Mirrors ``ndgpu.kernels._TRI_SRC`` / ``_TRI_Z_SRC`` line for line and must
    be updated with them. The tri layout is nastier than the Cartesian one --
    two sublattices interleaved on the third axis, three face families with
    ordered (a, b) weight pairs, and 2D handled as nz = 1 -- so this is where an
    index slip would otherwise only show up on a GPU.
    """
    ph, dg = phi.ravel(), diag.ravel()
    AH, BH = a_hyp.ravel(), b_hyp.ravel()
    AV, BV = a_v.ravel(), b_v.ravel()
    AHH, BHH = a_h.ravel(), b_h.ravel()
    WZ = None if wz is None else wz.ravel()
    out = np.empty(ph.size, dtype=np.result_type(phi, diag))
    for i in range(ph.size):
        z = i % nz
        t = (i // nz) % 2
        j = (i // (2 * nz)) % ny
        ii = i // (ny * 2 * nz)
        cidx = (ii * ny + j) * nz + z
        si, sj, st = ny * 2 * nz, 2 * nz, nz
        v = dg[i] * ph[i]
        if t == 0:
            v -= BH[cidx] * ph[i + st]
            if ii > 0:
                v -= BV[cidx - ny * nz] * ph[i - si + st]
            if j > 0:
                v -= BHH[(ii * (ny - 1) + j - 1) * nz + z] * ph[i - sj + st]
        else:
            v -= AH[cidx] * ph[i - st]
            if ii < nx - 1:
                v -= AV[cidx] * ph[i + si - st]
            if j < ny - 1:
                v -= AHH[(ii * (ny - 1) + j) * nz + z] * ph[i + sj - st]
        if WZ is not None:
            wzb = ((ii * ny + j) * 2 + t) * (nz - 1)
            if z < nz - 1:
                v -= WZ[wzb + z] * ph[i + 1]
            if z > 0:
                v -= WZ[wzb + z - 1] * ph[i - 1]
        out[i] = v
    return out.reshape(phi.shape)


def _tri_operator(nx, ny, nz=None, seed=0, **kw):
    from ndgpu.tri import TriGrid, TriGroupOperator

    rng = np.random.default_rng(seed)
    shape = (nx, ny, 2) if nz is None else (nx, ny, 2, nz)
    grid = TriGrid(shape=shape, side=1.5, height=2.0)
    D = 0.5 + rng.random(grid.shape)
    removal = 0.1 + rng.random(grid.shape)
    return TriGroupOperator(np, grid, D, removal, **kw), grid, rng


@pytest.mark.parametrize("nx,ny,nz", [(5, 4, None), (5, 4, 3), (1, 4, None),
                                      (4, 1, None), (2, 2, 1), (3, 3, 2)])
def test_tri_index_arithmetic_matches_slice_form(nx, ny, nz):
    op, grid, rng = _tri_operator(nx, ny, nz)
    args = op._fused_arrays(np.dtype(np.float64))
    phi = rng.standard_normal(grid.shape)
    got = emulate_tri(phi, *args)
    np.testing.assert_allclose(got, op.apply(phi), rtol=1e-13, atol=0)


def test_tri_index_arithmetic_with_discontinuity_factors():
    """Per-cell df makes the operator non-symmetric: a != b on every face."""
    nx, ny = 6, 5
    rng = np.random.default_rng(9)
    op, grid, _ = _tri_operator(nx, ny, seed=9,
                                df=0.7 + rng.random((nx, ny, 2)))
    assert not np.allclose(op.a_hyp, op.b_hyp)
    phi = rng.standard_normal(grid.shape)
    np.testing.assert_allclose(
        emulate_tri(phi, *op._fused_arrays(np.dtype(np.float64))),
        op.apply(phi), rtol=1e-13, atol=0)


def test_tri_supports_out_and_refuses_narrowing_dtypes():
    op, grid, rng = _tri_operator(5, 4, seed=11)
    phi = rng.standard_normal(grid.shape)
    buf = np.empty_like(phi)
    assert op.apply(phi, out=buf) is buf
    np.testing.assert_allclose(buf, op.apply(phi), rtol=0, atol=0)
    assert op._fusable(np.dtype(np.float64))

    # the noise solver's complex removal must not be cast down to a real phi
    from ndgpu.tri import TriGroupOperator

    cop = TriGroupOperator(np, grid, 0.5 + rng.random(grid.shape),
                           (0.1 + rng.random(grid.shape))
                           + 1j * rng.random(grid.shape))
    assert not cop._fusable(np.dtype(np.float64))
    assert cop._fusable(np.dtype(np.complex128))


def _block_ops():
    """One operator of each order-N block flavour, on a small grid.

    SDP3's tables admit no diagonal similarity, so its solver builds the
    congruence block; SDP2's builds the plain per-moment one. Taking them off
    the solvers keeps the coefficient plumbing honest.
    """
    from ndgpu import PWR_TWO_GROUP, SDP2EigenSolver, SDP3EigenSolver
    from ndgpu.spn import CongruentSDPNOperator, SDPNGroupOperator

    grid = Grid(shape=(6, 5, 4), size=(10.0, 8.0, 6.0))
    kw = dict(device="cpu", bc="reflective")
    ops = {"SDP2": SDP2EigenSolver(grid, PWR_TWO_GROUP, **kw).ops[0],
           "SDP3": SDP3EigenSolver(grid, PWR_TWO_GROUP, **kw).ops[0]}
    assert isinstance(ops["SDP3"], CongruentSDPNOperator)
    assert isinstance(ops["SDP2"], SDPNGroupOperator)
    return ops


def test_congruent_block_matches_the_expression_it_replaces():
    op = _block_ops()["SDP3"]
    rng = np.random.default_rng(17)
    u = rng.standard_normal((op.M,) + op._proj[0][1].shape)

    ref = np.zeros_like(u)                     # the original per-moment form
    for w, L in op._proj:
        s = None
        for i in range(op.M):
            if w[i] != 0.0:
                s = w[i] * u[i] if s is None else s + w[i] * u[i]
        Ls = L.apply(s)
        for i in range(op.M):
            if w[i] != 0.0:
                ref[i] += w[i] * Ls
    for (i, j), f in op._react.items():
        ref[i] += f * u[j]

    np.testing.assert_allclose(op.apply(u), ref, rtol=1e-12, atol=0)


def test_sdpn_block_matches_the_expression_it_replaces():
    op = _block_ops()["SDP2"]
    rng = np.random.default_rng(19)
    u = rng.standard_normal((op.M,) + op.moments[0].shape)

    ref = np.empty_like(u)
    for i in range(op.M):
        ref[i] = op.moments[i].apply(u[i])
        for j in range(op.M):
            if i != j:
                ref[i] = ref[i] + op.coupling[(i, j)] * u[j]

    np.testing.assert_allclose(op.apply(u), ref, rtol=1e-12, atol=0)


def test_block_kernel_cpu_fallbacks():
    """gather / scatter / dense-react, against the expressions they replace."""
    rng = np.random.default_rng(23)
    M, shape = 4, (5, 4, 3)
    u = rng.standard_normal((M,) + shape)
    w = np.array([0.0, 1.5, -0.25, 2.0])

    np.testing.assert_allclose(
        kernels.moment_gather(np, u, w),
        sum(w[m] * u[m] for m in range(M)), rtol=1e-14)

    out = rng.standard_normal((M,) + shape)
    s = rng.standard_normal(shape)
    ref = out + w[:, None, None, None] * s
    kernels.moment_scatter_add(np, out, s, w)
    np.testing.assert_allclose(out, ref, rtol=1e-14)

    pairs = {(0, 1): rng.standard_normal(shape), (2, 3): rng.standard_normal(shape),
             (3, 3): rng.standard_normal(shape)}
    out = rng.standard_normal((M,) + shape)
    ref = out.copy()
    for (i, j), f in pairs.items():
        ref[i] += f * u[j]
    kernels.dense_react_add(np, out, u, None, pairs)
    np.testing.assert_allclose(out, ref, rtol=1e-14)

    C = kernels.stack_pairs(np, pairs, M, shape, np.float64)
    assert C.shape == (M, M) + shape
    for (i, j), f in pairs.items():
        np.testing.assert_array_equal(C[i, j], f)
    assert C[1, 0].max() == 0.0                # unset entries stay zero


def test_operators_without_out_support_take_the_allocating_path():
    """The mesh stencil has no out=; the block operators must not assume it."""
    from ndgpu.mesh import _MeshGroupOperator
    from ndgpu.tri import TriGroupOperator

    assert not getattr(_MeshGroupOperator, "supports_out", False)
    assert GroupOperator.supports_out and TriGroupOperator.supports_out


def test_scatter_stack_matches_the_loop_it_replaces():
    """The batched in-scatter row must reproduce the per-pair Python loop.

    Including under transpose: only the adjoint solves read `sigma_s[g][gf]`,
    so a flipped index here would pass every forward test and quietly corrupt
    perturbation theory.
    """
    from ndgpu.solver import scatter_stack

    G, shape = 4, (3, 3, 2)
    rng = np.random.default_rng(29)
    # a realistic sparse pattern: downscatter plus one upscatter, some absent
    sigma_s = [[None] * G for _ in range(G)]
    for gf in range(G):
        for g in range(G):
            if g >= gf or (gf, g) == (3, 2):
                sigma_s[gf][g] = rng.random(shape)

    for adjoint in (False, True):
        S = scatter_stack(np, sigma_s, G, adjoint, shape, np.float64)
        assert S.shape == (G, G) + shape
        phi = rng.standard_normal((G,) + shape)
        for g in range(G):
            ref = np.zeros(shape)
            for gf in range(G):
                sc = sigma_s[g][gf] if adjoint else sigma_s[gf][g]
                if gf != g and sc is not None:
                    ref += sc * phi[gf]
            got = (S[g] * phi).sum(axis=0)
            np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0)
            np.testing.assert_array_equal(S[g, g], np.zeros(shape))


def test_group_batch_is_declined_on_cpu():
    """CPU keeps the sparse loop: it skips absent couplings instead of
    multiplying by materialized zeros, which is strictly better there."""
    from ndgpu import PWR_TWO_GROUP, DiffusionEigenSolver

    solver = DiffusionEigenSolver(Grid(shape=(4, 4, 4), size=(10.0, 10.0, 10.0)),
                                  PWR_TWO_GROUP, device="cpu")
    state = solver._initial_state()
    assert solver._make_group_batch(state, False, solver.nu_sigma_f) is None


def test_batched_group_loop_reproduces_the_unbatched_solve(monkeypatch):
    """End-to-end equivalence of the Phase 6 batched source assembly.

    The batched path is GPU-only, so nothing else here reaches it: `batch` is
    always None under NumPy. Forcing it on with a NumPy stand-in for the kernel
    exercises the parts that are pure bookkeeping and would be silent bugs --
    the Gauss-Seidel refresh of the stacked flux (a row must see the *updated*
    flux below g and the previous one above), the per-outer rescaling of that
    buffer alongside the state, and the batched fission source.
    """
    from ndgpu import DiffusionEigenSolver, PWR_TWO_GROUP
    from ndgpu.benchmarks import build_c5g7_2d

    prob = build_c5g7_2d(cells_per_pin=1)          # 7 groups, real upscatter-free set
    tol = dict(tol_k=1e-9, tol_source=1e-8)

    def solve():
        return DiffusionEigenSolver(prob.grid, prob.materials, prob.material_map,
                                    bc=prob.bc, device="cpu").solve(**tol)

    ref = solve()
    assert ref.converged

    def fake_accumulate(xp, out, W, P):
        for g in range(W.shape[0]):
            out += W[g] * P[g]
        return out

    monkeypatch.setattr(kernels, "group_accumulate", fake_accumulate)
    monkeypatch.setattr(kernels, "use_fused",
                        lambda xp, group=None: group == "groups")
    # Guard against this test silently becoming a no-op if the gates change:
    # assert the batched branch is actually taken before comparing.
    probe = DiffusionEigenSolver(prob.grid, prob.materials, prob.material_map,
                                 bc=prob.bc, device="cpu")
    assert probe._make_group_batch(probe._initial_state(), False,
                                   probe.nu_sigma_f) is not None
    got = solve()
    assert got.converged
    # Same arithmetic in a different association: k to well inside the solve
    # tolerance, and the same outer count (a stale flux buffer would change it).
    assert abs(got.k_eff - ref.k_eff) < 1e-9
    assert got.outer_iterations == ref.outer_iterations
    np.testing.assert_allclose(got.flux_numpy, ref.flux_numpy, rtol=1e-7)


def test_batched_matvec_matches_the_broadcast_form():
    """The 3x3 corner matvec the captured S_N sweep used to spell as a
    broadcast-multiply into an (K, 3, 3) temporary plus a reduction."""
    rng = np.random.default_rng(31)
    K, n = 7, 3
    A = rng.standard_normal((K, n, n))
    b = rng.standard_normal((K, n))
    ref = (A * b[:, None, :]).sum(axis=2)
    out = np.empty((K, n))
    kernels.batched_matvec(np, out, A, b)
    np.testing.assert_allclose(out, ref, rtol=1e-13, atol=0)
    np.testing.assert_allclose(out, np.einsum("krj,kj->kr", A, b), rtol=1e-13)


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_tri_sn_sweep_is_unchanged_by_the_fused_contractions(scheme):
    """The fused reductions in the levels sweep must not move the flux.

    They replace three weighted contractions that were written as
    broadcast-multiply + reduction to stay CUDA-graph-capturable. Both spellings
    compute the same sums, so a converged S_N solve must agree; this forces the
    fused path on under NumPy, where it is otherwise never taken.
    """
    from ndgpu.benchmarks import build_hpmr2d
    from ndgpu.tri_sn import TriSNTransportSolver

    p = build_hpmr2d(refine=4, drum_angle_deg=0.0, absorber="polar")
    kw = dict(active=p.active, bc="vacuum", scheme=scheme, engine="levels",
              device="cpu", n_polar=2, n_azi=12,
              mix_material=p.mix_material, mix_weight=p.mix_weight)

    def run(fused):
        s = TriSNTransportSolver(p.grid, p.materials, p.material_map, **kw)
        s._fused_reduce = fused          # the latched flag is the only switch
        return s, s.solve(tol_k=1e-7, tol_source=1e-6)

    s_ref, ref = run(False)
    assert not TriSNTransportSolver(
        p.grid, p.materials, p.material_map, **kw)._fused_reduce   # CPU default
    assert ref.converged

    s_new, got = run(True)
    assert s_new._fused_reduce
    assert got.converged
    assert abs(got.k_eff - ref.k_eff) < 1e-9
    np.testing.assert_allclose(np.asarray(got.flux), np.asarray(ref.flux),
                               rtol=1e-7, atol=0)
    assert got.n_sweeps == ref.n_sweeps      # same iteration path, not just same k


HAVE_CUPY = False
try:                                       # pragma: no cover - environment dependent
    import cupy as _cupy
    HAVE_CUPY = _cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    pass


@pytest.mark.skipif(not HAVE_CUPY, reason="needs a CUDA device")
@pytest.mark.parametrize("mod,attr,args", [
    ("ndgpu.sn", "SNTransportSolver", ()),
    ("ndgpu.tri_sn", "TriSNTransportSolver", (True,)),
])
def test_dsa_solver_accepts_a_host_residual(mod, attr, args):
    """The device DSA solve must take, and return, whichever backend it is given.

    The hybrid solvers are host-side and reach into a nested S_N solver for its
    DSA factor to use as a preconditioner, so "device solve, host residual" is a
    real call pattern. It used to fail deep inside ``pcg`` on ``b - apply_A(x)``
    mixing a NumPy ``b`` with a CuPy matvec.

    Genuinely GPU-only: with ``xp`` set to NumPy the factory returns the host LU
    and never reaches the branch under test, so a CPU version of this test would
    pass while checking nothing. It is skipped rather than faked.
    """
    import importlib
    import types

    import scipy.sparse as sp

    solver_cls = getattr(importlib.import_module(mod), attr)
    A = sp.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    b = np.array([1.0, 0.0])
    ref = np.linalg.solve(A.toarray(), b)

    fake = types.SimpleNamespace(xp=_cupy, dsa_on_device=True,
                                 dsa_rtol=1e-10, dsa_maxiter=500)
    solve = solver_cls._make_diff_solver.__get__(fake, solver_cls)(A, *args)

    out_host = solve(b)                    # host in -> host out
    assert isinstance(out_host, np.ndarray)
    np.testing.assert_allclose(out_host, ref, rtol=1e-6)

    out_dev = solve(_cupy.asarray(b))      # device in -> device out
    assert not isinstance(out_dev, np.ndarray)
    np.testing.assert_allclose(_cupy.asnumpy(out_dev), ref, rtol=1e-6)


def test_cmfd_lu_solver_bridges_the_bus_on_a_device_backend():
    """The default CMFD LU is a host factorization, but ``_cmfd_power`` runs on
    the solver's backend and hands it backend arrays.

    On GPU that raised ``TypeError: argument 1 must be numpy.ndarray, not
    ndarray`` from inside scipy -- a latent failure for the default
    ``cmfd_solver="lu"`` with ``device="gpu"``, since only the ``"mg"`` path was
    ever made device-resident. The bridge belongs in the factory, so the caller
    need not know where the solve lives. Exercised here with a stand-in backend,
    because the real one needs CuPy.
    """
    import types

    import scipy.sparse as sp

    from ndgpu.benchmarks import build_hpmr2d
    from ndgpu.tri_sn import TriSNTransportSolver

    p = build_hpmr2d(refine=3, drum_angle_deg=0.0, absorber="polar")
    s = TriSNTransportSolver(p.grid, p.materials, p.material_map, active=p.active,
                             bc="vacuum", engine="levels", device="cpu",
                             n_polar=2, n_azi=12, mix_material=p.mix_material,
                             mix_weight=p.mix_weight)
    A = sp.csr_matrix(np.array([[2.0, -1.0], [-1.0, 2.0]]))
    b = np.array([1.0, 0.0])

    plain = s._make_cmfd_solver(A)              # xp is numpy: no wrapper
    np.testing.assert_allclose(plain(b), np.linalg.solve(A.toarray(), b), rtol=1e-12)

    # A backend that is not numpy: the factory must wrap, and asnumpy() on a
    # numpy array is a no-op, so the wrapper round-trips cleanly here.
    s.xp = types.SimpleNamespace(asarray=np.asarray)
    wrapped = s._make_cmfd_solver(A)
    assert wrapped is not plain
    np.testing.assert_allclose(wrapped(b), np.linalg.solve(A.toarray(), b),
                               rtol=1e-12)


def test_module_of_identifies_numpy():
    assert kernels.module_of(np.zeros(3)) is np
