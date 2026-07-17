"""Symmetrized SDPN blocks and the CG path.

The SDPN A-block is non-symmetric (double-PN closure), but the SDP1/SDP2
tables admit a diagonal similarity r_i = sqrt(u_i/v_i) (from the rank-1
c^(1) = u v^T) that makes every c^(m) -- and with them the assembled block and
the transient time matrix -- symmetric, so the within-group solve runs on CG.
SDP3 admits no such similarity (its c^(4) couples (2,3) but not (3,2)) and
stays on BiCGStab. The standard SPN family is symmetric as-is, including its
Marshak boundary (K = g (diag(a)+g)^-1 diag(a) is symmetric when g is), so it
defaults to CG with no transform.

These tests pin (a) exact operator self-adjointness of every path that claims
it, (b) invariance of k_eff under the change of basis and solver, and (c) the
auto-selection logic.
"""

import numpy as np
import pytest

from ndgpu import Grid, SDP2EigenSolver, SDP3EigenSolver, SP5EigenSolver
from ndgpu.operator import (SDPNGroupOperator, _SPN_C, _SPN_G, _SDPN_C,
                            _diag_similarity)

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


def _rand_fields(shape, seed=3):
    rng = np.random.default_rng(seed)
    return (0.5 + rng.random(shape), 1.0 + rng.random(shape),
            0.1 + 0.3 * rng.random(shape))


def _self_adjointness(op, M, shape, seed=11):
    rng = np.random.default_rng(seed)
    u = rng.random((M,) + shape)
    v = rng.random((M,) + shape)
    return float(np.sum(v * op.apply(u)) - np.sum(u * op.apply(v)))


@pytest.mark.parametrize("order", [1, 2])
def test_symmetrized_sdpn_operator_is_self_adjoint(order):
    grid = Grid(shape=(6, 5, 1), size=(7.0, 6.0, 1.0))
    D1, st, rem = _rand_fields(grid.shape)
    bc = (("reflective", "vacuum"), ("zero-flux", "vacuum"), "reflective")
    plain = SDPNGroupOperator(np, grid, D1, st, rem, order=order, bc=bc)
    sym = SDPNGroupOperator(np, grid, D1, st, rem, order=order, bc=bc,
                            symmetrize=True)
    assert not plain.symmetric
    assert sym.symmetric
    assert abs(_self_adjointness(plain, order + 1, grid.shape)) > 1e-8
    assert abs(_self_adjointness(sym, order + 1, grid.shape)) < 1e-10


def test_sdp3_admits_no_diagonal_similarity():
    assert _diag_similarity(_SDPN_C[3]) is None
    grid = Grid(shape=(4, 4, 1), size=(4.0, 4.0, 1.0))
    D1, st, rem = _rand_fields(grid.shape)
    with pytest.raises(ValueError, match="no diagonal symmetrizing"):
        SDPNGroupOperator(np, grid, D1, st, rem, order=3, symmetrize=True)


def test_spn_block_is_symmetric_including_marshak():
    grid = Grid(shape=(6, 5, 1), size=(7.0, 6.0, 1.0))
    D1, st, rem = _rand_fields(grid.shape)
    bc = (("reflective", "vacuum"), ("reflective", "vacuum"), "reflective")
    for bg in (None, _SPN_G[2]):
        op = SDPNGroupOperator(np, grid, D1, st, rem, order=2, bc=bc,
                               coeffs=_SPN_C, boundary_g=bg)
        assert op.symmetric
        assert abs(_self_adjointness(op, 3, grid.shape)) < 1e-10


def test_symmetrized_transient_block_is_self_adjoint():
    # theta enters A through the conjugated time matrix, so the transient
    # SDP2 block is symmetric exactly when the steady one is.
    grid = Grid(shape=(6, 5, 1), size=(7.0, 6.0, 1.0))
    D1, st, rem = _rand_fields(grid.shape)
    op = SDPNGroupOperator(np, grid, D1, st, rem, order=2, bc="zero-flux",
                           theta=0.37, symmetrize=True)
    assert op.symmetric
    assert abs(_self_adjointness(op, 3, grid.shape)) < 1e-10


def _bl_problem(n=40):
    """Small Brantley-Larsen-style heterogeneous 2D core with vacuum faces."""
    from ndgpu import Material
    fuel = Material(name="fuel", diffusion=[1.0 / 4.5], sigma_a=[0.15],
                    nu_sigma_f=[0.24], total=[1.5], chi=[1.0])
    mod = Material(name="mod", diffusion=[1.0 / 3.0], sigma_a=[0.07],
                   nu_sigma_f=[0.0], total=[1.0], chi=[1.0])
    h = 10.0 / n
    xc = (np.arange(n) + 0.5) * h
    bar = np.zeros(n, bool)
    for lo, hi in [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)]:
        bar |= (xc > lo) & (xc < hi)
    mmap = np.where(bar[:, None] & (xc < 9.0)[None, :], 0, 1)[:, :, None]
    grid = Grid(shape=(n, n, 1), size=(10.0, 10.0, h))
    bc = (("reflective", "vacuum"), ("reflective", "vacuum"), "reflective")
    return grid, [fuel, mod], mmap.astype(int), bc


@pytest.mark.parametrize("cls", [SDP2EigenSolver, SDP3EigenSolver])
def test_k_invariant_under_symmetrization_choice(cls):
    """The similarity is a change of basis: k from the symmetrized CG path
    must equal k from the plain BiCGStab path to solver tolerance. (For SDP3
    auto resolves to the plain path, so this doubles as a no-regression
    check.)"""
    grid, mats, mmap, bc = _bl_problem()
    k_auto = cls(grid, mats, material_map=mmap, bc=bc,
                 device="cpu").solve(**TIGHT).k_eff
    k_plain = cls(grid, mats, material_map=mmap, bc=bc, device="cpu",
                  symmetrize=False, linear_solver="bicgstab").solve(
        **TIGHT).k_eff
    assert k_auto == pytest.approx(k_plain, abs=2e-8)


def test_congruent_sdp3_equals_plain_operator_in_transformed_basis():
    """The congruence operator must be EXACTLY R L_plain S: same physics, new
    basis. Verified action-by-action on random states (zero-flux bc)."""
    from ndgpu.operator import (CongruentSDPNOperator, _congruence_transform)

    grid = Grid(shape=(6, 5, 1), size=(7.0, 6.0, 1.0))
    D1, st, rem = _rand_fields(grid.shape)
    bc = (("reflective", "zero-flux"), ("zero-flux", "zero-flux"), "reflective")
    plain = SDPNGroupOperator(np, grid, D1, st, rem, order=3, bc=bc)
    cong = CongruentSDPNOperator(np, grid, D1, st, rem, order=3, bc=bc)
    tr = _congruence_transform(3)
    R, S = tr["R"], tr["S"]
    rng = np.random.default_rng(5)
    v = rng.random((4,) + grid.shape)
    u = np.einsum("ij,j...->i...", S, v)
    expect = np.einsum("ij,j...->i...", R, plain.apply(u))
    got = cong.apply(v)
    assert np.allclose(got, expect, atol=1e-11), np.abs(got - expect).max()
    assert abs(_self_adjointness(cong, 4, grid.shape)) < 1e-10


def test_congruent_sdp3_rejects_vacuum_faces():
    from ndgpu.operator import CongruentSDPNOperator

    grid = Grid(shape=(4, 4, 1), size=(4.0, 4.0, 1.0))
    D1, st, rem = _rand_fields(grid.shape)
    bc = (("reflective", "vacuum"), ("reflective", "zero-flux"), "reflective")
    with pytest.raises(ValueError, match="reflective/zero-flux"):
        CongruentSDPNOperator(np, grid, D1, st, rem, order=3, bc=bc)


def test_sdp3_k_invariant_under_congruence():
    """SDP3 in the congruence basis (+ CG) reproduces the plain BiCGStab
    eigenvalue on a zero-flux heterogeneous core. Auto selects the congruence
    here (all faces reflective/zero-flux), so it must match the explicit
    symmetrize=True path as well."""
    grid, mats, mmap, _ = _bl_problem(30)
    bc = (("reflective", "zero-flux"), ("reflective", "zero-flux"),
          "reflective")
    k_plain = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc,
                              device="cpu", symmetrize=False).solve(
        **TIGHT).k_eff
    k_sym = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu",
                            symmetrize=True).solve(**TIGHT).k_eff
    k_auto = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc,
                             device="cpu").solve(**TIGHT).k_eff
    assert k_sym == pytest.approx(k_plain, abs=2e-8)
    assert k_auto == pytest.approx(k_sym, abs=2e-8)


def test_auto_selection_paths():
    grid, mats, mmap, bc = _bl_problem(20)
    sdp2 = SDP2EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu")
    assert sdp2._symmetrize and all(op.symmetric for op in sdp2.ops)
    sdp3 = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu")
    assert not sdp3._symmetrize and not any(op.symmetric for op in sdp3.ops)
    sp5 = SP5EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu")
    assert not sp5._symmetrize and all(op.symmetric for op in sp5.ops)


def test_auto_congruence_from_boundary_conditions():
    """SDP3 auto (symmetrize=None) turns the congruence basis + CG on exactly
    when every boundary the operator sees is reflective/zero-flux."""
    from ndgpu.linalg import pcg, bicgstab
    from ndgpu.operator import CongruentSDPNOperator

    grid, mats, mmap, bc_vac = _bl_problem(20)
    bc_rz = (("reflective", "zero-flux"), ("reflective", "zero-flux"),
             "reflective")

    on = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc_rz, device="cpu")
    assert on._symmetrize and on._linsolve is pcg
    assert all(isinstance(op, CongruentSDPNOperator) for op in on.ops)

    off = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc_vac,
                          device="cpu")
    assert not off._symmetrize and off._linsolve is bicgstab

    # An active mask brings in mask_bc as a seventh boundary: the default
    # (vacuum) mask boundary must veto the congruence, a reflective one not.
    active = np.ones(grid.shape, dtype=bool)
    active[-1, -1, :] = False
    masked_vac = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc_rz,
                                 active=active, device="cpu")
    assert not masked_vac._symmetrize
    masked_ref = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc_rz,
                                 active=active, mask_bc="reflective",
                                 device="cpu")
    assert masked_ref._symmetrize and masked_ref._linsolve is pcg


def test_transient_auto_congruence_from_boundary_conditions():
    from ndgpu.linalg import pcg, bicgstab
    from ndgpu.transient import Kinetics, TransientSDPNSolver

    grid, mats, mmap, bc_vac = _bl_problem(10)
    kin = Kinetics(beta=[0.0065], decay=[0.08], velocities=[2.2e5])
    problem_at = lambda t: (mats, mmap)
    tr_on = TransientSDPNSolver(grid, problem_at, kin, order=3,
                                bc="zero-flux", device="cpu")
    assert tr_on._symmetrize and tr_on._linsolve is pcg
    tr_off = TransientSDPNSolver(grid, problem_at, kin, order=3, bc=bc_vac,
                                 device="cpu")
    assert not tr_off._symmetrize and tr_off._linsolve is bicgstab


def test_spn_preconditioned_bicgstab_k_invariant():
    """The SPN-companion Neumann preconditioner must not move the eigenvalue:
    same k as plain Jacobi-BiCGStab on the vacuum-face BL core (where the
    block stays non-symmetric), for both Krylov backends."""
    grid, mats, mmap, bc = _bl_problem()
    base = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu")
    k0 = base.solve(**TIGHT).k_eff
    pre = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu",
                          spn_precondition=2)
    assert len(pre._precond_ops) == base.n_groups
    assert all(op.symmetric for op in pre._precond_ops)
    k_pre = pre.solve(**TIGHT).k_eff
    assert k_pre == pytest.approx(k0, abs=2e-8)
    k_gmres = SDP3EigenSolver(grid, mats, material_map=mmap, bc=bc,
                              device="cpu", spn_precondition=2,
                              linear_solver="gmres").solve(**TIGHT).k_eff
    assert k_gmres == pytest.approx(k0, abs=2e-8)


def test_spn_precondition_marshak_and_symmetric_guard():
    grid, mats, mmap, bc = _bl_problem(20)
    # With Marshak the SDP2 block is non-symmetric; the companion carries
    # SPN's own (symmetric) g and k must be unchanged.
    base = SDP2EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu",
                           marshak_vacuum=True)
    k0 = base.solve(**TIGHT).k_eff
    k_pre = SDP2EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu",
                            marshak_vacuum=True, spn_precondition=2).solve(
        **TIGHT).k_eff
    assert k_pre == pytest.approx(k0, abs=2e-8)
    # On a symmetric/symmetrized path the companion lives in the wrong basis:
    # the combination is rejected.
    with pytest.raises(ValueError, match="non-symmetric SDPN path"):
        SDP2EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu",
                        spn_precondition=1)
    with pytest.raises(ValueError, match="non-symmetric SDPN path"):
        SP5EigenSolver(grid, mats, material_map=mmap, bc=bc, device="cpu",
                       spn_precondition=1)
