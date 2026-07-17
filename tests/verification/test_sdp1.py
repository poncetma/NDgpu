"""Simplified double-P1 (SDP1) solver verification.

SDP1 is the N=1 simplified double-PN approximation of Carreno, Vidal-Ferrandiz,
Ginestar & Verdu, Ann. Nucl. Energy 207 (2024) 110675. In ndgpu's symmetrized
two-moment (Phi1 = phi0 + 2 phi2, phi2) block it is algebraically identical to
SP3 except for the second-moment diffusion coefficient:

    SP3  (Brantley & Larsen):  D2 = 9/(35 Sigma_3)
    SDP1 (Carreno et al.):     D2 = 1/(5  Sigma_3)  = 7/9 * D2_SP3

These tests pin that relationship down and check the physics the paper reports:
SDP1 is a distinct, equal-cost transport approximation that pulls k further from
diffusion toward the transport reference than SP3 in strongly heterogeneous
media, while reducing exactly to diffusion (k_infinity) in an infinite medium.
"""

import numpy as np
import pytest

from ndgpu import (
    Grid,
    Material,
    PWR_TWO_GROUP,
    DiffusionEigenSolver,
    SP1EigenSolver,
    SP3EigenSolver,
    SP5EigenSolver,
    SP7EigenSolver,
    SDP1EigenSolver,
    SDP2EigenSolver,
    SDP3EigenSolver,
    k_infinite,
)
from ndgpu.operator import SP3GroupOperator
from ndgpu.solver import SDPNEigenSolver, SPNEigenSolver

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


# --- 1D BWR-like heterogeneous slab (cross sections from Table 1 of the paper,
# Rahnema & Nichita 1997): fuel assemblies separated by water gaps, the regime
# where the paper shows SDP1 beating SP3. ---------------------------------------

def _mat(name, D, sa, nsf, s12):
    return Material(name=name, diffusion=D, sigma_a=sa, nu_sigma_f=nsf,
                    sigma_s=[[0.0, s12], [0.0, 0.0]], chi=[1.0, 0.0])


def _hetero_slab(cells_per_cm=10):
    fuel = _mat("fuel", [1.4730, 0.3294], [0.0096, 0.0764],
                [0.0067, 0.1241], 0.0161)
    water = _mat("water", [1.7639, 0.2278], [0.0003, 0.0097],
                 [0.0, 0.0], 0.0380)
    seq = [water, fuel, water, fuel, water, fuel, water]
    thick = [1.158, 3.231, 1.158, 3.231, 1.158, 3.231, 1.158]
    idx = []
    for m, t in zip(seq, thick):
        idx += [0 if m is water else 1] * int(round(t * cells_per_cm))
    nx = len(idx)
    mmap = np.array(idx, dtype=int).reshape(nx, 1, 1)
    grid = Grid(shape=(nx, 1, 1), size=(sum(thick), 1.0, 1.0))
    return grid, [water, fuel], mmap


def _solve(solver_cls, grid, mats, mmap):
    r = solver_cls(grid, mats, material_map=mmap, bc="reflective",
                   device="cpu").solve(**TIGHT)
    assert r.converged, r
    return r.k_eff


# ------------------------------------------------------------------ operator ---

def test_sdp1_changes_only_the_second_moment_diffusion_by_7_over_9():
    """SDP1 leaves moment 1, the coupling and the moment-2 reaction identical to
    SP3, and scales the moment-2 leakage by exactly 7/9 (D2 = 1/(5 St) vs
    9/(35 St)). Probe with Phi1 = 0 so out[1] = 5*(leak + reaction*phi2)."""
    xp = np
    n = 12
    grid = Grid(shape=(n, 1, 1), size=(10.0, 1.0, 1.0))
    ones = np.ones(grid.shape)
    D1 = 0.9 * ones
    sigma_t = 1.3 * ones
    removal = 0.7 * ones
    reaction = sigma_t + 0.8 * removal  # moment-2 removal, same in both variants

    common = dict(bc="reflective")
    op3 = SP3GroupOperator(xp, grid, D1, sigma_t, removal, variant="sp3", **common)
    opd = SP3GroupOperator(xp, grid, D1, sigma_t, removal, variant="sdp1", **common)

    # A non-flat phi2 so the leakage (divergence of a gradient) is nonzero.
    phi2 = np.linspace(0.0, 1.0, n).reshape(n, 1, 1) ** 2
    u = np.stack([np.zeros(grid.shape), phi2])

    out3, outd = op3.apply(u), opd.apply(u)

    # Moment 1 (and the coupling, driven here only by phi2) is untouched.
    assert np.allclose(out3[0], outd[0], atol=1e-12)

    # Isolate the leakage operator: out[1]/5 - reaction*phi2.
    leak3 = out3[1] / 5.0 - reaction * phi2
    leakd = outd[1] / 5.0 - reaction * phi2
    assert np.any(np.abs(leak3) > 1e-6)  # leakage actually exercised
    assert np.allclose(leakd, (7.0 / 9.0) * leak3, rtol=1e-10, atol=1e-12)


def test_bad_variant_rejected():
    grid = Grid(shape=(4, 1, 1), size=(4.0, 1.0, 1.0))
    ones = np.ones(grid.shape)
    with pytest.raises(ValueError, match="variant"):
        SP3GroupOperator(np, grid, ones, ones, 0.5 * ones, variant="sp5")


# -------------------------------------------------------------------- solver ---

def test_sdp1_reflective_reduces_to_k_infinity():
    """In an infinite medium phi2 = 0 (flat flux), so the second-moment
    coefficient is irrelevant and SDP1 -- like SP3 -- collapses to diffusion."""
    grid = Grid(shape=(8, 8, 8), size=(90.0, 90.0, 90.0))
    r = SDP1EigenSolver(grid, PWR_TWO_GROUP, bc="reflective",
                        device="cpu").solve(**TIGHT)
    assert r.k_eff == pytest.approx(k_infinite(PWR_TWO_GROUP), abs=1e-6)


def test_sdp1_is_distinct_from_sp3_and_diffusion_on_a_finite_core():
    """With leakage present the three approximations give three different k."""
    grid = Grid(shape=(24, 24, 24), size=(40.0, 40.0, 40.0))
    kw = dict(device="cpu")
    kd = DiffusionEigenSolver(grid, PWR_TWO_GROUP, **kw).solve(**TIGHT).k_eff
    k3 = SP3EigenSolver(grid, PWR_TWO_GROUP, **kw).solve(**TIGHT).k_eff
    kdp = SDP1EigenSolver(grid, PWR_TWO_GROUP, **kw).solve(**TIGHT).k_eff
    assert abs(kdp - k3) > 1e-5, "SDP1 and SP3 unexpectedly identical"
    assert abs(kdp - kd) > 1e-5, "SDP1 and diffusion unexpectedly identical"


def test_sdp1_beats_sp3_toward_transport_in_heterogeneous_media():
    """The paper's headline result (1D BWR): both transport approximations pull
    k well below the leakage-free diffusion estimate, and SDP1 pulls slightly
    further than SP3 -- i.e. diffusion > SP3 > SDP1 for this heterogeneous
    slab."""
    grid, mats, mmap = _hetero_slab()
    kd = _solve(DiffusionEigenSolver, grid, mats, mmap)
    k3 = _solve(SP3EigenSolver, grid, mats, mmap)
    kdp = _solve(SDP1EigenSolver, grid, mats, mmap)
    assert kd > k3 > kdp, f"expected diffusion > SP3 > SDP1, got {kd}, {k3}, {kdp}"
    # The SP3->SDP1 shift is the transport correction from the 7/9 coefficient:
    # a real, few-tens-of-pcm effect, much smaller than the diffusion->SP3 gap.
    assert 5.0 < (k3 - kdp) * 1e5 < (kd - k3) * 1e5


# ------------------------------------------------------- higher orders SDP2/3 ---

class _SDP1General(SDPNEigenSolver):
    """SDP1 through the general U-form path (non-symmetric, BiCGStab)."""
    _order = 1


def test_general_uform_order1_matches_symmetric_sdp1():
    """The general M-moment U-form (used for SDP2/SDP3) reproduces the
    independently derived symmetric two-moment SDP1 block to round-off,
    validating the coefficient/source-distribution recipe at order 1."""
    grid, mats, mmap = _hetero_slab()
    k_sym = _solve(SDP1EigenSolver, grid, mats, mmap)
    k_gen = _solve(_SDP1General, grid, mats, mmap)
    assert k_gen == pytest.approx(k_sym, abs=1e-6)


@pytest.mark.parametrize("solver_cls", [SDP2EigenSolver, SDP3EigenSolver])
def test_higher_sdpn_reduce_to_k_infinity(solver_cls):
    """In an infinite medium every higher moment vanishes, so SDP2/SDP3 must
    collapse to k_infinity exactly -- a check that the M-moment coupling and the
    phi0 = c1[0].U recovery are mutually consistent."""
    grid = Grid(shape=(6, 6, 6), size=(90.0, 90.0, 90.0))
    r = solver_cls(grid, PWR_TWO_GROUP, bc="reflective",
                   device="cpu").solve(**TIGHT)
    assert r.k_eff == pytest.approx(k_infinite(PWR_TWO_GROUP), abs=1e-6)


@pytest.mark.parametrize("base", [SDPNEigenSolver, SPNEigenSolver])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_uform_power_iteration_matches_dense_eigensolve(base, order):
    """Gold standard, both families (SDPN and standard SPN = SP3/SP5/SP7): the
    Krylov power iteration reproduces a dense direct generalized eigensolve of
    the assembled M-moment block, for every order. Pins the operator (leakage +
    reaction coupling) and the fission/source wiring against an independent
    linear-algebra path."""
    sla = pytest.importorskip("scipy.linalg")
    fuel = Material(name="fuel", diffusion=[1.0], sigma_a=[0.1],
                    nu_sigma_f=[0.13], sigma_s=[[0.0]], chi=[1.0])
    n = 12
    grid = Grid(shape=(n, 1, 1), size=(20.0, 1.0, 1.0))

    class S(base):
        _order = order

    solv = S(grid, fuel, bc="zero-flux", device="cpu")
    M = order + 1
    op = solv.ops[0]
    N = M * n
    A = np.empty((N, N))
    for c in range(N):
        e = np.zeros(N)
        e[c] = 1.0
        A[:, c] = np.asarray(op.apply(e.reshape(M, n, 1, 1))).reshape(N)
    sw = np.array(op.src_weights)
    pw = np.array(op.phi0_weights)
    nsf = float(fuel.nu_sigma_f[0])
    F = np.zeros((N, N))
    for cell in range(n):
        for i in range(M):
            for j in range(M):
                F[i * n + cell, j * n + cell] = sw[i] * nsf * pw[j]
    w = sla.eig(np.linalg.solve(A, F), right=False)
    w = w[np.abs(w.imag) < 1e-9].real
    k_dense = float(np.max(w[w > 0]))
    k_pi = solv.solve(tol_k=1e-11, tol_source=1e-10).k_eff
    assert k_pi == pytest.approx(k_dense, abs=1e-6)


# ---------------------------------------------------- standard SPN (SP5/SP7) ---

class _SP3General(SPNEigenSolver):
    """SP3 through the general SPN U-form (order 1)."""
    _order = 1


def test_spn_uform_order1_matches_dedicated_sp3():
    """The general SPN U-form at order 1 reproduces ndgpu's dedicated symmetric
    SP3 solver -- validating the standard SPN coefficient matrices that SP5/SP7
    reuse."""
    grid, mats, mmap = _hetero_slab()
    assert _solve(_SP3General, grid, mats, mmap) == pytest.approx(
        _solve(SP3EigenSolver, grid, mats, mmap), abs=1e-6)


@pytest.mark.parametrize("solver_cls", [SP5EigenSolver, SP7EigenSolver])
def test_spn_reduce_to_k_infinity(solver_cls):
    """SP5/SP7 collapse to k_infinity exactly in an infinite medium."""
    grid = Grid(shape=(6, 6, 6), size=(90.0, 90.0, 90.0))
    r = solver_cls(grid, PWR_TWO_GROUP, bc="reflective",
                   device="cpu").solve(**TIGHT)
    assert r.k_eff == pytest.approx(k_infinite(PWR_TWO_GROUP), abs=1e-6)


@pytest.mark.parametrize("bc", ["zero-flux", "vacuum"])
def test_sp1_matches_diffusion(bc):
    """SP1 (the order-0 U-form) is the P1/diffusion equation, so it must
    reproduce DiffusionEigenSolver on a finite heterogeneous core under every
    boundary law -- an end-to-end check that the block machinery adds nothing
    at M = 1."""
    grid, mats, mmap = _hetero_slab(cells_per_cm=4)
    k = {}
    for name, cls in [("diffusion", DiffusionEigenSolver),
                      ("sp1", SP1EigenSolver)]:
        r = cls(grid, mats, material_map=mmap, bc=bc,
                device="cpu").solve(**TIGHT)
        assert r.converged, (name, r)
        k[name] = r.k_eff
    assert k["sp1"] == pytest.approx(k["diffusion"], abs=1e-8)


def test_sp1_marshak_reduces_to_robin_vacuum():
    """At M = 1 the coupled Marshak boundary degenerates to the alpha = 1/2
    Robin term, so SP1 with marshak_vacuum=True equals diffusion with the
    plain 'vacuum' bc."""
    grid, mats, mmap = _hetero_slab(cells_per_cm=4)
    k_dif = DiffusionEigenSolver(grid, mats, material_map=mmap, bc="vacuum",
                                 device="cpu").solve(**TIGHT).k_eff
    k_sp1 = SP1EigenSolver(grid, mats, material_map=mmap, bc="vacuum",
                           marshak_vacuum=True, device="cpu").solve(
        **TIGHT).k_eff
    assert k_sp1 == pytest.approx(k_dif, abs=1e-8)
