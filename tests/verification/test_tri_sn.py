"""Discrete-ordinates (S_N) transport on the triangular mesh (TriSNTransportSolver).

S_N on the body-fitted hex/triangular geometry, so transport runs on the actual
HP-MR core rather than a Cartesian stand-in. The per-ordinate streaming+collision
operator is assembled sparse (upwind/step differencing) and factorized once; a
sweep is a triangular solve.

Checks:
  * composition -- a homogeneous medium on a periodic lattice (a torus, i.e. an
    infinite medium) gives exactly k_inf with a flat flux. Flat flux makes the
    streaming term vanish edge-by-edge (the triangle's outward normals sum to
    zero), so this is exact and independently pins the geometry (edge normals,
    neighbour offsets) and the eigenvalue outer;
  * leakage -- a finite homogeneous tile with vacuum boundaries leaks, so
    k < k_inf, and refining the mesh (less numerical diffusion) moves k up
    toward it;
  * the n_azi = multiple-of-4 and bc guards.
"""

import numpy as np
import pytest

from ndgpu import Material, k_infinite
from ndgpu.tri import TriGrid
from ndgpu.tri_sn import TriSNTransportSolver

M1 = Material(diffusion=[1.0], sigma_a=[0.02], nu_sigma_f=[0.025],
              sigma_s=[[0.0]], name="m1")
M2 = Material(diffusion=[1.4, 0.4], sigma_a=[0.01, 0.10],
              nu_sigma_f=[0.007, 0.13], sigma_s=[[0.0, 0.018], [0.0, 0.0]],
              chi=[1.0, 0.0], name="m2")


@pytest.mark.parametrize("scheme", ["step", "scb"])
@pytest.mark.parametrize("mat", [M1, M2])
def test_periodic_homogeneous_is_kinf(mat, scheme):
    # Flat flux makes both schemes exact -- for SCB because each corner is a
    # closed sub-volume (its face normals sum to zero), so streaming vanishes.
    grid = TriGrid(shape=(6, 6, 2), side=3.0)
    r = TriSNTransportSolver(grid, mat, n_polar=2, n_azi=8, bc="periodic",
                             scheme=scheme).solve(tol_k=1e-9, tol_source=1e-9)
    assert r.converged
    assert r.k_eff == pytest.approx(k_infinite(mat), abs=2e-6)   # < 0.2 pcm
    flux = r.flux[0]
    assert flux.min() / flux.max() > 0.9999                     # flat


def test_scb_converges_faster_than_step():
    # On a smooth vacuum tile both schemes approach the same fine-mesh limit, but
    # the second-order SCB is closer to it than first-order step at equal mesh.
    m = Material(diffusion=[1.0], sigma_a=[0.05], nu_sigma_f=[0.07],
                 sigma_s=[[0.0]], name="smooth")
    size = 12.0

    def k(scheme, nrc):
        g = TriGrid(shape=(nrc, nrc, 2), side=size / nrc)
        return TriSNTransportSolver(g, m, n_polar=2, n_azi=8, bc="vacuum",
                                    scheme=scheme).solve(tol_k=1e-8,
                                                         tol_source=1e-7).k_eff

    ref = k("scb", 32)                                          # fine reference
    e_step = abs(k("step", 8) - ref)
    e_scb = abs(k("scb", 8) - ref)
    # On this small, boundary-layer-dominated tile the observed order is
    # pre-asymptotic, so SCB is ~1.7x more accurate here; the full second-order
    # payoff (correct HP-MR drum-worth sign two refinements sooner than step) is
    # in examples/hpmr_tri_sn.py.
    assert e_scb < 0.7 * e_step


def test_vacuum_tile_leaks_and_refines_toward_kinf():
    kinf = k_infinite(M1)
    ks = []
    for nrc in (4, 8):
        grid = TriGrid(shape=(nrc, nrc, 2), side=8.0 / nrc)      # same physical size
        r = TriSNTransportSolver(grid, M1, n_polar=2, n_azi=8,
                                 bc="vacuum").solve(tol_k=1e-6, tol_source=1e-5)
        assert r.converged
        ks.append(r.k_eff)
    assert ks[0] < kinf and ks[1] < kinf                        # leakage
    assert ks[1] > ks[0]                                        # refines up toward k_inf


def test_bad_n_azi_rejected():
    grid = TriGrid(shape=(4, 4, 2), side=3.0)
    with pytest.raises(ValueError, match="multiple of 4"):
        TriSNTransportSolver(grid, M1, n_polar=2, n_azi=6)


def test_bad_bc_rejected():
    grid = TriGrid(shape=(4, 4, 2), side=3.0)
    with pytest.raises(ValueError, match="vacuum.*periodic|periodic"):
        TriSNTransportSolver(grid, M1, n_polar=2, n_azi=8, bc="reflective")


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_dsa_matches_gmres_and_cuts_sweeps(scheme):
    # Scattering-dominated 2-group tile: every within-group acceleration must
    # converge to the same k (acceleration changes the iteration count, never
    # the fixed point), and DSA must beat plain source iteration by >5x sweeps.
    m = Material(diffusion=[1.4, 0.4], sigma_a=[0.005, 0.015],
                 nu_sigma_f=[0.004, 0.02], sigma_s=[[0.0, 0.025], [0.0, 0.0]],
                 chi=[1.0, 0.0], name="soft")
    n = 10
    grid = TriGrid(shape=(n, n, 2), side=24.0 / n)
    tols = dict(tol_k=1e-7, tol_source=1e-6)

    def run(acc):
        # plain power outers so the sweep count isolates the within-group scheme
        return TriSNTransportSolver(grid, m, n_polar=2, n_azi=8, bc="vacuum",
                                    scheme=scheme, acceleration=acc,
                                    outer_acceleration="power").solve(**tols)

    r_dsa = run("dsa")
    r_gm = run("gmres")
    r_si = run("si")
    r_pg = run("dsa-gmres")
    assert all(r.converged for r in (r_dsa, r_gm, r_si, r_pg))
    for r in (r_gm, r_si, r_pg):
        assert r.k_eff == pytest.approx(r_dsa.k_eff, abs=1e-6)
    assert r_si.n_sweeps > 5 * r_dsa.n_sweeps


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_cmfd_outer_matches_power_with_fewer_outers(scheme):
    # CMFD replaces the Anderson power update with a drift-corrected diffusion
    # eigensolve built from the schemes' own (conservative) face currents:
    # same fixed point, fewer transport outers.
    m = Material(diffusion=[1.4, 0.4], sigma_a=[0.005, 0.015],
                 nu_sigma_f=[0.004, 0.02], sigma_s=[[0.0, 0.025], [0.0, 0.0]],
                 chi=[1.0, 0.0], name="soft")
    n = 10
    grid = TriGrid(shape=(n, n, 2), side=24.0 / n)
    tols = dict(tol_k=1e-7, tol_source=1e-6)

    def run(outer):
        return TriSNTransportSolver(grid, m, n_polar=2, n_azi=8, bc="vacuum",
                                    scheme=scheme,
                                    outer_acceleration=outer).solve(**tols)

    r_pow = run("power")
    r_cmfd = run("cmfd")
    assert r_pow.converged and r_cmfd.converged
    assert r_cmfd.k_eff == pytest.approx(r_pow.k_eff, abs=1e-6)
    assert r_cmfd.outer_iterations < r_pow.outer_iterations


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_levels_engine_matches_lu(scheme):
    # The level-scheduled (GPU-oriented) sweep solves the same per-ordinate
    # systems as the LU engine in topological order: machine-precision equal
    # sweeps, current folds, and k, on a heterogeneous ragged-mask problem.
    rng = np.random.default_rng(5)
    fuel = Material(diffusion=[1.1, 0.4], sigma_a=[0.012, 0.1],
                    nu_sigma_f=[0.026, 0.1], sigma_s=[[0.0, 0.02], [0.0, 0.0]],
                    chi=[1.0, 0.0])
    absb = Material(diffusion=[0.9, 0.3], sigma_a=[0.20, 0.3],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.01], [0.0, 0.0]])
    nr = nc = 8
    grid = TriGrid(shape=(nr, nc, 2), side=1.5)
    mmap = rng.integers(0, 2, size=(nr, nc, 2))
    active = np.ones((nr, nc, 2), bool)
    active[0, :3, :] = False
    src = rng.random(nr * nc * 2)
    kw = dict(material_map=mmap, active=active, n_polar=2, n_azi=8,
              bc="vacuum", scheme=scheme)
    s_lu = TriSNTransportSolver(grid, [fuel, absb], engine="lu", **kw)
    s_lv = TriSNTransportSolver(grid, [fuel, absb], engine="levels", **kw)
    for g in range(2):
        assert np.max(np.abs(s_lu._sweep(g, src) - s_lv._sweep(g, src))) < 1e-12
        _, J1 = s_lu._sweep_currents(g, src)
        _, J2 = s_lv._sweep_currents(g, src)
        assert np.max(np.abs(J1 - J2)) < 1e-12
    r_lu = s_lu.solve(tol_k=1e-8, tol_source=1e-7)
    r_lv = s_lv.solve(tol_k=1e-8, tol_source=1e-7)
    assert r_lu.converged and r_lv.converged
    assert r_lv.k_eff == pytest.approx(r_lu.k_eff, abs=1e-10)


def test_levels_engine_rejects_periodic():
    grid = TriGrid(shape=(4, 4, 2), side=3.0)
    with pytest.raises(ValueError, match="cycles"):
        TriSNTransportSolver(grid, M1, n_polar=2, n_azi=8, bc="periodic",
                             engine="levels")


# ---- 3D: S_N on extruded triangular prisms (Phase 1, step differencing) -----

@pytest.mark.parametrize("mat", [M1, M2])
def test_3d_periodic_homogeneous_is_kinf(mat):
    # Full 3D torus (radial + axial periodic): flat flux, every streaming face
    # cancels (the prism's face normal-areas sum to zero), so k == k_inf exactly.
    # This pins the whole prism operator -- lateral tri edges AND axial caps.
    grid = TriGrid(shape=(6, 6, 2, 4), side=3.0, height=8.0)
    r = TriSNTransportSolver(grid, mat, n_polar=4, n_azi=8,
                             bc="periodic").solve(tol_k=1e-9, tol_source=1e-9)
    assert r.converged
    assert r.k_eff == pytest.approx(k_infinite(mat), abs=2e-6)
    flux = r.flux[0]
    assert flux.min() / flux.max() > 0.9999                      # flat in x,y,z


def test_3d_axial_slab_converges_to_diffusion():
    # radial-periodic + axial-vacuum reduces to a 1D axial slab (flux flat in
    # plane), so the ONLY leakage is axial -- this pins the axial cap scale.
    # As the slab thickens the axial leakage drops, so S_N (true transport)
    # converges to BOTH k_inf and the diffusion answer; the gap closes. A wrong
    # axial scale could not converge on both. DSA makes the low-leakage slab
    # tractable (plain source iteration stalls near dominance ratio 1).
    from ndgpu.tri import TriDiffusionEigenSolver
    mat = Material(diffusion=[1.0], sigma_a=[0.004], nu_sigma_f=[0.0055],
                   sigma_s=[[0.0]], name="d")
    kinf = k_infinite(mat)
    gaps = []
    for H, nz in [(60.0, 10), (240.0, 40)]:
        grid = TriGrid(shape=(2, 2, 2, nz), side=4.0, height=H)
        kd = TriDiffusionEigenSolver(
            grid, mat, bc=("reflective", "reflective", "vacuum"),
            device="cpu").solve(tol_k=1e-8, tol_source=1e-7).k_eff
        sn = TriSNTransportSolver(grid, mat, n_polar=6, n_azi=4,
                                  bc=("periodic", "vacuum")).solve(
            tol_k=1e-7, tol_source=1e-6, max_outer=800)
        assert sn.converged                      # DSA converges the low-leak slab
        assert sn.k_eff < kd < kinf              # transport leaks more; both < k_inf
        gaps.append(abs(sn.k_eff - kd))
    assert gaps[1] < 0.4 * gaps[0]               # gap closes as leakage drops


def test_3d_dsa_matches_source_iteration():
    # 3D DSA (Phase 3) must reproduce the plain source-iteration eigenvalue and
    # cut the outer count. Same operator, different within-group accelerator.
    grid = TriGrid(shape=(3, 3, 2, 6), side=3.0, height=36.0)
    kw = dict(n_polar=4, n_azi=4, bc="vacuum")
    si = TriSNTransportSolver(grid, M1, acceleration="si", **kw).solve(
        tol_k=1e-8, tol_source=1e-7, max_outer=2000)
    dsa = TriSNTransportSolver(grid, M1, acceleration="dsa", **kw).solve(
        tol_k=1e-8, tol_source=1e-7, max_outer=2000)
    assert si.converged and dsa.converged
    assert dsa.k_eff == pytest.approx(si.k_eff, abs=5e-6)
    assert dsa.outer_iterations <= si.outer_iterations


def test_3d_rejects_scb_and_levels():
    grid = TriGrid(shape=(4, 4, 2, 2), side=3.0, height=4.0)
    with pytest.raises(NotImplementedError, match="step"):
        TriSNTransportSolver(grid, M1, n_polar=2, n_azi=8, scheme="scb")


def test_3d_levels_engine_matches_lu():
    # The 3D level-scheduled sweep engine (engine="levels", the GPU path) solves
    # the same per-ordinate systems as the prism LU engine in topological order,
    # now with axial dependency edges (up to 5 inflow faces/cell). On NumPy both
    # run the identical arithmetic -> machine-precision equal.
    grid = TriGrid(shape=(5, 5, 2, 4), side=3.0, height=16.0)
    rng = np.random.default_rng(0)
    src = rng.random(5 * 5 * 2 * 4)
    kw = dict(n_polar=4, n_azi=4, bc="vacuum")
    lu = TriSNTransportSolver(grid, M1, engine="lu", **kw)
    lv = TriSNTransportSolver(grid, M1, engine="levels", **kw)
    assert np.max(np.abs(lu._sweep(0, src) - lv._sweep(0, src))) < 1e-12
    r_lu = lu.solve(tol_k=1e-8, tol_source=1e-7)
    r_lv = lv.solve(tol_k=1e-8, tol_source=1e-7)
    assert r_lu.converged and r_lv.converged
    assert r_lv.k_eff == pytest.approx(r_lu.k_eff, abs=1e-10)


def test_3d_levels_rejects_periodic():
    # Any periodic wrap (radial or axial) cycles the sweep dependency graph.
    grid = TriGrid(shape=(4, 4, 2, 3), side=3.0, height=9.0)
    with pytest.raises(ValueError, match="cycles"):
        TriSNTransportSolver(grid, M1, n_polar=2, n_azi=8,
                             bc=("periodic", "vacuum"), engine="levels")
