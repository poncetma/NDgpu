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
