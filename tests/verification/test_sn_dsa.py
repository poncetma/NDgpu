"""Diffusion synthetic acceleration (DSA) and the vectorized wavefront sweep.

Three trust anchors:

  * sweep equivalence -- the wavefront (diagonal-batched) sweep reproduces the
    legacy per-direction banded row sweep to machine precision, outgoing faces
    included, on a heterogeneous problem with nonzero incoming fluxes;
  * acceleration equivalence -- every within-group acceleration ("si", "gmres",
    "dsa", "dsa-gmres") converges to the same k-eigenvalue: acceleration must
    change the iteration count, never the fixed point;
  * acceleration effectiveness -- in a scattering-dominated medium (c ~ 0.97,
    where plain source iteration contracts at the scattering ratio) DSA cuts
    the transport-sweep count by well over 5x.
"""

import numpy as np
import pytest

from ndgpu import Grid, Material, SNTransportSolver, k_infinite


def _absorber_problem(n=20):
    grid = Grid(shape=(n, n, 1), size=(float(n), float(n), 1.0))
    fuel = Material(diffusion=[1.1], sigma_a=[0.012], nu_sigma_f=[0.026],
                    sigma_s=[[0.0]], name="fuel")
    absb = Material(diffusion=[0.9], sigma_a=[0.20], nu_sigma_f=[0.0],
                    sigma_s=[[0.0]], name="absorber")
    mmap = np.zeros((n, n, 1), int)
    lo, hi = int(n * 0.4), int(n * 0.6)
    mmap[lo:hi, lo:hi, 0] = 1
    return grid, [fuel, absb], mmap


def test_wavefront_sweep_matches_row_sweep():
    rng = np.random.default_rng(7)
    nx, ny = 7, 5
    grid = Grid(shape=(nx, ny, 1), size=(7.0, 6.0, 1.0))
    fuel = Material(diffusion=[1.1, 0.4], sigma_a=[0.012, 0.1],
                    nu_sigma_f=[0.026, 0.1], sigma_s=[[0.0, 0.02], [0.0, 0.0]],
                    chi=[1.0, 0.0])
    absb = Material(diffusion=[0.9, 0.3], sigma_a=[0.20, 0.3],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.01], [0.0, 0.0]])
    mmap = rng.integers(0, 2, size=(nx, ny, 1))
    kw = dict(material_map=mmap, n_polar=2, n_azi=8, bc="vacuum")
    s_wf = SNTransportSolver(grid, [fuel, absb], **kw)
    s_rw = SNTransportSolver(grid, [fuel, absb], sweep="rows", **kw)
    q = rng.random((nx, ny))
    inc = s_wf._zero_inc()
    for f in inc:                                # nonzero incoming on every face
        inc[f] = rng.random(inc[f].shape)
    for g in range(2):
        p_wf, o_wf = s_wf._sweep(q, s_wf.st[g], inc)
        p_rw, o_rw = s_rw._sweep(q, s_rw.st[g], inc)
        assert np.max(np.abs(p_wf - p_rw)) < 1e-12
        for f in o_wf:
            assert np.max(np.abs(o_wf[f] - o_rw[f])) < 1e-12


@pytest.mark.parametrize("acceleration", ["si", "gmres", "dsa-gmres"])
def test_accelerations_agree_with_dsa(acceleration):
    grid, mats, mmap = _absorber_problem(20)
    kw = dict(material_map=mmap, n_polar=2, n_azi=8, bc="vacuum")
    tols = dict(tol_k=1e-8, tol_source=1e-7)
    k_dsa = SNTransportSolver(grid, mats, acceleration="dsa", **kw).solve(**tols)
    k_alt = SNTransportSolver(grid, mats, acceleration=acceleration,
                              **kw).solve(**tols)
    assert k_dsa.converged and k_alt.converged
    assert k_alt.k_eff == pytest.approx(k_dsa.k_eff, abs=2e-7)   # same fixed point


def test_dsa_reflective_matches_kinf():
    # Reflective boundaries exercise the frozen-incoming fixed point around the
    # (vacuum-BC) DSA error operator; the flat-flux k_inf limit must survive it.
    mat = Material(diffusion=[1.4, 0.4], sigma_a=[0.01, 0.10],
                   nu_sigma_f=[0.007, 0.13], sigma_s=[[0.0, 0.018], [0.0, 0.0]],
                   chi=[1.0, 0.0])
    grid = Grid(shape=(8, 8, 1), size=(16.0, 16.0, 1.0))
    r = SNTransportSolver(grid, mat, n_polar=2, n_azi=8, bc="reflective",
                          acceleration="dsa").solve(tol_k=1e-9, tol_source=1e-9)
    assert r.converged
    assert r.k_eff == pytest.approx(k_infinite(mat), abs=2e-5)


def test_dsa_cuts_sweep_count_in_scattering_medium():
    # c = sigma_s / sigma_t ~ 0.97: plain source iteration contracts at ~0.97
    # per sweep, DSA at <~ 0.23 c. Same answer, far fewer sweeps.
    n = 24
    grid = Grid(shape=(n, n, 1), size=(24.0, 24.0, 1.0))
    m = Material(diffusion=[1.0 / 3.0], sigma_a=[0.03], nu_sigma_f=[0.04],
                 sigma_s=[[0.0]])                        # Sigma_t = 1, c = 0.97
    # plain power outers, so the sweep count isolates the within-group scheme
    kw = dict(n_polar=2, n_azi=8, bc="vacuum", outer_acceleration="power")
    tols = dict(tol_k=1e-7, tol_source=1e-6)
    r_dsa = SNTransportSolver(grid, m, acceleration="dsa", **kw).solve(**tols)
    r_si = SNTransportSolver(grid, m, acceleration="si", **kw).solve(**tols)
    assert r_dsa.converged and r_si.converged
    assert r_dsa.k_eff == pytest.approx(r_si.k_eff, abs=1e-6)
    assert r_si.n_sweeps > 5 * r_dsa.n_sweeps


def test_cmfd_cuts_outers_at_same_answer():
    # CMFD replaces the power-iteration update with a drift-corrected diffusion
    # eigensolve: same fixed point (the drift terms reproduce the transport
    # balance exactly at convergence), far fewer transport outers. The large
    # tile keeps the dominance ratio high so the contrast is unambiguous.
    grid, mats, mmap = _absorber_problem(48)
    kw = dict(material_map=mmap, n_polar=2, n_azi=8, bc="vacuum")
    tols = dict(tol_k=1e-7, tol_source=1e-6)
    r_pow = SNTransportSolver(grid, mats, outer_acceleration="power",
                              **kw).solve(**tols)
    r_cmfd = SNTransportSolver(grid, mats, outer_acceleration="cmfd",
                               **kw).solve(**tols)
    assert r_pow.converged and r_cmfd.converged
    assert r_cmfd.k_eff == pytest.approx(r_pow.k_eff, abs=1e-6)
    assert r_cmfd.outer_iterations * 3 < r_pow.outer_iterations
    assert r_cmfd.n_sweeps * 3 < r_pow.n_sweeps


def test_cmfd_stable_with_optically_thick_cells():
    # 4 mfp per cell: plain CMFD oscillates; the odCMFD-style face damping
    # (beta += theta) must keep it convergent, and the reflective-boundary
    # k_inf composition limit must survive the CMFD outer as well.
    n = 12
    grid = Grid(shape=(n, n, 1), size=(48.0, 48.0, 1.0))
    m = Material(diffusion=[1.0 / 3.0], sigma_a=[0.2], nu_sigma_f=[0.26],
                 sigma_s=[[0.0]])                            # Sigma_t h = 4
    r = SNTransportSolver(grid, m, n_polar=3, n_azi=12, bc="vacuum",
                          outer_acceleration="cmfd").solve(tol_k=1e-7,
                                                           tol_source=1e-6)
    assert r.converged
    mat = Material(diffusion=[1.4, 0.4], sigma_a=[0.01, 0.10],
                   nu_sigma_f=[0.007, 0.13], sigma_s=[[0.0, 0.018], [0.0, 0.0]],
                   chi=[1.0, 0.0])
    grid2 = Grid(shape=(8, 8, 1), size=(16.0, 16.0, 1.0))
    r2 = SNTransportSolver(grid2, mat, n_polar=2, n_azi=8, bc="reflective",
                           outer_acceleration="cmfd").solve(tol_k=1e-9,
                                                            tol_source=1e-9)
    assert r2.converged
    assert r2.k_eff == pytest.approx(k_infinite(mat), abs=2e-5)
