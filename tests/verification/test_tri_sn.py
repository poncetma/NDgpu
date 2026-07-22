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


@pytest.mark.parametrize("mat", [M1, M2])
def test_periodic_homogeneous_is_kinf(mat):
    grid = TriGrid(shape=(6, 6, 2), side=3.0)
    r = TriSNTransportSolver(grid, mat, n_polar=2, n_azi=8,
                             bc="periodic").solve(tol_k=1e-9, tol_source=1e-9)
    assert r.converged
    assert r.k_eff == pytest.approx(k_infinite(mat), abs=2e-6)   # < 0.2 pcm
    flux = r.flux[0]
    assert flux.min() / flux.max() > 0.9999                     # flat


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
