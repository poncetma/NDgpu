"""OECD/NEA C5G7 MOX benchmark (2D quarter core), pin-cell homogenized.

Checks the geometry construction (pin counts, symmetry, reaction-rate
conservation of the homogenization), that homogenized diffusion lands within
~2% of the published transport reference K_REFERENCE_2D, and that the whole
SPN/SDPN angular hierarchy behaves physically on C5G7. The hierarchy test
deliberately instantiates every solver so it always exercises the *current*
coefficient tables (_SPN_C/_SDPN_C in ndgpu.operator) -- coefficient
regressions (like the paper's SDP2/SDP3 table errors) surface here as broken
ordering or a blown eigenvalue, without pinning any solver output value.
"""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, SP3EigenSolver, SP5EigenSolver,
                   SP7EigenSolver, SDP1EigenSolver, SDP2EigenSolver,
                   SDP3EigenSolver)
from ndgpu.benchmarks import build_c5g7_2d
from ndgpu.benchmarks.c5g7 import FUEL_FRACTION, K_REFERENCE_2D


def test_c5g7_geometry():
    prob = build_c5g7_2d(cells_per_pin=2)
    assert prob.grid.shape == (102, 102, 1)
    assert prob.material_map.shape == (102, 102, 1)
    assert len(prob.materials) == 7
    # 4 assemblies x (264 fuel + 24 GT + 1 FC) pin cells; the rest is water.
    counts = np.bincount(prob.pin_map.ravel(), minlength=7)
    assert counts[4] == 4 * 24            # guide tubes
    assert counts[5] == 4 * 1             # fission chambers
    assert counts[:4].sum() == 4 * 264    # fuel pins
    assert counts[6] == 51 * 51 - 4 * 289
    # quarter-core diagonal symmetry of the layout
    assert np.array_equal(prob.pin_map, prob.pin_map.T)
    # homogenization conserved reaction rates at the mixing level
    uo2_cell = prob.materials[0]
    from ndgpu.benchmarks._c5g7_data import C5G7_XS
    expected = FUEL_FRACTION * np.array(C5G7_XS["UO2"]["nu_fission"])
    assert np.allclose(uo2_cell.nu_sigma_f, expected)


def test_c5g7_solves_to_reasonable_k():
    prob = build_c5g7_2d(cells_per_pin=1)
    res = DiffusionEigenSolver(prob.grid, prob.materials, prob.material_map,
                               bc=prob.bc, device="cpu").solve(
        tol_k=1e-6, tol_source=1e-5)
    assert res.converged
    # Homogenized diffusion on a coarse mesh: expect within ~2% of transport.
    assert abs(res.k_eff - K_REFERENCE_2D) / K_REFERENCE_2D < 0.02, res.k_eff


@pytest.fixture(scope="module")
def family_keff():
    """k_eff of the full angular hierarchy at cells_per_pin=1 (~6 s total)."""
    prob = build_c5g7_2d(cells_per_pin=1)
    out = {}
    for name, cls in [("diffusion", DiffusionEigenSolver),
                      ("sp3", SP3EigenSolver), ("sp5", SP5EigenSolver),
                      ("sp7", SP7EigenSolver), ("sdp1", SDP1EigenSolver),
                      ("sdp2", SDP2EigenSolver), ("sdp3", SDP3EigenSolver)]:
        r = cls(prob.grid, prob.materials, prob.material_map, bc=prob.bc,
                device="cpu").solve(tol_k=1e-6, tol_source=1e-5)
        assert r.converged, (name, r)
        out[name] = r.k_eff
    return out


def test_c5g7_family_lands_near_transport(family_keff):
    """Every angular approximation within 500 pcm of the OpenMC/MCNP reference
    (a blown coefficient table misses by thousands of pcm)."""
    for name, k in family_keff.items():
        assert abs(k - K_REFERENCE_2D) / K_REFERENCE_2D < 5e-3, (name, k)


def test_c5g7_transport_cluster(family_keff):
    """The C5G7 signature (paper Sec. 4.3): the six SPN/SDPN eigenvalues
    stagnate -- they cluster within ~150 pcm of each other."""
    ks = [family_keff[n] for n in ("sp3", "sp5", "sp7", "sdp1", "sdp2", "sdp3")]
    assert max(ks) - min(ks) < 1.5e-3, ks


def test_c5g7_hierarchies_are_monotone(family_keff):
    """Both families approach the transport reference monotonically from below
    on this mesh, and each SDPN beats its matched-DoF SPN partner -- the
    orderings that broke under the (pre-fix) SDP2/SDP3 coefficient errors."""
    k = family_keff
    assert k["sp3"] < k["sp5"] < k["sp7"] < K_REFERENCE_2D
    assert k["sdp1"] < k["sdp2"] < k["sdp3"] < K_REFERENCE_2D
    for sdp, sp in (("sdp1", "sp3"), ("sdp2", "sp5"), ("sdp3", "sp7")):
        assert abs(k[sdp] - K_REFERENCE_2D) < abs(k[sp] - K_REFERENCE_2D), \
            (sdp, k[sdp], sp, k[sp])
