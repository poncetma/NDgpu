"""OECD/NEA C5G7 MOX benchmark (2D quarter core), pin-cell homogenized.

Checks the geometry construction (pin counts, symmetry, reaction-rate
conservation of the homogenization) and that homogenized diffusion lands
within ~2% of the published transport reference K_REFERENCE_2D.
"""

import numpy as np

from ndgpu import DiffusionEigenSolver
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
