"""Per-face boundary conditions and the C5G7 benchmark geometry."""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, ONE_GROUP_DEMO
from ndgpu.benchmarks import build_c5g7_2d
from ndgpu.benchmarks.c5g7 import FUEL_FRACTION, K_REFERENCE_2D
from ndgpu.operator import normalize_bc

TIGHT = dict(tol_k=1e-9, tol_source=1e-8)


def test_normalize_bc_forms():
    full = (("reflective", "zero-flux"), ("zero-flux", "zero-flux"),
            ("reflective", "reflective"))
    assert normalize_bc("zero-flux") == tuple(("zero-flux",) * 2 for _ in range(3))
    assert normalize_bc(("reflective", "zero-flux", "reflective")) == (
        ("reflective",) * 2, ("zero-flux",) * 2, ("reflective",) * 2)
    assert normalize_bc([("reflective", "zero-flux"), "zero-flux", "reflective"]) == full
    assert normalize_bc(["reflective", "zero-flux", "zero-flux", "zero-flux",
                         "reflective", "reflective"]) == full
    # "vacuum" and float albedo coefficients are valid face specs.
    assert normalize_bc("vacuum") == tuple(("vacuum",) * 2 for _ in range(3))
    assert normalize_bc(0.4695) == tuple((0.4695,) * 2 for _ in range(3))
    assert normalize_bc(("vacuum", "reflective", 0.4695)) == (
        ("vacuum",) * 2, ("reflective",) * 2, (0.4695,) * 2)
    with pytest.raises(ValueError):
        normalize_bc("bogus")
    with pytest.raises(ValueError):
        normalize_bc(-1.0)
    with pytest.raises(ValueError):
        normalize_bc(["zero-flux"] * 5)


def test_reflective_symmetry_plane_equals_full_domain():
    # A half-domain with a reflective symmetry plane must reproduce the
    # full-domain k exactly (the fundamental mode is symmetric).
    L, n = 90.0, 32
    full = DiffusionEigenSolver(
        Grid(shape=(n, n, n), size=(L, L, L)), ONE_GROUP_DEMO, device="cpu"
    ).solve(**TIGHT)
    half = DiffusionEigenSolver(
        Grid(shape=(n // 2, n, n), size=(L / 2, L, L)), ONE_GROUP_DEMO,
        bc=(("zero-flux", "reflective"), "zero-flux", "zero-flux"), device="cpu",
    ).solve(**TIGHT)
    assert full.converged and half.converged
    assert half.k_eff == pytest.approx(full.k_eff, abs=1e-7)


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
