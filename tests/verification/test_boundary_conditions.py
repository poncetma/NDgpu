"""Boundary-condition specs and exact symmetry invariants.

normalize_bc must accept every documented spec form, and a half-domain with a
reflective symmetry plane must reproduce the full-domain k exactly (the
fundamental mode is symmetric) -- an exact invariant, independent of mesh.
"""

import pytest

from ndgpu import DiffusionEigenSolver, Grid, ONE_GROUP_DEMO
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
