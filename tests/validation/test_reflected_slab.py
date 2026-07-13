"""Reflected slab reactor: the low-level and Model-API solves against the exact
analytic eigenvalue.

The one-group reflected slab (Lamarsh, *Introduction to Nuclear Reactor Theory*,
Ch. 7) has a closed-form eigenvalue from the transcendental condition
``D_c B tan(B a) = (D_r / L) coth(b / L)`` -- so this benchmark checks the
discretization error against exact mathematics, not another code. It is a filled
1-D geometry, so it is solved two ways: the low-level ``DiffusionEigenSolver`` on
the benchmark's grid, and the high-level ``ndgpu.Model``; both must converge to
the analytic value at second order, and both must agree.
"""
import numpy as np
import pytest

import ndgpu
from ndgpu import DiffusionEigenSolver
from ndgpu.benchmarks import (bare_k, build_reflected_slab, reflected_k,
                             SLAB_CORE, SLAB_REFLECTOR)

TIGHT = dict(tol_k=1e-10, tol_source=1e-9)


def test_reflector_savings_is_positive():
    # A reflector returns leaked neutrons, so the reflected core is more reactive
    # than the same core with a bare (zero-flux) edge, but still below k_inf.
    k_inf = SLAB_CORE.nu_sigma_f[0] / SLAB_CORE.sigma_a[0]
    assert bare_k() < reflected_k() < k_inf
    assert (reflected_k() - bare_k()) * 1e5 > 1000.0     # a substantial savings


def test_low_level_solver_converges_to_analytic_second_order():
    # Interface pinned at 25 cm (cells multiples of 9), so the analytic reference
    # is fixed and the FV error must fall ~4x per mesh doubling.
    err = {}
    for cells in (45, 90, 180):
        p = build_reflected_slab(cells=cells)
        assert p.core_half_width == pytest.approx(25.0)   # interface on a cell edge
        k = DiffusionEigenSolver(p.grid, p.materials, material_map=p.material_map,
                                 bc=p.bc, device="cpu").solve(**TIGHT).k_eff
        err[cells] = abs(k - p.k_reference)
    assert err[180] < 5e-6                                # < 0.5 pcm at 180 cells
    assert 3.5 < err[45] / err[90] < 4.5                  # second order
    assert 3.5 < err[90] / err[180] < 4.5


def test_model_api_matches_analytic_and_low_level():
    # The same slab through ndgpu.Model (1-D, filled geometry).
    a, b = 25.0, 20.0
    k_model = (ndgpu.Model(size=(a + b,), cells=(90,))
               .fill(SLAB_CORE).add_box(SLAB_REFLECTOR, x=(a, a + b))
               .set_boundary(x=("reflective", "zero-flux"))
               .run(**TIGHT).k_eff)
    p = build_reflected_slab(cells=90)
    k_low = DiffusionEigenSolver(p.grid, p.materials, material_map=p.material_map,
                                 bc=p.bc, device="cpu").solve(**TIGHT).k_eff
    assert k_model == pytest.approx(reflected_k(a=a, b=b), abs=1e-5)   # vs analytic
    assert k_model == pytest.approx(k_low, abs=1e-9)                   # same physics as low level
