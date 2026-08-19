"""Definition and static convergence gates for ANL-7416 LRA-2D."""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver
from ndgpu.benchmarks import build_lra2d, lra2d_static_keff
from ndgpu.benchmarks.lra import (GEOMETRIC_CORE_AREA_CM2, K_REFERENCE,
                                  PAPER_CORE_AREA_CM2, CHEREZOV_BY_FEM_ORDER,
                                  LRAAdiabaticState, LRATransientResult)


def test_lra_map_and_control_region_match_original_specification():
    p = build_lra2d()
    assert p.material_map.shape == (11, 11, 1)
    assert np.count_nonzero(p.core_mask) == 78
    assert np.count_nonzero(p.control_mask) == 4
    assert GEOMETRIC_CORE_AREA_CM2 == 17550.0
    assert PAPER_CORE_AREA_CM2 == 17750.0  # documented paper inconsistency
    np.testing.assert_array_equal(np.argwhere(p.control_mask)[:, :2],
                                  [[7, 5], [7, 6], [8, 5], [8, 6]])


def test_cherezov_spatial_order_ladder_is_not_mislabeled_as_bdf_order():
    assert sorted(CHEREZOV_BY_FEM_ORDER) == list(range(1, 9))
    assert CHEREZOV_BY_FEM_ORDER[1]["first_peak_time_s"] == 1.352
    assert CHEREZOV_BY_FEM_ORDER[4]["power_at_3s_w_cm3"] == 99.0


@pytest.mark.parametrize("state", ["in", "out"])
def test_lra_control_material_endpoints(state):
    p = build_lra2d(control=state, axial_buckling=False)
    mats, mmap = p.problem_at(123.0)
    r = mats[int(mmap[7, 5, 0])]
    expected = 8.3440e-2 if state == "in" else 7.3324e-2
    assert r.sigma_a[1] == pytest.approx(expected)

    literal = build_lra2d(control=state)
    mats_b, mmap_b = literal.problem_at(123.0)
    r_b = mats_b[int(mmap_b[7, 5, 0])]
    assert r_b.sigma_a[1] == pytest.approx(expected + 0.2091e-4)


def test_lra_control_worth_scale_is_explicit_and_does_not_change_rods_in():
    scale = 1.125
    p = build_lra2d(control="transient", control_worth_scale=scale,
                    axial_buckling=False)
    mats_in, mmap = p.problem_at(0.0)
    mats_out, _ = p.problem_at(2.0)
    index = int(mmap[7, 5, 0])
    assert mats_in[index].sigma_a[1] == pytest.approx(8.3440e-2)
    assert mats_out[index].sigma_a[1] == pytest.approx(
        8.3440e-2 + scale * (7.3324e-2 - 8.3440e-2))
    with pytest.raises(ValueError, match="control_worth_scale"):
        build_lra2d(control_worth_scale=0.0)


def test_lra_reflector_uses_original_anl_values_not_table_typos():
    p = build_lra2d(axial_buckling=False)
    mats, _ = p.problem_at(0.0)
    reflector = mats[4]
    assert reflector.sigma_a[0] == pytest.approx(6.034e-4)
    assert reflector.sigma_a[1] == pytest.approx(1.911e-2)
    assert reflector.removal[0] == pytest.approx(
        6.034e-4 + 4.754e-2)


def _keff(problem):
    mats, mmap = problem.problem_at(0.0)
    result = DiffusionEigenSolver(
        problem.grid, mats, mmap, bc=problem.bc, device="cpu"
    ).solve(tol_k=2e-9, tol_source=2e-8)
    assert result.converged
    return result.k_eff


def test_lra_static_endpoints_match_reference_with_specified_buckling():
    # Four cells per assembly edge is the 3.75 cm reference mesh used by the
    # original fine-difference calculation. Rods-out is a raw eigenvalue in
    # the literature, not the critical-adjusted ratio used by a transient.
    k_in = lra2d_static_keff(refine=4, control="in")
    k_out_raw = lra2d_static_keff(refine=4, control="out")
    assert k_in == pytest.approx(0.99623051, abs=2e-7)
    assert k_out_raw == pytest.approx(1.01481578, abs=2e-7)
    assert k_in == pytest.approx(K_REFERENCE["rods_in"], abs=1.1e-4)
    assert k_out_raw == pytest.approx(K_REFERENCE["rods_out"], abs=6.5e-4)


def test_lra_static_endpoint_helper_rejects_transient_control():
    with pytest.raises(ValueError, match="static control"):
        lra2d_static_keff(control="transient")


def test_lra_omitting_axial_buckling_is_pinned_and_distinct():
    # This diagnostic used to hide the reflector typo by shifting k upward.
    k = _keff(build_lra2d(refine=4, control="in", axial_buckling=False))
    assert k == pytest.approx(0.99997297, abs=2e-7)
    assert abs(k - K_REFERENCE["rods_in"]) > 3e-3


def test_lra_adiabatic_heat_and_doppler_laws_are_exact():
    p = build_lra2d()
    state = LRAAdiabaticState.uniform(p.grid.shape)
    q = np.full(p.grid.shape, 2.0)
    state.advance(q, 0.25)
    expected = 300.0 + 0.25 * (3.83e-11 / 3.204e-11) * 2.0
    assert np.all(state.temperature == pytest.approx(expected))


def test_lra_doppler_does_not_scale_axial_leakage():
    from ndgpu.solver import Fields

    p = build_lra2d(refine=1)
    mats, mmap = p.problem_at(0.0)
    cold = Fields(np, p.grid, mats, mmap, np.float64)
    state = LRAAdiabaticState.uniform(p.grid.shape, 600.0)
    hot = Fields(np, p.grid, mats, mmap, np.float64,
                 xs_update=state.hook(axial_buckling2=1e-4))
    factor = 1.0 + 3.034e-3 * (np.sqrt(600.0) - np.sqrt(300.0))
    leakage = cold.diffusion[0] * 1e-4
    np.testing.assert_allclose(hot.sigma_a[0] - leakage,
                               (cold.sigma_a[0] - leakage) * factor)


def test_lra_reference_peak_temperature_is_assembly_averaged():
    class Transient:
        times = np.array([0.0, 1.0, 2.0, 3.0])

    result = LRATransientResult(
        transient=Transient(),
        average_power_w_cm3=np.array([1.0, 3.0, 2.0, 1.0]),
        average_temperature_k=np.array([300.0, 400.0, 500.0, 600.0]),
        peak_assembly_temperature_k=np.array([300.0, 450.0, 650.0, 800.0]),
        peak_temperature_k=np.array([300.0, 500.0, 750.0, 900.0]),
        temperature=np.full((1, 1, 1), 900.0),
        coupling="test",
    )
    metrics = result.metrics()
    assert metrics["peak_temperature_at_3s_k"] == 800.0
    assert metrics["peak_cell_temperature_at_3s_k"] == 900.0
