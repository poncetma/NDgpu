"""TWIGL (2D) and Langenbuch/LMW (3D) kinetics benchmarks.

Reference eigenvalues and power histories live with the problem definitions
in ndgpu.benchmarks.twigl / .langenbuch; these tests march the transients on
coarse meshes and check against them.
"""

import numpy as np
import pytest

from ndgpu import TransientSolver
from ndgpu.benchmarks import build_langenbuch, build_twigl
from ndgpu.benchmarks.langenbuch import PEAK_REFERENCE
from ndgpu.benchmarks.twigl import K_REFERENCE, P_REFERENCE


def test_twigl_step_and_ramp():
    results = {}
    for pert in ("step", "ramp"):
        prob = build_twigl(perturbation=pert, cells_per_8cm=1)
        solver = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                                 bc=prob.bc, device="cpu")
        res = solver.solve(t_end=0.5, dt=2e-3)
        results[pert] = res
        assert abs(res.k0 - K_REFERENCE) < 5e-3, res.k0
        assert res.power[-1] == pytest.approx(res.power[-2], rel=0.05)  # settled

    # This coarse 10x10 mesh lands ~1% high of the converged values (the
    # refined cells_per_8cm=4 solve reproduces them to <1e-3, see the README).
    step, ramp = results["step"], results["ramp"]
    i100 = np.searchsorted(step.times, 0.1)
    assert step.power[i100] == pytest.approx(P_REFERENCE["step"][0.1], abs=0.05)
    assert ramp.power[i100] == pytest.approx(P_REFERENCE["ramp"][0.1], abs=0.05)
    assert step.power[-1] == pytest.approx(P_REFERENCE["step"][0.5], abs=0.05)
    assert ramp.power[-1] == pytest.approx(P_REFERENCE["ramp"][0.5], abs=0.05)
    assert step.power[i100] > 1.5 * ramp.power[i100]  # step leads early on
    assert np.all(np.diff(ramp.power) > -1e-9)        # ramp power monotone


def test_langenbuch_rod_maps():
    prob = build_langenbuch()
    mats, mmap = prob.problem_at(0.0)
    # bank 1 half inserted: 4 rods x 4 fully-rodded cells (z = 100..180)
    assert int(np.sum(mmap == 3)) == 16
    assert int(np.sum(mmap == 4)) == 9  # rod guides in the top reflector
    mats2, mmap2 = prob.problem_at(5.0)  # tip at 115 cm: partial cell appears
    assert len(mats2) == 6
    assert int(np.sum(mmap2 == 5)) == 4
    mats3, mmap3 = prob.problem_at(1000.0)  # bank 1 out, bank 2 at 60 cm
    assert int(np.sum(mmap3 == 3)) == 5 * 6  # 5 rods cover z = 60..180
    assert prob.problem_at(1000.0) is prob.problem_at(1000.0)  # cached


def test_langenbuch_transient_shape():
    # Coarse mesh, big steps: just verify the classic rise-then-fall shape.
    prob = build_langenbuch()
    solver = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                             bc=prob.bc, device="cpu")
    res = solver.solve(t_end=60.0, dt=0.25)
    p = res.power
    t_ref, p_ref = PEAK_REFERENCE
    t_peak = res.times[np.argmax(p)]
    assert 0.985 < res.k0 < 1.01, res.k0
    assert p.max() == pytest.approx(p_ref, abs=0.15), p.max()  # bank-1 withdrawal
    assert t_ref - 3.0 < t_peak < t_ref + 3.0, t_peak  # peak while bank 2 inserts
    assert p[-1] < 0.5 * p.max(), p[-1]                # driven back down
