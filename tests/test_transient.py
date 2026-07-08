"""Transient solver validation.

The cornerstone is the point-kinetics comparison: for a *uniform* fissile
perturbation of a bare homogeneous core, the flux shape never changes, so the
exact solution of the space-time diffusion equations is the point-kinetics
ODE system — a rigorous, independent reference for the whole transient stack
(precursor treatment, critical adjustment, time stepping).
"""

import numpy as np
import pytest

from ndgpu import Grid, Kinetics, Material, TransientSolver
from ndgpu.benchmarks import build_langenbuch, build_twigl

ONE_GROUP = Material(name="1g", diffusion=[1.3], sigma_a=[0.030], nu_sigma_f=[0.035])
KIN = Kinetics(velocities=[2200.0], beta=[0.0065], decay=[0.08])
GRID = Grid(shape=(12, 12, 12), size=(90.0, 90.0, 90.0))


def uniform_step_problem(eps):
    """nuSigma_f -> (1 + eps) * nuSigma_f everywhere for t > 0."""
    m0 = [ONE_GROUP]
    m1 = [Material(name="pert", diffusion=[1.3], sigma_a=[0.030],
                   nu_sigma_f=[0.035 * (1.0 + eps)])]
    return lambda t: ((m0 if t <= 0 else m1), None)


def test_unperturbed_transient_stays_steady():
    solver = TransientSolver(GRID, lambda t: ([ONE_GROUP], None), KIN, device="cpu")
    res = solver.solve(t_end=0.5, dt=0.05)
    assert np.allclose(res.power, 1.0, atol=1e-5), res.power


def test_matches_point_kinetics_for_uniform_perturbation():
    from scipy.integrate import solve_ivp

    eps = 1e-3  # +100 pcm, well below prompt critical (beta = 650 pcm)
    solver = TransientSolver(GRID, uniform_step_problem(eps), KIN, device="cpu")
    res = solver.solve(t_end=0.2, dt=5e-4)

    # Exact point kinetics in one-group form: with a = nuSigma_f/k0 and the
    # critical adjustment, DB^2 + Sigma_a = a exactly (also in the discrete
    # problem), so:
    #   (1/v) dP/dt = a [(1-beta)(1+eps) - 1] P + lam C,  dC/dt = beta a (1+eps) P - lam C
    v, beta, lam = KIN.velocities[0], KIN.beta[0], KIN.decay[0]
    a = 0.035 / res.k0
    rhs = lambda t, y: [
        v * (a * ((1 - beta) * (1 + eps) - 1.0) * y[0] + lam * y[1]),
        beta * a * (1 + eps) * y[0] - lam * y[1],
    ]
    ref = solve_ivp(rhs, (0, 0.2), [1.0, beta * a / lam], method="Radau",
                    t_eval=res.times, rtol=1e-10, atol=1e-12)

    err = np.max(np.abs(res.power - ref.y[0]) / ref.y[0])
    assert err < 2e-3, f"max relative deviation from point kinetics: {err:.2e}"
    assert res.power[-1] > 1.01  # the transient actually went somewhere


def test_twigl_step_and_ramp():
    k0_expected = 0.91321  # published TWIGL quarter-core eigenvalue
    results = {}
    for pert in ("step", "ramp"):
        prob = build_twigl(perturbation=pert, cells_per_8cm=1)
        solver = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                                 bc=prob.bc, device="cpu")
        res = solver.solve(t_end=0.5, dt=2e-3)
        results[pert] = res
        assert abs(res.k0 - k0_expected) < 5e-3, res.k0
        assert res.power[-1] == pytest.approx(res.power[-2], rel=0.05)  # settled

    # Published TWIGL 2D reference (seed-blanket, e.g. Hageman & Yasinsky):
    #   step: P(0.1)=2.06, P(0.5)=2.13;  ramp: P(0.1)=1.31, P(0.5)=2.11.
    # This coarse 10x10 mesh lands ~1% high of the converged values (the
    # refined cells_per_8cm=4 solve reproduces them to <1e-3, see the README).
    step, ramp = results["step"], results["ramp"]
    i100 = np.searchsorted(step.times, 0.1)
    assert step.power[i100] == pytest.approx(2.06, abs=0.05), step.power[i100]
    assert ramp.power[i100] == pytest.approx(1.31, abs=0.05), ramp.power[i100]
    assert step.power[-1] == pytest.approx(2.13, abs=0.05), step.power[-1]
    assert ramp.power[-1] == pytest.approx(2.11, abs=0.05), ramp.power[-1]
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
    t_peak = res.times[np.argmax(p)]
    # Published LMW (Langenbuch-Maurer-Werner) reference: near-critical core,
    # power peaks ~1.6x around t=21 s, then bank 2 drives it well down.
    assert 0.985 < res.k0 < 1.01, res.k0
    assert p.max() == pytest.approx(1.6, abs=0.15), p.max()  # bank-1 withdrawal
    assert 18.0 < t_peak < 24.0, t_peak         # peak while bank 2 inserts
    assert p[-1] < 0.5 * p.max(), p[-1]         # driven back down
