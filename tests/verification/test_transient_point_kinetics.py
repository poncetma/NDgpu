"""Transient solver against exact references.

The cornerstone is the point-kinetics comparison: for a *uniform* reactivity
perturbation of a bare homogeneous core, the flux shape never changes, so the
space-time diffusion equations reduce *exactly* to the point-kinetics ODEs — a
rigorous, independent reference for the whole transient stack (precursor
treatment, critical adjustment, time stepping).

The perturbation is a uniform **absorption** step (not a fission step): a small
drop in Sigma_a inserts positive reactivity while leaving nu*Sigma_f fixed, so
res.power (proportional to nu*Sigma_f * phi, i.e. the neutron population) tracks
the point-kinetics amplitude n(t) *exactly*. A fission (nu*Sigma_f) step would
instead make res.power a production-rate proxy that jumps by (1+eps) the instant
the cross section changes, offsetting it from n(t) by ~eps and masking the true
agreement. With a physical neutron speed the prompt jump to beta/(beta-rho) is
resolved in-window, and the match tightens to a few 1e-4 (backward-Euler O(dt)).
"""

import numpy as np
import pytest

from ndgpu import Grid, Kinetics, Material, TransientSolver

D, SIGMA_A, NU_SIGMA_F = 1.3, 0.030, 0.035
BASE = Material(name="1g", diffusion=[D], sigma_a=[SIGMA_A], nu_sigma_f=[NU_SIGMA_F])
# Physical thermal-neutron speed (2200 m/s = 2.2e5 cm/s): the prompt generation
# time Lambda = 1/(v*a) ~ 1.3e-4 s is short enough that the prompt jump develops
# within the transient window.
V, BETA, LAM = 2.2e5, 0.0065, 0.08
KIN = Kinetics(velocities=[V], beta=[BETA], decay=[LAM])
GRID = Grid(shape=(12, 12, 12), size=(90.0, 90.0, 90.0))


def absorption_step_problem(delta_sigma_a):
    """Sigma_a -> Sigma_a - delta everywhere for t > 0 (positive reactivity)."""
    m0 = [BASE]
    m1 = [Material(name="pert", diffusion=[D], sigma_a=[SIGMA_A - delta_sigma_a],
                   nu_sigma_f=[NU_SIGMA_F])]
    return lambda t: ((m0 if t <= 0 else m1), None)


def test_unperturbed_transient_stays_steady():
    solver = TransientSolver(GRID, lambda t: ([BASE], None), KIN, device="cpu")
    res = solver.solve(t_end=0.5, dt=0.05)
    assert np.allclose(res.power, 1.0, atol=1e-5), res.power


def test_matches_point_kinetics_for_uniform_perturbation():
    from scipy.integrate import solve_ivp

    # Get the initial eigenvalue for the exact critical adjustment (a = nuSf/k0),
    # then size the absorption drop to a +$0.50 step (well below prompt critical).
    k0 = TransientSolver(GRID, lambda t: ([BASE], None), KIN, device="cpu").solve(
        t_end=1e-3, dt=1e-3).k0
    a = NU_SIGMA_F / k0
    rho_dollars = 0.5
    d_sigma_a = rho_dollars * BETA * a

    t_end, dt = 0.5, 1e-4
    res = TransientSolver(GRID, absorption_step_problem(d_sigma_a), KIN,
                          device="cpu").solve(t_end=t_end, dt=dt)

    # Exact point kinetics for this absorption step. With the critical
    # adjustment DB^2 + Sigma_a = a, dropping Sigma_a by d_sigma_a gives
    #   (1/v) dn/dt = (d_sigma_a - beta a) n + lam C,  dC/dt = beta a n - lam C.
    rhs = lambda t, y: [V * ((d_sigma_a - BETA * a) * y[0] + LAM * y[1]),
                        BETA * a * y[0] - LAM * y[1]]
    ref = solve_ivp(rhs, (0, t_end), [1.0, BETA * a / LAM], method="Radau",
                    t_eval=res.times, rtol=1e-11, atol=1e-13)

    err = np.max(np.abs(res.power - ref.y[0]) / ref.y[0])
    assert err < 5e-4, f"max relative deviation from point kinetics: {err:.2e}"
    # The prompt jump is real and resolved: power reaches the beta/(beta-rho)
    # plateau (= 2.0 at $0.50) well before the delayed rise carries it past.
    plateau = 1.0 / (1.0 - rho_dollars)
    i_plateau = np.searchsorted(res.times, 0.2)
    assert res.power[i_plateau] == pytest.approx(plateau, abs=0.05), res.power[i_plateau]
    assert res.power[-1] > plateau  # delayed supercritical rise continues
