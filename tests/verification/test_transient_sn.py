"""Time-dependent S_N transport (TransientSNSolver) against exact references.

Phase 1 of the transient-transport extension: the Cartesian S_N engine driven
by the shared backward-Euler / delayed-precursor machinery. Two references, both
independent of the solver:

* **Exact fixed point.** The critically adjusted steady S_N state is an exact
  equilibrium of the discrete transient equations, so with no perturbation the
  power holds at 1 to solver tolerance. This is the transport analogue of the
  diffusion suite's cornerstone and validates the whole time-stepping /
  precursor / critical-adjustment stack (including the angular-flux time source,
  which diffusion never exercises).

* **Point kinetics.** For a *uniform* perturbation the flux shape never changes,
  so the space-angle transport equations collapse exactly onto the point-kinetics
  ODEs. Unlike diffusion -- whose fundamental mode is geometry-locked -- a
  transport mode's shape depends on the *scattering ratio* c = Sigma_s/Sigma_t, so
  a Sigma_a step subtly reshapes it. A **nu*Sigma_f step** leaves the streaming +
  scattering operator (hence the mode) untouched, giving a transport-exact point
  kinetics reference. The only bookkeeping: res.power tracks nu*Sigma_f * phi, so
  after the (1+eps) production step it reads (1+eps) n(t); dividing it out
  recovers the population n(t). A low-leakage slab keeps the mode nearly
  isotropic so the naive generation time Lambda = 1/(v a) is accurate; in a leaky
  slab the transport Lambda departs from 1/(v a) (the isotropic fission source
  cannot balance the anisotropic time term pointwise) and the match loosens to a
  physical ~1e-3 -- a transport-kinetics effect, not a discretization error.
"""

import numpy as np
import pytest

from ndgpu import Grid, Kinetics, Material, TransientSNSolver
from ndgpu.sn import SNTransportSolver

D, SIGMA_A, NU_SIGMA_F = 1.3, 0.030, 0.035
BASE = Material(name="1g", diffusion=[D], sigma_a=[SIGMA_A], nu_sigma_f=[NU_SIGMA_F])
V, BETA, LAM = 2.2e5, 0.0065, 0.08
KIN = Kinetics(velocities=[V], beta=[BETA], decay=[LAM])
# ~1D slab: many cells in x, one huge cell in y so the y-vacuum leakage (and its
# anisotropy) is negligible -- the S_N sweep sees an essentially 1D problem.
SLAB = Grid(shape=(60, 1, 1), size=(500.0, 1e5, 1.0))
QUAD = dict(n_polar=2, n_azi=4, acceleration="dsa")


def nsf_step(eps):
    """nu*Sigma_f -> nu*Sigma_f (1 + eps) everywhere for t > 0 (a shape-preserving
    reactivity insertion: the transport mode is unchanged, only production scales)."""
    m1 = [Material(name="pert", diffusion=[D], sigma_a=[SIGMA_A],
                   nu_sigma_f=[NU_SIGMA_F * (1.0 + eps)])]
    return lambda t: (([BASE] if t <= 0 else m1), None)


# --- Phase 0 engine hook: the angular-flux sweep -------------------------------

def test_sweep_ang_matches_steady_sweep_and_psi_sums_to_phi():
    """The transient's per-ordinate sweep (_sweep_ang) must reproduce the steady
    wavefront sweep's scalar flux for the same source, and the angular flux it
    returns must integrate back to that scalar flux (Sum_m w_m psi_m = phi)."""
    eng = SNTransportSolver(SLAB, BASE, bc="vacuum", **QUAD)
    q = np.linspace(0.5, 1.5, eng.nx)[:, None] * np.ones((eng.nx, eng.ny))
    st = eng.st[0]
    inc = eng._zero_inc()
    phi_ref, _ = eng._sweep_wavefront(q, st, inc)
    phi, _, psi = eng._sweep_ang(q, st, inc, want_psi=True)
    assert np.allclose(phi, phi_ref, atol=1e-11), np.max(np.abs(phi - phi_ref))
    phi_from_psi = np.einsum("m,mxy->xy", eng.w, psi)
    assert np.allclose(phi_from_psi, phi, atol=1e-11)


def test_reflective_transient_not_supported():
    """Reflective boundaries in the transient sweep are Phase 2; the engine hook
    rejects them clearly rather than silently leaking."""
    eng = SNTransportSolver(SLAB, BASE, bc="reflective", **QUAD)
    with pytest.raises(NotImplementedError):
        eng._solve_group_transient(np.ones((eng.nx, eng.ny)), eng.ss_self[0],
                                   eng.st[0], eng._zero_inc(),
                                   np.zeros((eng.M, eng.nx, eng.ny)), None, 1e-6, 0)


# --- Phase 1 transient driver --------------------------------------------------

def test_sn_unperturbed_transient_stays_steady():
    res = TransientSNSolver(SLAB, lambda t: ([BASE], None), KIN, bc="vacuum",
                            **QUAD).solve(t_end=0.2, dt=0.02)
    assert np.allclose(res.power, 1.0, atol=1e-5), res.power


def test_sn_transient_tracks_point_kinetics():
    """A +$0.50 nu*Sigma_f step in the low-leakage slab: the S_N power (population
    part) tracks exact point kinetics to backward-Euler O(dt), and the prompt jump
    reaches the beta/(beta-rho) = 2.0 plateau."""
    from scipy.integrate import solve_ivp

    k0 = TransientSNSolver(SLAB, lambda t: ([BASE], None), KIN, bc="vacuum",
                           **QUAD).solve(t_end=1e-3, dt=1e-3).k0
    a = NU_SIGMA_F / k0
    rho_dollars = 0.5
    # nu*Sigma_f -> (1 + eps): k -> k0 (1 + eps), so rho$ = (eps/(1+eps))/beta.
    eps = rho_dollars * BETA / (1.0 - rho_dollars * BETA)

    t_end, dt = 0.08, 1e-4
    res = TransientSNSolver(SLAB, nsf_step(eps), KIN, bc="vacuum",
                            **QUAD).solve(t_end=t_end, dt=dt, tol_step=1e-8)

    # Exact point kinetics for the production step: prompt production
    # (1-beta) a (1+eps), loss a (unchanged operator), delayed lam C, with the
    # precursors starting from the *unperturbed* equilibrium beta a / lam.
    ap = a * (1.0 + eps)
    rhs = lambda t, y: [V * (((1.0 - BETA) * ap - a) * y[0] + LAM * y[1]),
                        BETA * ap * y[0] - LAM * y[1]]
    ref = solve_ivp(rhs, (0, res.times[-1]), [1.0, BETA * a / LAM], method="Radau",
                    t_eval=res.times, rtol=1e-11, atol=1e-13)

    # res.power = (1+eps) n(t) once production has stepped; the t=0 sample is the
    # unperturbed steady state (power 1), excluded from the population comparison.
    n = res.power / (1.0 + eps)
    err = np.max(np.abs(n[1:] - ref.y[0][1:]) / ref.y[0][1:])
    assert err < 5e-4, f"max relative deviation from point kinetics: {err:.2e}"
    # The prompt jump is real and resolved: the population rises well past the
    # delayed-only rate toward the beta/(beta-rho) = 2.0 plateau (the approach
    # time constant ~ Lambda/(beta-rho) puts the full plateau past this window),
    # climbing monotonically. The tight reactivity check is the PK match above.
    plateau = 1.0 / (1.0 - rho_dollars)
    assert 1.6 < n[-1] < plateau + 0.05, n[-1]
    assert np.all(np.diff(n) > -1e-9)  # monotone supercritical rise
