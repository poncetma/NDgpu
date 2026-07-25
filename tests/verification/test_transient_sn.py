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
  ODEs. A **nu*Sigma_f step** is used because it leaves the streaming +
  scattering operator -- hence the transport mode -- exactly untouched, while a
  Sigma_a step at fixed Sigma_t = 1/3D perturbs the scattering ratio
  c = Sigma_s/Sigma_t and so could reshape it (diffusion is immune: its
  fundamental mode is geometry-locked). The only bookkeeping: res.power tracks
  nu*Sigma_f * phi, so after the (1+eps) production step it reads (1+eps) n(t);
  dividing it out recovers the population n(t).

  What actually limits the match is the **generation time**, not the mode shape.
  A low-leakage slab keeps the flux nearly isotropic, so the naive
  Lambda = 1/(v a) assumed by the reference ODE is accurate and the deviation is
  pure backward-Euler O(dt) (measured 9.4e-4, 4.5e-4, 2.0e-4, 8.0e-5 for
  dt = 4e-4 .. 5e-5). In a leaky slab the true transport Lambda is
  adjoint-weighted and departs from 1/(v a), so the deviation *saturates* at a
  dt-independent ~1.3e-3 -- physics, not a discretization error. Measured with
  both perturbation types, the two agree to < 0.5% in each regime, confirming
  the floor tracks leakage (anisotropy) and that mode reshaping is negligible
  here. See examples/transient_sn_pk_benchmark.py.
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


@pytest.mark.parametrize("bc", ["vacuum", "reflective"])
def test_wavefront_transient_sweep_matches_row_reference(bc):
    """Phase 2: the vectorized wavefront sweep carries the per-ordinate time
    source and returns the angular flux, and must reproduce the per-ordinate row
    reference exactly -- scalar flux, angular flux and outgoing edges -- with
    non-trivial incoming fluxes on every face."""
    grid = Grid(shape=(9, 7, 1), size=(20.0, 15.0, 1.0))
    ew = SNTransportSolver(grid, BASE, bc=bc, n_polar=2, n_azi=8,
                           sweep="wavefront")
    er = SNTransportSolver(grid, BASE, bc=bc, n_polar=2, n_azi=8, sweep="rows")
    rng = np.random.default_rng(3)
    q = rng.random((9, 7))
    q_ang = rng.random((ew.M, 9, 7))
    inc = ew._zero_inc()
    if bc == "reflective":
        for f in inc:
            inc[f] = rng.random(inc[f].shape)
    pw, ow, psw = ew._sweep_wavefront(q, ew.st[0], inc, q_ang=q_ang,
                                      want_psi=True)
    pr, orr, psr = er._sweep_ang(q, er.st[0], inc, q_ang, want_psi=True)
    assert np.allclose(pw, pr, atol=1e-12), np.max(np.abs(pw - pr))
    assert np.allclose(psw, psr, atol=1e-12), np.max(np.abs(psw - psr))
    for f in ow:
        assert np.allclose(ow[f], orr[f], atol=1e-12), f
    assert np.allclose(np.einsum("m,mxy->xy", ew.w, psw), pw, atol=1e-12)


def test_reflective_transient_stays_steady_and_is_kinf():
    """Phase 2: with reflective boundaries the transient's boundary fixed point
    must hold the steady state exactly -- and for a homogeneous medium that state
    is the non-leaking k_inf solution, so this also pins the reflective transient
    against an analytic eigenvalue."""
    grid = Grid(shape=(6, 6, 1), size=(30.0, 30.0, 1.0))
    res = TransientSNSolver(grid, lambda t: ([BASE], None), KIN,
                            bc="reflective", **QUAD).solve(t_end=0.1, dt=0.02)
    k_inf = NU_SIGMA_F / SIGMA_A
    assert res.k0 == pytest.approx(k_inf, rel=1e-8), res.k0
    assert np.allclose(res.power, 1.0, atol=1e-5), res.power


def test_cmfd_step_acceleration_matches_plain_iteration():
    """Phase 2: the drift-corrected coarse solve accelerates the within-step
    fission fixed point without moving it -- the CMFD and plain power traces must
    agree, at far fewer fixed-point iterations per step. CMFD pays most at large
    dt, where the 1/(v dt) shift no longer damps the fixed point."""
    eps = 0.5 * BETA / (1.0 - 0.5 * BETA)
    grid = Grid(shape=(12, 12, 1), size=(60.0, 60.0, 1.0))
    kw = dict(t_end=0.2, dt=5e-2, tol_step=1e-9)
    runs = {}
    for acc in ("none", "cmfd"):
        runs[acc] = TransientSNSolver(grid, nsf_step(eps), KIN, bc="vacuum",
                                      step_acceleration=acc, **QUAD).solve(**kw)
    assert np.allclose(runs["cmfd"].power, runs["none"].power, rtol=0, atol=1e-5), \
        np.max(np.abs(runs["cmfd"].power - runs["none"].power))
    its_c = np.mean(runs["cmfd"].step_iterations)
    its_n = np.mean(runs["none"].step_iterations)
    assert its_c < 0.6 * its_n, (its_c, its_n)


def test_cmfd_rejects_row_sweep_and_tri_engine():
    """CMFD needs the wavefront sweep's face currents; asking for it with the
    row sweep is an error rather than a silent downgrade."""
    solver = TransientSNSolver(SLAB, lambda t: ([BASE], None), KIN, bc="vacuum",
                               step_acceleration="cmfd", sweep="rows", **QUAD)
    with pytest.raises(ValueError, match="face"):
        solver.solve(t_end=2e-3, dt=1e-3)


# --- Phase 3: the tri / prism engine -------------------------------------------
# The same driver, a different transport engine: TriSNTransportSolver prefactors
# its per-ordinate operator, so the backward-Euler shift Sigma_t + theta is folded
# in at construction (sigma_t_shift) rather than passed per sweep. A *periodic*
# tri lattice is an infinite medium -- zero leakage, flat isotropic flux -- which
# makes point kinetics an EXACT reference with no spatial error and no anisotropy
# floor at all: the only remaining error is backward Euler. That is a sharper test
# than any Cartesian slab can give.

TRI_MAT = Material(name="1g", diffusion=[D], sigma_a=[SIGMA_A],
                   nu_sigma_f=[NU_SIGMA_F], sigma_s=[[0.0]])
TRI_QUAD = dict(n_polar=2, n_azi=8)


def tri_nsf_step(eps):
    pert = [Material(name="pert", diffusion=[D], sigma_a=[SIGMA_A],
                     nu_sigma_f=[NU_SIGMA_F * (1.0 + eps)], sigma_s=[[0.0]])]
    return lambda t: (([TRI_MAT] if t <= 0 else pert), None)


def tri_grid(is3d=False):
    from ndgpu.tri import TriGrid
    # Minimal axial extent: psi is retained between steps, so nz drives memory.
    return (TriGrid(shape=(2, 2, 2, 3), side=4.0, height=12.0) if is3d
            else TriGrid(shape=(4, 4, 2), side=3.0))


def tri_solver(grid, problem, scheme, **kw):
    from ndgpu.tri_sn import TriSNTransportSolver
    return TransientSNSolver(grid, problem, KIN, bc="periodic",
                             engine_cls=TriSNTransportSolver, scheme=scheme,
                             **TRI_QUAD, **kw)


@pytest.mark.parametrize("is3d", [False, True])
@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_tri_transient_periodic_is_kinf_and_stays_steady(scheme, is3d):
    """On a periodic tri/prism torus the critically adjusted steady state is an
    exact equilibrium of the discrete transient equations: k0 is k_inf to round-off
    and the power holds at 1 to machine precision. For SCB this also pins the
    *corner-resolved* time source -- theta*psi_old enters each corner sub-volume's
    own balance, and only the correct layout cancels the theta shift exactly."""
    res = tri_solver(tri_grid(is3d), lambda t: ([TRI_MAT], None),
                     scheme).solve(t_end=0.06, dt=0.02)
    k_inf = NU_SIGMA_F / SIGMA_A
    assert res.k0 == pytest.approx(k_inf, abs=1e-9), res.k0
    assert np.allclose(res.power, 1.0, atol=1e-12), np.max(np.abs(res.power - 1))


@pytest.mark.parametrize("scheme", ["step", "scb"])
def test_tri_transient_matches_point_kinetics_to_first_order(scheme):
    """Zero leakage and a flat flux make both schemes spatially exact, so the
    deviation from exact point kinetics is *purely* backward Euler and must halve
    with dt (measured 8.81e-4, 4.42e-4, 2.21e-4 for dt = 4e-4 .. 1e-4, i.e.
    ratios of 2.00). step and scb agree to all printed digits, as they must."""
    from scipy.integrate import solve_ivp

    eps = 0.5 * BETA / (1.0 - 0.5 * BETA)
    grid = tri_grid()
    errs = []
    for dt in (4e-4, 2e-4, 1e-4):
        res = tri_solver(grid, tri_nsf_step(eps), scheme).solve(
            t_end=0.02, dt=dt, tol_step=1e-9)
        a = NU_SIGMA_F / res.k0
        ap = a * (1.0 + eps)
        rhs = lambda t, y: [V * (((1.0 - BETA) * ap - a) * y[0] + LAM * y[1]),
                            BETA * ap * y[0] - LAM * y[1]]
        ref = solve_ivp(rhs, (0, res.times[-1]), [1.0, BETA * a / LAM],
                        method="Radau", t_eval=res.times, rtol=1e-12,
                        atol=1e-14).y[0]
        n = res.power / (1.0 + eps)
        errs.append(float(np.max(np.abs(n[1:] - ref[1:]) / ref[1:])))
    assert errs[-1] < 5e-4, errs
    for coarse, fine in zip(errs, errs[1:]):        # first order in dt
        assert coarse / fine == pytest.approx(2.0, abs=0.15), errs


def test_tri_transient_vacuum_leaks_and_stays_steady():
    """Vacuum (leaky) tri transient: k0 drops well below k_inf and the critically
    adjusted state is still an exact equilibrium."""
    from ndgpu.tri import TriGrid
    from ndgpu.tri_sn import TriSNTransportSolver

    res = TransientSNSolver(TriGrid(shape=(8, 8, 2), side=3.0),
                            lambda t: ([TRI_MAT], None), KIN, bc="vacuum",
                            engine_cls=TriSNTransportSolver,
                            **TRI_QUAD).solve(t_end=0.1, dt=0.02)
    assert res.k0 < 0.8 * NU_SIGMA_F / SIGMA_A          # real leakage
    assert np.allclose(res.power, 1.0, atol=1e-8), np.max(np.abs(res.power - 1))


def test_tri_transient_rejects_levels_engine():
    """The level-scheduled sweep carries one shared cell source per ordinate, so a
    per-ordinate time source needs wider level tables (Phase 3b): refuse clearly
    rather than silently dropping the time source."""
    from ndgpu.tri import TriGrid
    from ndgpu.tri_sn import TriSNTransportSolver

    solver = TransientSNSolver(TriGrid(shape=(8, 8, 2), side=3.0),
                               lambda t: ([TRI_MAT], None), KIN, bc="vacuum",
                               engine_cls=TriSNTransportSolver, engine="levels",
                               **TRI_QUAD)
    with pytest.raises(NotImplementedError, match="engine='lu'"):
        solver.solve(t_end=0.02, dt=0.02)


def test_tri_sigma_t_shift_shifts_only_total_xs():
    """The construction-time shift must land on Sigma_t alone -- within-group
    scattering untouched -- or the transient would silently change the medium."""
    from ndgpu.tri_sn import TriSNTransportSolver

    grid = tri_grid()
    base = TriSNTransportSolver(grid, TRI_MAT, bc="periodic", **TRI_QUAD)
    shifted = TriSNTransportSolver(grid, TRI_MAT, bc="periodic",
                                   sigma_t_shift=[0.37], **TRI_QUAD)
    assert np.allclose(shifted.st - base.st, 0.37)
    assert np.allclose(shifted.ss_self, base.ss_self)
    assert np.allclose(shifted.nsf, base.nsf)
    with pytest.raises(ValueError, match="one value per group"):
        TriSNTransportSolver(grid, TRI_MAT, bc="periodic",
                             sigma_t_shift=[0.1, 0.2], **TRI_QUAD)


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
