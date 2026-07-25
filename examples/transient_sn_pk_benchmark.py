"""Time-dependent S_N transport against exact point kinetics.

The validation benchmark for :class:`ndgpu.TransientSNSolver`. A *uniform*
perturbation of a bare slab leaves the flux shape untouched, so the space-angle
transport equations collapse exactly onto the point-kinetics ODEs -- an
independent reference for the whole transient stack (angular-flux time source,
precursor treatment, critical adjustment, time stepping).

Two subtleties that this benchmark is built around, both transport-specific:

1. Use a **nu*Sigma_f step**, not a Sigma_a step. A transport mode's shape
   depends on the scattering ratio c = Sigma_s/Sigma_t, so changing Sigma_a
   (at fixed Sigma_t = 1/3D) reshapes it; scaling nu*Sigma_f leaves the
   streaming + scattering operator -- and hence the mode -- exactly unchanged.
   (Diffusion is immune: its fundamental mode is geometry-locked.)

2. The naive generation time Lambda = 1/(v a) assumed by the reference ODE is
   exact only as the anisotropy vanishes. On a low-leakage core the deviation is
   pure backward-Euler O(dt); on a leaky core it saturates at a dt-INDEPENDENT
   floor -- the true Lambda is adjoint-weighted. Both regimes are printed below;
   the saturating column is physics, not a discretisation error.

Run:  python examples/transient_sn_pk_benchmark.py [--quick]
"""

import sys

import numpy as np
from scipy.integrate import solve_ivp

from ndgpu import Grid, Kinetics, Material, TransientSNSolver

D, SIGMA_A, NU_SIGMA_F = 1.3, 0.030, 0.035
V, BETA, LAM = 2.2e5, 0.0065, 0.08
BASE = Material(name="1g", diffusion=[D], sigma_a=[SIGMA_A], nu_sigma_f=[NU_SIGMA_F])
KIN = Kinetics(velocities=[V], beta=[BETA], decay=[LAM])
QUAD = dict(n_polar=2, n_azi=4, acceleration="dsa")
RHO_DOLLARS = 0.5

# One huge transverse cell makes the S_N sweep effectively 1-D.
GRIDS = {"low-leak (500 cm)": Grid(shape=(60, 1, 1), size=(500.0, 1e5, 1.0)),
         "leaky (80 cm)":     Grid(shape=(40, 1, 1), size=(80.0, 1e5, 1.0))}


def nsf_step(eps):
    """nu*Sigma_f -> nu*Sigma_f (1 + eps) for t > 0: shape-preserving."""
    pert = [Material(name="pert", diffusion=[D], sigma_a=[SIGMA_A],
                     nu_sigma_f=[NU_SIGMA_F * (1.0 + eps)])]
    return lambda t: (([BASE] if t <= 0 else pert), None)


def point_kinetics(times, a, eps):
    """Exact point kinetics for the production step, from the *unperturbed*
    precursor equilibrium: prompt production (1-beta) a (1+eps), loss a."""
    ap = a * (1.0 + eps)
    rhs = lambda t, y: [V * (((1.0 - BETA) * ap - a) * y[0] + LAM * y[1]),
                        BETA * ap * y[0] - LAM * y[1]]
    return solve_ivp(rhs, (0, times[-1]), [1.0, BETA * a / LAM], method="Radau",
                     t_eval=times, rtol=1e-11, atol=1e-13).y[0]


def run(grid, dt, t_end, eps, k0):
    res = TransientSNSolver(grid, nsf_step(eps), KIN, bc="vacuum",
                            **QUAD).solve(t_end=t_end, dt=dt, tol_step=1e-8)
    # res.power tracks nu*Sigma_f phi, so it reads (1 + eps) n(t) after the step.
    n = res.power / (1.0 + eps)
    ref = point_kinetics(res.times, NU_SIGMA_F / k0, eps)
    err = float(np.max(np.abs(n[1:] - ref[1:]) / ref[1:]))   # t=0 is pre-step
    return res, n, ref, err


def main(quick=False):
    dts = [2e-4, 1e-4] if quick else [4e-4, 2e-4, 1e-4, 5e-5]
    t_end = 0.06
    eps = RHO_DOLLARS * BETA / (1.0 - RHO_DOLLARS * BETA)
    print(f"+${RHO_DOLLARS:.2f} nu*Sigma_f step (eps = {eps:.6f}), "
          f"t_end = {t_end} s, S{2 * 4} quadrature\n")

    k0s = {}
    for name, grid in GRIDS.items():
        k0s[name] = TransientSNSolver(grid, lambda t: ([BASE], None), KIN,
                                      bc="vacuum", **QUAD).solve(
                                          t_end=1e-3, dt=1e-3).k0

    head = "  ".join(f"{n:>22s}" for n in GRIDS)
    print(f"{'dt (s)':>10s}  {head}")
    print(f"{'':>10s}  " + "  ".join(f"{'max dev from PK':>22s}" for _ in GRIDS))
    print("-" * (12 + 24 * len(GRIDS)))
    last = {}
    for dt in dts:
        cells = []
        for name, grid in GRIDS.items():
            _, _, _, err = run(grid, dt, t_end, eps, k0s[name])
            trend = ""
            if name in last:
                trend = f"  ({last[name] / err:4.2f}x)"
            last[name] = err
            cells.append(f"{err:.3e}{trend:>10s}")
        print(f"{dt:>10.0e}  " + "  ".join(f"{c:>22s}" for c in cells))

    print("\nk_eff (S_N steady): " +
          ", ".join(f"{n} = {k:.6f}" for n, k in k0s.items()))
    print("\nlow-leak: halving dt halves the deviation -- backward-Euler O(dt),")
    print("          the solver is limited only by the time discretisation.")
    print("leaky:    the deviation SATURATES -- the naive Lambda = 1/(v a) in the")
    print("          reference ODE is not the transport generation time. Physics,")
    print("          not a discretisation error (it does not shrink with dt).")

    # Prompt jump on the low-leakage core, resolved in-window.
    name = "low-leak (500 cm)"
    res, n, ref, err = run(GRIDS[name], 1e-4, 0.08, eps, k0s[name])
    plateau = 1.0 / (1.0 - RHO_DOLLARS)
    print(f"\nprompt jump ({name}, dt = 1e-4): n(0.08 s) = {n[-1]:.4f} "
          f"vs beta/(beta-rho) = {plateau:.2f}")
    print(f"  max deviation {err:.2e} over {len(res.times) - 1} steps, "
          f"{res.total_inner_iterations} sweeps, "
          f"{np.mean(res.step_iterations):.1f} fixed-point its/step, "
          f"{res.solve_seconds:.1f} s on {res.device}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
