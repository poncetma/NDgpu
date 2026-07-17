"""Transient on the 2D Brantley-Larsen problem across the angular hierarchy.

The steady benchmark (sdpn_brantley_larsen_2d.py, Carreno et al. 2024 Fig. 3)
shows a ~2800 pcm spread between SP1 and transport on this strongly
heterogeneous one-group core. Here the same geometry is driven through a
transient: at t = 0+ the absorption of the CENTRAL fuel bar drops by
d_sigma_a -- a localized reactivity insertion sitting exactly where the
angular treatment matters -- and each method (SP1, SP3, SP5, SDP1, SDP3)
marches its own kinetics from its own critically-adjusted steady state.

Every method sees the same physical perturbation but assigns it a different
static worth rho (the transport effect on the perturbed flux overlap), so the
prompt jumps and delayed rises separate accordingly. Point kinetics with each
method's own rho predicts its plateau beta/(beta - rho); the printed
PK-plateau column checks that each transient is internally consistent with
its static worth.

Usage: python examples/sdpn_brantley_larsen_transient.py [n] [--quick]
       n = cells per side (default 80); --quick = coarse dt, short window.
"""

import sys
import time

import numpy as np

from ndgpu import (Grid, Kinetics, Material, TransientSDPNSolver,
                   TransientSPNSolver)
from ndgpu.solver import SDPNEigenSolver, SPNEigenSolver

# One-group data (paper Table 5) + a perturbed central bar (index 2).
D_SIGMA_A = 0.002                    # central-bar absorption drop at t = 0+
                                     # (~ +$0.45: sub-prompt-critical, so the
                                     # transient shows a resolved prompt jump
                                     # to ~beta/(beta-rho) then a delayed rise)
FUEL = Material(name="fuel", diffusion=[1.0 / (3 * 1.5)], sigma_a=[1.5 - 1.35],
                nu_sigma_f=[0.24], total=[1.5], chi=[1.0])
MOD = Material(name="mod", diffusion=[1.0 / (3 * 1.0)], sigma_a=[1.0 - 0.93],
               nu_sigma_f=[0.0], total=[1.0], chi=[1.0])
PERT = Material(name="fuel-pert", diffusion=[1.0 / (3 * 1.5)],
                sigma_a=[1.5 - 1.35 - D_SIGMA_A], nu_sigma_f=[0.24],
                total=[1.5], chi=[1.0])
BARS = [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)]      # central bar = index 1
M0 = [FUEL, MOD, FUEL]
M1 = [FUEL, MOD, PERT]

# One-group kinetics: thermal speed, one delayed family.
V, BETA, LAM = 2.2e5, 0.0065, 0.08
KIN = Kinetics(velocities=[V], beta=[BETA], decay=[LAM])


def build(n):
    """Material map with the central bar as its own index (2)."""
    h = 10.0 / n
    xc = (np.arange(n) + 0.5) * h
    mmap = np.ones((n, n), int)                       # moderator
    fuel_row = (xc < 9.0)[None, :]
    for i, (lo, hi) in enumerate(BARS):
        col = ((xc > lo) & (xc < hi))[:, None]
        mmap[np.broadcast_to(col & fuel_row, (n, n))] = 2 if i == 1 else 0
    grid = Grid(shape=(n, n, 1), size=(10.0, 10.0, h))
    bc = (("reflective", "vacuum"), ("reflective", "vacuum"), "reflective")
    return grid, mmap[:, :, None], bc


def _eig_cls(family, order):
    return type(f"_{family}{order}",
                (SPNEigenSolver if family == "SPN" else SDPNEigenSolver,),
                {"_order": order})


METHODS = [                     # (label, family, order)
    ("SP1", "SPN", 0), ("SP3", "SPN", 1), ("SP5", "SPN", 2),
    ("SDP1", "SDPN", 1), ("SDP3", "SDPN", 3),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    n = int(args[0]) if args else 80
    t_end, dt = (0.05, 5e-4) if quick else (0.5, 2e-4)
    grid, mmap, bc = build(n)
    problem = lambda t: ((M0 if t <= 0 else M1), mmap)

    print(f"Brantley-Larsen 2D transient: central-bar d_sigma_a = -{D_SIGMA_A}"
          f" at t=0+,\n{n}x{n} mesh, dt = {dt:g} s, t_end = {t_end:g} s, "
          f"v = {V:g} cm/s, beta = {BETA}, lambda = {LAM}/s\n")
    hdr = (f"{'method':7s}{'k0':>9s}{'rho ($)':>10s}{'PK plat.':>9s}"
           f"{'P(0.1s)':>9s}{'P(0.25s)':>9s}{'P(end)':>9s}{'time':>8s}")
    print(hdr)
    print("-" * len(hdr))

    checkpoints = {}
    for label, family, order in METHODS:
        cls = _eig_cls(family, order)
        # Static worth of the step for THIS angular order.
        k0 = cls(grid, M0, material_map=mmap, bc=bc, device="cpu").solve(
            tol_k=1e-9, tol_source=1e-8).k_eff
        k1 = cls(grid, M1, material_map=mmap, bc=bc, device="cpu").solve(
            tol_k=1e-9, tol_source=1e-8).k_eff
        rho = (k1 - k0) / (k0 * k1)
        plateau = BETA / (BETA - rho) if rho < BETA else np.inf

        tcls = TransientSPNSolver if family == "SPN" else TransientSDPNSolver
        t0 = time.perf_counter()
        res = tcls(grid, problem, KIN, order=order, bc=bc,
                   device="cpu").solve(t_end=t_end, dt=dt)
        wall = time.perf_counter() - t0
        def p_at(tq):
            i = np.searchsorted(res.times, tq)
            return res.power[min(i, len(res.power) - 1)]
        print(f"{label:7s}{res.k0:>9.5f}{rho / BETA:>10.4f}{plateau:>9.3f}"
              f"{p_at(0.1):>9.3f}{p_at(0.25):>9.3f}{res.power[-1]:>9.3f}"
              f"{wall:>7.1f}s", flush=True)
        checkpoints[label] = res

    print("\nrho ($) = static (k1-k0)/(k0 k1)/beta for the method's own pair of"
          "\nsteady solves; PK plat. = beta/(beta-rho), the point-kinetics"
          "\nprompt plateau that P(0.1s) should approach if the transient is"
          "\nconsistent with the static worth. P(end) includes the delayed rise.")


if __name__ == "__main__":
    main()
