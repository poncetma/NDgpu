"""Benchmark: DSA, CMFD and the vectorized wavefront sweep in the S_N solvers.

Cartesian: two problems, each solved by every (acceleration, sweep, outer)
combination:

  * absorber -- the resolved absorber block in fuel from the verification suite
    at 48 x 48 (vacuum, 1 group): moderate scattering, a genuine transport
    problem with steep gradients;
  * scattering -- a homogeneous scattering-dominated box (c = 0.99, 32 x 32,
    vacuum): the regime where plain source iteration contracts at ~c per sweep
    and synthetic acceleration is the textbook fix.

The reference for the pcm columns is the pre-DSA solver: acceleration="gmres"
with the per-direction banded row sweep (the original implementation). All
combinations discretize the identical problem, so Delta-k should be at the
convergence-tolerance level (<~ 0.1 pcm); the cost columns (outers, transport
sweeps, wall time) are where the schemes differ.

Triangular: a scattering-dominated 2-group vacuum tile (Be-reflector-like
thermal scattering ratio) on step and SCB differencing, comparing the same
within-group accelerations (reference: "gmres", the original tri scheme; the
tri outer is already Anderson-accelerated, so no CMFD there).

Run:  python examples/sn_dsa_benchmark.py
"""

import numpy as np

from ndgpu import Grid, Material, SNTransportSolver
from ndgpu.tri import TriGrid
from ndgpu.tri_sn import TriSNTransportSolver

TOLS = dict(tol_k=1e-7, tol_source=1e-6)
CASES = [
    ("gmres + rows (ref)", dict(acceleration="gmres", sweep="rows")),
    ("gmres + wavefront", dict(acceleration="gmres", outer_acceleration="power")),
    ("si    + wavefront", dict(acceleration="si", outer_acceleration="power")),
    ("dsa   + wavefront", dict(acceleration="dsa", outer_acceleration="power")),
    ("dsa-gmres + wavefront", dict(acceleration="dsa-gmres",
                                   outer_acceleration="power")),
    ("dsa + wavefront + cmfd", dict(acceleration="dsa")),      # the defaults
]


def absorber_problem():
    n = 48
    grid = Grid(shape=(n, n, 1), size=(float(n), float(n), 1.0))
    fuel = Material(diffusion=[1.1], sigma_a=[0.012], nu_sigma_f=[0.026],
                    sigma_s=[[0.0]], name="fuel")
    absb = Material(diffusion=[0.9], sigma_a=[0.20], nu_sigma_f=[0.0],
                    sigma_s=[[0.0]], name="absorber")
    mmap = np.zeros((n, n, 1), int)
    lo, hi = int(n * 0.4), int(n * 0.6)
    mmap[lo:hi, lo:hi, 0] = 1
    return grid, [fuel, absb], mmap, "absorber 48x48, 1g, vacuum"


def scattering_problem():
    n = 32
    grid = Grid(shape=(n, n, 1), size=(float(n), float(n), 1.0))
    m = Material(diffusion=[1.0 / 3.0], sigma_a=[0.01], nu_sigma_f=[0.015],
                 sigma_s=[[0.0]])                       # Sigma_t = 1, c = 0.99
    return grid, [m], np.zeros((n, n, 1), int), "homogeneous 32x32, c=0.99, vacuum"


def run(problem):
    grid, mats, mmap, label = problem()
    print(f"\n== {label} ==")
    print(f"{'scheme':<24} {'k_eff':>10} {'dpcm':>7} {'outers':>7} "
          f"{'sweeps':>7} {'time [s]':>9}")
    k_ref = None
    for name, kw in CASES:
        s = SNTransportSolver(grid, mats, material_map=mmap,
                              n_polar=2, n_azi=8, bc="vacuum", **kw)
        r = s.solve(**TOLS)
        if k_ref is None:
            k_ref = r.k_eff
        flag = "" if r.converged else "  NOT CONVERGED"
        print(f"{name:<24} {r.k_eff:>10.6f} {(r.k_eff - k_ref) * 1e5:>7.2f} "
              f"{r.outer_iterations:>7d} {r.n_sweeps:>7d} "
              f"{r.solve_seconds:>9.2f}{flag}")


def run_tri():
    m = Material(diffusion=[1.4, 0.4], sigma_a=[0.005, 0.015],
                 nu_sigma_f=[0.004, 0.02], sigma_s=[[0.0, 0.025], [0.0, 0.0]],
                 chi=[1.0, 0.0], name="soft")
    for scheme, n in (("step", 14), ("scb", 10)):
        grid = TriGrid(shape=(n, n, 2), side=24.0 / n)
        print(f"\n== tri {scheme} {n}x{n}x2, 2g scattering-dominated, vacuum ==")
        print(f"{'scheme':<24} {'k_eff':>10} {'dpcm':>7} {'outers':>7} "
              f"{'sweeps':>7} {'time [s]':>9}")
        k_ref = None
        for acc in ("gmres", "si", "dsa", "dsa-gmres"):
            s = TriSNTransportSolver(grid, m, n_polar=2, n_azi=8, bc="vacuum",
                                     scheme=scheme, acceleration=acc)
            r = s.solve(**TOLS)
            if k_ref is None:
                k_ref = r.k_eff
            name = acc + (" (ref)" if acc == "gmres" else "")
            flag = "" if r.converged else "  NOT CONVERGED"
            print(f"{name:<24} {r.k_eff:>10.6f} {(r.k_eff - k_ref) * 1e5:>7.2f} "
                  f"{r.outer_iterations:>7d} {r.n_sweeps:>7d} "
                  f"{r.solve_seconds:>9.2f}{flag}")


if __name__ == "__main__":
    run(absorber_problem)
    run(scattering_problem)
    run_tri()
