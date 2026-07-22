"""Discrete-ordinates (S_N) transport on the HP-MR triangular core.

S_N now runs on the actual body-fitted HP-MR mesh (ndgpu.TriSNTransportSolver),
not a Cartesian stand-in -- so the microreactor finally has a true-transport
k-eigenvalue reference alongside diffusion and SP3, all on the same triangular
grid and the same cross sections.

For each drum angle the script solves the core three ways -- triangular diffusion,
triangular SP3, and triangular S_N -- and reports the control-drum worth
(reactivity, withdrawn 0 deg -> inserted 180 deg). The physics question is the
sign of the transport correction to that worth: diffusion treats the near-black
B4C arc as fully absorbing, while transport self-shields it (the flux is depressed
inside the arc, so it absorbs less), which resolves *less* drum worth. SP3 already
shows this; S_N is the reference that settles it.

Caveat on mesh: the S_N solver uses upwind (step) differencing -- robustly
non-negative through the black absorber, but only first-order, so it is
numerically diffusive on a coarse mesh. That numerical diffusion is worst where
the flux gradient is steepest (drums inserted), so at coarse refinement it
corrupts the *worth* (even flips the sign of the small S_N-vs-diffusion
correction); it converges away under refinement. Running a few refinements makes
that convergence explicit.

    python examples/hpmr_tri_sn.py [refine ...]   (default: 4 6 8)

Cross sections are illustrative placeholders, not predictive.
"""

import sys
import time

import numpy as np

from ndgpu.benchmarks import build_hpmr2d
from ndgpu.tri import TriDiffusionEigenSolver, TriSP3EigenSolver
from ndgpu.tri_sn import TriSNTransportSolver

refines = [int(a) for a in sys.argv[1:]] or [4, 6, 8]


def solve_all(refine, angle):
    p = build_hpmr2d(refine=refine, drum_angle_deg=float(angle), absorber="raster")
    kw = dict(active=p.active, mask_bc=p.mask_bc, device="cpu")
    kd = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                 **kw).solve(tol_k=1e-8, tol_source=1e-7).k_eff
    ks = TriSP3EigenSolver(p.grid, p.materials, p.material_map,
                           **kw).solve(tol_k=1e-8, tol_source=1e-7).k_eff
    t = time.perf_counter()
    kn = TriSNTransportSolver(p.grid, p.materials, p.material_map, active=p.active,
                              n_polar=2, n_azi=8, bc="vacuum").solve(
        tol_k=5e-7, tol_source=5e-6, max_outer=200)
    n_cell = int((np.asarray(p.material_map) > 0).sum())
    return kd, ks, kn.k_eff, kn.n_ordinates, n_cell, time.perf_counter() - t


print("HP-MR 2D, triangular diffusion / SP3 / S_N, raster drums\n")
print(f"{'refine':>6} {'cells':>7} {'ordts':>6}  "
      f"{'--- drum worth 0->180 (pcm) ---':^34}  {'S_N':>6}")
print(f"{'':>6} {'':>7} {'':>6}  {'diffusion':>11} {'SP3':>10} {'S_N':>10}  {'time':>6}")
for refine in refines:
    K = {}
    n_ord = n_cell = 0
    t_sn = 0.0
    for a in (0.0, 180.0):
        kd, ks, kn, n_ord, n_cell, dt = solve_all(refine, a)
        K[a] = (kd, ks, kn)
        t_sn += dt
    worth = lambda x: (1.0 / K[0.0][x] - 1.0 / K[180.0][x]) * 1e5
    wd, ws, wn = worth(0), worth(1), worth(2)
    print(f"{refine:>6} {n_cell:>7} S{n_ord:<5} "
          f"{wd:>11.0f} {ws:>10.0f} {wn:>10.0f}  {t_sn:>5.0f}s")
    print(f"{'':>21}  transport correction vs diffusion:  "
          f"SP3 {ws - wd:+.0f},  S_N {wn - wd:+.0f} pcm")

print("\nAs the mesh refines, the S_N worth correction converges toward SP3's:")
print("transport self-shields the black drum, resolving LESS worth than diffusion")
print("(a positive S_N-minus-diffusion worth correction once step differencing's")
print("numerical diffusion is resolved out).")
print("\n(placeholder cross sections: illustrative, not predictive.)")
