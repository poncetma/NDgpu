"""Pin-resolved C5G7 spatial-convergence sweep (area-weighted cylinders).

Refines cells_per_pin for a chosen method and streams k_eff, dk vs the OpenMC
reference, and wall time after each run, plus a running Richardson
extrapolation fitting k(s) = k_inf + a/s + b/s^2 (the interface band mixes a
first- and a second-order error term; on this problem the h^2 term dominates).
The goal is a spatially converged pin-resolved k that can be compared directly
with Carreno et al. (2024) Table 7.

Usage: python examples/c5g7_meshconv.py METHOD s1 s2 s3 ...
       METHOD in diffusion|sp3|sp5|sp7|sdp1|sdp2|sdp3
"""

import sys
import time

import numpy as np

from ndgpu import (DiffusionEigenSolver, SP1EigenSolver, SP3EigenSolver,
                   SP5EigenSolver, SP7EigenSolver, SDP1EigenSolver,
                   SDP2EigenSolver, SDP3EigenSolver)
from ndgpu.benchmarks import K_REFERENCE_2D, build_c5g7_2d

SOLVERS = {"diffusion": DiffusionEigenSolver, "sp1": SP1EigenSolver,
           "sp3": SP3EigenSolver, "sp5": SP5EigenSolver, "sp7": SP7EigenSolver,
           "sdp1": SDP1EigenSolver, "sdp2": SDP2EigenSolver,
           "sdp3": SDP3EigenSolver}
PAPER = {"sp1": 1.18336, "sp3": 1.18271, "sp5": 1.18221, "sp7": 1.18231,
         "sdp1": 1.18242, "sdp2": 1.18219, "sdp3": 1.18286}


def richardson(ss, ks):
    """k_inf from a least-squares fit of k(s) = k_inf + a/s + b/s^2 over the
    (up to 4) finest points; with only two points, a plain 1/s^2 fit."""
    if len(ss) < 2:
        return None
    x = 1.0 / np.array(ss[-4:], float)
    y = np.array(ks[-4:], float)
    cols = [np.ones_like(x), x**2] if len(x) < 3 else [np.ones_like(x), x, x**2]
    kinf = np.linalg.lstsq(np.vstack(cols).T, y, rcond=None)[0][0]
    return kinf


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "diffusion"
    sizes = [int(x) for x in sys.argv[2:]] or [8, 12, 16, 20, 24, 28]
    cls = SOLVERS[method]
    pk = PAPER.get(method)
    ptxt = f"  paper k={pk:.5f}" if pk else ""
    print(f"C5G7 pin-resolved, method={method}, ref k={K_REFERENCE_2D:.5f}{ptxt}\n")
    print(f"{'s':>3} {'grid':>10} {'k_eff':>10} {'dk (pcm)':>10} "
          f"{'Richardson':>11} {'time':>9}")
    print("-" * 58)
    ss, ks = [], []
    for s in sizes:
        prob = build_c5g7_2d(cells_per_pin=s, pin_resolved=True)
        t0 = time.perf_counter()
        res = cls(prob.grid, prob.materials, prob.material_map, bc=prob.bc,
                  device="cpu", mix_material=prob.mix_material,
                  mix_weight=prob.mix_weight).solve(tol_k=1e-7, tol_source=1e-6)
        dt = time.perf_counter() - t0
        ss.append(s); ks.append(res.k_eff)
        rich = richardson(ss, ks)
        rtxt = f"{rich:.5f}" if rich is not None else "--"
        dk = (res.k_eff - K_REFERENCE_2D) / K_REFERENCE_2D * 1e5
        print(f"{s:>3} {prob.grid.shape[0]:>5}^2 {res.k_eff:>10.5f} "
              f"{dk:>10.0f} {rtxt:>11} {dt:>8.1f}s", flush=True)


if __name__ == "__main__":
    main()
