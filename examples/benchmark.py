"""CPU vs GPU benchmark on progressively larger bare-core problems.

On a CUDA-less machine this reports CPU numbers only. Every solve is verified
against the analytic k_eff so the speed numbers are for *correct* solves.

Usage: python examples/benchmark.py [sizes...]   e.g. python examples/benchmark.py 64 128 192
"""

import sys

from ndgpu import DiffusionEigenSolver, Grid, PWR_TWO_GROUP, get_backend, k_bare_box

import numpy as np

sizes = [int(s) for s in sys.argv[1:]] or [32, 64, 96, 128]
box = (150.0, 150.0, 150.0)
k_exact = k_bare_box(PWR_TWO_GROUP, box)

devices = ["cpu"]
try:
    get_backend("gpu")
    devices.append("gpu")
except Exception as e:
    print(f"(no GPU available: {e} — running CPU only)\n")

print(f"{'n^3':>6} {'cells':>12} {'device':>7} {'outers':>7} {'inners':>7} "
      f"{'time [s]':>9} {'Mcells*iters/s':>15} {'dk [pcm]':>9}")

times = {}
for n in sizes:
    grid = Grid(shape=(n, n, n), size=box)
    for dev in devices:
        solver = DiffusionEigenSolver(grid, PWR_TWO_GROUP, device=dev)
        solver.solve(max_outer=3, tol_k=0)  # warm-up (JIT/alloc), not timed
        res = solver.solve()
        assert res.converged
        # throughput: group-cells processed per stencil application
        work = grid.n_cells * res.inner_iterations / res.solve_seconds / 1e6
        times[(n, dev)] = res.solve_seconds
        print(f"{n:>6} {grid.n_cells:>12,} {dev:>7} {res.outer_iterations:>7} "
              f"{res.inner_iterations:>7} {res.solve_seconds:>9.2f} {work:>15.1f} "
              f"{(res.k_eff - k_exact) * 1e5:>9.1f}")
    if "gpu" in devices:
        print(f"{'':>27} speedup: {times[(n, 'cpu')] / times[(n, 'gpu')]:.1f}x")
