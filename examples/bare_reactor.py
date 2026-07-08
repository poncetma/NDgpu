"""Solve a bare homogeneous two-group PWR-like core and compare with the
exact analytic k_eff. Runs on GPU automatically when CUDA + CuPy are present.

Usage: python examples/bare_reactor.py [n_cells_per_axis] [cpu|gpu|auto]
"""

import sys

from ndgpu import DiffusionEigenSolver, Grid, PWR_TWO_GROUP, k_bare_box

n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
device = sys.argv[2] if len(sys.argv) > 2 else "auto"

size = (150.0, 150.0, 150.0)  # cm
grid = Grid(shape=(n, n, n), size=size)

solver = DiffusionEigenSolver(grid, PWR_TWO_GROUP, device=device)
print(f"Bare {size[0]:.0f} cm cube, {n}^3 = {grid.n_cells:,} cells x 2 groups "
      f"on {solver.device}")

res = solver.solve(verbose=False)
k_exact = k_bare_box(PWR_TWO_GROUP, size)

print(res)
print(f"  k_eff (numeric) = {res.k_eff:.6f}")
print(f"  k_eff (exact)   = {k_exact:.6f}")
print(f"  difference      = {(res.k_eff - k_exact) * 1e5:+.1f} pcm")

flux = res.flux_numpy
mid = n // 2
fast, thermal = flux[0, :, :, mid], flux[1, :, :, mid]
print(f"  fast/thermal flux ratio at core center: "
      f"{flux[0, mid, mid, mid] / flux[1, mid, mid, mid]:.2f}")
print(f"  midplane thermal flux peaking factor:   "
      f"{thermal.max() / thermal.mean():.2f}")
