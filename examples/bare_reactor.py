"""Bare homogeneous two-group PWR-like cube via the simplified Model API.

Compares the computed k_eff with the exact analytic value and prints NDgpu's
human-readable solution report. Runs on GPU automatically when CUDA + CuPy are
present.

Usage: python examples/bare_reactor.py [n_cells_per_axis] [cpu|gpu|auto]
"""

import sys

import ndgpu
from ndgpu import PWR_TWO_GROUP, k_bare_box

n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
device = sys.argv[2] if len(sys.argv) > 2 else "auto"

size = (150.0, 150.0, 150.0)  # cm

# Define the reactor: one material filling a cube, zero-flux (bare) boundary.
model = ndgpu.Model(size=size, cells=(n, n, n)).fill(PWR_TWO_GROUP).set_boundary("zero-flux")
result = model.run(device=device)

print(result)                                    # the transparent solution report

k_exact = k_bare_box(PWR_TWO_GROUP, size)
print(f"\n  analytic k_eff : {k_exact:.6f}")
print(f"  difference     : {(result.k_eff - k_exact) * 1e5:+.1f} pcm "
      f"(2nd-order discretization error at {n}^3)")
