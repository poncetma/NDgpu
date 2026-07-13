"""Reflected slab reactor: an analytic diffusion benchmark, solved two ways.

The one-group reflected slab (Lamarsh, Introduction to Nuclear Reactor Theory,
Ch. 7) has an exact eigenvalue from a transcendental condition, so it checks the
code against mathematics rather than another code. Because the geometry is a
filled 1-D slab, it runs on both the low-level DiffusionEigenSolver and the
high-level Model API; both are compared with the analytic value here, and the
reflector's effect (reflector savings) is shown against a bare core.

Usage: python examples/reflected_slab.py [cells] [cpu|gpu|auto]
"""

import sys

import ndgpu
from ndgpu import DiffusionEigenSolver
from ndgpu.benchmarks import (bare_k, build_reflected_slab, reflected_k,
                             SLAB_CORE, SLAB_REFLECTOR)

cells = int(sys.argv[1]) if len(sys.argv) > 1 else 90
device = sys.argv[2] if len(sys.argv) > 2 else "auto"

k_ref = reflected_k()
k_bare = bare_k()
print("Reflected slab reactor (one group)")
print(f"  core: k_inf = {SLAB_CORE.nu_sigma_f[0] / SLAB_CORE.sigma_a[0]:.4f}, "
      f"25 cm half-width; reflector: 20 cm\n")
print(f"  analytic reflected k = {k_ref:.6f}")
print(f"  analytic bare-core k = {k_bare:.6f}   "
      f"(reflector savings = {(k_ref - k_bare) * 1e5:+.0f} pcm)\n")

# --- without the Model API: the low-level solver on the benchmark's grid -----
p = build_reflected_slab(cells=cells)
k_low = DiffusionEigenSolver(p.grid, p.materials, material_map=p.material_map,
                             bc=p.bc, device=device).solve(tol_k=1e-10, tol_source=1e-9).k_eff
print(f"  DiffusionEigenSolver ({cells} cells) : k = {k_low:.6f}   "
      f"({(k_low - p.k_reference) * 1e5:+.2f} pcm vs analytic)")

# --- with the Model API: the same slab as a 1-D Model ------------------------
a, b = p.core_half_width, p.reflector_width
model = (ndgpu.Model(size=(a + b,), cells=(cells,))
         .fill(SLAB_CORE).add_box(SLAB_REFLECTOR, x=(a, a + b))
         .set_boundary(x=("reflective", "zero-flux")))
print(model.run(device=device, tol_k=1e-10, tol_source=1e-9))
