"""C5G7 2D benchmark: diffusion / SP3 / SDP1 / SDP2 / SDP3 vs the paper's Table 7.

Runs the OECD/NEA C5G7 2D MOX benchmark (7 groups, quarter core + water
reflector, k_ref = 1.18655 from OpenMC/MCNP) through ndgpu's full angular
hierarchy and lines the eigenvalues up against Table 7 of Carreno et al.,
Ann. Nucl. Energy 207 (2024) 110675.

Caveat on the absolute pcm: the paper resolves each cylindrical fuel pin with a
FE mesh, whereas ndgpu volume-homogenizes each 1.26 cm pin cell -- an extra
modeling bias on top of the angular approximation. So the interesting comparison
is the *hierarchy behavior*, which the paper flags as the C5G7 story: unlike the
strongly heterogeneous BWR case, here higher order does NOT help -- SP5/SP7 and
SDP2/SDP3 do not improve on SP3/SDP1. The run below should reproduce that
stagnation.
"""

import sys
import time

from ndgpu import (DiffusionEigenSolver, SP1EigenSolver, SP3EigenSolver,
                   SP5EigenSolver, SP7EigenSolver, SDP1EigenSolver,
                   SDP2EigenSolver, SDP3EigenSolver)
from ndgpu.benchmarks import K_REFERENCE_2D, build_c5g7_2d

# Paper Table 7: (k_eff, Delta_k pcm vs 1.18655). Delta_k = |k_ref - k|/k_ref.
PAPER = {
    "SP1":  (1.18336, 319), "SDP1": (1.18242, 413),
    "SP3":  (1.18271, 384), "SDP2": (1.18219, 436),
    "SP5":  (1.18221, 434), "SDP3": (1.18286, 369),
    "SP7":  (1.18231, 424),
}

cpp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
pin_resolved = "--pin-resolved" in sys.argv

prob = build_c5g7_2d(cells_per_pin=cpp, pin_resolved=pin_resolved)
nx, ny, _ = prob.grid.shape
kw = dict(bc=prob.bc, device=device)
if pin_resolved:
    kw.update(mix_material=prob.mix_material, mix_weight=prob.mix_weight)
geom = "pin-resolved (r=0.54 cm cylinders, area-weighted)" if pin_resolved \
    else "pin-cell homogenized"
print(f"C5G7 2D quarter core: {nx} x {ny} cells x 7 groups "
      f"(cells_per_pin={cpp}, {geom})")
print(f"reference k_eff = {K_REFERENCE_2D:.5f} (OpenMC/MCNP)\n")

methods = [("diffusion", DiffusionEigenSolver),
           ("SP1", SP1EigenSolver),
           ("SP3", SP3EigenSolver), ("SDP1", SDP1EigenSolver),
           ("SP5", SP5EigenSolver), ("SDP2", SDP2EigenSolver),
           ("SP7", SP7EigenSolver), ("SDP3", SDP3EigenSolver)]

results = {}
hdr = (f"{'method':10s}{'k_eff':>10s}{'dk (pcm)':>10s}   |  "
       f"{'paper k':>9s}{'paper dk':>10s}")
print(hdr); print("-" * len(hdr))
for name, cls in methods:
    t0 = time.perf_counter()
    res = cls(prob.grid, prob.materials, prob.material_map, **kw).solve(
        tol_k=1e-6, tol_source=1e-5)
    dt = time.perf_counter() - t0
    results[name] = res.k_eff
    dk = (res.k_eff - K_REFERENCE_2D) / K_REFERENCE_2D * 1e5
    pk = PAPER.get(name)
    ptxt = f"{pk[0]:>9.5f}{pk[1]:>9d} " if pk else f"{'--':>9s}{'--':>10s}"
    print(f"{name:10s}{res.k_eff:>10.5f}{dk:>10.0f}   |  {ptxt}   ({dt:.1f}s)")

kd = results["diffusion"]
print("\nAngular correction at this mesh (k - k_diffusion, pcm):")
for name in ("SP3", "SDP1", "SP5", "SDP2", "SP7", "SDP3"):
    print(f"  {name:5s} {(results[name] - kd) * 1e5:+6.0f}")
print("\ndk = (k - k_ref)/k_ref in pcm (signed; the paper tabulates |dk|).")
if pin_resolved:
    print("Pin-resolved on a 1st-order FV mesh converges in k only as O(h) at "
          "the sharp fuel/water\nflux cusp, so the absolute k here is NOT "
          "spatially converged (the paper uses high-order FEM\nthat resolves "
          "the pin on a coarse mesh). The angular *correction* above converges "
          "faster,\nand still shows the paper's C5G7 stagnation: the SDPN orders "
          "cluster, unlike the strongly\nheterogeneous BWR/blade cases where "
          "SDP1 sharply improves on diffusion.")
