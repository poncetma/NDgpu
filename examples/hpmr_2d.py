"""2D HP-MR microreactor: k-eigenvalue and control-drum worth curve.

Assembly-level radial model of the ANL/INL heat-pipe microreactor (VTB
reference design) on a body-fitted triangular mesh, with the 12 control
drums' B4C arcs rotated together from fully inserted (0 deg, arcs facing the
core) to fully withdrawn (180 deg, arcs facing outward).

    python examples/hpmr_2d.py [refine] [absorber] [device]

refine (default 6) sets triangles per hex (6 refine^2). absorber (default
"polar") selects how the B4C arc is represented: "raster" stamps whole cells
by centroid (staircase worth curve); "polar" volume-mixes the exact polar area
fraction into the drum cells, giving a smooth worth-vs-angle curve that
converges faster and works below the raster's refine>=4 floor. Cross sections
are placeholders; swap in SPH-corrected sets via build_hpmr2d(materials=).
"""

import sys

from ndgpu.benchmarks import build_hpmr2d
from ndgpu.tri import TriDiffusionEigenSolver

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 6
absorber = sys.argv[2] if len(sys.argv) > 2 else "polar"
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"

print(f"HP-MR 2D, refine={refine} ({6 * refine**2} triangles/assembly), "
      f"absorber={absorber!r}, device={device}\n")
print(f"{'drum angle':>10} {'k_eff':>10} {'reactivity vs 0 (pcm)':>22}")

k0 = None
for angle in (0, 30, 60, 90, 120, 150, 180):
    p = build_hpmr2d(refine=refine, drum_angle_deg=float(angle), absorber=absorber)
    res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                  active=p.active, mask_bc=p.mask_bc,
                                  mix_material=p.mix_material, mix_weight=p.mix_weight,
                                  device=device).solve(tol_k=1e-8, tol_source=1e-7)
    if not res.converged:
        raise SystemExit(f"not converged at {angle} deg: {res}")
    k0 = k0 or res.k_eff
    rho = (1 / k0 - 1 / res.k_eff) * 1e5
    print(f"{angle:>9}° {res.k_eff:>10.5f} {rho:>22.0f}")

print("\n(placeholder cross sections: shapes and worths are illustrative,"
      "\n not predictive; see ndgpu/benchmarks/hpmr.py)")
