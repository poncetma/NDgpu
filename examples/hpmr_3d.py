"""3D HP-MR microreactor: k-eigenvalue, drum worth, axial flux profile.

Full-height model of the ANL/INL heat-pipe microreactor: the 2D radial core
extruded to 200 cm as triangular prisms (160 cm fueled + 20 cm Be axial
reflectors, drums and their B4C arcs running the full height, vacuum z faces).

    python examples/hpmr_3d.py [refine] [nz] [absorber] [device]

absorber (default "polar") selects the drum B4C-arc treatment: "polar"
volume-mixes the exact polar area fraction (smooth, mesh-convergent worth) and
"raster" stamps whole cells. Cross sections are placeholders; swap in
SPH-corrected FEMFFUSION sets via ndgpu.femffusion + build_hpmr3d(materials=...).
"""

import sys

import numpy as np

from ndgpu.benchmarks import build_hpmr3d
from ndgpu.benchmarks.hpmr import FUEL
from ndgpu.tri import TriDiffusionEigenSolver

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
nz = int(sys.argv[2]) if len(sys.argv) > 2 else 20
absorber = sys.argv[3] if len(sys.argv) > 3 else "polar"
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"


def solve(angle):
    p = build_hpmr3d(refine=refine, nz=nz, drum_angle_deg=angle, absorber=absorber)
    res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                  active=p.active, mask_bc=p.mask_bc, bc=p.bc,
                                  mix_material=p.mix_material,
                                  mix_weight=p.mix_weight, device=device).solve(
        tol_k=1e-7, tol_source=1e-6)
    if not res.converged:
        raise SystemExit(f"not converged at {angle} deg: {res}")
    return p, res


p, r_out = solve(0.0)
print(f"HP-MR 3D, refine={refine}, nz={nz} (dz={p.grid.dz:g} cm), "
      f"{p.grid.n_cells} cells, absorber={absorber!r}, device={device}\n")
print(f"drums out (0°):   k_eff = {r_out.k_eff:.5f}   [{r_out.solve_seconds:.1f} s]")
_, r_in = solve(180.0)
worth = (1 / r_in.k_eff - 1 / r_out.k_eff) * 1e5
print(f"drums in (180°):  k_eff = {r_in.k_eff:.5f}   [{r_in.solve_seconds:.1f} s]")
print(f"total drum worth: {worth:.0f} pcm\n")

fuel = p.material_map == FUEL
prof = (r_out.flux_numpy[1] * fuel).sum(axis=(0, 1, 2))
prof /= prof.max()
print("axial thermal-flux profile in the fuel (z upward):")
for k in range(nz - 1, -1, -1):
    z = (k + 0.5) * p.grid.dz
    tag = "axial refl" if prof[k] == 0 else ""
    print(f"  z = {z:5.1f} cm  {'#' * int(round(prof[k] * 50)):<50} {tag}")

print("\n(placeholder cross sections: shapes and worths are illustrative,"
      "\n not predictive; see ndgpu/benchmarks/hpmr.py)")
