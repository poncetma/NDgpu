"""3D HP-MR microreactor, discrete-ordinates (S_N) transport.

The 2D radial core extruded to triangular prisms (160 cm fuel + 20 cm Be axial
reflectors, drums and their B4C arcs full height, vacuum z faces) solved by the
prism S_N transport solver rather than diffusion -- the transport reference on
the real body-fitted core.

    python examples/hpmr_3d_sn.py [refine] [nz] [scheme] [device]

* scheme   : "step" (1st order, robust; the levels/GPU sweep) or "scb"
             (2nd order; CPU/LU today, GPU via the levels engine).
* device   : "cpu" or "gpu" (levels engine + CUDA graphs + device DSA/MG-CMFD).
* cmfd_solver="mg" (multigrid, O(N)) is used automatically for the larger prism
  counts where the drift-matrix sparse LU is O(N^2)/O(N^4/3) fill and blows up.

Cross sections are illustrative placeholders (swap SPH-corrected FEMFFUSION sets
via build_hpmr3d(materials=...)); report drum WORTH, not absolute k.
"""

import sys
import time

import numpy as np

from ndgpu.benchmarks import build_hpmr3d
from ndgpu.tri import TriDiffusionEigenSolver
from ndgpu.tri_sn import TriSNTransportSolver

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 2
nz = int(sys.argv[2]) if len(sys.argv) > 2 else 10
scheme = sys.argv[3] if len(sys.argv) > 3 else "step"
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"

# multigrid CMFD once the prism count makes the drift-matrix LU (O(N^2),
# O(N^4/3) fill) expensive; LU is faster on the smaller meshes.
CMFD = "mg" if (refine >= 4 or nz >= 20) else "lu"
TOL = dict(tol_k=5e-6, tol_source=5e-5, max_outer=100)


def solve_sn(angle):
    p = build_hpmr3d(refine=refine, nz=nz, drum_angle_deg=angle, absorber="polar")
    # bc="vacuum": the physical boundary is the excised void border + vacuum z
    # faces (the reflective grid edge in p.bc is never reached).
    s = TriSNTransportSolver(
        p.grid, p.materials, p.material_map, active=p.active,
        bc="vacuum", mix_material=p.mix_material, mix_weight=p.mix_weight,
        n_polar=2, n_azi=8, scheme=scheme, engine="levels", device=device,
        cmfd_solver=CMFD)
    t = time.perf_counter()
    r = s.solve(**TOL)
    return p, r, time.perf_counter() - t


print(f"HP-MR 3D S_N: refine={refine} nz={nz} scheme={scheme} device={device} "
      f"cmfd={CMFD}")
p, r, dt = solve_sn(0.0)                                  # drums inserted (arc at core)
print(f"  active prisms : {int(p.active.sum())}")
print(f"  k_eff (drums IN) = {r.k_eff:.5f}  ({r.outer_iterations} outers, "
      f"{r.n_sweeps} sweeps, {dt:.1f}s, converged={r.converged})")

# axial flux shape (group-1 fast flux, core-averaged per layer) -- the 20 cm Be
# reflectors flatten the ends; the fuel span cosines.
fz = r.flux[0].sum(axis=(0, 1, 2))
print("  axial flux (per layer, normalised): "
      + " ".join(f"{x:.2f}" for x in fz / fz.max()))

# drum worth: withdrawn (arc outward, 180 deg) minus inserted, in pcm of 1/k
_, r_out, _ = solve_sn(180.0)
worth = (1.0 / r.k_eff - 1.0 / r_out.k_eff) * 1e5
print(f"  k_eff (drums OUT) = {r_out.k_eff:.5f}   drum worth = {worth:+.0f} pcm")

# diffusion cross-check on the same core (its P1 limit)
kd = TriDiffusionEigenSolver(
    p.grid, p.materials, p.material_map, active=p.active, mask_bc=p.mask_bc,
    bc=p.bc, mix_material=p.mix_material, mix_weight=p.mix_weight,
    device="cpu").solve(tol_k=1e-6, tol_source=1e-5).k_eff
print(f"  3D diffusion k (drums IN) = {kd:.5f}   S_N - diffusion = "
      f"{1e5 * (r.k_eff - kd):+.0f} pcm")
