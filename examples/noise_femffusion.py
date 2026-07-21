"""Cross-check the frequency-domain noise solver against FEMFFUSION.

Replicates FEMFFUSION's 1D two-group noise regression case (test/1D_noise_SPN,
diffusion: a 1 Hz absorption fluctuation in one cell of a 300 cm slab) and
compares the complex flux noise -- field by field -- against FEMFFUSION's own
output, reporting accuracy and wall time.

Usage: python examples/noise_femffusion.py [cpu|gpu|auto] [cells]
"""

import sys
import time

import numpy as np

from ndgpu import NoiseSolver, NoiseSource
from ndgpu.benchmarks import build_femffusion_1d_noise

device = sys.argv[1] if len(sys.argv) > 1 else "auto"
cells = int(sys.argv[2]) if len(sys.argv) > 2 else 60
# "diffusion" vs FEMFFUSION diffusion; "sp3" vs FEMFFUSION Full_SPN (coupled
# Marshak vacuum).
angular = sys.argv[3] if len(sys.argv) > 3 else "diffusion"

bench = build_femffusion_1d_noise(cells=cells, angular=angular)
ns = NoiseSolver(bench.grid, bench.materials, bench.material_map,
                 kinetics=bench.kinetics, bc=bench.bc, angular=bench.angular,
                 marshak_vacuum=bench.marshak_vacuum, device=device)

ts = []
for _ in range(3):
    t0 = time.perf_counter()
    res = ns.solve(NoiseSource(d_sigma_a=bench.d_sigma_a),
                   2.0 * np.pi * bench.frequency_hz, tol=1e-10)
    ts.append(time.perf_counter() - t0)
t_solve = min(ts)

# Match FEMFFUSION's static-flux normalization (least squares on group 1), then
# compare the absolute complex noise. delta-phi is compared at the 60 reference
# cell centres; if the mesh is finer, interpolate the ndgpu field onto them.
xc = (np.arange(cells) + 0.5) * (300.0 / cells)
xref = (np.arange(60) + 0.5) * 5.0
interp = lambda z: np.interp(xref, xc, z.real) + 1j * np.interp(xref, xc, z.imag)
phi1 = interp(np.asarray(ns.flux0[0]).ravel().astype(complex)).real
scale = np.dot(bench.static_flux_ref, phi1) / np.dot(phi1, phi1)

bnd = "Marshak vacuum" if bench.marshak_vacuum else "Robin vacuum"
print(f"FEMFFUSION 1D 2-group noise, angular={angular} ({bnd}), {cells} cells, "
      f"{bench.frequency_hz} Hz, on {ns.device}")
print(f"  k_eff: ndgpu {ns.k_eff:.6f}  FEMFFUSION {bench.k_eff_ref:.6f}  "
      f"(dk {abs(ns.k_eff - bench.k_eff_ref) * 1e5:.2f} pcm)")
print(f"  noise solve: {t_solve * 1e3:.1f} ms, {res.sweeps} sweeps, "
      f"{res.inner_iterations} inner COCG iters   "
      f"(FEMFFUSION reference run: ~13 ms, 47 GMRES iters)")
for g in range(2):
    d = scale * interp(np.asarray(res.d_flux_numpy()[g]).ravel())
    ref = bench.d_flux_ref[g]
    rel_l2 = np.linalg.norm(d - ref) / np.linalg.norm(ref)
    ip = int(np.argmax(np.abs(ref)))
    da = abs(d[ip]) / abs(ref[ip]) - 1.0
    dp = np.degrees(np.angle(d[ip] / ref[ip]))
    print(f"  group {g + 1}: delta-phi rel L2 = {rel_l2:.3e}   "
          f"peak |delta-phi| diff {da * 100:+.3f}%   phase diff {dp:+.4f} deg")
