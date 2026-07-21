"""Frequency-domain neutron noise: reproduce the zero-power reactor transfer
function, then show the global-to-local transition in a heterogeneous core.

Usage: python examples/noise_transfer_function.py [cpu|gpu|auto]

Part 1 sweeps frequency for a homogeneous, fully reflected one-group reactor
driven by a uniform absorption fluctuation, and compares the flat flux-noise
amplitude delta-phi/phi_0 to point kinetics, G(w) * delta-rho -- an exact,
analytic check of the i w/v term and the delayed-neutron feedback in chi_eff(w).

Part 2 puts a localized (vibrating) absorber in the near-critical 2D TWIGL core
and reports how the response migrates from a global fundamental-mode shape at
low frequency to a locally peaked one at high frequency.
"""

import sys

import numpy as np

from ndgpu import (DiffusionEigenSolver, Grid, Kinetics, Material, NoiseSolver,
                   NoiseSource, zero_power_transfer_function)
from ndgpu.perturbation import first_order_reactivity

device = sys.argv[1] if len(sys.argv) > 1 else "auto"

# --- Part 1: point-kinetics transfer function --------------------------------
mat = Material(name="crit", diffusion=[1.2], sigma_a=[0.020], nu_sigma_f=[0.020])
grid = Grid(shape=(3, 3, 1), size=(15.0, 15.0, 15.0))
kin = Kinetics(
    velocities=[2.2e5],
    beta=[0.00021, 0.00142, 0.00127, 0.00257, 0.00075, 0.00027],
    decay=[0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])

ns = NoiseSolver(grid, mat, kinetics=kin, bc="reflective", device=device)
Lam = ns.generation_time()
print(f"one-group critical reactor: k_eff = {ns.k_eff:.6f}, "
      f"Lambda = {Lam:.3e} s, on {ns.device}")

eps = 1e-6
matp = Material(name="p", diffusion=[1.2], sigma_a=[0.020 + eps], nu_sigma_f=[0.020])
fwd, adj = ns.eig.solve(), ns.eig.solve(adjoint=True)
d_rho = ns.k_eff * first_order_reactivity(
    ns.eig, fwd, adj, DiffusionEigenSolver(grid, matp, None, bc="reflective"))
print(f"uniform absorber d_rho = {d_rho:.3e}\n")

src = NoiseSource(d_sigma_a=[eps])
print(f"{'f [Hz]':>10} {'|dphi/phi|':>14} {'phase [deg]':>12} "
      f"{'|rel err|':>11}")
for f in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
    res = ns.solve(src, 2.0 * np.pi * f, tol=1e-11)
    amp = complex(np.mean(res.relative()[0]))
    predicted = zero_power_transfer_function(2.0 * np.pi * f, kin, Lam) * d_rho
    print(f"{f:10.3g} {abs(amp):14.6e} {np.degrees(np.angle(amp)):12.3f} "
          f"{abs(amp / predicted - 1.0):11.2e}")

# --- Part 2: global-to-local transition in a heterogeneous core --------------
from ndgpu.benchmarks import build_twigl                       # noqa: E402

p = build_twigl("none", cells_per_8cm=2)
mats, mmap = p.problem_at(0.0)
nsc = NoiseSolver(p.grid, mats, mmap, kinetics=p.kinetics, bc=p.bc, device=device)
nx, ny, _ = p.grid.shape
damp = np.zeros(p.grid.shape, dtype=complex)
damp[nx // 2, ny // 2, 0] = 5e-4                               # local thermal absorber
src2 = NoiseSource(d_sigma_a=[np.zeros(p.grid.shape), damp])
phi0 = np.asarray(nsc.flux0[1]).ravel()

print(f"\n2D TWIGL core: k_eff = {nsc.k_eff:.5f}, localized absorber at "
      f"cell ({nx // 2}, {ny // 2})")
print(f"{'f [Hz]':>10} {'sweeps':>7} {'shape ~ fundamental':>20}")
for f in (0.01, 0.1, 1.0, 10.0, 100.0):
    res = nsc.solve(src2, 2.0 * np.pi * f, tol=1e-8)
    d = np.abs(np.asarray(res.d_flux_numpy()[1]).ravel())
    cos = float(np.dot(d, phi0) / (np.linalg.norm(d) * np.linalg.norm(phi0)))
    print(f"{f:10.3g} {res.sweeps:7d} {cos:20.4f}")
print("(cosine 1 = global fundamental shape; smaller = localized around the "
      "perturbation)")
