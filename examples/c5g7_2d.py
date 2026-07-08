"""Solve the OECD/NEA C5G7 2D MOX benchmark (pin-cell homogenized) with both
diffusion and SP3, and compare against the transport reference.

Usage: python examples/c5g7_2d.py [cells_per_pin] [cpu|gpu|auto]
"""

import sys

import numpy as np

from ndgpu import DiffusionEigenSolver, SP3EigenSolver
from ndgpu.benchmarks import K_REFERENCE_2D, build_c5g7_2d

s = int(sys.argv[1]) if len(sys.argv) > 1 else 2
device = sys.argv[2] if len(sys.argv) > 2 else "auto"

prob = build_c5g7_2d(cells_per_pin=s)
nx, ny, _ = prob.grid.shape


def pin_powers(res):
    """(51, 51) pin-cell fission rates normalized to a fuel-pin average of 1."""
    flux = res.flux_numpy[:, :, :, 0]                  # (G, nx, ny)
    fis = prob.fission_xs[prob.material_map[:, :, 0]]  # (nx, ny, G)
    rate = np.einsum("gxy,xyg->xy", flux, fis)
    pins = rate.reshape(nx // s, s, ny // s, s).sum(axis=(1, 3))
    fuel = prob.pin_map.T < 4
    return pins / pins[fuel].mean(), fuel


print(f"C5G7 2D quarter core: {nx} x {ny} cells x 7 groups")
print(f"transport reference k_eff = {K_REFERENCE_2D:.5f}, max pin power ~2.50\n")

for name, cls in [("diffusion", DiffusionEigenSolver), ("SP3", SP3EigenSolver)]:
    solver = cls(prob.grid, prob.materials, prob.material_map,
                 bc=prob.bc, device=device)
    res = solver.solve(tol_k=1e-6, tol_source=1e-5)
    power, fuel = pin_powers(res)
    print(f"{name:>9}:  {res}")
    print(f"{'':>9}   k_eff = {res.k_eff:.5f} "
          f"({(res.k_eff - K_REFERENCE_2D) * 1e5:+.0f} pcm vs transport)   "
          f"max pin power = {power[fuel].max():.3f}")
