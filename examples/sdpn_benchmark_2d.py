"""2D benchmark: diffusion vs SDP1/SDP2/SDP3 on a strongly rodded assembly.

A 2-group reflective lattice with a cross-shaped gadolinia control blade (Fuel
IIg, thermal Sigma_a = 0.49 cm^-1) cutting through a fuel assembly, flanked by
water gaps. The blade carves a deep, sharply-cornered thermal-flux depression --
a 2D field with steep gradients and near-discontinuous angular flux, exactly
where diffusion is weakest and the simplified double-PN corrections matter.

A converged 2D transport (S_N) reference is expensive, so here the highest-order
method available, SDP3, is used as the reference proxy: the table reports how far
diffusion / SDP1 / SDP2 sit from SDP3 in eigenvalue and thermal-flux shape, plus
wall time and iteration count. The companion 1D benchmark (sdpn_benchmark_1d.py)
scores the same hierarchy against a true S16 transport solution.
"""

import time

import numpy as np

from ndgpu import (Grid, Material, DiffusionEigenSolver, SDP1EigenSolver,
                   SDP2EigenSolver, SDP3EigenSolver)


def _mat(name, D, sa, nsf, s12):
    return Material(name=name, diffusion=D, sigma_a=sa, nu_sigma_f=nsf,
                    sigma_s=[[0.0, s12], [0.0, 0.0]], chi=[1.0, 0.0])

WATER    = _mat("water",    [1.7639, 0.2278], [0.0003, 0.0097], [0.0, 0.0],       0.0380)
FUEL     = _mat("fuel_I",   [1.4730, 0.3294], [0.0096, 0.0764], [0.0067, 0.1241], 0.0161)
BLADE    = _mat("fuel_IIg", [1.5342, 0.3143], [0.0135, 0.4873], [0.0056, 0.0187], 0.0136)
MATS = [WATER, FUEL, BLADE]
W, F, B = 0, 1, 2

SIZE = 25.68            # cm, a 8x8 arrangement of 3.21 cm cells (fuel/water/blade)
CELL = 3.21


def material_map(n):
    """n x n cell map: fuel background, a '+' gadolinia blade with water cladding
    down the central rows/cols, reflective assembly edges."""
    g = np.full((n, n), F, dtype=int)
    c = n // 2
    hw = max(1, n // 32)                    # half-width of the blade in cells
    wg = max(1, n // 16)                    # water gap flanking the blade
    lo, hi = c - hw, c + hw
    wl, wh = lo - wg, hi + wg
    g[wl:wh, :] = W; g[:, wl:wh] = W        # water channels
    g[lo:hi, :] = B; g[:, lo:hi] = B        # gadolinia cross
    return g


def solve(solver_cls, n):
    grid = Grid(shape=(n, n, 1), size=(SIZE, SIZE, 1.0))
    mmap = material_map(n)[:, :, None]
    t0 = time.perf_counter()
    r = solver_cls(grid, MATS, material_map=mmap, bc="reflective",
                   device="cpu").solve(tol_k=1e-8, tol_source=1e-7)
    dt = time.perf_counter() - t0
    flux = np.asarray(r.flux_numpy).reshape(2, n, n)
    return r.k_eff, flux, dt, r.outer_iterations, r.inner_iterations


def main():
    n = 128
    print(f"Assembly: {SIZE:.1f} x {SIZE:.1f} cm, {n}x{n} cells, gadolinia '+' "
          f"blade, reflective BC\n")

    results = {}
    for name, cls in [("diffusion", DiffusionEigenSolver), ("SDP1", SDP1EigenSolver),
                      ("SDP2", SDP2EigenSolver), ("SDP3", SDP3EigenSolver)]:
        results[name] = solve(cls, n)

    k_ref, flux_ref, *_ = results["SDP3"]
    th_ref = flux_ref[1]
    th_ref_n = th_ref / th_ref.mean()

    hdr = (f"{'method':10s}{'k':>11s}{'dk vs SDP3':>12s}{'th-flux RMSE':>14s}"
           f"{'time (s)':>10s}{'in-iter':>9s}")
    print(hdr); print("-" * len(hdr))
    for name in ("diffusion", "SDP1", "SDP2", "SDP3"):
        k, flux, dt, no, ni = results[name]
        dk = (k - k_ref) / k_ref * 1e5
        th_n = flux[1] / flux[1].mean()
        rmse = float(np.sqrt(np.mean(((th_n - th_ref_n) / th_ref_n.max()) ** 2))) * 100
        tag = "  (ref)" if name == "SDP3" else ""
        print(f"{name:10s}{k:11.6f}{dk:12.0f}{rmse:13.3f}%{dt:10.3f}{ni:9d}{tag}")

    print("\n(dk in pcm vs SDP3; th-flux RMSE = normalized thermal-flux shape "
          "error vs SDP3, % of peak.)")
    print("Transport effect (diffusion vs SDP3): "
          f"{abs(results['diffusion'][0] - k_ref) * 1e5:.0f} pcm in k, "
          f"peak thermal-flux depression at blade centre "
          f"phi_min/phi_max = {th_ref.min() / th_ref.max():.3f}.")


if __name__ == "__main__":
    main()
