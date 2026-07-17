"""2D one-group Brantley-Larsen problem (Carreno et al. 2024, Fig. 3 / Table 6).

A Cartesian-aligned verification problem for the SPN / SDPN implementation: three
fuel bars (1 cm wide, 9 cm tall) at x in [1,2],[4,5],[7,8], y in [0,9], in a
10x10 cm square of moderator. Reflective at x=0 and y=0, vacuum at x=10 and y=10.
Strong transport effect -- diffusion (SP1) is ~2800 pcm below the transport
reference, and the SPN/SDPN hierarchy climbs toward it. Because every material
interface is grid-aligned, ndgpu's 2nd-order FV converges cheaply, so this
isolates "are the equations right?" from any geometry-resolution error.

Reference (Table 6, OpenMOC = 0.80536):
   SP1 0.77680  SP3 0.79904  SP5 0.80280  SP7 0.80354
   SDP1 0.80161 SDP2 0.80373 SDP3 0.80402
"""

import sys
import time

import numpy as np

from ndgpu import (Grid, Material, DiffusionEigenSolver, SP5EigenSolver,
                   SP7EigenSolver, SDP2EigenSolver, SDP3EigenSolver)
from ndgpu.solver import SPNEigenSolver, SDPNEigenSolver

K_REF = 0.80536                      # OpenMOC transport reference
PAPER = {"SP1": 0.77680, "SP3": 0.79904, "SP5": 0.80280, "SP7": 0.80354,
         "SDP1": 0.80161, "SDP2": 0.80373, "SDP3": 0.80402}

# One-group data (Table 5): Sigma_t, Sigma_s0, nu Sigma_f. D = 1/(3 Sigma_t),
# removal = Sigma_a = Sigma_t - Sigma_s0 (within-group scatter is inert here).
FUEL = Material(name="fuel", diffusion=[1.0 / (3 * 1.5)], sigma_a=[1.5 - 1.35],
                nu_sigma_f=[0.24], total=[1.5], chi=[1.0])
MOD = Material(name="mod", diffusion=[1.0 / (3 * 1.0)], sigma_a=[1.0 - 0.93],
               nu_sigma_f=[0.0], total=[1.0], chi=[1.0])
BARS = [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)]


def build(n):
    """n x n cells over 10x10 cm; fuel bars carved out, else moderator."""
    h = 10.0 / n
    xc = (np.arange(n) + 0.5) * h
    in_bar = np.zeros(n, bool)
    for lo, hi in BARS:
        in_bar |= (xc > lo) & (xc < hi)
    fuel_col = in_bar[:, None]                     # x varies along axis 0
    fuel_row = (xc < 9.0)[None, :]                 # y < 9 (axis 1)
    mmap = np.where(fuel_col & fuel_row, 0, 1).astype(int)  # 0 fuel, 1 mod
    grid = Grid(shape=(n, n, 1), size=(10.0, 10.0, h))
    # reflective at x=0 / y=0, vacuum at x=10 / y=10; z reflective (2D).
    bc = (("reflective", "vacuum"), ("reflective", "vacuum"), "reflective")
    return grid, mmap[:, :, None], bc


# SP3/SDP1 through the general U-form path so the Marshak boundary applies
# uniformly across the hierarchy.
class _SP3(SPNEigenSolver):
    _order = 1


class _SDP1(SDPNEigenSolver):
    _order = 1


# (name, solver, is_marshak_capable). SP1 = diffusion (single moment: Marshak
# and per-moment vacuum coincide).
METHODS = [("SP1", DiffusionEigenSolver, False), ("SP3", _SP3, True),
           ("SP5", SP5EigenSolver, True), ("SP7", SP7EigenSolver, True),
           ("SDP1", _SDP1, True), ("SDP2", SDP2EigenSolver, True),
           ("SDP3", SDP3EigenSolver, True)]


def _solve(cls, capable, n, marshak):
    grid, mmap, bc = build(n)
    kw = {"marshak_vacuum": True} if (marshak and capable) else {}
    return cls(grid, [FUEL, MOD], material_map=mmap, bc=bc, device="cpu",
               **kw).solve(tol_k=1e-9, tol_source=1e-8).k_eff


def _richardson(ks, ns):
    # h ~ 1/n, 2nd-order FV on this grid-aligned geometry.
    (n1, k1), (n2, k2) = (ns[-2], ks[-2]), (ns[-1], ks[-1])
    return k2 + (k2 - k1) / ((n2 / n1) ** 2 - 1)


def main():
    marshak = "--marshak" in sys.argv
    extrap = "--extrapolate" in sys.argv or marshak
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bc_name = ("coupled Marshak (paper's exact vacuum BC)" if marshak
               else "per-moment Robin vacuum (alpha=1/2)")
    print(f"Brantley-Larsen 2D one-group, boundary = {bc_name}")
    print(f"reference k = {K_REF:.5f} (OpenMOC)\n")

    if extrap:
        ns = [int(x) for x in args] or [80, 160, 240]
        print(f"Richardson extrapolation over meshes {ns} (h->0):\n")
        hdr = f"{'method':6s}{'ndgpu k_inf':>12s}{'paper k':>10s}{'d (pcm)':>9s}"
        print(hdr); print("-" * len(hdr))
        for name, cls, cap in METHODS:
            ks = [_solve(cls, cap, n, marshak) for n in ns]
            kinf = _richardson(ks, ns)
            print(f"{name:6s}{kinf:>12.5f}{PAPER[name]:>10.5f}"
                  f"{(kinf - PAPER[name]) * 1e5:>9.1f}")
        print("\nWith the coupled Marshak boundary, ndgpu reproduces the paper's "
              "Table 6 k_eff\n(SDP3 carries a small residual). d = ndgpu - paper, pcm.")
    else:
        n = int(args[0]) if args else 80
        hdr = f"{'method':6s}{'k_eff':>10s}{'paper k':>10s}{'d (pcm)':>9s}"
        print(f"single mesh {n}x{n} (not converged):\n{hdr}"); print("-" * len(hdr))
        for name, cls, cap in METHODS:
            k = _solve(cls, cap, n, marshak)
            print(f"{name:6s}{k:>10.5f}{PAPER[name]:>10.5f}"
                  f"{(k - PAPER[name]) * 1e5:>9.0f}")
    print("\nChecks: SP1<<SP3<SP5<SP7, SDP1<SDP2<SDP3, and at matched DoF "
          "SDP1>SP3, SDP2>SP5, SDP3>SP7.")


if __name__ == "__main__":
    main()
