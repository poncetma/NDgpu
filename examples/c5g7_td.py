"""Run a 2D C5G7-TD transient (OECD/NEA time-dependent C5G7 benchmark) with
diffusion kinetics, pin-cell homogenized or pin-resolved.

Usage: python examples/c5g7_td.py [case] [cells_per_pin] [cpu|gpu|auto]
                                  [--pin-resolved] [--t-end T] [--dt DT]

e.g.   python examples/c5g7_td.py TD1-1 2 auto --t-end 10 --dt 0.01
       python examples/c5g7_td.py TD0-1 10 gpu --pin-resolved
"""

import sys

import numpy as np

from ndgpu import TransientSolver
from ndgpu.benchmarks import C5G7TD_CASES, build_c5g7_td

args = [a for a in sys.argv[1:] if not a.startswith("--")]
case = args[0] if len(args) > 0 else "TD1-1"
s = int(args[1]) if len(args) > 1 else 2
device = args[2] if len(args) > 2 else "auto"
pin_resolved = "--pin-resolved" in sys.argv
opt = {k: float(sys.argv[sys.argv.index(k) + 1])
       for k in ("--t-end", "--dt") if k in sys.argv}
t_end, dt = opt.get("--t-end", 10.0), opt.get("--dt", 0.01)

prob = build_c5g7_td(case, cells_per_pin=s, pin_resolved=pin_resolved)
nx, ny, _ = prob.grid.shape
mode = "pin-resolved" if pin_resolved else "pin-cell homogenized"
print(f"C5G7-TD {case} ({C5G7TD_CASES[case]}), {mode}, "
      f"{nx} x {ny} cells x 7 groups, dt = {dt} s to {t_end} s")

solver = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                         bc=prob.bc, device=device,
                         mix_material=prob.mix_material,
                         mix_weight=prob.mix_weight)
res = solver.solve(t_end=t_end, dt=dt, verbose=True)
print(res)
print(f"k0 = {res.k0:.5f}   P_min = {res.power.min():.4f} at "
      f"t = {res.times[np.argmin(res.power)]:.2f} s   "
      f"P(end) = {res.power[-1]:.4f}")

# Fractional core fission rate at the benchmark's reporting instants.
marks = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
print("t (s) :", "  ".join(f"{t:7.2f}" for t in marks if t <= t_end))
pw = np.interp([t for t in marks if t <= t_end], res.times, res.power)
print("P/P0  :", "  ".join(f"{p:7.4f}" for p in pw))
