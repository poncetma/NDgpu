"""CPU vs GPU cost of an **uncoupled** HP-MR transient, across problem sizes.

    python examples/hpmr_transient_bench.py --device cpu
    python examples/hpmr_transient_bench.py --device gpu --sizes 2d:4 3d:4x20
    python examples/hpmr_transient_bench.py --check-every 8 --dtype float32

No thermal coupling, no feedback: just the neutronics time loop, so what is
timed is the diffusion transient itself and nothing else. The harness and the
reasoning behind the columns live in
:mod:`ndgpu.benchmarks.hpmr_transient_bench`; this is the command line over it.

**us/CG-iter is the column that says what to optimize.** One CG iteration is a
fixed number of kernels over N-element arrays. If it is flat in problem size,
the run is launch-bound and the lever is fewer, larger launches -- fusion, graph
capture, batching the group loop, fewer host syncs. If it grows with size, the
run is bandwidth-bound and the lever is fewer bytes -- float32, fewer
temporaries, a preconditioner that trades passes for iterations. The crossover
between those regimes is where the GPU starts to pay, and it is a property of
this problem on this card, so it is measured rather than assumed.

``cg/step`` is the control. It is pure algorithm and must not move between a CPU
and a GPU leg; if it does, the wall-time ratio is not a speedup.
"""

import argparse

import numpy as np

from ndgpu.benchmarks.hpmr_transient_bench import (HEADER, SIGMA_A_SCALE,
                                                   format_row, parse_size,
                                                   transient_bench)

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--sizes", nargs="+",
                default=["2d:3", "2d:4", "2d:6", "3d:4x10"])
ap.add_argument("--groups", choices=("2", "11"), default="11")
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--dt", type=float, default=0.02)
ap.add_argument("--tol-step", type=float, default=1e-6)
ap.add_argument("--max-sweeps", type=int, default=4000)
ap.add_argument("--anderson-depth", type=int, default=1)
ap.add_argument("--rebalance", action="store_true", default=True)
ap.add_argument("--no-rebalance", dest="rebalance", action="store_false")
ap.add_argument("--sigma-a-scale", type=float, default=SIGMA_A_SCALE)
ap.add_argument("--device", default="auto")
ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
ap.add_argument("--precond-degree", type=int, default=0)
ap.add_argument("--check-every", type=int, default=1)
ap.add_argument("--no-warmup", dest="warmup", action="store_false", default=True)
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

print(HEADER)
rows = []
for spec in args.sizes:
    refine, nz = parse_size(spec)
    r = transient_bench(refine, nz, groups=args.groups, steps=args.steps,
                        dt=args.dt, tol_step=args.tol_step,
                        max_sweeps=args.max_sweeps,
                        anderson_depth=args.anderson_depth,
                        rebalance=args.rebalance,
                        sigma_a_scale=args.sigma_a_scale, device=args.device,
                        dtype=np.float32 if args.dtype == "float32" else np.float64,
                        precond_degree=args.precond_degree,
                        check_every=args.check_every, warmup=args.warmup,
                        verbose=args.verbose)
    rows.append(r)
    print(format_row(r, spec), flush=True)

print(f"\n  device {rows[0]['device']}, {args.dtype}, {args.groups} groups, "
      f"{args.steps} steps of dt = {args.dt:g} s, "
      f"precond_degree={args.precond_degree}, check_every={args.check_every}, "
      f"anderson_depth={args.anderson_depth}, rebalance={args.rebalance}")
print("  us/cg flat in dof  => launch-bound (fuse, capture, batch, desync);\n"
      "  us/cg rising with dof => bandwidth-bound (float32, fewer passes).")
