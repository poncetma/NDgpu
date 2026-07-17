---
name: benchmark
description: Benchmarking protocol for ndgpu solver comparisons (solver variants, orders, preconditioners, meshes, devices). Use BEFORE running any performance or accuracy comparison and when writing up its results — every reported row must pair accuracy (k_eff + Δpcm vs a named reference) with cost (wall time + iteration counts).
---

# ndgpu benchmarking protocol

Core rule: **never report speed without accuracy, and never report accuracy
without cost.** A solver comparison that shows only one axis is not a result;
every table row carries both.

## Protocol

1. **Clean machine.** Never benchmark with background jobs running — a
   contended run once showed SDP1 at 2.3× SP3 purely from contention. Check
   with `ps`/`top` before timing anything.
2. **Best-of-N.** Run each configuration N ≥ 3 times identically and report
   the fastest run (least contended). One process, one device, same dtype.
3. **Identical, tight, stated tolerances.** Same grid, same
   `tol_k`/`tol_source` (typically `1e-8`/`1e-7`) for every variant, quoted
   in the write-up. Assert `result.converged` before a run counts.
4. **Accuracy.** Report `k_eff` to 7 digits and Δ = 1e5·(k − k_ref) in pcm
   against a **named** reference — a published value, the finest mesh, or the
   highest-order method; say which. Add flux RMSE / max relative error when
   flux shape matters. Configurations that must be mathematically equivalent
   (change of basis, preconditioner, linear-solver swap) must agree to
   |Δk| ≤ 2e-8 — a preconditioner that moves k is a bug, not a trade-off.
5. **Cost.** Report wall seconds **and** iteration counts: outers, inners
   (inners = operator applies; the counts are comparable across CG, GMRES and
   BiCGStab by construction). The counts separate algorithmic gains from
   machine effects: applies down but time up means per-apply cost grew — say
   so explicitly (e.g. "GPU-aimed, CPU-neutral") instead of dropping the
   variant.
6. **Row format.** One row per configuration:
   `label | k_eff | Δpcm vs <ref> | outers | inners | t [s]`
   followed by 1–2 sentences of interpretation: what wins, on which axis,
   and why.

## Template

```python
r = solver.solve(tol_k=1e-8, tol_source=1e-7)
assert r.converged
print(f"{label:40s} k={r.k_eff:.7f}  dk={1e5*(r.k_eff-k_ref):+7.1f} pcm  "
      f"outers={r.outer_iterations:3d}  inners={r.inner_iterations:6d}  "
      f"t={r.solve_seconds:6.2f}s")
```

## Environment notes

- Run python from the repo directory (`~/claude-tests/ndgpu`) or imports fail.
- This box is CPU-only (numpy backend, 8 cores, single-threaded stencils):
  polynomial/heavier preconditioners often cut applies but lose wall time
  here — report both numbers and note the GPU outlook rather than discarding
  them.
- Timing scripts are throwaway: put them in the job tmp dir, keep only the
  numbers (in the write-up / memory / commit message).
