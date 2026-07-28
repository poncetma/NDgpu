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

## Hybrid S_N/diffusion: what a mask sweep must show

The hybrid solvers (`sn_mask=` on `HybridSNDiffusionSolver`, `hybrid_mask=` on
the SPN family) need more care than "empty mask == diffusion, full mask == S_N".

1. **The exact limits do not test the coupling.** An empty mask has no transport
   boxes; a full mask leaves `_active` all-False, so there is no diffusion bulk.
   Both bypass the interface entirely. Reporting "both limits are exact" as
   verification of a hybrid is wrong -- it verifies the two degenerate paths.
   Test the interface with a *partial* mask: unperturbed equilibrium (power must
   hold at 1) exercises it end to end.
2. **Sweep the mask size, and expect non-monotonic k.** The box's incoming
   angular flux is reconstructed *isotropically* from the neighbouring bulk
   scalar flux (`_box_incoming`), which is only valid where the flux is nearly
   isotropic. Measured on a 20x20 absorber-in-fuel problem (dk vs S_N, pure
   diffusion = +524 pcm):

       mask   4x4    6x6    10x10   14x14   18x18   full
       dk    -190   +18.5   +60.3  -185.5  -529.4    0

   There is a **usable window**: the interface must sit a few cells clear of both
   the strong absorber and the outer boundary. Crowding either end is *worse
   than pure diffusion*. More transport is not monotonically better.
3. **k and transient accuracy are different functionals -- validate both.** k is
   sensitive to the absorber; a transient's growth rate is the fission-weighted
   generation time over the *fuel*, which an absorber-centred mask never
   touches (the absorber has nu_Sigma_f = 0). On the same problem a mask that
   closed 96% of the k gap closed only 46% of the transient rate gap, and the
   collar that fixed k did nothing for the rate. **A mask tuned on drum worth
   does not transfer to a transient** -- quote transient accuracy from a
   transient measurement.
4. **Do not reach for the interface tolerance first.** Tightening the interface
   GMRES from 1e-4 to 1e-11 moved end-of-transient power by 0.06 pcm against a
   120 pcm discrepancy. Rule it out with one run before theorising about it.

## Long runs: make them observable

A refined mesh or a fine time step can turn a benchmark into a multi-hour run,
and a *stalled* run looks exactly like a slow one from outside. Never launch a
long configuration blind.

1. **Turn verbosity on** for anything expected to take more than a few minutes
   (`verbose=True`, or the code's per-iteration log — OpenSN's
   `verbose_outer_iterations`). Keep the **full** log in a file; a driver that
   pipes through `grep RESULT` throws away the only evidence of why a run
   stalled, and the convergence *status* line with it.
2. **Check progress at intervals**, not just at the end. Useful signals: the
   residual/`k_eff_change` still descending (not oscillating around a floor),
   iteration count advancing, RSS not climbing without bound, CPU near 100%
   (a compute-bound process is not hung — look at the numbers, not the clock).
3. **If it has stalled, cancel it.** Do not wait it out: a stalled run at the
   head of a queue blocks everything behind it, and unvalidated new code is
   worth more machine time than one extra data point. Report the diagnosis.
4. **Tolerance is the usual culprit.** Every stall in this repo so far has been
   a tolerance tighter than the iteration can reach, which presents as a hang
   rather than an error: `k_tol=1e-10` on a power iteration that floors at
   1e-8, an *absolute* `l_abs_tol` that stops being reachable as the mesh
   refines and per-cell flux shrinks, `tol_step=1e-10` against a fixed point
   that floors near 3e-10. Before tightening a tolerance, find the floor —
   run once loose and look at the residual it settles on.
5. **Never set an acceptance threshold tighter than the solver's own
   tolerance.** An accuracy check at 1e-3 pcm against a solve converged to
   1e-9 measures iteration noise, and a refinement study whose error falls
   below that floor reports a meaningless "order". Quote the floor
   (~`tol / t_span` for a rate, ~`tol * n_steps` for an accumulated ratio) and
   keep every measured point at least ~100x above it.

### The launch rule (not optional)

Items 1-2 above kept being honoured for the *first* run of a session and quietly
dropped on the fifth, when a driver got rewritten or a tolerance tightened. State
it as a rule with no judgement call in it:

**A run expected to exceed ~2 minutes is launched together with a Monitor. Not
`&` plus hand-polling.** Hand-polling has failed the same way every time: it
burns a tool call per check, it samples at whatever interval the conversation
happens to allow, and it cannot distinguish "slow" from "stalled" without a
second sample — so stalls are found minutes late, if at all. Concretely:

1. The driver writes a per-iteration line (`verbose=True`) to a **file**.
2. The same call that launches it arms a Monitor over that file whose filter
   emits: per-case results, a heartbeat carrying the latest iterate, an explicit
   `STALLED` line when the heartbeat is unchanged across two polls, and the
   failure signatures (`FAILED|Traceback|Error`). Silence must be impossible —
   a filter that matches only success is indistinguishable from a crashloop.
3. Exit the monitor on job exit **and** on the final-result marker, so it does
   not sit armed after the work is done.

Checklist before launching a sweep, each learned from a run that had to be
thrown away:

- Verbosity actually reaches the log (`grep -c` the iterate marker) — a rewritten
  driver that dropped `verbose=` looks identical to a healthy silent one.
- Tolerance is one the iteration can reach (item 4), and `xtol/ftol` are loose
  enough to let the optimizer *terminate*: `tol=1e-10` on `least_squares` runs to
  `max_nfev` every time, which cost 9 minutes on one variant where `1e-8`
  self-terminated in 87 s.
- The whole sweep's cost is estimated first: cases x evaluations x seconds. If
  it exceeds ~15 minutes, cut the case list rather than discovering the cost by
  waiting through it.
- Kill a sweep as soon as the pattern is established. Three variants showing the
  same pathology is a result; running the remaining three to confirm it a fourth
  and fifth time is not.

## Environment notes

- Run python from the repo directory (`~/claude-tests/ndgpu`) or imports fail.
- This box is CPU-only (numpy backend, 8 cores, single-threaded stencils):
  polynomial/heavier preconditioners often cut applies but lose wall time
  here — report both numbers and note the GPU outlook rather than discarding
  them.
- Timing scripts are throwaway: put them in the job tmp dir, keep only the
  numbers (in the write-up / memory / commit message).
