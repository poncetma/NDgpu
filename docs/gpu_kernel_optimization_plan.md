# GPU kernel-fusion and low-level optimization plan

Written 2026-07-28 from a static read of the codebase. No GPU or CuPy was
available in the environment where this was written, so every performance number
below is an *estimate* derived from kernel and allocation counts, not a
measurement. Phase 0 exists to fix that.

## Status (2026-07-29)

| phase | state | where |
|---|---|---|
| 0 — instrumentation | **done** | `ndgpu/profiling.py` |
| 1 — fused 7-point stencil | **done** | `ndgpu/kernels.py`, `stencil.py:apply` |
| 2 — fused SPN/SDPN block | **done** | `sp3.py:apply`, `spn.py:apply` (both blocks) |
| 3 — fused/de-synchronized Krylov | **done** (except pipelining) | `kernels.py`, `linalg.py:pcg`, `cocg` |
| 1b — fused **triangular** stencil | **done** | `tri.py:TriGroupOperator.apply` |
| 4 — CUDA graph capture | not started | |
| 5 — Cartesian SN host round-trips | **done** (DSA + GMRES; sweep edges left, see below) | `sn.py` |
| 6 — batched group loops | **done** (in-scatter + fission source) | `solver.py`, `kernels.group_accumulate` |
| 6b — batched groups in the **transient** | **done**, measured: 1.03x | `transient.py:_group_batch` |
| precision (float32 / SN dtype) | not started | |

All fused kernels are GPU-only and dispatched through `ndgpu.kernels`; the
NumPy path is unchanged and is what runs when the backend is NumPy, when dtypes
would have to be narrowed, or when fusion is switched off with `NDGPU_FUSED=0`
/ `kernels.set_fused(False)`. That switch is the A/B lever the benchmarks use.

## First measurement — Tesla T4, Colab, 2026-07-29

From `notebooks/colab_gpu_fusion_phase1.ipynb`. Correctness held everywhere:
fused vs. unfused agreed to ~1e-16 relative on all twelve operator variants,
CG iteration counts were unchanged, and `k_eff` moved by less than the 2e-8
equivalence bound on all six end-to-end cases.

**The launch-bound premise is confirmed.** One trivial kernel costs **10.8 us**
on this machine (Colab's host CPU is weak and this includes CuPy's per-operation
Python dispatch). In units of that probe kernel, per apply:

| operator | unfused | fused |
|---|---|---|
| `GroupOperator` | 50.5 | 7.9 |
| `SP3GroupOperator` | 187 | 25.7 |
| `SDPNGroupOperator` order 3 | 738 | **195** |

Read the ratios, not the absolute counts: a stencil operation costs several
probe-kernels' worth of dispatch, so these run high against the 13-and-1 you get
from reading the source. The residual 195 for SDP3 was the still-unfused order-N
block; Phase 2 has since addressed it, and re-measuring that row is the first
item under "next actions". Note this run predates Phase 1b, so nothing here
covers the triangular stencil or HP-MR.

**The stencil speedup does not decay with grid size**, contrary to the
prediction below: 7.3x at 8^3, 6.4x at 128^3, and roughly flat in between. The
two regimes cost about the same. Small grids are pure launch overhead; by 96^3
the fused kernel is at ~79% of the T4's 320 GB/s, so the win there is instead
the ~6x memory traffic saved by not materializing seven temporaries. The
crossover sits between 64^3 and 96^3. This is a better outcome than expected and
it weakens the case for Phase 4 on large grids specifically (graph capture only
attacks launch overhead, and large grids are no longer launch-bound).

**One fused kernel was slower than the CuPy expression it replaced.** The first
two-leg run showed end-to-end wins of 1.3-2.5x on the 2D and small cases but
**0.82-0.85x** on the large 3D diffusion solves (96^3, 144^3) — a regression
monotone in problem size, i.e. a per-element cost, not launch overhead. Adding
independent group switches and re-running placed it exactly:

| case | stencil | krylov | all |
|---|---|---|---|
| bare box 48^3 | 1.32 | **0.77** | 1.01 |
| bare box 96^3 | 1.68 | 1.07 | 1.91 |
| bare box 144^3 | 1.78 | 1.09 | 2.04 |
| IAEA-3D (51,51,57) | 1.26 | 1.05 | 1.33 |
| C5G7-2D diffusion | 1.41 | 1.21 | 1.92 |
| C5G7-2D SP3 | 1.90 | 1.12 | 2.41 |

The **stencil wins on every case**; the Krylov kernels were the problem, and the
culprit is the `dot` reduction, exactly as suspected. Per-kernel, fused/generic:

|  n | 1024 | 4096 | 16384 | 65536 | 262144 | 1048576 | 4194304 |
|---|---|---|---|---|---|---|---|
| `dot` | 2.16 | 1.98 | 2.07 | 1.56 | **0.36** | 0.19 | 0.18 |
| `cg_update` | 6.64 | 4.53 | 4.30 | 4.16 | 1.42 | 1.70 | 1.67 |
| `cg_direction` | 2.00 | 2.01 | 1.93 | 1.85 | 2.00 | 1.69 | 1.70 |

Above ~1e5 elements the generic two-input `ReductionKernel` runs at roughly a
**fifth** of CuPy's `sum`, which dispatches to CUB. `kernels.DOT_FUSED_MAX`
(now 2^16, the largest measured size where the fused version still wins) hands
back to `xp.sum` above the crossover, and that turned 0.82x into **2.04x** at
144^3. The other two Krylov kernels win at every size and are not thresholded.

The generalizable lesson: **"fused is faster" is not true kernel by kernel.**
CuPy's generic paths are hand-tuned in places, so a custom kernel has to beat
CUB, not just beat a naive loop — and only the per-group switches made that
visible instead of averaging it into a single "fusion" number.

## Second measurement — T4, after Phases 1b and 2

Same notebook, same protocol (tol_k=1e-8, tol_source=1e-7, best of 3, four legs).
The Cartesian rows are unchanged from the first run, which is the control; the
new rows are the point.

| case | stencil | krylov | all | wall (none -> all) |
|---|---|---|---|---|
| bare box 48^3 | 1.32 | 1.03 | 1.41 | 0.34 -> 0.24 s |
| bare box 96^3 | 1.69 | 1.08 | 1.91 | 1.67 -> 0.87 s |
| bare box 144^3 | 1.79 | 1.09 | 2.05 | 7.31 -> 3.57 s |
| IAEA-3D (51,51,57) | 1.28 | 1.03 | 1.15 | 0.89 -> 0.77 s |
| C5G7-2D diffusion | 1.35 | 1.13 | 1.94 | 4.89 -> 2.52 s |
| C5G7-2D SP3 | 1.88 | 1.13 | 2.30 | 8.30 -> 3.61 s |
| **C5G7-2D SDP3** | **3.83** | 0.99 | **4.23** | **30.7 -> 7.27 s** |
| **HP-MR 2D tri** (24642 cells) | **1.40** | 1.15 | **1.79** | 0.58 -> 0.32 s |

**Phase 2 is the biggest single win in the project so far**: C5G7-2D SDP3 goes
4.23x, 30.7 s to 7.3 s. That is the congruence block, which was ~195
probe-kernels per apply. Note the `stencil` leg carries the block kernels too
(both are operator kernels), so the 3.83 is the two together — the legs cannot
separate them, which is a limitation of how they were defined.

**Phase 1b delivers what it was for**: HP-MR 2D on the tri mesh at 1.79x, from a
stencil that Phases 1-2 could not reach at all. Accuracy held everywhere.

The one soft row is IAEA-3D, where `all` (1.15) came in below `stencil` (1.28).
Small case, and within the run-to-run spread seen elsewhere; not worth chasing
without a repeat.

## Does CUDA graph capture hurt large problems? (2026-07-30)

Asked directly, and the answer is not the obvious one.

**Replay itself is never slower.** A captured graph runs the same kernels on the
same buffers; `cudaGraphLaunch` replaces N driver-side launches with one. Per
iteration it is a strict improvement, and the improvement shrinks toward zero as
the kernels get long enough to hide the launch cost. So there is no regime where
replaying a graph costs more than launching its kernels individually.

**But capture is not free, because of what it forbids.** A captured region may
not allocate and may not call cuBLAS. `tri_sn`'s level sweep contains three
weighted contractions -- the step-scheme flux reduction, the SCB flux reduction,
and the per-cell 3x3 corner matvec -- and to stay capturable all three were
written as a broadcast-multiply into a full-size temporary followed by a
reduction, with comments saying exactly why ("NOT matmul -- cuBLAS calls cannot
be captured"). That form does **~3x the memory traffic** of the contraction it
computes, on every level of every sweep, and it holds scratch buffers as large as
the angular flux (`psiw` is (M, N); the SCB `bm` blocks total three times it).

So the cost of capture was paid in *bandwidth and footprint*, which is what
dominates large problems, to buy *launch overhead*, which only matters on small
ones. Capture was also unconditional on CuPy, so a large case paid the whole
penalty for a benefit that had already gone to zero. That is a real inversion of
the intended priority -- and it is invisible if you only look at the capture code
itself, because the damage is in the arithmetic the capture constraint mandated.

**The fix was not to gate or remove capture, but to remove the deoptimization.**
All three contractions are now single hand-written kernels
(`kernels.moment_gather`, `kernels.batched_matvec`): one pass, no temporary, and
still perfectly capturable -- an `ElementwiseKernel` is recordable, only cuBLAS
is not. So the large-case path gets the traffic of a matmul *and* keeps the
small-case graph replay; nothing is traded.

Measured saving in sweep buffer footprint (HP-MR 2D, refine 4, S2/12, CPU count
of the same allocations the GPU makes), scaling linearly with problem size:

| scheme | broadcast form | fused form | saved |
|---|---|---|---|
| step | 9.57 MB | 7.41 MB | 23% |
| scb | 33.63 MB | 23.49 MB | 30% |

The broadcast form is kept as the fallback for when the fused kernels are
switched off, because it is allocation-free and therefore still capture-safe --
the generic NumPy-style fallbacks inside `ndgpu.kernels` are not, which is why
`_fused_reduce` is latched once at setup and must not flip between capturing a
graph and replaying it.

Still unmeasured on a GPU: the *time* saved. The footprint number above is
exact (it is just allocation accounting); the bandwidth claim follows from the
traffic ratio but wants a T4 confirmation.

## Third measurement — T4, the transport path, 2026-07-30

From `notebooks/colab_tri_sn_capture.ipynb`.

**The graph-capture question, answered.** Capture engages (2 graphs, one per
(group, iface) pair). The contraction rework is exact -- dk = 4e-16, identical
sweep counts -- and removes 22.6% (step) / 30.2% (SCB) of the sweep buffers,
matching the CPU prediction exactly. Capture itself is worth **2.32x at 6.5k
cells, 1.82x at 11k, 1.42x at 25k**: a clean decay with size, and never below
1.0. So capture stays, ungated -- removing the deoptimization instead of gating
capture was the right call, and the two wins compound.

**The rework pays per scheme, not per size.** SCB gains 1.23-1.39x at every size
and in 3D; `step` gains nothing (0.97-1.02x). SCB's 3x3 matvec sits *inside* the
level loop, paid per level per sweep; step has one flux reduction at the end of a
sweep, one kernel among hundreds. The prediction that the win would *grow* with
size was wrong -- a fixed traffic ratio gives a constant speedup once
bandwidth-bound.

**Fusion on the transport path** (four legs, 11250 cells):

| case | stencil | krylov | groups | all |
|---|---|---|---|---|
| hybrid tri-SN/diffusion, 2 group | 1.04 | 1.08 | 0.97 | **1.29** |
| tri-SN, 11 group | **1.45** | 1.02 | 1.00 | **1.50** |

30.2 s -> 20.2 s on the 11-group case, essentially all of it the fused tri
stencil. (The hybrid's legs multiply to 1.08 against an `all` of 1.29; best-of-2
on a ~10 s solve is thin enough that this is probably variance rather than a real
interaction.)

### Extending Phase 6 into tri_sn is NOT worth doing -- measured

`x_groups` is 1.00 on both rows, because `TriSNTransportSolver.solve` has its own
O(G^2) in-scatter loop that Phase 6 never touched. The question was whether to
wire it in. Bounding it settles it:

- 11-group tri-S_N converges in **8 outers** (CMFD-accelerated).
- Its assembly is 2 x (2G^2 + G) = 506 launches per outer, so 4,048 total.
- At the measured 10.8 us/launch that is **44 ms of a 20.15 s solve = 0.22%**.
- Ceiling if the assembly were free: **1.002x**.

The general rule this exposes: the O(G^2) assembly costs in proportion to *outer
count*, and CMFD-accelerated transport needs ~25x fewer outers than the diffusion
power iteration (8 vs 212 on the same 11-group core). Phase 6 pays precisely
where it was implemented and nowhere else. Retired.

## The uncoupled HP-MR transient — where its time actually goes (2026-08-05)

Asked directly: how much does a GPU buy on a neutronics-only HP-MR transient?
Measuring the CPU side first turned up an answer that reorders the whole
question.

Harness: `ndgpu/benchmarks/hpmr_transient_bench.py`, driven by
`examples/hpmr_transient_bench.py` and `notebooks/colab_hpmr_transient_gpu.ipynb`.
The manoeuvre is a uniform +0.5 $ absorption step at t=0 rather than a drum
rotation, because a drum's absorber area fraction lands differently on every
mesh and its iteration counts would not be comparable across a size sweep.

**The cost of a step factorizes, and only one factor is about the bus:**

    ms/step  =  (sweeps/step) x (G x subsweeps) x (CG iters/solve) x (cost/iter)

Measured on this box (CPU, 8 cores, float64, 11-group ENDF/B-8, `rebalance=True,
anderson_depth=1`, 3 steps of dt = 0.02 s):

| case | cells | dof | ms/step | sweeps/step | cg/step | cg per group-sweep | us/cg | ns/cg/dof | P(end) |
|---|---|---|---|---|---|---|---|---|---|
| 2D refine 2 | 1,320 | 14,520 | 3,888 | 252 | 36,210 | 13.1 | 107.4 | 7.40 | 1.75992 |
| 2D refine 3 | 2,970 | 32,670 | 8,161 | 249 | 53,172 | 19.4 | 153.5 | 4.70 | 1.76193 |
| 2D refine 4 | 5,280 | 58,080 | 19,388 | 240 | 69,346 | 26.3 | 279.6 | 4.81 | 1.76195 |
| 2D refine 6 | 11,880 | 130,680 | 60,489 | 253 | 111,364 | 40.0 | 543.2 | 4.16 | 1.75945 |

(`P(end)` spans 0.15% over a 9x cell range and is not monotone in the mesh — as
expected for a power stopped on a relative source change of 1e-6 after ~250
sweeps. It is here as a *control*, to show the four rows solved the same
problem, not as a convergence study.)

Three things fall out.

**1. The within-step fixed point takes ~250 sweeps, on every step, at every
mesh.** Not a first-step artifact: per-step CG counts at refine 2 are 35,689 /
35,258 / 37,682 / 41,837 over the first four steps. This is the dominant factor
by an order of magnitude and **no bus changes it**. The cause is structural: at
dt = 0.02 s the time term 1/(v dt) is ~5e-8 against Sigma_a ~ 1e-2, so the
backward-Euler operator is barely distinguishable from the steady one, and the
source iteration converges at the core's dominance ratio.

`rebalance=True` is what makes it tractable at all — it is a **one-cell CMFD**,
killing the fundamental amplitude mode. Measured at refine 3 (3 steps):

| outer scheme | ms/step | cg/step | P(end) |
|---|---|---|---|
| rebalance + Picard | **7,029** | 159,515 | 1.761934 |
| Anderson 5 | 28,056 | 579,813 | 1.760362 |
| Anderson 5 + rebalance | diverged at 4,000 sweeps | — | — |
| Picard alone | 36,959 | 780,867 | 1.761764 |

5.3x for the rebalance, at an answer agreeing to 4e-4 — and the documented
Anderson/rebalance incompatibility reproduced as an outright divergence, not a
slowdown.

What is left after the amplitude mode is the **shape** modes, and the standard
instrument for those is a spatial CMFD. `TransientSNSolver` already has one
(`_cmfd_step`, worth 2.4-3.1x there on a fixed point the theta shift had
*already* damped). The diffusion transient has no equivalent. Generalizing
`rebalance` from one coarse cell to a coarse mesh is the single largest lever
identified here, and it is bus-independent: it multiplies with whatever the GPU
gives.

**2. The CPU is already bandwidth-bound.** `ns/cg/dof` settles at 4.2-4.8 from
refine 3 on (the 7.40 at refine 2 is fixed overhead on a cache-resident
problem). One CG iteration touches ~10 arrays, so ~4.5 ns/dof is ~18 GB/s
effective — about right for this box. That number is the CPU asymptote a GPU has
to beat, and it is flat, so CPU cost is now simply linear in unknowns: `ms/step`
goes 3.9 -> 8.2 -> 19.4 -> 60.5 s as `dof` goes 14.5k -> 131k.

**3. The 2D core is small enough that the GPU will barely win.** A CG iteration
is a fused stencil apply plus ~5 vector kernels and a sync. The units to use are
this document's own T4 numbers: one trivial kernel is 10.8 us, and a *fused*
`GroupOperator.apply` measured **7.9 probe-kernels** — dispatch on Colab's weak
host, not arithmetic. That puts the launch-bound floor for one CG iteration
somewhere around **150-250 us**, not the ~75 us a naive one-kernel-per-op count
suggests. Against the measured CPU column:

| case | dof | cpu us/cg | gpu us/cg (predicted) | speedup (predicted) | **measured** |
|---|---|---|---|---|---|
| 2D refine 2 | 14,520 | 107 | 150-250 (launch floor) | **0.4-0.7x — a loss** | — |
| 2D refine 3 | 32,670 | 154 | 150-250 (launch floor) | ~0.6-1.0x | **0.50x** |
| 2D refine 4 | 58,080 | 280 | 150-250 (launch floor) | 1.1-1.9x | **0.77x** |
| 2D refine 6 | 130,680 | 543 | 150-250 (launch floor) | 2.2-3.6x | ~1.5x (est) |
| 3D refine 4 / nz 20 | 1,161,600 | ~5,200 (extrapolated) | ~370-450 (bandwidth) | ~11-14x | ~10.5x (est) |

Two things follow. The smallest 2D case is predicted to be **slower on the GPU
than on the CPU** — worth stating up front, because a benchmark that only
reported the 3D row would look like a uniform win. And the launch/bandwidth
crossover (~93 MB of traffic per CG iteration at 1.16M unknowns, ~370 us at
250 GB/s effective) sits close to 1M unknowns, so even the 3D cases are only
just into the regime the GPU is built for.

**Scored against measurement** (right-hand column, added after the run): the
*shape* was right — 2D loses, 3D wins big, crossover near 1M — but the launch
floor was optimistic. It measured **~370-400 us/cg**, not 150-250, so every 2D
prediction was too generous by roughly the ratio, and 2D refine 4 turned out to
be a **loss (0.77x)** where 1.1-1.9x was predicted. The 3D estimate was good
(~10.5x measured-extrapolated against ~11-14x predicted).

The error is instructive: the floor was estimated from `GroupOperator`'s 7.9
probe-kernels, but the transient's CG iteration also carries the Krylov kernels,
the Jacobi apply and a host sync, and the *triangular* stencil's probe-kernel
count had never been measured separately. Under-counting the kernels in the loop
under-counts the floor.

### The ~250 sweeps per step were SPECTRAL, not spatial (2026-08-05)

The section below identifies the within-step sweep count as the dominant cost
and proposes a spatial CMFD for it. **That diagnosis was wrong, and measuring it
properly turned up a one-parameter fix worth ~7x.**

`scatter_subsweeps` is the number of Gauss-Seidel passes over the energy groups
per fixed-point evaluation; the auto rule picks 3 whenever upscatter exists.
On the 11-group HP-MR that is far too few. Sweeps / total CG for one step:

| subsweeps | 3 | 4 | 5 | **6** | 8 |
|---|---|---|---|---|---|
| refine 2 | 248 / 35,689 | 144 / 23,421 | 43 / 7,661 | **27 / 5,285** | 25 / 5,536 |
| refine 3 | 247 / 52,864 | 144 / 34,609 | 30 / 8,068 | **26 / 7,583** | 25 / 8,254 |

**3 -> 6 is ~7x fewer CG iterations and ~3x less wall time, at roughly half the
error** (refine 3, one step: P = 1.401807 at subsweeps 3 against a converged
1.401552, i.e. 2.6e-4; subsweeps 6 gives 1.401604, 5.2e-5). End to end the
benchmark at 2D refine 3 went **11,346 -> 1,748 ms/step**, sweeps 265 -> 24.
The optimum is 6 with a 5-8 plateau, identically at both mesh sizes, and the
4->5 transition is a *cliff*: below a threshold the group cascade cannot
propagate within one fixed-point evaluation and the outer iteration is left to
carry it. That is the whole origin of the "hundreds of sweeps".

**UPDATE 2026-08-06: the default WAS raised to 6, and the mechanism turned out
not to be what this section says.** Three follow-ups settled it.

*(1) The default is safe.* The check that licenses it is that different subsweep
counts reach the SAME fixed point, which they do on both benchmarks under a
tol_step ladder: HP-MR subsweeps 3/6/12 give 1.3990072 / 1.3990070 / 1.3990067
at tol_step 1e-9, and C5G7 subsweeps 3/8 give 0.7814288 / 0.7814287. So the
count is a cost knob, not a physics one. At the production tolerance the HP-MR
errors are 2.5e-4 / 7.1e-5 / 1.8e-4 and the cost to converge is 43,371 /
**11,449** / 26,387 CG -- 6 is simultaneously the cheapest and the most
accurate, and the plateau has a top edge as well as a bottom one.
`_UPSCATTER_SUBSWEEPS = 6` in `transient.py`.

*(2) The subsweeps are not fixing spectral convergence -- they are feeding the
REBALANCE.* With `rebalance=False` the dramatic subsweep sensitivity disappears:

| subsweeps | rebalance ON | rebalance OFF |
|---|---|---|
| 3 | 248 | 1347 |
| 5 | 43 | 1129 |
| 6 | 27 | 1080 |
| 8 | — | 1027 |

Subsweeps alone are worth 1.3x and the rebalance alone ~5x, but together they
are worth ~50x. The rebalance's correction assumes the swept flux satisfies a
neutron balance with the source it was given; under Gauss-Seidel that holds only
once the groups stop moving relative to one another. **The rebalance has a
spectral-consistency precondition**, which is what the subsweeps supply and what
produces the otherwise inexplicable cliff between 4 and 5 passes (a geometric
contraction at the measured upscatter loop gain of ~0.1 per pass cannot make a
cliff).

*(3) Two acceleration ideas were tried and both failed.* Symmetric Gauss-Seidel
over energy -- alternating ascending and descending passes, on the argument that
an ascending pass resolves downscatter exactly so a descending one resolves
upscatter exactly -- is **worse at every count** (613/362/298 sweeps at n_sub
2/3/4 against 403/248/144 plain). A descending pass solves each group from stale
faster-group fluxes and so discards the downscatter cascade, and downscatter is
~5x the stronger coupling here. And pinning the inner CG tolerance looked like a
2.6x-cheaper, 25x-more-accurate win in a single run, but the cost-accuracy curve
shows it is **false convergence**: pinned runs stall (identical CG and identical
answers at tol_step 1e-8 and 1e-9) because CG noise becomes the fixed point's
floor, converging to 1.3987584 (rtol 1e-6) or 1.3989827 (rtol 1e-7) instead of
the true 1.3990070. The shipping adaptive policy is the only one that actually
converges, and its docstring rationale is now measured rather than asserted.

The original C5G7 discussion below is kept for the reasoning; its conclusion is
superseded by (1).

**C5G7-TD was then measured, and the default was still NOT raised.** TD1-1,
2 cells/pin, 0.4 s at dt = 0.05 (explicit `subsweeps=3` reproduces `auto`
bit-for-bit, confirming the rule picks 3 there):

| subsweeps | sweeps | cg | wall | P(end) | max abs dP vs auto |
|---|---|---|---|---|---|
| auto (=3) | 399 | 151,258 | 25.7 s | 0.781553 | — |
| 4 | 246 | 106,190 | 19.0 s | 0.781518 | 2.61e-4 |
| **6** | **190** | **87,713** | **15.8 s** | 0.781514 | 2.47e-4 |
| 8 | 162 | 81,157 | 14.4 s | 0.781493 | 1.97e-4 |

C5G7 gains too -- **1.7x fewer CG at 6** -- but nothing like HP-MR's 7x, and the
power history moves by ~2.5e-4. The reason for holding: on HP-MR a tolerance
ladder *proved* the converged value and showed higher subsweeps is closer to it,
whereas C5G7's P(end) drifts monotonically (0.781553 / 0.781518 / 0.781514 /
0.781493) instead of stepping to a plateau, so these rows alone do not say which
end is right. C5G7-TD is validated against FEMFFUSION; shifting that comparison
by 2.5e-4 on an unproven accuracy argument is not warranted.

So the auto rule stays at 3, and 6 is set explicitly on the two HP-MR paths
where the 7x lives and the accuracy gain is established:
`hpmr_transient_bench.SUBSWEEPS` and `coupling.py`'s multigroup branch (which is
the same 11-group HP-MR the measurement was made on). Raising the global default
wants a tolerance ladder on C5G7 first -- that is the outstanding item.

**Coarse-mesh rebalance was built, measured and REMOVED.** A full multigroup CMR
(per-region correction factors from a dense coarse balance with fission
implicit, reducing to the scalar rebalance at one region) was implemented and
verified -- P(end) came out identical to 6 digits across 1, 8, 21, 58 and 193
regions, which is exactly the invariant saying the correction cannot move the
fixed point. It simply does not work here:

* 1 -> 193 regions (near the fine mesh, ~7 cells each) bought only **1.24x**;
* once the spectral fix is applied it *regresses* the solve, 26 sweeps / 7,583
  CG becoming 190 / 54,558;
* and it is not more accurate either -- it lands P = 1.401353 where the true
  converged value is 1.401552, i.e. 2.0e-4 off, on the far side.

The structural reason is instructive: summing the balance over all groups
cancels scattering identically, so a coarse-mesh rebalance is **blind to
spectral error by construction** -- it could never have touched the actual slow
mode. The code was deleted rather than left as an unrecommended option.

**Two lessons worth carrying.** First, a rank-1 correction helping (the scalar
rebalance is worth 5.4x) says the mode has a large global component; it does
*not* say the residual is spatially structured, and inferring the latter is what
sent this down the wrong path. Second, Anderson **saturates** here -- depths 5,
20 and 50 all land near 850 sweeps, and pinning the inner CG tolerance to make
the map stationary changes 938 to 895, i.e. nothing. When a Krylov method
saturates and a physics correction does not, the slow mode is not a few isolated
eigenvalues.

### First GPU result, and what it cost to interpret (2026-08-05)

**T4, Colab, 2D refine 3, 11 groups, 4 steps** — the GPU is **1.9x SLOWER**:

| bus | ms/step | sweeps | cg/step | us/cg | ns/cg/dof | P(end) |
|---|---|---|---|---|---|---|
| cpu | 12,126 | 265 | 56,665 | 214.0 | 6.55 | 1.84234 |
| gpu | 23,041 | 265 | 56,664 | 406.6 | 12.45 | 1.84231 |

This is the launch-bound regime confirmed on the machine: `us/cg` on the GPU is
**406.6 us**, above the 150-250 us band predicted above, so the launch floor is
worse than estimated and the break-even point is further out than the prediction
table says. Refine 3 at 0.53x sits between the predicted 0.4-0.7x (refine 2) and
1.1-1.9x (refine 4), so the *shape* of the prediction holds even though the
floor was optimistic. Treat the 2D rows of that table as upper bounds.

**The correctness gate fired, and it was the gate that was wrong.** It asserted
`|dP| < 1e-8` and `|dk0| < 2e-8`; the run gave 3.12e-5 and 8.04e-8. The
diagnosis, and the general lesson:

- The *work* columns matched almost exactly — 265 vs 265 sweeps, 56,665 vs
  56,664 CG. A defect in the batched assembly changes the fixed point, so it
  surfaces there first. It did not.
- A tolerance ladder settled it. On 2D refine 2, P(end) at tol_step
  1e-5 / 1e-6 / 1e-7 / 1e-8 is 1.625123 / 1.626469 / 1.626604 / 1.626618:
  first-order convergence to ~1.6266196, i.e. **error ~ 150 x tol_step**. At the
  default 1e-6 the solver's own answer is uncertain at **1.5e-4** — so the
  CPU/GPU gap of 3.1e-5 is *five times smaller than the convergence error*. The
  buses agree better than the tolerance guarantees.
- Two hypotheses were tested and **rejected** on the way, which is why the
  ladder was worth running: (a) that the batched assembly reorders the group
  sums — it does not, batch-on vs batch-off is bit-identical on CPU at both 1e-6
  and 1e-8; (b) that rounding-scale perturbations get amplified — a one-off
  1e-15 kick moves P by only ~1e-14, with identical sweep counts, at every
  tolerance. The real mechanism is discrete: persistent per-reduction rounding
  differences make the two buses cross the `tol_step` threshold a sweep apart,
  and one sweep's correction near convergence is worth ~1e-5.

**Generalizable:** a stopping criterion is not an error bound, and the gap
between them is the *fixed point's* amplification, not the floating-point
format's. Before asserting agreement between two implementations, measure what
the solver's own tolerance actually buys — otherwise the gate encodes a
precision the code never promised. Both `tol_step` and `tol_k` were being read
as error bounds here.

### The full sweep — T4, 2026-08-05. Where the GPU starts to pay

`notebooks/colab_hpmr_transient_gpu.ipynb`, 11 groups, 4 steps, +0.5 $ step.
Sweeps and CG counts matched CPU exactly wherever both ran (265 vs 265, 56,665
vs 56,664; 258 vs 258, 74,554 vs 74,554), so the wall times are comparable.

| case | dof | cpu ms/step | gpu ms/step | speedup | gpu us/cg | ns/cg/dof | achieved GB/s |
|---|---|---|---|---|---|---|---|
| 2D r3 | 32,670 | 11,346 | 22,637 | **0.50x** | 399.5 | 12.23 | 6.5 |
| 2D r4 | 58,080 | 21,548 | 28,139 | **0.77x** | 377.4 | 6.50 | 12.3 |
| 2D r6 | 130,680 | — | 42,735 | ~1.5x (est) | 366.9 | 2.81 | 28.5 |
| 3D r4/nz10 | 580,800 | — | 37,619 | ~5.4x (est) | 459.2 | 0.79 | 101 |
| 3D r4/nz20 | 1,161,600 | — | 31,623 | ~10.5x (est) | 473.8 | 0.41 | 196 |
| 3D r6/nz20 | 2,613,600 | — | 71,382 | ~16x (est) | 699.4 | 0.27 | 299 |

**The regime question is answered, and both regimes are present.** `us/cg` is
flat at **~370-400 us from 33k to 131k unknowns** — that is the launch floor,
and it is where the two 2D losses come from. Above ~500k it starts rising, and
the achieved bandwidth (10 arrays x 8 B per unknown per CG iteration) climbs
2% -> 9% -> 32% -> 61% -> **93% of the T4's 320 GB/s**. So the crossover sits
near 0.5-1M unknowns, and by 2.6M the kernels are running essentially at the
card's memory-bandwidth limit — there is nothing left to win there without
moving fewer bytes.

The `(est)` speedups extrapolate the CPU column, whose `ns/cg/dof` had settled
to ~5 (6.13 at 33k, 4.98 at 58k, matching the 4.2-4.8 asymptote measured
locally); `ms/step = us/cg x cg/step`, and `cg/step` is measured. They are
estimates, not measurements — `CPU_DOF_MAX` in the notebook has since been
raised to 140k so the 2D r6 row, the one that pins the low end, gets measured
on the next run.

In wall-clock terms, for the 1.16M-unknown 3D core and a 30 s transient at
dt = 0.02 s (1,500 steps): **~5.8 days on CPU, ~13 h on GPU, ~10 h with
`precond_degree=1`** — and roughly **25 minutes** if a spatial CMFD cuts the 230
sweeps per step to ~10. The ordering of the levers is unchanged by the GPU
result: CMFD ~23x, GPU ~10x, preconditioner ~1.4x.

Two smaller observations. **3D is cheaper per unknown than fine 2D**: `cg/g/sw`
is 26.2 at 3D refine 4 against 40.0 at 2D refine 6, because the axial leakage
and the 1/(v dt) shift improve the conditioning. And the sweep count *falls*
slightly in 3D (230-284 vs 265), so the 9x jump in unknowns from 2D r6 to
3D r4/nz20 actually costs *fewer* CG iterations per step (66,739 vs 116,462).

### Phase 6b measured: the batched transient assembly is worth 1.03x

A/B at 2D refine 3 (`set_fused_group("groups", False)`): 23,855 -> 23,097
ms/step, identical sweeps (265) and identical CG counts (56,664), power apart by
2.7e-5 (rounding, well inside the convergence floor). **1.03x.**

That is far less than the launch-count argument for it suggested. Two reasons,
both worth remembering:

1. The dense `(G, G, *grid)` stack reads **G values per cell per row regardless
   of sparsity**, where the Python loop touched only the non-zero couplings. So
   the kernel buys launches with *memory traffic* — the same trade graph capture
   was found to be making in `tri_sn`, arrived at from a different direction.
2. The assembly was a smaller share of the launch budget than the O(G^2) count
   implied: at ~19 CG iterations per group-sweep, the CG kernels outnumber the
   assembly kernels several times over.

It stays (it is never worse, and the case for it grows with G and with anything
that cuts CG iterations — `precond_degree=1` nearly halves them, which raises
the assembly's share), but it is a 3% effect, not the ~20% predicted. This is
the third instance in this document of the same lesson: **a launch-count
argument is a hypothesis about where time goes, not a measurement of it.**

### The levers, measured on the GPU (3D r4/nz20, 1.16M unknowns)

| leg | ms/step | vs base | cg/step | vs base | P(end) |
|---|---|---|---|---|---|
| baseline | 31,593 | 1.00x | 66,739 | — | 1.814281 |
| `check_every=2` | 29,254 | 1.08x | 68,623 | +2.8% | 1.814279 |
| `check_every=3` | 28,542 | 1.11x | 71,711 | +7.4% | 1.814280 |
| `check_every=4` | 28,276 | 1.12x | 73,095 | +9.5% | 1.814280 |
| `precond_degree=1` | 23,291 | **1.36x** | 34,772 | **-47.9%** | 1.814281 |
| `precond_degree=2` | 27,146 | 1.16x | 37,343 | -44.0% | 1.814280 |

**`precond_degree=1` is the win, exactly as the CPU pre-measurement predicted.**
It was singled out above as "the most GPU-shaped lever" on the grounds that it
deletes reductions and pays in applies, and that it was a net loss on CPU; on
the GPU that trade flips to **1.36x**. Degree 2 removes slightly more iterations
but pays for them in a third apply and nets *worse* than degree 1 — so degree 1
is the sweet spot, not "more is better".

`check_every` behaves as predicted too, but modestly: the sync saving does
outrun the wasted iterations, and unlike on CPU it keeps winning out to 4
(1.12x at +9.5% iterations). Note the two levers overlap — `precond_degree`
removes CG iterations, hence the syncs `check_every` exists to skip — so they
should be measured together rather than assumed to compound.

**Recommendation for GPU transients: `precond_degree=1`.** It is a
one-parameter change, the answer is unchanged to all printed digits
(1.814281 both ways), and it is worth more than everything else in this table
combined.

### The cost side of every GPU lever, measured on CPU first

Each of these levers trades *iterations* (which CPU can measure exactly, and
which are bus-independent) for *per-iteration cost* (which only a GPU can
settle). Measuring the iteration side up front means the GPU run has one
unknown per leg instead of two, and it sets the bar each leg has to clear.
2D refine 1, 11 groups, 1 step, negative insertion (Sigma_a x 1.01):

| leg | cg/step | vs base | us/cg | ms/step | sweeps | P(end) |
|---|---|---|---|---|---|---|
| baseline | 10,800 | — | 105.1 | **1,135** | 142 | 0.51053 |
| `check_every=3` | 12,879 | **+19%** | 93.9 | 1,209 | 142 | 0.51053 |
| `precond_degree=1` | 7,012 | **-35%** | 167.5 | 1,175 | 142 | 0.51053 |
| `precond_degree=2` | 6,126 | **-43%** | 200.6 | 1,229 | 142 | 0.51053 |
| float32 | 16,478 | **+53%** | 90.0 | 1,482 | 140 | 0.51052 |

Sweeps and the answer are unchanged in every case (float32 differs by one unit
in the 5th figure), so these are pure inner-solver trades — exactly the property
that makes them safe to A/B on wall time alone. Every leg is a net loss on CPU,
which is what makes them interesting: each buys something the CPU does not pay
for.

Reading them:

- **`precond_degree` is the most GPU-shaped lever here.** It removes 35-43% of
  the CG iterations — i.e. 35-43% of the *reductions and syncs* — and pays in
  extra operator applies, which on GPU are one fused kernel each. On CPU that
  trade is very nearly a wash: degree 1 raises `us/cg` 59% (105 -> 168) against
  a 35% iteration cut, netting only **3.5% slower**. On a bus where a reduction
  costs a pipeline stall and an apply costs one launch, the same trade should
  land clearly positive. (This is a much better CPU showing than the 37%
  slowdown recorded earlier on the coupled path — that was a bigger, 11-group
  refine-3 case, so the two are not directly comparable, but it does mean the
  bar the GPU has to clear is low.)
- **`check_every` has to clear +19%** of CG time from removed stalls alone.
- **float32 is not a free 2x.** Halving the bytes is worth at most 2x, but it
  costs +53% iterations at the same relative tolerance, so the ceiling is more
  like 1.3x — and only in the bandwidth-bound regime.

**Caveat on this table specifically:** it is the *smallest* case (330 cells,
3,630 unknowns), chosen so the five legs are cheap to run. That is fine for the
iteration counts, which barely move with mesh size, but it understates float32:
at this size the arrays are cache-resident, so halving them buys much less than
it would at 1M unknowns. Re-measure the dtype leg at size before trusting its
ratio.
- **float32 *converges* on the transient path**, which refines the earlier note
  that it fails: that was the *eigensolve* at `tol_k=1e-10, tol_source=1e-9`.
  The transient's defaults are `tol_k=1e-8, tol_source=1e-7` with
  `tol_step=1e-6`, which float32 has the headroom for. The lever is a tolerance
  question, not a dtype impossibility.

### What landed alongside the measurement

- **Batched group-source assembly in the transient** (`_group_batch`), the same
  trade `_PowerIterationSolver._make_group_batch` already made for the steady
  outer loop — but the transient needs it more, assembling a right-hand side per
  group per subsweep per sweep with ~250 sweeps in a step. O(G^2) launches per
  subsweep becomes O(G). GPU-only; the CPU path is unchanged by construction
  because `use_fused` is False on NumPy.
- **`linsolve_kwargs`** on `TransientSolver.solve`, which makes `pcg`'s
  `check_every` reachable. It is documented as *the* fix for per-iteration GPU
  stalls and was used by `sn.py`/`tri_sn.py` but by nothing on the diffusion
  path. **The cost side is now measured**, which sets the bar the GPU saving has
  to clear: at `check_every=3` on 2D refine 1, CG iterations per step go
  10,800 -> 12,879 (**+19%**) with the sweep count and the answer bit-unchanged
  (142 sweeps, P = 0.51053 both ways), and on CPU — which pays nothing for a
  sync — it is a straight **loss**, 1,668 -> 1,814 ms/step. So on GPU the
  removed stalls must be worth more than 19% of CG time or the leg is negative.
  Each group solve is only ~4-9 iterations (`cg per group-sweep` / 3 subsweeps),
  which is why the waste is this steep; expect 2-3 to be the whole usable range.
- **`step_iterations` populated for the diffusion transient.** The field existed
  on `TransientResult` but only `TransientSNSolver` ever filled it, so
  sweeps/step — the factor that turns out to dominate — was silently unavailable
  on the path where it matters most.
- **A NumPy path for `kernels.group_accumulate`.** Test-only: it is what lets
  the batched arithmetic, above all the scattering transpose, be checked without
  a GPU (`tests/verification/test_batched_group_source.py`, including an
  end-to-end run with the batch forced on and a negative control that asserts a
  transposed stack *would* be caught).

## Starting point

Two observations frame the whole plan.

1. **There is no custom-kernel code in the repo.** No `cupy.fuse`, no
   `ElementwiseKernel`, no `RawKernel`, no `ReductionKernel` anywhere under
   `ndgpu/`. Everything is generic NumPy-API code that CuPy dispatches one
   operation at a time. That is exactly what `backend.py` was designed for and
   it is the right default — this plan is about selectively breaking that
   uniformity where it costs the most.

2. **`tri_sn.py` is the exception and the template.** It already has
   device-resident sweeps (`_sweep_dev`), persistent preallocated buffers driven
   by `out=` kernels (`_run_levels`), a level schedule that batches the sweep
   into a short fixed kernel sequence, and **CUDA graph capture**
   (`_levels_exec`, `tri_sn.py:877-921`) — including the subtle detail of
   avoiding cuBLAS calls inside the captured region because graph capture cannot
   record them (`tri_sn.py:849`). None of that discipline has propagated to
   `stencil.py`, `spn.py`, `sp3.py`, `solver.py`, `linalg.py`, or the Cartesian
   `sn.py`.

So this is largely a matter of applying an approach already validated in one
module to the rest of the code.

## The core issue: launch-bound, not bandwidth-bound

`GroupOperator.apply` (`stencil.py:268-279`) is the innermost operation of every
diffusion, SP3, SPN, DSA and CMFD solve:

```python
out = self._stencil_diag * phi
out[1:, :, :] -= self.wx * phi[:-1, :, :]
...  # five more shifted multiply-adds
```

That is **13 kernel launches and 7 full-size allocations** per apply. Counting up
the stack:

| Operator | Kernels per apply |
|---|---|
| `GroupOperator` (diffusion), `stencil.py:268` | ~13 |
| `SP3GroupOperator`, `sp3.py:161-165` | ~32 |
| `SDPNGroupOperator` order 3, `spn.py:107-123` | **~145** |

The SDPN figure comes from `apply()` nesting a full `L.apply()` inside a loop
over projections, then a dense M x M reaction loop. At a typical ~5 us launch
cost that is ~0.7 ms of pure overhead per operator apply, *independent of grid
size*. On the HP-MR 2D grids (order 1e4-1e5 cells) the actual arithmetic is a
few microseconds. The overhead-to-work ratio is roughly 100:1.

`pcg` (`linalg.py:84-94`) compounds this: three dot products per iteration, each
written `xp.sum(u * v)` — a full-size temporary allocation plus two kernels,
where a fused reduction is one kernel and no temporary.

Conclusion: for the problem sizes this code actually runs, **kernel count is the
dominant cost, not memory bandwidth**. That is what makes fusion and graph
capture the right tools, in that order.

---

## Phase 0 — Instrument before optimizing (prerequisite) — DONE

Nothing below should be merged on faith.

*Implemented* as `ndgpu/profiling.py`. The useful piece turned out not to be
`cupyx.profiler.benchmark` but `effective_launches`: time an operation on a grid
small enough to be pure overhead, divide by the measured cost of one trivial
kernel, and read off how many kernels it launched. No profiler, no privileges,
and it works in Colab where `nsys` does not. `nvtx_range` is there for when a
real trace is available; `ab_compare` runs any callable with fusion off then on.

- Add a profiling harness: `cupyx.profiler.benchmark` on individual operator
  applies, plus `nvtx` ranges around sweep / PCG / outer iteration so `nsys`
  traces are readable.
- Extend `examples/speed_benchmark.py` to sweep grid size and report **kernel
  count** alongside wall time.

This immediately tells you, per solver, whether it is launch-bound or
bandwidth-bound — which decides whether fusion or graph capture is the right fix.
Pair accuracy and cost per the repo's benchmark-reporting protocol
(`.claude/skills/benchmark`).

## Phase 1 — Fuse the stencil apply (highest value, contained) — DONE

*Implemented* as `kernels.stencil7_apply`, an `ElementwiseKernel` over a flat
index that decodes (i, j, k) and reaches the three face-coupling arrays through
their own strides (each is one cell short on a *different* axis, which is the
part that is easy to get wrong). `GroupOperator.apply` dispatches to it when the
backend is CuPy, `phi` is C-contiguous, and `A phi` would not have to be
narrowed to `phi`'s dtype — that last guard is what keeps the noise solver's
complex removal from being silently truncated against a real flux.

`GroupOperator` also grew `apply(phi, out=...)` and a `supports_out` flag, so
block operators can write each row of their state in place. `cupy.fuse` was not
used: it does not handle the shifted-slice pattern.

Notes from doing it: the cached kernel arrays are built lazily per dtype
(`_fused_arrays`), so a solver that only ever runs float64 never materializes a
float32 copy. Grids one cell thick on an axis give a zero-size coupling array,
which becomes a 1-element dummy so the kernel always has a valid pointer.

Original plan text follows.

Replace the six shifted multiply-adds in `GroupOperator.apply` with a single
`cupy.RawKernel` (or an `ElementwiseKernel` over a flat index that decodes
i, j, k). One launch, one allocation, one pass over `phi` with cache reuse across
the seven stencil points instead of seven separate streaming reads.

- Expected: **13 -> 1 launches**, roughly 3-4x less memory traffic.
- Keep the existing NumPy path and dispatch to the kernel only when
  `xp is cupy`, so CPU behaviour and the whole validation suite stay untouched.
- This propagates automatically into SP3, SPN/SDPN, DSA, CMFD, transient and
  noise, since they all reach the stencil through `op_cls`.

`cupy.fuse` is worth a 20-minute experiment first, but it is expected to
disappoint: it does not handle the shifted-slice pattern well and tends to fall
back. The RawKernel is the real answer.

Watch out for: the cylindrical `row_scale` / `rhs_weight` variants, the
`active`-mask decoupling, and the non-symmetric divergence form
(`symmetric=False`). All must stay bit-comparable with the NumPy path — the
ANL 8-A1 and VVER cases are the regression guards.

## Phase 2 — Fuse the SPN/SDPN block apply — DONE

All three block operators now go through `ndgpu.kernels`:

- **SP3/SDP1 (2x2)**, `sp3.py:apply` — each moment's leakage is written straight
  into its row via `apply(out=...)`, then one `kernels.sp3_couple` pass over
  both rows. 3 kernels instead of ~32.
- **`SDPNGroupOperator` (per-moment order-N)**, `spn.py:apply` — M leakages
  written in place, then the whole M x M off-diagonal reaction in one
  `kernels.dense_react_add`. M+1 kernels instead of ~M(M+1).
- **`CongruentSDPNOperator` (the ~195-launch worst case)**, `spn.py:apply` —
  per projection: `moment_gather` (s = sum_m w_m u_m), one stencil,
  `moment_scatter_add` (out_m += w_m L s); then one `dense_react_add` for the
  dense reaction. Three kernels per projection plus one, instead of ~2M+13.

Implementation notes worth keeping:

- The dense (M, M, \*grid) coupling stack is built **lazily on first fused
  apply**, so the CPU path never allocates it and keeps walking the sparse
  `{(i, j): field}` dict. `kernels.dense_react_add` takes both forms and
  dispatches, which keeps one call site instead of branching in each operator.
- The fused reaction accumulator is initialized from `out` rather than from
  zero, so the summation order matches the per-pair expression it replaces.
- Operators without `supports_out` (the triangular and mesh stencils) take the
  allocating path with identical arithmetic, so tri-SP3/SDPN is unaffected.

**Not yet measured.** Correctness is pinned on CPU
(`test_fused_kernels.py` compares each block against a transcription of the
expression it replaced); the launch count and end-to-end effect come from the
next notebook run, where SDP3's ~195 is the row to watch and C5G7-2D SDP3 has
been added end to end (it selects the congruence block).

## Phase 1b — Fuse the triangular stencil — DONE

`TriGroupOperator.apply` was ~15 kernels of shifted slices over a
(nx, ny, 2\[, nz\]) up/down-triangle layout, and everything HP-MR runs on it —
the drum-worth work, the hybrid SPN masks, the tri-SPN solvers — so Phases 1-2
did not reach any of it. Now one `ElementwiseKernel`, plus `supports_out = True`
so the block operators write each moment into its row.

Harder than the Cartesian stencil in three ways, all of which the CPU
transcription test (`emulate_tri`) exists to catch:

- **Two interleaved sublattices.** Down and up triangles share the third axis,
  so the neighbour offsets depend on which sublattice the cell is in and the
  branch on `t` cannot be folded away.
- **Ordered weight pairs.** Each of the three face families (hypotenuse,
  vertical, horizontal) carries `(a, b)` rather than one symmetric weight,
  because discontinuity factors make the operator non-symmetric — so the two
  directions across a face read *different* arrays, and swapping them is a bug
  that constant-coefficient test data would not reveal.
- **2D as nz = 1.** A C-contiguous (nx, ny, 2) array has exactly the flat layout
  of (nx, ny, 2, 1), so one kernel covers slabs and extruded prisms; the axial
  coupling is a second compiled variant.

The `_fusable` dtype guard from Phase 1 is carried over — the noise solver
builds tri operators with a complex removal, and casting those down to a real
flux would silently drop the imaginary part.

The hex stencil (`hex.py`) is the same shape of problem and can follow; nothing
currently depends on it being fast.

`spn.py:107-123` is the worst offender and gates the SP5/SP7 and hybrid-drum
work. The block is M coupled fields on the same grid; the projected leakage plus
the dense M x M reaction can be one kernel templated on M, holding a cell's M
moments in registers.

- Expected: **~145 -> 1 launch** for SDP3.
- This is the single largest speedup available in the repo, and it is the path
  the HP-MR drum-worth work runs on.

Do this after Phase 1 so the fused scalar stencil already exists to build on.

## Phase 3 — Fuse and de-synchronize the Krylov solvers — DONE (except pipelining)

*Implemented*: `pcg` and `cocg` now route their vector updates and reductions
through `ndgpu.kernels` — `x += alpha*p; r -= alpha*Ap` is one kernel
(`cg_update`), `p <- z + beta*p` is one (`cg_direction`), the Neumann
preconditioner's damped-Jacobi sweep is one (`neumann_step`), and `xp.sum(u*v)`
became a `ReductionKernel` (`dot`) rather than cuBLAS, per the graph-capture
note below. The updated vectors are CuPy **out-params**, which bind by
reference, so the updates are genuinely in place and the CG loop no longer
allocates a direction vector per iteration — the Phase 4 prerequisite.

The coefficients stay 0-d device scalars throughout; only the `check_every`
convergence test touches the host. **Chronopoulos-Gear pipelining is not done**
and remains the open item in this phase.

Original plan text follows.

Three separable changes in `linalg.py`:

- **Fuse the vector updates.** `x += alpha*p; r -= alpha*Ap` into one
  `ElementwiseKernel`; `z = inv_diag*r` combined with the `r.z` reduction;
  `p = z + beta*p` fused. Same for the Neumann preconditioner's
  `z += inv_diag * (r - apply_A(z))` (`linalg.py:45`).
- **Replace `xp.sum(u*v)` with a fused `ReductionKernel`.** Prefer this over
  cuBLAS `dot`: cuBLAS is faster standalone but blocks graph capture in Phase 4,
  per the existing note at `tri_sn.py:849`.
- **Halve the synchronization points.** Restructure to Chronopoulos-Gear
  (pipelined) CG, which merges the two per-iteration reductions into one. The
  `linalg.py` module docstring already names global reductions as the only sync
  points; this attacks them directly. Slightly less numerically robust — gate it
  behind a flag and validate against the full benchmark suite before defaulting
  it on.

## Phase 4 — Extend CUDA graph capture beyond `tri_sn`

Once Phases 1-3 land, the PCG inner loop between convergence checks
(`check_every=k`) is a fixed sequence of kernels on fixed buffers — exactly the
shape `_levels_exec` already captures. Capture k iterations as one graph and
replay it.

Prerequisite: convert `pcg` to persistent preallocated work vectors with `out=`,
which Phase 3 mostly does anyway. Reuse the capture/fallback structure from
`tri_sn._levels_exec` verbatim, including the memory-pool warm-up (allocating
during capture is illegal).

This is what makes *small* grids fast, and small grids are most of the
validation suite.

## Phase 5 — Kill the host round-trips in the Cartesian SN path — DONE (partly)

Done: the **DSA solve** and the **within-group GMRES**.

- `_make_diff_solver` (new, mirroring `tri_sn`'s) moves the DSA diffusion
  operator to the device once and solves it with Jacobi-CG. It replaces a
  per-source-iteration D2H -> host sparse LU back-solve -> H2D. The transient
  DSA factor goes through the same path.
- `_gmres_solve` now runs `ndgpu.linalg.gmres` over grid-shaped device arrays
  instead of scipy's over host vectors, so the Krylov space stays on the GPU and
  a matvec no longer round-trips the flux. ndgpu preconditions on the *right*,
  so the Arnoldi residual is the true residual and the stopping rule is
  unchanged; the DSA preconditioner is linear, which is what that requires.
  `linalg.gmres` gained `raise_on_fail` (matching `pcg`) so an inexact
  within-group solve returns its iterate rather than raising.
- `dsa_on_device=False` restores the old host path — the A/B lever, and the
  fallback if the iterative DSA solve ever proves too weak.
- `_si_solve`'s two per-iteration `float()` reductions became one transfer.

**Not done, deliberately: the sweep's boundary arrays.** The plan listed the
`_sweep_wavefront` host round-trip beside the DSA one, but they are three orders
of magnitude apart. The boundary edge fluxes are (M, ny) and (M, nx) — 8 small
transfers per sweep, against a diagonal loop of (nx+ny-1) x ~8 batched kernels.
That is ~1% of a sweep, and making it device-resident means moving `_reflect`,
`_flat_inc`/`_unflat_inc` and the boundary fixed point onto the device too.
Not worth the regression risk for 1%; the host LU it sat next to was the real
defect.

Original plan text follows.

In `sn.py`:

- `_dsa_apply` (`sn.py:720-724`) does `asnumpy(r)` -> scipy host LU solve ->
  `xp.asarray(...)` **per source iteration**: a D2H + host solve + H2D in the
  innermost loop.
- `_gmres_solve` (`sn.py:864-892`) runs scipy GMRES on the host, calling
  `asnumpy(p).ravel()` on every matvec.
- `_sweep_wavefront` returns edge fluxes as host numpy (`sn.py:632`), `_reflect`
  operates on host, and the next sweep copies them back
  (`sn.py:586-589`) — a full host round-trip per sweep, 8 H2D copies per call.
- `_si_solve` (`sn.py:849`) syncs twice per iteration on `float(xp.max(...))`.
  Check every few iterations instead, as `pcg` already does.

`tri_sn.py` already solved all of this: `_make_diff_solver`
(`tri_sn.py:936-968`) moves the operator to the device once and solves with
device PCG/BiCGStab, and `_sweep_dev` keeps the sweep device-resident. Port both
patterns to `sn.py` and `hybrid_sn.py`. Keeping edge fluxes on-device is a
contained, self-checking change.

Consider promoting this above Phase 2 if the Cartesian SN path is the active
workload — it is less an optimization than "the GPU path is not really on the
GPU".

## Phase 6 — Batch the group loops — DONE (source assembly)

*Implemented* in `_PowerIterationSolver.solve`. The scattering matrix is stacked
into `(G, G, *grid)` (`solver.scatter_stack`) and the per-group scalar fluxes
into `(G, *grid)`, so one `kernels.group_accumulate` walks a whole in-scatter
row and another does the entire fission source. O(G^2) launches per outer
becomes O(G).

What made it safe rather than a wide refactor:

- **The Gauss-Seidel sweep is preserved exactly.** The stacked flux buffer is
  refreshed immediately after each group's solve, so a row reads the *updated*
  flux for g' < g and the previous one for g' > g -- what the Python loop did.
  The buffer is also rescaled with the state at the end of each outer (phi0 is
  linear in the state), which is the one place a stale-cache bug could hide.
- **`Fields` was not restructured.** The plan proposed storing cross sections
  as `(G, ...)` arrays throughout, which touches transient, noise and SPH. The
  stack is built once per solve from the existing sparse lists instead, so
  nothing outside `solve()` changes.
- **CPU keeps the sparse loop**, which skips absent couplings rather than
  multiplying by materialized zeros -- strictly better where launches are free.
- **Gated on G >= 3 and on memory**: the `(G, G, *grid)` stack is skipped if it
  would exceed an eighth of free device memory. At G = 2 the saving is two
  kernels and not worth the footprint.

The transpose (`sigma_s[gf][g]` forward, `sigma_s[g][gf]` adjoint) is the part a
batched rewrite silently gets backwards -- only adjoint solves would notice --
so `scatter_stack` is a module-level function with a CPU test pinning both
senses against the loop.

**Not yet measured**, and the case that will decide it is the **11-group HP-MR
core** now in the notebook (vendored ENDF/B-8 data, ~200 outers, 121 in-scatter
pairs per outer). Note the default HP-MR placeholder materials are only
2-group, so every other HP-MR row will show `x_groups` = 1.0 by construction.

Original plan text follows.


`Fields` (`solver.py:102-222`) stores per-group data as Python lists of separate
arrays, so every group operation is a Python loop of small kernels. The
in-scatter assembly in `solver.py:407-416` is **G^2 kernel launches per outer
iteration** — 121 for the 11-group HP-MR library. Storing cross sections as
`(G, nx, ny, nz)` arrays and doing the scatter as one batched contraction over
the group axis collapses that to a handful.

Wider refactor than Phases 1-3; touches transient, noise and SPH. Sequence it
last despite the good payoff. Note the within-group solves are Gauss-Seidel over
groups and stay sequential — only the source assembly batches.

---

## Precision — check this independently of everything above

`solver.py:245` claims float32 "roughly doubles GPU throughput". That is about
right on a datacenter card (A100/H100); on any GeForce card FP64 runs at
**1/32 to 1/64** of FP32.

**Resolved for the T4** (the card the 2026-07-29 run used, FP64 at 1/32 of
FP32): the docstring is right and my earlier framing was wrong. These stencils
are *bandwidth*-bound, not FLOP-bound — the fused apply hits ~79% of peak
bandwidth at 96^3 — so float32 buys the halved memory traffic, about 2x, not the
32x arithmetic ratio. Worth having, but it does not outrank Phases 2-6 the way
the raw FP64:FP32 number suggested. The S_N sweeps may differ and have not been
measured.

Independently of the ratio, `sn.py`/`tri_sn.py` still have no `dtype` parameter
at all, which is a gap regardless of how much float32 turns out to buy.

Relatedly, `sn.py` and `tri_sn.py` have **no `dtype` parameter at all** — the SN
sweeps are hard-wired float64 (`xp.zeros(...)` with no dtype throughout). Since a
sweep is a transport recurrence with mild error growth, FP32 sweeps with FP64
accumulation of the scalar flux and k is a standard, safe mixed-precision split
and would be a large win on consumer hardware. Diffusion and SPN already plumb
`dtype` through; SN should match.

## Notebooks

Two, and they cover disjoint code:

- `notebooks/colab_gpu_fusion_phase1.ipynb` — diffusion and SPN. Launch counts,
  the correctness gates, the stencil/Krylov/block/groups leg A/B, the `dot`
  crossover, the Cartesian S_N DSA-on-device A/B (Phase 5), and end-to-end rows
  for the bare box, IAEA-3D, C5G7-2D (diffusion/SP3/SDP3), HP-MR 2D and 3D, and
  the 11-group HP-MR core.
- `notebooks/colab_tri_sn_capture.ipynb` — the **transport** path, which the
  first notebook never touches (`TriSNTransportSolver` does not appear in it).
  Whether capture engages at all, the broadcast-vs-fused contraction gate on
  both schemes, buffer footprint, wall time vs problem size and ordinate count,
  a `graphs=` on/off A/B, the 3D prisms, and the four-leg fusion A/B on the
  hybrid tri-S_N/diffusion solver plus an 11-group tri-S_N row.

- `notebooks/colab_hpmr_transient_gpu.ipynb` — the **uncoupled HP-MR transient**
  (see the section below). CPU/GPU agreement gate, the batched-assembly A/B, a
  size sweep from 14.5k to 2.6M unknowns reporting `us/cg` so the launch-bound /
  bandwidth-bound regime is read off rather than assumed, and one-at-a-time legs
  for `check_every`, `precond_degree` and float32.

**Known gap the second notebook is built to expose:** Phase 6 batched the
in-scatter assembly in `solver.py`'s power iteration, but
`TriSNTransportSolver.solve` has its *own* O(G^2) group loop (twice -- the group
solves and the CMFD pass) which was not touched. On a pure tri-S_N solve the
`groups` leg therefore measures only the diffusion/CMFD sub-solves. The
11-group tri-S_N row is the evidence for whether extending Phase 6 there pays.

## How to measure what has landed

```
python tools/build_src_zip.py          # notebooks install this; a stale zip
                                       # silently runs old code
```
then open `notebooks/colab_gpu_fusion_phase1.ipynb` on a GPU runtime and upload
`dist/ndgpu-src.zip`. Sections in order: launch cost and kernels-per-apply; a
hard correctness gate (fused vs. unfused to round-off, including the awkward
cases — one-cell-thick grids, `active` masks, every boundary law, the
cylindrical divergence form, a complex operator); the stencil A/B vs. grid size;
the CG A/B with an assertion that the iteration count is unchanged; and
end-to-end `k_eff` + cost on the bare box, IAEA-3D and C5G7-2D with the
|Δk| <= 2e-8 equivalence bound enforced.

The shape of the section-3 curve is the decisive result: a speedup that is large
on small grids and *decays* toward a small constant confirms launch-bound, and
the level it decays to is the bandwidth-bound ratio. Flat and small would mean
this GPU is not launch-bound at these sizes and Phases 4-6 should be re-ordered.

## Suggested ordering

Phase 0 -> 1 -> 3 -> 2 captures most of the benefit, and each step is
independently verifiable against the existing benchmark suite. Slot Phase 5
early if Cartesian SN is the active workload.

## Open questions that would change the ordering

- ~~**Which GPU is the target?**~~ Answered: Tesla T4 on Colab. See the
  precision section — it turned out not to dominate, because these kernels are
  bandwidth-bound.
- **Which solver path matters most right now** — SPN/hybrid drum work (Phase 2)
  or Cartesian SN (Phase 5)? Still open, and now the main thing setting the
  order. Phase 2 is sized at 195 probe-kernels per SDP3 apply.

## Next actions (as of 2026-08-05)

**Ahead of everything below, for the transient specifically: a spatial CMFD.**
The measurement section above found ~250 outer sweeps per step at every mesh
size, which is 1-2 orders of magnitude more leverage than any kernel work in
this document and is bus-independent. `rebalance` is a one-cell CMFD and is
already worth 5.3x; `TransientSNSolver._cmfd_step` is a working spatial one to
port from. Kernel-level GPU work multiplies with it, so the order is not
either/or — but doing the GPU side first optimizes the smaller factor.

Phases 0, 1, 1b, 2, 3, 5 and 6 are landed. Everything except Phase 5 and the
transient's 6b is measured: **1.4x-4.2x on whole solves**, with the stencil leg
winning on every case. Remaining, in order:

0. **Run `notebooks/colab_hpmr_transient_gpu.ipynb`.** It is the only unmeasured
   thing on the diffusion transient path and it carries three unmeasured
   changes: the batched transient assembly (6b), `check_every` reachability, and
   the launch-vs-bandwidth regime call for this problem. Its size sweep is also
   the first CPU/GPU comparison in this repo that asserts the two buses did the
   *same iteration count* before dividing their wall times.

1. **Measure Phases 5 and 6.** The notebook's section 6 A/Bs `dsa_on_device` on a
   C5G7-sized S_N problem for both `dsa` and `dsa-gmres`. It is the only
   unmeasured change in the repo, and unlike the fusion phases it is a defect
   repair, so the expected shape is *iterations up, wall time down* — report
   both. It also now carries a **3D HP-MR** end-to-end row (extruded prisms,
   ~65k cells at refine=4/nz=20), the largest and most representative case here.
2. **Confirm the `dot` threshold** on any second GPU before trusting 2^16 as a
   constant; the crossover is a bandwidth/launch balance and will move. A
   size-dispatched constant is a stopgap — the principled fix is a hand-written
   multi-block reduction that beats CUB, which is real work for a ~2x gain on
   one of five kernels, and is deliberately deferred.
3. **The hex stencil** (`hex.py`) — the same shape of problem as Phase 1b, an
   afternoon's work, but nothing currently depends on it being fast.
4. **Phase 4 (extending graph capture)** is less attractive than it looked: it
   only attacks launch overhead, and large grids turned out to be
   bandwidth-bound. It is still right for the small-grid regime, which is most
   of the validation suite -- but see the section above on what capture costs
   when it dictates the arithmetic inside the captured region. Extend it only
   where the captured code needs no cuBLAS *and* no full-size temporaries.
5. **Precision** unchanged — see the section above.
