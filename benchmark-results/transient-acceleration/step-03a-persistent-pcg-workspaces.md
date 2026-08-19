# Step 03a: persistent PCG workspaces

Date: 2026-08-11  
Device: CPU / NumPy, FP64  
Decision: **accepted as a CPU-verified prerequisite; GPU performance gate pending**

## Change

`PCGWorkspace` now owns stable `x`, residual, preconditioned residual,
direction, and operator-output arrays. The diffusion transient keeps one
workspace per energy group. Cartesian and triangular finite-volume operators
write directly into the persistent `A p` buffer, and the native Neumann/Jacobi
preconditioner uses persistent output and scratch buffers as well.

The default remains the same PCG recurrence and residual check. Setting
`reuse_krylov_workspaces=False` restores the allocationful execution path for
an A/B measurement. Other Krylov methods are unchanged.

This is intentionally not CUDA graph capture yet. Stable storage is a
necessary capture condition, but capture and replay must be measured on a real
GPU before being enabled.

## Correctness

Focused regressions compare allocationful and workspace PCG, including a
degree-2 Neumann preconditioner. They verify identical solutions and iteration
counts, stable array identity across solves, use of every supported `out=`
operator call, and rejection of incompatible workspace shapes/dtypes.

A three-step diffusion transient is bit-for-bit identical with reuse disabled
and enabled:

- identical power history and final flux;
- identical inner-iteration total;
- identical fixed-point sweeps per step.

Focused result: **34 passed in 25.98 s**.
Full repository result: **492 passed, 5 skipped, 13 deselected in 480.62 s**.

## Tier-B 11-group HP-MR insertion

Configuration: 2-D refinement 2, 1,320 active cells, 14,520 unknowns, four
20 ms steps, six energy subsweeps, whole-core rebalance, `tol_step=1e-6`.
Each leg was run three times; the table reports the median wall time.

| Storage | Wall samples (s) | Median (s) | Inner iterations | Sweeps/step | Final P/P0 |
|---|---:|---:|---:|---:|---:|
| Allocationful | 2.974, 2.903, 3.030 | 2.974 | 17,786 | 23.0 | 1.8413575583 |
| Persistent | 2.998, 3.200, 2.941 | 2.998 | 17,786 | 23.0 | 1.8413575583 |

The 0.8% CPU median difference is noise/neutral, as expected: NumPy allocation
is not the target and the five workspace validations add a small fixed host
cost. Work and physics are exactly invariant.

## Repeated-solve allocation probe

Eighty nearby right-hand sides were solved on a `32 x 32 x 4` stencil, with
2,922 total PCG iterations in every sample. Five-run medians:

| Storage | Median time (s) | Traced peak bytes during march |
|---|---:|---:|
| Allocationful | 0.6209 | 328,952 |
| Persistent | 0.6035 | 98,568 |

Persistent storage reduced the measured transient allocation high-water mark
by 70.0% and was 2.9% faster in this allocator-focused CPU probe. This is
supporting evidence, not a GPU speed claim.

## GPU acceptance still required

The updated Colab transient notebook now includes an allocationful/workspace
A/B leg. Before proceeding to graph replay, require on both a launch-bound 2-D
case and a larger 3-D case:

1. identical `cg/step`, sweeps, power, and final shape;
2. lower or neutral `us/cg` with workspaces;
3. lower CuPy memory-pool growth after warm-up;
4. no graph-capture implementation accepted until graph construction is
   amortized by the reported transient duration.

## Colab GPU results

The first reported 2-D refinement-3 leg has 2,970 active cells and 32,670
unknowns. CPU and GPU performed the same work to within 0.008%:

| Device | ms/step | Sweeps/step | CG/step | us/CG | P(end) |
|---|---:|---:|---:|---:|---:|
| CPU | 2,743.6 | 22 | 6,342 | 432.6 | 1.84263 |
| GPU | 4,048.2 | 22 | 6,342 | 638.3 | 1.84260 |

The GPU is therefore 1.48x slower at this size, despite comparable work and a
power difference (`3.13e-5`) inside the configured solver error. This fails the
launch-bound 2-D performance gate and confirms that reducing GPU dispatch cost
is necessary. Batched group-source assembly improved the GPU leg from 4,371.6
to 4,048.2 ms/step (1.08x), but is not enough on its own.

The size sweep locates the CPU/GPU crossover between 58,080 and 130,680
unknowns:

| Case | Unknowns | CPU ms/step | GPU ms/step | GPU/CPU speedup | GPU us/CG |
|---|---:|---:|---:|---:|---:|
| 2-D refine 3 | 32,670 | 1,910.9 | 4,062.9 | 0.47x | 640.6 |
| 2-D refine 4 | 58,080 | 3,972.0 | 5,476.7 | 0.73x | 620.7 |
| 2-D refine 6 | 130,680 | 11,559.2 | 7,987.7 | 1.45x | 597.5 |
| 3-D refine 4 x 10 | 580,800 | -- | 7,807.3 | -- | 644.3 |
| 3-D refine 4 x 20 | 1,161,600 | -- | 6,959.0 | -- | 718.7 |
| 3-D refine 6 x 20 | 2,613,600 | -- | 15,393.0 | -- | 1,082.0 |

`us/CG` is nearly flat through roughly 600k unknowns, then rises as memory
bandwidth begins to matter. Raw milliseconds across the 3-D rows should not be
compared without work counters: refine 4 x 20 needed fewer sweeps and CG
iterations than refine 4 x 10.

On the 1,161,600-unknown case, the latest run reduced 7,543.4 to 7,289.1
ms/step, a **1.035x** gain at identical work and power. This accepts the
workspace mechanism on the large GPU case. A small 2-D workspace A/B and CuPy
pool high-water measurement remain desirable, but persistent storage stays as
the default and enables the graph-capture experiment.

The same 3-D tuning sweep found `check_every=3` worth 1.08x and degree-1
Neumann preconditioning worth 1.20x independently. Their combined effect is
added to the next notebook revision.
