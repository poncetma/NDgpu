# Step 03b: fixed-block CUDA-graph PCG

Date: 2026-08-12  
Decision: **implemented; CPU fallback verified, GPU gate pending**

## Motivation from Colab

GPU cost stayed nearly flat at roughly 600--640 microseconds per CG iteration
from 32,670 through 580,800 unknowns, while the GPU remained 1.48x slower than
CPU at 32,670 unknowns. This is the signature of launch/synchronization cost,
not arithmetic throughput. At 1.16M and 2.61M unknowns the cost rose to 719 and
1,082 microseconds per iteration as bandwidth began to matter.

## Implementation

`pcg(..., graph_block=k)` captures exactly `k` complete PCG recurrences in one
CUDA graph. It requires `graph_block == check_every`, so captured and ordinary
paths inspect convergence after exactly the same iteration numbers.

The captured recurrence uses persistent arrays for:

- solution, residual, direction, preconditioned residual, and `A*p`;
- `r.z`, new `r.z`, `p.A.p`, alpha, and beta device scalars;
- operator and native/mixed-preconditioner output buffers.

Dot reductions write into caller-provided scalar buffers. Scalar division and
copy are explicit one-element kernels, so no array allocation or host scalar
read occurs inside capture. The first real iteration block warms kernel
compilation and the memory pool; graph construction is therefore included in
the measured transient but does not perform a fake extra block.

Capture is currently restricted to CuPy, fused Krylov kernels, `out=` capable
operators, allocation-free preconditioners, persistent workspaces, and at most
65,536 cells per group. Above that size `kernels.dot` deliberately uses CuPy's
faster CUB reduction, whose allocation behavior is not graph-safe. Unsupported
or failed capture falls back permanently and records the reason.

Telemetry is exposed as:

- `TransientResult.cuda_graph_captures`;
- `TransientResult.cuda_graph_replays`;
- `TransientResult.cuda_graph_errors`;
- matching benchmark keys and coupled profile counters.

## CPU gate

Focused Krylov/fused/transient result: **85 passed, 2 skipped in 59.65 s**.
Extended focused result including coupling/model API: **105 passed, 2 skipped
in 87.05 s**. Full repository result: **501 passed, 5 skipped, 13 deselected
in 538.81 s**.

A real 11-group HP-MR refinement-2 insertion with `check_every=3` was run with
and without `graph_block=3` on CPU. CPU correctly took the fallback path while
retaining exactly:

- 6,204 inner iterations;
- 27 fixed-point sweeps;
- final power `1.3990792349574892`.

Unit tests also pin cadence validation, scalar-buffer operations, transparent
CPU fallback, unchanged iterations, and unchanged solution arrays.

## GPU gate added to Colab

The notebook runs the launch-bound 2-D refinement-3 case at block/cadence 1
and 3, each against an uncaptured leg with the same cadence. It prints graph
captures, replays, errors, time, and speedup, and asserts identical CG work and
fixed-point sweeps.

Acceptance requires:

1. nonzero captures/replays and no capture errors;
2. identical work, power, and shape at the same cadence;
3. reduced microseconds per CG iteration for both block sizes;
4. graph construction amortized within the four-step reported duration;
5. no enabling by default until the faster block is known on the test GPU.
