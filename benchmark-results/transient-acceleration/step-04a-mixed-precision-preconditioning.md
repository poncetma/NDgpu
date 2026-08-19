# Step 04a: FP32 polynomial preconditioning

Date: 2026-08-11  
Device: CPU / NumPy, FP64 outer solve  
Decision: **accuracy accepted; fused-cast GPU retest rejected on performance**

## Numerical design

`precond_dtype="float32"` builds a shadow FP32 diffusion operator for the
Jacobi/Neumann polynomial only. The following remain FP64:

- neutron state and physical diffusion operator;
- true residual evaluations and convergence decisions;
- CG recurrence coefficients and vector state;
- fission/scattering fixed point, precursors, and power edits.

The low-precision preconditioner owns persistent FP32 residual, correction,
and operator-output arrays. PCG periodically replaces its recursive residual
with `b - A*x` evaluated by the FP64 operator (default interval 32). If a
checked residual becomes non-finite or the mixed solve reaches its iteration
limit, it restarts safely from zero with the native FP64 preconditioner.
Low-precision operators with non-finite or non-positive inverse diagonals are
rejected during setup and use FP64 immediately. FP16 is deliberately not
exposed.

## CPU verification

Focused Krylov, preconditioner, transient, coupling, and model-API result:
**63 passed in 66.68 s**.
Full repository result: **497 passed, 5 skipped, 13 deselected in 501.28 s**.
After the fused-cast and full-FP32 dtype repairs: **499 passed, 5 skipped,
13 deselected in 493.37 s**.

Regressions cover:

- FP64 true-residual accuracy with a degree-2 FP32 polynomial;
- periodic residual replacement;
- deliberate NaN injection and successful automatic FP64 restart;
- invalid dtype rejection;
- FP32-versus-FP64 transient power and final-flux accuracy.

## Tier-B 11-group HP-MR insertion

Configuration: 2-D refinement 2, four 20 ms steps, degree-1 Neumann
preconditioning, whole-core rebalance, and six energy subsweeps. Three CPU
runs per leg:

| Preconditioner | Wall samples (s) | Median (s) | Inner iterations | Sweeps/step | Final P/P0 |
|---|---:|---:|---:|---:|---:|
| FP64 | 3.060, 3.009, 3.402 | 3.060 | 10,455 | 23.0 | 1.8413555045127423 |
| FP32 | 3.373, 3.512, 3.605 | 3.512 | 10,455 | 23.0 | 1.8413555045123060 |

Work is exactly invariant and final power differs by `4.4e-13`. FP32 is 14.8%
slower on CPU because casting and the shadow operator cannot benefit from GPU
bandwidth; this is expected and prevents claiming a device-independent win.

## GPU gate

The updated Colab notebook adds `mixed-precond-fp32` immediately after the
degree-1 FP64 row. Accept mixed preconditioning only if:

1. power/shape remain inside the FP64 accuracy gate and fallback count is zero;
2. sweeps and total inner work remain materially unchanged;
3. it beats the degree-1 FP64 preconditioner on both the 2-D and 3-D GPU legs;
4. memory traffic/time saved by FP32 exceeds cast-kernel overhead.

## First GPU result

On the 1,161,600-unknown 3-D refine-4 x 20 case:

| Preconditioner | ms/step | CG/step | P(end) | vs baseline |
|---|---:|---:|---:|---:|
| Degree 0 FP64 baseline | 7,289.1 | 9,683 | 1.814890 | 1.00x |
| Degree 1 FP64 | 5,899.3 | 5,169 | 1.814890 | 1.24x |
| Degree 1 mixed FP32 | 6,175.1 | 5,169 | 1.814890 | 1.18x |

Mixed precision preserved work and power but was **4.7% slower than the
equivalent degree-1 FP64 preconditioner**, so the original implementation
fails its performance gate. Its apparent 1.18x gain against degree zero came
from the polynomial, not FP32.

Profiling identified two conversion launches around every FP32 polynomial
application. Fusing the residual down-cast and initial Jacobi multiply removed
one, but the retest above still lost. Mixed preconditioning therefore remains
available only as an experimental option; development moved to the monolithic
multigroup solve.

The full-FP32 leg exposed a separate fused group-accumulation dtype mismatch.
The source assembly now explicitly retains solver dtype, the contraction
accepts distinct weight/flux/output dtypes while accumulating in output
precision, and precursor kinetics no longer silently promote to FP64. A new
CPU regression verifies FP32 flux and precursor storage. Full FP32 is also
rejected on the measured GPU: 11,294 ms/step, twice the Krylov work, and a
larger power deviation than FP64.
