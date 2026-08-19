# Step 02: guard-aware adjoint reuse

Date: 2026-08-11  
Device: CPU / NumPy, FP64  
Decision: **accepted on CPU; retain opt-in public controls**

## Audit result

The original roadmap assumed a hard residual event paid for an IQS corrector
and then replayed the interval with full diffusion. Code tracing showed that
this was already avoided: the projected forward residual makes the fallback
decision before the IQS macro corrector.

Profiling the asymmetric guarded benchmark instead found:

| Phase | Seconds | Share of 13.55 s march |
|---|---:|---:|
| Adjoint shape solves | 7.30 | 53.9% |
| IQS shape solves | 5.49 | 40.5% |
| Full-diffusion fallback | 0.65 | 4.8% |

The useful optimization target was therefore adjoint reuse, not fallback
replay.

## Implementation

`projected_adjoint_residual` evaluates the defect of the current adjoint in the
energy-transposed eigenproblem. Its eigenvalue/amplitude component is projected
out against the current forward shape, so arbitrary adjoint normalization and
pure eigenvalue motion do not trigger a refresh.

`adjoint_residual_tol` checks this defect at every shape correction:

- a defect above the tolerance refreshes the adjoint early;
- `adjoint_every` remains a hard maximum reuse age;
- a full-diffusion fallback always refreshes the adjoint;
- no tolerance preserves the previous fixed-cadence behavior.

The result records residual values/times and evaluation, refresh, and maximum-
defect counters. The residual cost was 0.006–0.008 s for 16–18 evaluations,
against seconds for the avoided eigen solves.

The calibrated two-group benchmark uses `adjoint_every=6` and
`adjoint_residual_tol=0.005`. These are benchmark controls, not universal
reactor-design defaults.

## Full three-case benchmark

| Case | Previous guarded dynamic [s] | New [s] | Adjoint solves old → new | Power error old → new | Fallbacks |
|---|---:|---:|---:|---:|---:|
| Slow symmetric | 4.58 | 4.23 | 9 → 6 | 2.98% → 2.98% | 0 |
| Fast symmetric | 4.96 | 4.15 | 10 → 7 | 2.46% → 2.62% | 0 |
| Asymmetric | 15.24 | 13.77 | 11 → 9 | 1.26% → 1.97% | 1 |

Cross-run wall deltas are indicative because CPU load varied; the deterministic
adjoint counts establish the work reduction. Against the full solves from the
new run, guarded dynamic speedups are 2.16x, 2.05x, and 0.96x. The asymmetric
guard is now close to parity rather than 20% slower, while remaining below the
calibrated 3% maximum power-error envelope.

Threshold exploration showed the expected trade-off. With a six-update maximum
age, no residual-triggered refresh gave about 1.99% asymmetric error; a 0.01
threshold gave 2.03%; 0.005 retained more refreshes and stayed at 1.97%. The
selected value also behaved consistently on both symmetric ramps.

## Verification

Regression coverage checks:

- an exact adjoint has negligible residual;
- forward and adjoint rescaling do not change the residual;
- a localized absorber perturbation produces a nonzero defect;
- the residual can force refresh before maximum age;
- non-positive tolerances are rejected;
- hard fallback behavior and existing IQS tests remain intact.

The next roadmap phase targets persistent GPU workspaces and CUDA-graph-ready
Krylov iteration batches.

