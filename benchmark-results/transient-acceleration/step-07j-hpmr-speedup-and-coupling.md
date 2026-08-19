# Step 07j — why HP-MR shows less BDF speedup, and coupled feedback

Date: 2026-08-18

## Outcome

The smaller HP-MR speedup is real, but the earlier 1.49--2.00x figures were
not matched-method speedups. They compared adaptive BDF with a deliberately
fine 64-step backward-Euler reference. At matched controller tolerance and
observable error, adaptive backward Euler is as fast as or slightly faster
than adaptive BDF on these small, smooth drum maneuvers.

The generic thermal coupler now supports unequal accepted neutron widths. It
inserts thermal exchange times as exact adaptive endpoints without restarting
BDF history, integrates power with `sum(q_i * dt_i) / sum(dt_i)`, and advances
feedback only after accepted steps. Rejected neutron work and adaptive history
are exposed in the coupled result and profiler counters.

## Speedup diagnosis

On the uncoupled 11-group slow-symmetric case at `RTOL=1e-3`:

| method | acc./rej. | order counts | inner | CPU (s) | max power error |
|---|---:|---:|---:|---:|---:|
| adaptive BE | 29/8 | 29 at q=1 | 69,155 | 14.74 | 0.0279% |
| automatic BDF5 | 29/9 | 23/6 at q=1/2 | 71,143 | 14.63 | 0.0210% |
| fixed-maximum BDF5 | 42/7 | 4/4/4/4/26 | 87,723 | 17.43 | 0.0128% |

Automatic BDF and BE therefore have essentially identical wall time; BDF uses
2.9% more inner work for a modestly smaller error. Forcing order five makes the
defect more restrictive and is slower. Removing the four trajectory-knot
restarts does not help: automatic BDF still chooses order one for 23 of 28
steps and becomes slower than BE because of additional rejections.

The contrast with LRA has four causes:

1. **Benchmark contract.** LRA's headline is a matched-error comparison:
   BE needs 3180 accepted steps while BDF needs 201. The first HP-MR headline
   used a 64-step BE reference, only 2.2--3.8 times the adaptive count.
2. **Temporal physics.** LRA contains a sharp prompt-supercritical peak and a
   second rod-stop peak. The present HP-MR motion changes normalized power by
   only a few percent and is close to linear between raster knots.
3. **Selected order.** LRA spends most of its history at high order. HP-MR's
   full-state predictor defect consistently selects q=1 or q=2; higher-degree
   extrapolation adds little and is more sensitive to spatial/raster noise.
4. **Cost per attempt.** Every HP-MR attempt carries a similar 11-group spatial
   solve. Nine BDF rejections erase the small saving available from only 35
   fewer accepted steps than the fine reference.

Thus the 1.5--2.0x number remains a valid cost reduction relative to a chosen
fine reference, but it must not be described as a BDF-over-BE advantage.

## Coupled feedback gates

The accelerated-thermal two-group stress test (`heat capacity x 0.005`) gives:

| method | acc./rej. | inner | CPU (s) | max power error | max mean-T error |
|---|---:|---:|---:|---:|---:|
| adaptive BE | 17/4 | 5,625 | 0.999 | 0.0139% | 0.0119 K |
| automatic BDF5 | 18/4 | 5,800 | 1.056 | 0.0176% | 0.0129 K |

The physical-mass 11-group coupled gate gives:

| method | acc./rej. | inner | CPU (s) | max power error | max mean-T error |
|---|---:|---:|---:|---:|---:|
| adaptive BE | 29/8 | 69,171 | 13.58 | 0.0278% | 0.000127 K |
| automatic BDF5 | 29/9 | 70,951 | 14.29 | 0.0208% | 0.000132 K |

Both coupled gates are stable, positive, and agree closely with their 64-step
BE reference. Adaptive BDF is 5--6% slower than adaptive BE here, consistent
with its order-one-dominated history. The 11-group physical case changes mean
fuel temperature by only 0.004 K over 0.8 s, so it verifies coupling mechanics
and numerical stability rather than strong physical feedback.

## Decision

Adaptive width control passes the coupled 2-D gate, but high-order BDF has no
performance advantage for the current mild HP-MR maneuvers. Keep automatic
order because its fallback to q=1--2 is safe; report method-to-method speedup
only at matched error. The next informative CPU experiment is a larger-worth
or longer maneuver that produces genuine nonlinear thermal feedback. The next
production-scale gates remain 3-D motion and GPU throughput.
