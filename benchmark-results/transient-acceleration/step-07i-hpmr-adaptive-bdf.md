# Step 07i — adaptive BDF on moving HP-MR drums

Date: 2026-08-18

## Benchmark correction

The legacy fixed-step HP-MR BDF harness had two problems that made it unsuitable
for an adaptive stability gate:

1. its refinement-1, 4-degree drum trajectory produced identical mixture
   weights at every frame, so the nominal motion was actually unresolved; and
2. it selected pre-rasterized frames as piecewise-constant coefficient jumps.

The harness now rejects an unresolved trajectory, defaults to refinement 2,
and linearly interpolates compatible cached mixture weights between frame
knots. BDF history still restarts at the four knots because the piecewise-linear
trajectory derivative changes there. The fine reference scheme is explicit and
defaults to the required backward Euler rather than the former BDF2.

## Eleven-group CPU stability gate

All cases use the real 55-site triangular HP-MR geometry, the 11-group
ENDF/B-VIII-derived constants, polar drum mixing, monolithic multigroup FGMRES,
and automatic BDF1--5 at `RTOL=1e-3`. The adaptive initial width corresponds to
eight uniform steps. Accuracy is measured against a 64-step backward-Euler run
on the identical interpolated control trajectory.

| maneuver | acc./rej. | order 1/2 | max power error | final flux error | CPU (s) | BE CPU (s) | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| slow symmetric, 0.8 s | 29/9 | 23/6 | 0.0210% | 0.00352% | 14.74 | 22.29 | 1.51x |
| fast symmetric, 0.08 s | 24/11 | 17/7 | 0.0544% | 0.0119% | 13.39 | 19.91 | 1.49x |
| fast asymmetric, 0.08 s | 17/4 | 13/4 | 0.0322% | 0.0000285% | 11.70 | 23.38 | 2.00x |

Every power history remains finite and positive. No candidate selects an order
above two: frequent trajectory knots and localized shape changes correctly
make the automatic controller conservative. This is useful behavior, not a
failure to reach the configured order-five ceiling.

The error-scaled/cap-0.5 rejection policy was also compared directly with
factor-two rejection on the slow maneuver. It changes 28/13 accepted/rejected
steps to 29/9, reduces inner work from 77,066 to 71,143 (7.7%), and lowers the
single-run time from 15.47 to 13.81 s while slightly reducing both power and
flux errors. This transfers the Step-07h optimization to a spatially changing
11-group problem; the wall result is a single A/B and the deterministic work
reduction is the stronger evidence.

## Decision and next gate

The two-dimensional moving-control stability gate passes. Adaptive BDF remains
opt-in, but there is no evidence of positivity loss, misleading high-order
selection, or asymmetric-shape instability. The next progression is coupled
thermal feedback on these maneuvers, followed by the 3-D moving-drum and GPU
throughput gates. A refinement/polar-sampling convergence check should precede
any claim about the physical drum-worth history; this checkpoint assesses time
integration stability and performance, not HP-MR design prediction.

Reproduce the table with:

```bash
PYTHONPATH=. python examples/hpmr_bdf_cpu_benchmark.py \
  --refine 2 --reference-steps 64 --reference-scheme bdf1 \
  --candidate-steps 8 --schemes bdf5 --adaptive-rtols 1e-3 \
  --rejection-strategy error \
  --json-output benchmark-results/transient-acceleration/hpmr-adaptive-bdf-refine2.json
```
