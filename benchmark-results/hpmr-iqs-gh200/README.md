# Guarded adaptive IQS on GH200

Decision date: 2026-08-20

Decision: **guarded adaptive IQS is the preferred method for production HP-MR
coupled transients.** Fixed-cadence IQS remains a benchmark/reference mode, and
full diffusion remains the accuracy reference.

The machine-readable record is
[`adaptive-fallback-r4-nz20.json`](adaptive-fallback-r4-nz20.json).

## Workload

The calibration used the serious 3-D HP-MR drum transient:

- 105,600 active cells at radial refinement 4 and 20 axial layers;
- the given 11-group ENDF/B-VIII cross sections and kinetics;
- all 12 drums moving from 90 to 90.50 degrees;
- measured insertion of +0.248 dollar;
- 4 s ramp starting at 1 s, followed through 200 s;
- `dt=0.05 s`, `dt_thermal=0.5 s`, and maximum `shape_dt=2 s`;
- one NVIDIA GH200 120GB GPU and one CPU task.

Guard settings:

```text
residual_tol       = 0.002
fallback_residual  = 0.01
iqs_predictor_tol  = 0.02
adjoint_every      = 5
```

The submitted command was equivalent to:

```bash
python -u examples/hpmr_coupled_transient.py \
  --refine 4 --nz 20 --groups 11 \
  --t-end 200 --dt 0.05 --dt-thermal 0.5 \
  --drum-from 90 --dollars 0.3076923076923077 \
  --t-start 1 --t-ramp 4 --n-angles 13 \
  --device gpu --profile \
  --quasistatic-shape-dt 2 --shape-method iqs --adjoint-every 5 \
  --residual-tol 0.002 --fallback-residual 0.01 \
  --iqs-predictor-tol 0.02
```

## Result

| quantity | guarded IQS | fixed-cadence IQS |
|---|---:|---:|
| Slurm job | 196799 | 196460 |
| transient | 3,343.2 s | 2,663.3 s |
| shape updates | 131 | 100 |
| IQS shape solves | 129 | 100 |
| adjoint solves | 29 | 21 |
| full-diffusion fallbacks | 2 | 0 |
| peak P/P0 | 5.2228 at 45 s | 5.2235 at 46 s |
| final P/P0 | 1.3580 | 1.3635 |

The guard added 29 IQS shape solves and two hard full-diffusion fallback steps.
Fallback work itself was only 16.4 s; most of the 25.5% transient overhead came
from the extra shape and adjoint solves. The independent predictor disagreement
never exceeded 0.59%, so the 2% predictor guard did not reduce the scheduled
interval. The adaptive behavior in this run was driven by the projected shape
residual.

## Accuracy decision

The full-diffusion reference job 196622 was preserved through physical time
110 s before cancellation. At 10 s sampling intervals through that point:

- guarded IQS maximum relative power error was **0.173%**;
- fixed-cadence IQS maximum relative power error was **5.708%**;
- guarded IQS stayed within 0.11% through the drum ramp and within 0.18%
  afterward;
- fixed-cadence IQS started 5.7% low and settled near 1% high late in the
  available reference.

Representative values:

| time | full diffusion | guarded IQS | guarded error | fixed IQS | fixed error |
|---:|---:|---:|---:|---:|---:|
| 10 s | 2.16912 | 2.17041 | +0.059% | 2.04530 | -5.708% |
| 20 s | 3.44761 | 3.45098 | +0.098% | 3.32260 | -3.626% |
| 60 s | 4.81154 | 4.80323 | -0.173% | 4.84990 | +0.797% |
| 110 s | 2.64905 | 2.64666 | -0.090% | 2.67650 | +1.036% |

This accuracy improvement is large enough to justify the 25.5% transient cost
increase. For this calibrated HP-MR problem, unguarded IQS should no longer be
used as the production answer merely because it is faster.

## Going-forward policy

- Use guarded adaptive IQS for production HP-MR coupled transients.
- Use the calibrated `0.002/0.01/0.02` guard profile as the starting point for
  comparable r4, 11-group drum maneuvers.
- Keep full diffusion as a shorter-window reference whenever the mesh,
  cross-section set, feedback model, or maneuver severity changes materially.
- Report guard settings, shape updates, fallbacks, adjoints, wall time, and the
  maximum error against the available reference.
- Use `--unguarded-iqs` only for fixed-cadence method studies and regression
  comparisons.
- Do not make these HP-MR-calibrated thresholds global low-level API defaults
  for unrelated reactor models.

## Limitations

The full reference is available only through 110 s. The completed application
log reports two hard fallbacks but did not serialize their exact times. Future
production result artifacts should include `fallback_times`, `shape_times`, and
`shape_reasons` so those events can be audited directly.
