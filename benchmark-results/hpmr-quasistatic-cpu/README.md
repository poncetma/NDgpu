# 2D HP-MR quasi-static CPU benchmark

Run on 11 August 2026 after the adjoint-weighted amplitude, conservative
precursor-shape, and predictor-cadence corrections, using an
Intel Core i5-1145G7 (4 cores / 8 threads), Python 3.13.5, NumPy 2.1.3, and the
CPU backend. The complete benchmark took 88.0 seconds.

## Executive result

IQS now separates the adjoint-weighted neutron-population amplitude from
physical fission power, retains the corrector's spatial precursor distribution
after matching every family's accepted adjoint projection, and never adopts
the corrector's independent coarse-amplitude inventory. Plain IQS consequently
agrees closely with adiabatic QS in global power while resolving the final
spatial shape 21–133x more accurately.

Adiabatic QS remains the fastest approximation at 2.1–2.4x dynamic speedup.
Plain IQS provides 1.3–2.1x and nearly the same maximum power-history error.
Both have peak-power bias below 0.4%; their larger in-ramp errors occur between
scheduled shape updates and are corrected at the next macro boundary.

Guarded IQS combines a 0.012 soft residual threshold, a 0.020 hard fallback,
and a 2% amplitude-predictor tolerance that halves subsequent shape intervals.
It lowers maximum power error to 3.0%, 2.5%, and 1.3% for the slow, fast, and
asymmetric cases. It remains about 1.3x faster for the symmetric ramps; the
asymmetric run is 20% slower than full diffusion because it invokes one hard
fallback. Fallback is a safety mechanism rather than a guaranteed acceleration.

![Power and temperature histories](histories.png)

![Performance and accuracy trade-off](tradeoffs.png)

## Cases

All cases use the real 55-site 2D HP-MR geometry, polar area-fraction control
drums, 2,970 active triangular cells, delayed-neutron kinetics, conduction, and
Doppler feedback. Each has 120 fine neutron steps and 12 scheduled shape macro
intervals.

| Case | Drum motion | Static worth | Timing | Fine / thermal / shape dt |
|---|---|---:|---|---|
| Slow symmetric | all 12, 90° → 95° | +204.7 pcm (+0.315 $) | ramp 1–9 s; end 12 s | 0.10 / 0.50 / 1.00 s |
| Fast symmetric | all 12, 90° → 95° | +204.7 pcm (+0.315 $) | ramp 0.25–0.75 s; end 3 s | 0.025 / 0.25 / 0.25 s |
| Asymmetric | four adjacent +x drums, 90° → 110° | +145.7 pcm (+0.224 $) | ramp 0.5–2.5 s; end 6 s | 0.05 / 0.50 / 0.50 s |

The slow ramp uses 17 cached geometry frames; the other two use 11. The same
frame objects are reused between control changes, so operator rebuild identity
semantics match the intended coupled application.

## Performance

“Dynamic” time excludes the coupled steady-state solve but includes IQS
initialization and adjoint work. It is the fairest end-to-end comparison after
the common equilibrium calculation.

| Case | Method | Dynamic time [s] | Speedup | Shape updates | Fallbacks |
|---|---|---:|---:|---:|---:|
| Slow | Full diffusion | 6.11 | 1.00x | 0 | 0 |
|  | Adiabatic QS | 2.87 | 2.13x | 12 | 0 |
|  | Time-dependent IQS | 3.75 | 1.63x | 12 | 0 |
|  | Guarded IQS | 4.58 | 1.33x | 16 | 0 |
| Fast | Full diffusion | 6.48 | 1.00x | 0 | 0 |
|  | Adiabatic QS | 2.65 | 2.44x | 12 | 0 |
|  | Time-dependent IQS | 3.05 | 2.13x | 12 | 0 |
|  | Guarded IQS | 4.96 | 1.31x | 18 | 0 |
| Asymmetric | Full diffusion | 12.72 | 1.00x | 0 | 0 |
|  | Adiabatic QS | 6.08 | 2.09x | 12 | 0 |
|  | Time-dependent IQS | 9.60 | 1.33x | 12 | 0 |
|  | Guarded IQS | 15.24 | 0.83x | 18 | 1 |

The asymmetric full solve costs about twice the symmetric solve despite the
same mesh and step count, because the tilted spatial source needs substantially
more fixed-point/Krylov work. This is exactly the kind of workload where
shape acceleration should eventually pay off.

## Accuracy against full transient diffusion

The power metric is the maximum pointwise relative error over the entire
history. Peak bias compares the maximum power reached. Shape error is the L2
difference between final multigroup fluxes after unit-L2 normalization over
active cells.

| Case | Method | Max power error | Peak bias | Max mean-T error [K] | Final flux-shape L2 error |
|---|---|---:|---:|---:|---:|
| Slow | Adiabatic QS | 3.45% | -0.38% | 0.018 | 3.42e-4 |
|  | Time-dependent IQS | 3.46% | -0.39% | 0.018 | 1.63e-5 |
|  | Guarded IQS | 2.98% | -0.09% | 0.004 | 2.03e-5 |
| Fast | Adiabatic QS | 11.51% | -0.17% | 0.006 | 5.14e-4 |
|  | Time-dependent IQS | 11.53% | -0.18% | 0.006 | 6.85e-6 |
|  | Guarded IQS | 2.46% | -0.03% | 0.001 | 4.48e-6 |
| Asymmetric | Adiabatic QS | 5.10% | -0.32% | 0.011 | 4.93e-3 |
|  | Time-dependent IQS | 5.18% | -0.30% | 0.011 | 3.70e-5 |
|  | Guarded IQS | 1.26% | -0.03% | 0.001 | 6.75e-6 |

Adiabatic QS and plain IQS peak agreement is much better than their maximum
history error because they catch up at scheduled shape updates. IQS has the
more faithful spatial state at those boundaries. The corrector's independent
amplitude-predictor disagreement reaches 3.7%, 12.9%, and 5.7%, closely marking
the cases with large between-update error. In guarded IQS the earlier updates
reduce that disagreement to 4.0%, 2.5%, and 1.7%, respectively.

Temperature differences are small because these 3–12 s runs are much shorter
than the approximately 268 s fuel thermal time constant. They validate the
coupling path but do not replace a minute-scale feedback benchmark.

## Recommended next implementation work

1. Calibrate residual and predictor thresholds on the one-minute 3-D 11-group
   case rather than treating these 2-D values as universal defaults.
2. Reduce the cost of asymmetric fallback, which currently erases the CPU
   acceleration, by replaying only the rejected region/interval where possible.
3. Fuse the extra adjoint-weighted reductions on GPU and keep their scalar
   telemetry asynchronous where possible.
4. Repeat macro-step convergence sweeps on GPU. `iqs_substeps` now traverses
   actual cached control frames, but 2–4 CPU substeps cost much more than the
   small accuracy change justified.
5. Repeat first with 11 groups on GPU, then with the one-minute 3-D case.

## Reproduction and raw data

From the repository root:

```bash
PYTHONPATH=. python examples/hpmr_quasistatic_benchmark.py \
  --refine 3 --output-dir benchmark-results/hpmr-quasistatic-cpu
```

- [`summary.csv`](summary.csv): one row per case and method, including timings,
  errors, solver work, residuals, and fallback counts.
- [`histories.csv`](histories.csv): power and temperature histories.
- [`metadata.json`](metadata.json): machine-independent case definitions and
  measured cold-static worths.
- [`histories.png`](histories.png) and [`tradeoffs.png`](tradeoffs.png): figures
  used above.

The benchmark uses the repository's illustrative two-group HP-MR material set.
It is suitable for algorithm comparison but not for predictive reactor-safety
claims. The 11-group ENDF/B-VIII-derived set is available through `--groups 11`
but was deliberately not used for this CPU matrix.
