# 2D HP-MR quasi-static CPU benchmark

Run on 11 August 2026 after the adjoint-weighted amplitude, conservative
precursor-shape, predictor-cadence, and adjoint-residual corrections, using an
Intel Core i5-1145G7 (4 cores / 8 threads), Python 3.13.5, NumPy 2.1.3, and the
CPU backend. The complete benchmark took 92.6 seconds.

## Executive result

IQS now separates the adjoint-weighted neutron-population amplitude from
physical fission power, retains the corrector's spatial precursor distribution
after matching every family's accepted adjoint projection, and never adopts
the corrector's independent coarse-amplitude inventory. Plain IQS consequently
agrees closely with adiabatic QS in global power while resolving the final
spatial shape 21–133x more accurately.

Adiabatic QS remains the fastest approximation at 2.1–3.2x dynamic speedup.
Plain IQS provides 1.5–2.6x and nearly the same maximum power-history error.
Both have peak-power bias below 0.4%; their larger in-ramp errors occur between
scheduled shape updates and are corrected at the next macro boundary.

Guarded IQS combines a 0.012 soft residual threshold, a 0.020 hard fallback,
and a 2% amplitude-predictor tolerance that halves subsequent shape intervals.
Adjoints have a six-correction maximum age and refresh early at a 0.005
transposed-eigenproblem residual; hard fallback still forces refresh. The guard
lowers maximum power error to 3.0%, 2.6%, and 2.0% for the slow, fast, and
asymmetric cases. It is about 2.1x faster for the symmetric ramps; the
asymmetric run is within 4% of full diffusion despite invoking one hard
fallback. Fallback remains a safety mechanism rather than a guaranteed
acceleration.

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
| Slow | Full diffusion | 9.13 | 1.00x | 0 | 0 |
|  | Adiabatic QS | 3.70 | 2.47x | 12 | 0 |
|  | Time-dependent IQS | 4.02 | 2.27x | 12 | 0 |
|  | Guarded IQS | 4.23 | 2.16x | 16 | 0 |
| Fast | Full diffusion | 8.52 | 1.00x | 0 | 0 |
|  | Adiabatic QS | 2.69 | 3.17x | 12 | 0 |
|  | Time-dependent IQS | 3.26 | 2.61x | 12 | 0 |
|  | Guarded IQS | 4.15 | 2.05x | 18 | 0 |
| Asymmetric | Full diffusion | 13.25 | 1.00x | 0 | 0 |
|  | Adiabatic QS | 6.37 | 2.08x | 12 | 0 |
|  | Time-dependent IQS | 9.11 | 1.45x | 12 | 0 |
|  | Guarded IQS | 13.77 | 0.96x | 18 | 1 |

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
|  | Guarded IQS | 2.62% | -0.02% | 0.001 | 6.25e-6 |
| Asymmetric | Adiabatic QS | 5.10% | -0.32% | 0.011 | 4.93e-3 |
|  | Time-dependent IQS | 5.18% | -0.30% | 0.011 | 3.70e-5 |
|  | Guarded IQS | 1.97% | -0.03% | 0.001 | 6.75e-6 |

Adiabatic QS and plain IQS peak agreement is much better than their maximum
history error because they catch up at scheduled shape updates. IQS has the
more faithful spatial state at those boundaries. The corrector's independent
amplitude-predictor disagreement reaches 3.7%, 12.9%, and 5.7%, closely marking
the cases with large between-update error. In guarded IQS the earlier updates
reduce that disagreement to 4.0%, 2.7%, and 2.5%, respectively.

The adjoint residual was evaluated 16–18 times for only milliseconds of total
cost. It triggered three early refreshes in each symmetric case and four in the
asymmetric case, while the six-update maximum age and hard fallback limited
reuse. The guarded runs used 6, 7, and 9 total adjoint solves respectively,
instead of the previous every-second-update policy's 9, 10, and 11.

Temperature differences are small because these 3–12 s runs are much shorter
than the approximately 268 s fuel thermal time constant. They validate the
coupling path but do not replace a minute-scale feedback benchmark.

## Recommended next implementation work

1. Calibrate residual and predictor thresholds on the one-minute 3-D 11-group
   case rather than treating these 2-D values as universal defaults.
2. Develop a cheaper adjoint solve or multi-mode importance update. Residual-
   controlled reuse has brought the asymmetric guard near full-diffusion parity,
   but adjoint and IQS solves still dominate its profile.
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
