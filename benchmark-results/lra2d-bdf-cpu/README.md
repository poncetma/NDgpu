# LRA-2D coupled BDF benchmark (CPU)

Date: 2026-08-18. The rods/control discrepancy is resolved. This remains a
finite-volume comparison to Cherezov et al., not a claim to reproduce their
fourth-order B-spline finite-element space.

For the complete accuracy, convergence, controller-history, and matched-cost
analysis, see the rendered
[adaptive-BDF LRA report](../lra2d-adaptive-bdf-report/adaptive-bdf-lra-benchmark.pdf).

## Reproduce

```bash
PYTHONPATH=. python examples/lra2d_bdf_benchmark.py \
    --refine 4 --cherezov-controls
```

The paper preset uses `h0=1e-6 s`, `RTOL=1e-5`, maximum BDF order five,
fully implicit adiabatic/Doppler feedback, assembly thermal zones, and the
specified axial buckling. It also performs a separate rods-out eigenvalue
solve and reports the raw value alongside rods-in. This prevents the raw
`k_out` reference from being confused with `k_out/k_in`.

## Specification audit and resolved discrepancy

The original ANL-7416/DIF3D input uses reflector absorptions
`Sigma_a1=6.034e-4 cm-1` and `Sigma_a2=1.911e-2 cm-1`. Cherezov Table 4 prints
both one decade too large. The earlier ndgpu checkpoint corrected only the
fast value, retaining `Sigma_a2=1.911e-1`; it then disabled the specified
`Bz^2=1e-4 cm-2` axial leakage because those two errors happened to offset in
the rods-in eigenvalue. The cancellation failed for rods-out and produced the
apparent 377 pcm control-worth deficit.

ndgpu now restores both original reflector absorptions and includes `D_g Bz^2`
by default. The Doppler edit also scales only physical fast absorption, leaving
the axial-leakage contribution unchanged. `--no-axial-buckling` remains an
explicit diagnostic and no longer defines the benchmark.

Two unrelated specification details remain explicit:

- the 78 non-reflector assemblies give a geometric core area of 17,550 cm2,
  whereas Cherezov Eq. (51) prints 17,750 cm2;
- the reference `1.01546` is a raw rods-out eigenvalue, not a critical-adjusted
  endpoint ratio.

## Static convergence

| FV cell width | refine | rods in | rods out |
|---:|---:|---:|---:|
| 15 cm | 1 | 1.00389122 | 1.02826090 |
| 7.5 cm | 2 | 0.99749892 | 1.01677507 |
| 3.75 cm | 4 | 0.99623051 | 1.01481578 |
| 1.875 cm | 8 | 0.99624987 | 1.01508744 |
| 1 cm | 15 | 0.99632537 | 1.01532020 |
| original reference | -- | 0.99633 | 1.01546 |
| Cherezov FEMCORE | -- | 0.99639 | 1.01550 |

The 1 cm rods-in result reproduces the archived DIF3D value `0.996325`. At the
3.75 cm transient mesh, errors against the original reference are -10 pcm
rods-in and -64 pcm rods-out; the residual is ordinary FV mesh convergence,
not an anomalous control worth.

## Coupled transient result

| quantity | ndgpu FV 3.75 cm | original reference | Cherezov FEMCORE |
|---|---:|---:|---:|
| rods in | 0.99623051 | 0.99633 | 0.99639 |
| rods out | 1.01481578 | 1.01546 | 1.01550 |
| accepted / rejected steps | 461 / 21 | -- | 399 / 7 (FEM4) |
| first peak time (s) | 1.47008 | 1.436 | 1.439 |
| first peak power (W/cm3) | 5599.35 | 5411 | 5641 |
| second peak power (W/cm3) | 807.20 | 784 | 820 |
| power at 3 s (W/cm3) | 98.8317 | 96.2 | 98.7 |
| mean temperature at 3 s (K) | 1084.68 | 1087 | 1122 |
| peak temperature at 3 s (K) | 2953.17 | 2948 | 3068 |

Relative to the original reference, the transient differences are +2.37% in
first-peak time, +3.48% in first-peak power, +2.96% in second-peak power,
+2.74% in 3 s power, -0.21% in mean temperature, and +0.18% in peak
temperature. Relative to FEMCORE, the power metrics are within 1.6%; its
temperature result is 3.3--3.7% higher than ndgpu and the original reference.

This run used 461 accepted and 21 rejected steps, 14,889 total FGMRES
applications, 352,629 inner iterations, and 496 feedback constituent solves.
Transient march time was 31.07 s on the development CPU; wall time is
host-specific, while step and iteration counts are the portable cost record.

## Interpretation and next gate

Correcting the benchmark data removes the proposed need for an SPH or
control-worth calibration. The default perturbation scale remains exactly
one. The earlier endpoint-worth diagnostic and all transient rows generated
with the mistyped reflector are superseded by this report.

The remaining comparison gaps are well-behaved spatial and temporal-method
differences. Automatic `q-1/q/q+1` selection and the matched-error
backward-Euler gate are now complete: at a roughly 3% first-peak error envelope,
BDF is 5.18x faster and uses 5.02x fewer inner iterations. See
[Step 07g](../transient-acceleration/step-07g-automatic-order-matched-be.md).
The next LRA task is structured tolerance-ladder output and rejection-overhead
reduction; a refined-space run should then quantify the remaining 64 pcm
rods-out and 2.37% peak-time offsets.

## Sources

- [Original ANL-7416 Benchmark 14](https://corephysics.com/benchmarks/anl7416_benchmark14.pdf)
- [Archived fine-mesh DIF3D input and results](https://corephysics.com/benchmarks/dif3d_lra.txt)
- [Cherezov, Vasiliev, and Ferroukhi (2025)](https://doi.org/10.1016/j.anucene.2024.110837)
