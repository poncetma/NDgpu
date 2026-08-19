# Step 07f — LRA reflector erratum and control-worth resolution

Date: 2026-08-18

## Outcome

The 377 pcm rods-out discrepancy was not caused by cell-centred interface
leakage or by the control interpolation. It was a benchmark-data error.

Cherezov Table 4 prints both reflector absorptions a factor of ten above the
original ANL-7416 values. The earlier implementation corrected
`Sigma_a1=6.034e-4 cm-1` but retained the mistyped
`Sigma_a2=1.911e-1 cm-1`. It also omitted the specified
`Bz^2=1e-4 cm-2`; that omission accidentally compensated the reflector error
for rods-in but not rods-out.

The model now uses the original reflector values
`(6.034e-4, 1.911e-2) cm-1`, includes `D_g Bz^2` by default, and excludes the
axial-leakage loss from the Doppler absorption multiplier. Tests pin the two
reflector values, both raw endpoints, the no-buckling diagnostic, and the
leakage/Doppler separation.

## Evidence

The archived 1 cm DIF3D input/result reports `k_in=0.996325`. ndgpu gives
`0.99632537` at the same cell width. Its endpoint convergence is:

| FV cell width | rods in | rods out |
|---:|---:|---:|
| 15 cm | 1.00389122 | 1.02826090 |
| 7.5 cm | 0.99749892 | 1.01677507 |
| 3.75 cm | 0.99623051 | 1.01481578 |
| 1.875 cm | 0.99624987 | 1.01508744 |
| 1 cm | 0.99632537 | 1.01532020 |
| reference | 0.99633 | 1.01546 |

The corrected 3.75 cm paper-control transient uses 461 accepted / 21 rejected
steps and produces first peak `1.47008 s / 5599.35 W cm-3`, second peak
`807.20 W cm-3`, and `P(3 s)=98.8317 W cm-3`. These differ from the original
reference by +2.37%, +3.48%, +2.96%, and +2.74%, respectively. The former
result (`1.6486 s`, `5317 W cm-3`, `415 W cm-3`, `83.83 W cm-3`) is invalid
because it used the mistyped reflector.

## Decision

The rods/control discrepancy is closed. No control-worth scale or interface
correction is justified. The Cherezov preset now includes a separate raw
rods-out solve in its JSON contract. Automatic BDF order selection and an
accuracy-matched backward-Euler comparison remain open acceleration work.

Sources: [original ANL benchmark](https://corephysics.com/benchmarks/anl7416_benchmark14.pdf),
[archived DIF3D input/result](https://corephysics.com/benchmarks/dif3d_lra.txt),
and [Cherezov et al.](https://doi.org/10.1016/j.anucene.2024.110837).
