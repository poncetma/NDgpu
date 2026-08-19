# Step 07e — full-state LRA defect and paper-comparison contract

Date: 2026-08-18

> Superseded by [Step 07f](step-07f-lra-control-discrepancy.md). The adaptive
> machinery and comparison contract remain valid, but the numerical LRA rows
> below used a factor-of-ten reflector thermal-absorption typo and are not
> benchmark results.

This checkpoint closes the first default-readiness item from Step 07d. The
adaptive BDF acceptance norm now concatenates the multigroup flux, both LRA
precursor-family fields, and the fully implicit endpoint temperature. All use
the same nonuniform polynomial time nodes. Rejected attempts still leave flux,
precursor, thermal BDF, and external accepted-step state untouched.

## Reproducible paper controls

The benchmark driver now provides:

```bash
PYTHONPATH=. python examples/lra2d_bdf_benchmark.py \
    --refine 1 --cherezov-controls
```

The preset selects the paper's reported initial width `1e-6 s`, `RTOL=1e-5`,
maximum BDF order five, fully implicit temperature feedback, assembly thermal
zones, and a `0.1 s` maximum width. Its JSON records two remaining differences:
ndgpu currently holds the maximum attainable BDF order instead of comparing
`q-1/q/q+1`, and its spatial discretization is cell-centred finite volume.

## CPU result

The 15 cm assembly mesh and 3.75 cm refined FV mesh produced:

| quantity | ndgpu FV 15 cm | ndgpu FV 3.75 cm | Cherezov FEM1 | Cherezov FEM4 |
|---|---:|---:|---:|---:|
| accepted/rejected steps | 521 / 24 | 443 / 24 | 403 / 5 | 399 / 7 |
| first peak time (s) | 1.34462 | 1.64860 | 1.352 | 1.440 |
| first peak power (W/cm3) | 5908.9 | 5317.4 | 5820 | 5644 |
| second peak power (W/cm3) | 963.3 | 415.1 | 854 | 824 |
| power at 3 s (W/cm3) | 108.46 | 83.83 | 109.9 | 99.0 |

The 15 cm transient solve took 8.58 s on the development CPU, with 13,750
FGMRES applications including rejected attempts and 567 constituent feedback
solves. The 3.75 cm run took 26.95 s, 14,288 FGMRES applications, and 488
constituent solves. Wall time is host-specific; step and work counts are the
comparison metrics.

Relative to the paper's first-order spatial row, first-peak time is -0.55%,
first-peak power +1.53%, and 3 s power -1.31%, while the second peak is +12.8%.
That pattern supports the earlier finding that the coarse ndgpu history is a
low-order spatial result. It does not turn the FV mesh into a first-order FEM
equivalent: its raw rods-in eigenvalue is 1.006500, and the refined FV
rods-out eigenvalue remains 377 pcm below the published reference.

The refined run sharpens that conclusion. Its rods-in eigenvalue is within
13 pcm of FEMCORE, but the first peak is 14.6% late, the second peak 49.4%
low, and 3 s power 15.1% low. Time adaptivity therefore does not cure the
control-worth/interface error seen under fixed 10 ms stepping.

## Decision

The full-state error norm and benchmark labeling are accepted. Adaptive BDF
remains experimental. A defensible Cherezov performance comparison still
requires:

1. automatic `q-1/q/q+1` order/width selection;
2. tolerance convergence using that controller;
3. a spatial method or explicit correction matching both raw endpoint
   eigenvalues before comparison to the fourth-order FEM headline;
4. matched-history-error backward-Euler work, not equal-width speedup.
