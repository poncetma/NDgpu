# Adaptive BDF on LRA-2D: rendered benchmark report

The primary artifact is
[adaptive-bdf-lra-benchmark.pdf](adaptive-bdf-lra-benchmark.pdf). It evaluates
accuracy against the original ANL-7416/2 reference data, uses FEMCORE as a
secondary comparison, and reports temporal convergence plus an
accuracy-matched backward-Euler speedup. The updated accuracy study separates
Cherezov's element/assembly-average temperature approximation from cell-local
feedback and follows the latter from 15 cm to 1 cm FV resolution. It also
compares measured CPU time with Cherezov et al.'s published FEM-order ladder,
without forming a cross-host speedup. It also records the conservative
error-scaled rejection experiment and its three-sample CPU comparison. The
current edition also bounds transferability of the fivefold LRA result with a
matched adaptive-BE/BDF comparison on coupled 11-group HP-MR motion and a
repeated Tesla T4 preconditioner/performance gate.

Rebuild the figures and PDF from the stored machine-readable histories with:

```bash
cd benchmark-results/lra2d-adaptive-bdf-report
make
```

Inputs and generated assets:

- `benchmark-summary.json`: aggregate ANL, FEMCORE, CPU, and matched-work data;
- `data/bdf5-cell-refine*-rtol-1e-5.json`: the cell-local spatial ladder;
- `data/*cell*rtol-1e-{3,4,6}.json`: matched-error and tight cell-local runs;
- `data/*.json`: complete histories and controller telemetry where requested;
- `data/rejection-policy-summary.json`: rejection-policy A/B work and timing
  samples;
- `make_figures.py`: deterministic figure generator;
- `report.tex`: LaTeX source;
- `figures/`: PDF and PNG renderings used by the report.

Validation at this checkpoint: `575 passed, 5 skipped, 13 deselected`; the
latest affected coupling/HP-MR/LRA/BDF regression is `119 passed`.
