# Step 07g — automatic BDF order and matched-error backward Euler

Date: 2026-08-18

## Outcome

Adaptive BDF now compares normalized full-state predictor defects for
`q-1`, `q`, and `q+1` after every accepted step. Each candidate is converted
to a bounded next-width proposal; the largest safe proposal wins, subject to
a 5% hysteresis against order chatter. Candidate evaluation reuses the
accepted endpoint and requires no extra diffusion solve. Rejection still
rolls back neutron, precursor, and external thermal histories, while the LRA
driver keeps its thermal BDF recurrence on the selected neutron order.

Adaptive backward Euler now uses the same acceptance/width controller, making
an accuracy-matched baseline possible.

## Corrected LRA result at Cherezov controls

The 3.75 cm finite-volume case, `h0=1e-6 s`, `RTOL=1e-5`, maximum order five,
fully implicit feedback, produced:

| controller | accepted/rejected | order counts 1/2/3/4/5 | FGMRES | inner | time (s) |
|---|---:|---:|---:|---:|---:|
| fixed maximum order | 461 / 21 | 2/2/2/2/453 | 14,889 | 352,629 | 31.07 |
| automatic order | 414 / 29 | 15/8/12/30/349 | 14,385 | 341,182 | 29.63 |

The automatic result is `1.469974 s / 5599.09 W cm-3` at the first peak,
`807.191 W cm-3` at the second, and `98.8317 W cm-3` at 3 s. These are
numerically indistinguishable at reporting precision from the fixed-maximum
result. Automatic selection reduces accepted steps 10.2%, FGMRES work 3.4%,
inner work 3.2%, and measured transient time 4.7%. The accepted count is 3.8%
above Cherezov's 399-step fourth-order FEM result, though ndgpu still has more
rejections (29 versus 7).

## Tolerance convergence

Automatic BDF5 on the same spatial model gives:

| RTOL | accepted/rejected | FGMRES | inner | time (s) | t1 (s) | P1 | P2 | P(3 s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-3 | 199 / 22 | 12,575 | 301,861 | 25.26 | 1.465891 | 5409.93 | 806.418 | 98.82496 |
| 1e-4 | 284 / 22 | 12,306 | 294,134 | 24.12 | 1.468178 | 5563.33 | 807.078 | 98.82945 |
| 1e-5 | 414 / 29 | 14,385 | 341,182 | 29.63 | 1.469974 | 5599.09 | 807.191 | 98.83172 |
| 1e-6 | 605 / 28 | 17,920 | 421,475 | 37.21 | 1.469777 | 5601.56 | 807.202 | 98.83190 |

The first prompt peak controls convergence. Relative to `1e-6`, the `1e-5`
peak differs by -0.044%, `1e-4` by -0.68%, and `1e-3` by -3.42%; the second
peak, tail power, and temperatures converge sooner. The non-monotone wall time
between `1e-3` and `1e-4` reflects fewer rejected/feedback solves at `1e-4`;
iteration counts, not a single host timing, are the portable evidence.

## Accuracy-matched backward-Euler comparison

The matched envelope uses the `1e-6` BDF result above as the temporal
reference. BDF5 at `RTOL=1e-3` has -3.42% first-peak error; backward Euler at
`RTOL=1e-4` has -2.92%. Their first-peak time errors are both about -0.25%,
and tail/temperature differences are smaller than the peak error.

| method | RTOL | accepted | FGMRES | inner | time (s) | P1 error |
|---|---:|---:|---:|---:|---:|---:|
| automatic BDF5 | 1e-3 | 199 | 12,575 | 301,861 | 25.26 | -3.42% |
| backward Euler | 1e-4 | 3,187 | 66,849 | 1,515,351 | 130.83 | -2.92% |

At comparable sharp-peak accuracy, automatic BDF is 5.18x faster, uses 5.32x
fewer FGMRES applications, 5.02x fewer inner iterations, and 16.0x fewer
accepted steps.

## Decision

The CPU automatic-order and matched-error gates are accepted. Adaptive BDF
remains opt-in until equivalent GPU behavior is measured. The next work is to
reduce rejection overhead and carry the verified controller into the staged
HP-MR transient progression.

The structured ladder is reproducible with:

```bash
PYTHONPATH=. python examples/lra2d_bdf_tolerance_benchmark.py --refine 4
```
