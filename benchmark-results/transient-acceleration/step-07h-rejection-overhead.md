# Step 07h — adaptive-BDF rejection overhead

Date: 2026-08-18

## Outcome

Rejected attempts now expose their endpoint, attempted width, selected order,
and solver-work counters. The controller also has an opt-in error-scaled retry
policy. It applies the usual order-dependent defect formula but never retries
with more than half the failed width; defects large enough to warrant a more
severe cut can shrink as far as the existing 0.2 lower factor. The original
factor-two rule remains the default and is forced by `--cherezov-controls`.

An initially tested 0.8 maximum retry factor was rejected. At `RTOL=1e-5` it
caused a BDF1 reject/accept oscillation: 140 rejected and 623 accepted steps,
versus 27/410 for the factor-two baseline. Capping the retry at 0.5 removes
that failure mode while still responding more strongly to very large defects.

## Corrected LRA A/B benchmark

The benchmark is the 3.75 cm, cell-local-feedback LRA calculation with fully
implicit coupling, automatic BDF1--5, `h0=1e-6 s`, and `hmax=0.1 s`.

| `RTOL=1e-3` policy | accepted/rejected | FGMRES | rejected FGMRES | inner | feedback | median CPU (s) |
|---|---:|---:|---:|---:|---:|---:|
| factor two | 201/30 | 13,129 | 2,523 | 315,381 | 387 | 26.876 |
| error-scaled, cap 0.5 | 201/25 | 12,395 | 2,144 | 297,831 | 356 | 25.494 |

The optimized policy removes all repeated rejection streaks in this run (the
baseline has two such accepted-step starts and a maximum three-attempt
streak). It reduces total FGMRES and inner work by 5.59% and 5.56%, rejected
FGMRES work by 15.0%, and the three-sample median CPU time by 5.14% (1.054x).
Raw CPU samples are stored in `data/rejection-policy-summary.json` in the
rendered-report directory because the individual samples overlap.

The policy changes the admissible loose-tolerance time grid, as it should. The
first peak changes from 5248.71 to 5313.11 W/cm3, moving closer to the ANL
5411 W/cm3 value (error -3.00% to -1.81%). First-peak time changes from
1.46459 to 1.46625 s. Second-peak power, 3 s power, mean temperature, and peak
temperature change by at most 0.043%.

At the tighter `RTOL=1e-5` accuracy gate, the policies are effectively
neutral: 410/27 versus 413/26 accepted/rejected steps, 14,159 versus 14,160
FGMRES applications, and 335,854 versus 335,917 inner iterations. First-peak
time and power are identical at stored precision. Single CPU timings are
28.62 and 29.18 s and are treated as noise, not as a speedup or slowdown.

## Decision

Retain the error-scaled/cap-0.5 policy as an opt-in acceleration for practical
tolerances. Keep factor-two rejection as the library default and for the
paper-faithful Cherezov comparison. The tight gate shows no material penalty,
and the practical gate shows deterministic work reduction, but this policy is
not yet promoted until it passes the staged HP-MR transients. The next Phase-7
work is therefore the uniform-insertion and slow/fast/asymmetric HP-MR
stability progression.
