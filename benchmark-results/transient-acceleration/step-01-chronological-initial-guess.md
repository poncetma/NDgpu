# Step 01: chronological raw-flux initial guess

Date: 2026-08-11  
Device: CPU / NumPy, FP64  
Decision: **rejected; no production code retained**

## Question

Can the last two accepted flux fields reduce transient work through

```text
phi_predict = max(phi_n + w (phi_n - phi_n-1), 0)?
```

The trial initialized the fission source consistently from the predicted flux,
but retained the exact accepted `phi_n` in the backward-Euler time source. It
did not alter the equations, tolerances, or convergence tests. Weights 0.25,
0.5, and 1.0 were examined after the undamped trial proved unfavorable.

The experiment was motivated by Austin, Chalmers, and Warburton's GPU study of
initial guesses for sequences of PDE linear systems
([doi:10.1137/20M1368677](https://doi.org/10.1137/20M1368677)). NDgpu's outer
problem differs materially: its group solves live inside a rebalanced,
stateful upscatter/fission-source fixed point.

## 11-group HP-MR bulk insertion

Configuration: 2-D refinement 2, 1,320 active cells, 14,520 unknowns, four
20 ms steps, six energy subsweeps, whole-core rebalance, `tol_step=1e-6`.

| Predictor | Mean sweeps/step | Inner iterations/step | ms/step | Final P/P0 |
|---|---:|---:|---:|---:|
| Previous accepted flux | 23 | 4,446 | 675.6 | 1.84136 |
| Linear, w=0.25 | 67 | 12,957 | 1,669.2 | 1.84129 |
| Linear, w=0.50 | 121 | 23,238 | 2,833.2 | 1.84125 |
| Linear, w=1.00 | 28 | 5,485 | 802.2 | 1.84202 |

Even the best extrapolated leg performed 23% more inner work; damping could
make the interaction with rebalance substantially worse rather than smoothly
interpolate toward the baseline.

## Coupled cached-drum ramps

Configuration: quick two-group HP-MR cases, 2-D refinement 2, conduction and
temperature feedback enabled. Work counters are more reliable than the short
single-run wall measurements and drive the decision.

### Slow symmetric ramp

| Predictor | Total sweeps | Inner iterations | March seconds | Final P/P0 |
|---|---:|---:|---:|---:|
| Previous accepted flux | 185 | 3,668 | 0.486 | 1.124413487 |
| Linear, w=0.25 | 189 | 3,732 | 0.515 | 1.124351706 |
| Linear, w=0.50 | 207 | 4,076 | 0.539 | 1.124329551 |
| Linear, w=1.00 | 227 | 4,488 | 0.622 | 1.124632677 |

### Asymmetric four-drum ramp

| Predictor | Total sweeps | Inner iterations | March seconds | Final P/P0 |
|---|---:|---:|---:|---:|
| Previous accepted flux | 615 | 13,268 | 1.932 | 1.540561755 |
| Linear, w=0.25 | 585 | 12,548 | 1.743 | 1.540501822 |
| Linear, w=0.50 | 659 | 14,116 | 1.903 | 1.540608316 |
| Linear, w=1.00 | 623 | 13,344 | 1.781 | 1.540995117 |

Weight 0.25 reduced asymmetric inner work by 5.4%, but it failed to help the
slow ramp and catastrophically increased the stiff 11-group insertion work.
The different final powers are alternate stopping points inside the configured
fixed-point tolerance; at `tol_step=1e-8`, the undamped predictor differed from
the baseline by `3.6e-7` relative, consistent with the solver's documented
tolerance amplification but too large for a bitwise-equivalence gate.

## Interpretation and next action

Raw flux mixes the rapidly changing amplitude with the slowly changing spatial
shape. Extrapolating it perturbs the whole-core rebalance correction and the
incoming state of the finite-upscatter Gauss-Seidel map. This is not the same
sequence-of-linear-systems setting in which polynomial prediction is normally
effective.

No solver/API/test changes from this experiment were retained. A future reuse
experiment should project out the adjoint-weighted amplitude first, predict
only the normalized shape, or recycle a Krylov subspace within the actual
linear group/block solve. The next roadmap step is guarded-IQS fallback reuse,
which targets measured duplicated work without changing the full-diffusion
iteration path.

