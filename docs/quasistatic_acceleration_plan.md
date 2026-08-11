# Quasi-static transient acceleration plan

## Objective

Accelerate slow, coupled full-core transients in which neutron population and
delayed precursors change much faster than flux shape and temperature. The
acceptance workload is the optional one-minute 3-D HP-MR case in
`notebooks/colab_coupled_transient_gpu_latest.ipynb`:

- real 11-group SPH-ready constants;
- 52,800 active cells at radial refinement 4 and 10 axial layers;
- all twelve control drums withdrawing together by approximately +0.25 dollar;
- a 15 s drum ramp beginning at 5 s, followed through 60 s;
- 0.2 s neutron output steps, 1.0 s thermal steps, and 31 cached drum frames;
- conduction and temperature-dependent cross sections on the GPU.

The current fully spatial method advances 300 neutron steps. Quasi-static (QS)
acceleration should solve a small amplitude/precursor system at those output
times while updating the expensive 11-group spatial shape only when its error
requires it.

This is an acceleration method, not a new physics model. It must converge to
the existing backward-Euler transient as the shape-update interval is reduced.

## Implementation status (2026-08-11)

Phase 0 and the fixed-shape portion of Phase 1 are now executable:

- `TransientSolver.solve(initial_steady=...)` accepts a compatible eigenpair,
  and `coupled_transient` automatically hands off its converged hot coupled
  eigenpair instead of repeating the time-zero power iteration.
- `project_effective_kinetics` applies the current loss/fission operator to a
  forward anchor and projects `rho`, `Lambda`, and delayed fractions with an
  adjoint anchor. Global and per-material kinetics are supported.
- `integrate_point_kinetics` and `advance_point_kinetics` provide the matching
  backward-Euler amplitude/precursor march.
- `fixed_shape_coupled_transient` performs one initial adjoint solve, projects
  changed control/feedback operators, keeps the fixed fission-power shape and
  temperature on the selected device, and advances the existing conduction
  solver at `dt_thermal` cadence.
- CPU regression gates cover exact homogeneous projection, normalization
  invariance, analytic absorption worth, stationary point kinetics, equality
  with the full spatial transient for a shape-preserving insertion, stationary
  coupled equilibrium, and initial-state reuse.

The current result type reports `shape_updates == 0` intentionally. It is an
explicit fixed-shape prototype, suitable for exact shape-preserving cases and
small-perturbation experiments—not yet the production path for a large black
absorber movement. Phase 2 starts with warm-started forward re-anchoring,
continuity at shape updates, and interval refinement against 2-D HP-MR ramps.

## Mathematical split

Write the multigroup flux as

```text
phi_g(r,t) = A(t) psi_g(r,t)
```

where `A` is the global amplitude (reported as relative power) and `psi` is a
slowly varying spatial/energy shape. Normalize the shape with a discrete
adjoint-weighted condition, for example

```text
<psi*, v^-1 psi> = 1,
```

and rescale it after every shape update so that amplitude remains continuous.
The finite-volume inner product must include active-cell volume and the
tri-grid operator's existing right-hand-side weight.

Projection with the adjoint shape gives the effective point-kinetics system

```text
dA/dt = ((rho - beta_eff) / Lambda) A + sum_i lambda_i zeta_i
dzeta_i/dt = (beta_eff_i / Lambda) A - lambda_i zeta_i.
```

`rho`, `Lambda`, `beta_eff_i`, and delayed spectra are computed from the
current forward/adjoint shapes and the same material/blend fields used by the
spatial solver. Per-material velocity and delayed-neutron tables must remain
supported.

For a state between shape solves, estimate reactivity by applying the current
loss and fission operators to the last forward shape and projecting with the
last adjoint. Re-anchor this Rayleigh estimate at every shape solve. That makes
each perturbation local in drum angle and temperature; it avoids applying
first-order perturbation theory across the entire motion of a black absorber,
where it is known to fail.

## Proposed algorithm

### Initial state

1. Compute or accept a converged coupled hot equilibrium.
2. Reuse its forward flux and `k_eff`; do not repeat the time-zero eigen solve.
3. Solve one adjoint problem at the same material, drum, and temperature state.
4. Normalize the forward/adjoint pair and initialize amplitude and delayed
   precursor state consistently with the existing critical adjustment.
5. Keep shape, spatial precursor information, temperature, material fields,
   and operator coefficients on the GPU.

### One shape interval

1. Select the next interval endpoint from the maximum shape interval, cached
   control-frame boundaries, thermal exchanges, and any error-triggered event.
2. Using the current shape and adjoint, project the operator at each control or
   thermal event to obtain effective reactivity and kinetics parameters.
3. Integrate the small amplitude/precursor system adaptively. A stable implicit
   or matrix-exponential update is cheap enough to take substeps without any
   full-core kernels.
4. Form thermal power as amplitude times the normalized fission-power shape.
   Accumulate it over the thermal window and advance conduction with the
   existing device-resident solver.
5. Update feedback fields only at thermal cadence. Project their reactivity
   effect immediately, without forcing a spatial shape solve.
6. At the interval endpoint, perform a spatial shape corrector. Warm-start it
   from the previous shape and use the known amplitude evolution to remove the
   fundamental amplitude mode.
7. Renormalize the corrected shape and adjust amplitude/precursors so total
   power and neutron population are continuous across the update.
8. Refresh the adjoint only when the shape residual, accumulated reactivity,
   or a maximum adjoint age requires it.

The first production version should be an adiabatic QS method: the corrector
uses the instantaneous forward eigen shape. The next version should implement
improved quasi-static (IQS): a large-step shape equation retains the shape time
derivative and the spatial delayed-source correction. Both use the same public
driver and result type.

## Adaptive shape-update policy

Fixed shape intervals are useful for convergence studies, but production needs
event/error control. Trigger a new shape solve when any configured condition is
met:

- maximum interval reached, initially 2 s for the HP-MR minute case;
- a new cached drum frame has accumulated more than a specified reactivity
  change, initially 0.02 dollar since the last shape anchor;
- maximum fuel-temperature change since the anchor exceeds 2 K;
- projected spatial residual exceeds its tolerance;
- predicted and corrected amplitude disagree beyond the requested power error;
- a control step, material discontinuity, or user-declared event occurs.

The spatial residual should remove the adjoint/amplitude component before its
norm is evaluated; otherwise a harmless normalization error will look like a
shape error. Record why every update occurred.

If the residual cannot be reduced, reactivity approaches prompt critical, or
controls move discontinuously beyond a configured limit, fall back to the full
spatial stepper for that interval. The accelerated and reference paths should
be composable in one run.

## Software architecture

### Refactor the existing stepper first

Extract the reusable state and one-step operation from `TransientSolver.solve`:

```text
TransientState
  flux, fission_source, spatial_precursors, time, k0, fields/operators

TransientSolver.initialize(...) -> TransientState
TransientSolver.advance(state, t_next, problem, feedback, ...) -> StepInfo
```

This refactor must reproduce existing transient histories before QS logic is
added. It also enables reuse of the coupled equilibrium and removes the current
duplicate initial critical solve.

### New module

Add `ndgpu/quasistatic.py` with:

```text
QuasiStaticConfig
  shape_dt_max, amplitude tolerances, residual thresholds,
  adjoint refresh policy, fallback policy

QuasiStaticState
  amplitude, effective precursors, forward shape, adjoint shape,
  optional spatial precursor correction, temperature, control state

QuasiStaticResult
  ordinary coupled histories plus shape-update times/reasons,
  projected parameters, errors, phase timings, and counters

QuasiStaticSolver
  initialize(), advance_interval(), solve()
```

Expose it first as an explicit advanced solver. After validation, add
`mode="quasistatic"` or a `core.quasistatic_transient(...)` convenience method
to `TriReactor`.

### Projection kernels

Implement batched GPU reductions for:

- adjoint normalization;
- effective generation time and delayed fractions;
- loss/fission Rayleigh quotients;
- projected shape residual;
- normalized fission-power shape.

Use one packed reduction per projection event rather than one host
synchronization per group or coefficient. Projection happens at shape/thermal
cadence, never at every amplitude substep.

### Precursors

Milestone 1 may use effective precursor amplitudes with a frozen spatial
precursor shape. IQS must retain the spatial precursor correction. A practical
GPU representation is the existing per-family precursor field, updated
analytically from the amplitude-weighted fission shape over each amplitude or
thermal interval. This costs a few elementwise kernels per family and avoids an
11-group spatial solve.

## Delivery phases

### Phase 0 — trustworthy baseline

- Separate coupled equilibrium, initial eigen solve, and time-march profiling.
- Reuse a compatible coupled steady state in the transient initializer.
- Record eigen and march work independently.
- Freeze the minute-case inputs and archive its full-spatial result.

Exit criterion: stationary coupled histories remain unchanged and the startup
timings add up without hidden work.

### Phase 1 — fixed-shape point-kinetics prototype

- Implement adjoint normalization and effective kinetics projection.
- Integrate amplitude and delayed precursors with a frozen shape.
- Drive the existing thermal solver with amplitude-scaled power.
- Support material feedback and cached control frames through projected
  reactivity, but perform no shape updates.

Exit criterion: uniform bulk perturbations match analytic point kinetics and
the full solver; stationary coupled power remains exactly one.

### Phase 2 — adiabatic shape updates

- Warm-start forward shape solves at configurable macro intervals.
- Re-anchor reactivity and kinetics after each solve.
- Preserve amplitude, power, and precursor continuity during renormalization.
- Add lagged/adaptive adjoint refresh.

Exit criterion: shape-interval refinement converges monotonically to the full
transient on TWIGL and 2-D HP-MR drum ramps.

### Phase 3 — improved quasi-static corrector

- Add the shape time-derivative term.
- Retain spatial delayed-source corrections.
- Add predictor/corrector error estimation and full-step fallback.
- Align shape events with control and thermal discontinuities.

Exit criterion: IQS is materially more accurate than adiabatic QS at the same
number of shape solves for a localized drum manoeuvre.

### Phase 4 — coupled GPU production path

- Fuse projection reductions and keep shape/thermal fields resident.
- Add the high-level `TriReactor` entry point and Colab comparison.
- Tune adaptive thresholds against error and wall time.
- Support restart/checkpoint of amplitude, shape, adjoint, precursors, and
  temperature.

Exit criterion: the one-minute 3-D HP-MR case meets its accuracy targets and
shows a useful end-to-end speedup including the initial coupled equilibrium.

## Verification and validation matrix

| Problem | Purpose | Required comparison |
|---|---|---|
| Unperturbed homogeneous/HP-MR | Invariance | `P/P0=1`, no temperature drift, no unnecessary shape update |
| Uniform absorption insertion | Amplitude equations | Analytic point kinetics and current full transient |
| Material-dependent kinetics | Projection correctness | Existing per-material transient regression cases |
| TWIGL step and ramp | Published space-time result | Published powers and full solver |
| 2-D HP-MR 0.25-dollar drum ramp | Localized shape change | Full coupled transient at decreasing shape intervals |
| Reduced 3-D HP-MR | Axial shape and feedback | Full solver on CPU/GPU |
| One-minute 3-D HP-MR | Production acceptance | QS interval refinement plus selected full-spatial checkpoints |

For the production case, target initially:

- peak and final power within 0.5% of the full spatial reference;
- peak time within one amplitude output step;
- mean/peak fuel temperature within 0.2 K;
- integrated fission energy within 0.2%;
- reactivity and power continuous at shape updates;
- CPU/GPU histories within the existing backend tolerances.

Tighten these only after a shape-interval convergence curve demonstrates that
the requested accuracy is attainable and measured rather than assumed.

## Performance measurements and targets

Report:

- coupled-equilibrium and initialization time;
- number of amplitude substeps, thermal steps, forward shape solves, adjoint
  solves, fallbacks, and operator rebuilds;
- time in amplitude integration, projection, forward/adjoint shape correction,
  thermal solve, transfers, and result assembly;
- power/temperature error versus full spatial reference;
- speedup at equal error, not merely at equal nominal time step.

The first target is to reduce 300 fully spatial steps to no more than 31
forward shape corrections and 6 adjoint refreshes. The adaptive method should
do better during the 40 s post-motion hold. A useful acceptance target is at
least 5× faster time marching and 3× faster end-to-end on a T4-class GPU while
meeting the accuracy limits above. The end-to-end target is deliberately lower
because the initial coupled equilibrium remains a significant fixed cost.

## Principal risks

- First-order drum-worth projection fails if allowed to span a large absorber
  movement. Mitigation: small cached frames, local re-anchoring, residual
  triggers, and full-step fallback.
- A stale adjoint biases effective kinetics. Mitigation: age/residual policy and
  comparison against refreshing every shape solve.
- Pure adiabatic QS misses spatial delayed-source history. Mitigation: treat it
  as an intermediate milestone and implement the IQS precursor correction.
- Shape solves can cost more than easy full time steps during a very slow ramp.
  Mitigation: adaptive rather than frame-by-frame shape updates, and measure
  speedup at equal error on the minute case.
- Thermal feedback can move reactivity while shape is frozen. Mitigation:
  project feedback at every thermal exchange and trigger a shape update on
  accumulated temperature/reactivity change.
