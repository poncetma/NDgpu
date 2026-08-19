# Step 07a: multi-level BDF foundation and CPU stability gate

Date: 2026-08-12  
Device: CPU / NumPy, FP64  
Decision: **constant/nonuniform BDF1--6 and event restart accepted as opt-in;
automatic order/step control not yet accepted**

## What “multi-level” means here

Cherezov, Vasiliev, and Ferroukhi organize a coupled transient into three
iteration levels: BDF time integration is the global level, the implicit
neutronics/thermal fixed point is the constituent level, and multigroup plus
within-group iterations form the inner level. ndgpu previously had BDF2 only
in the discrete-ordinates transient; the production diffusion/coupling and
monolithic multigroup paths were backward-Euler-only.

This phase puts the global BDF history representation around either diffusion
in-step solver. It does not conflate time order with the spatial multilevel
preconditioner planned separately in Phase 6.

## Implementation

- Shared constant-step BDF coefficients for orders 1--6, with order ramping as
  history becomes available.
- Nonuniform-step coefficients computed from the derivative of the Lagrange
  interpolant on the accepted time nodes. `dt` may be a scalar or an explicit
  positive sequence summing to `t_end` in `TransientSolver`.
- The same BDF history combination is used for flux and analytically eliminated
  precursor fields. A critical equilibrium remains exact for every order and
  step ratio.
- Both `step_solver="fixed-point"` and `"monolithic"` consume the same time
  shift and carried history field.
- `bdf_restart_times` discards flux and precursor history before a known
  discontinuity, forces BDF1 on that step, and ramps order again.
- `TransientResult` and `CoupledTransientResult` report `time_scheme` and the
  actual `time_orders`; coupled profiling reports `neutron_bdf_max_order`.
- The coupled driver forwards BDF controls through `transient_kwargs`. It still
  uses constant neutron and thermal grids because adaptive rejection needs
  transactional rollback of both physics states.

The algebra follows Section 2.3 and Table 1 of Cherezov et al., while using a
direct history-coefficient representation rather than Nordsieck storage at
this stage. The paper's Nordsieck predictor/error estimator is the next layer.

## CPU correctness gates

- BDF1--6 coefficients differentiate every polynomial through their order.
- Nonuniform BDF coefficients satisfy the same identities on irregular nodes.
- Manufactured stiff decay demonstrates the expected order with exact startup.
- Analytic precursor elimination preserves a critical equilibrium through
  startup and every order.
- Diffusion BDF1--6 preserves an unperturbed core and reports the expected
  startup orders.
- BDF3 fixed-point and monolithic histories/fluxes agree to solver tolerance.
- A nonuniform BDF5 diffusion schedule preserves equilibrium.
- An event restart reduces `[1,2,3,...]` to order one at the event without a
  power disturbance.
- The coupled driver forwards and reports BDF order.

Focused result after implementation: **32 BDF/time-scheme tests passed**;
focused coupled BDF/monolithic/stationary result: **3 passed**.

## 11-group HP-MR diagnostic

Configuration: 2-D refinement 1, 330 active cells / 3,630 unknowns, real
11-group upscattering library, monolithic FGMRES in-step solve, drums 150 to
154 degrees. A fine 32-step BDF2 solution is the reference. Four cached control
intervals are common to every run and every discontinuity triggers history
restart. The asymmetric case moves three adjacent drums and leaves nine fixed.

The table reports representative 16-step candidates. `speedup` includes the
time march but reuses the common initial eigenstate. `dP(end)` and `dflux` are
relative to the fine BDF2 solution.

| Case | Scheme | dt | Wall | Speedup | Inner PCG | dP(end) | dflux |
|---|---|---:|---:|---:|---:|---:|---:|
| slow symmetric, 0.8 s | BDF1 | 0.050 s | 1.348 s | 1.61x | 9,823 | 4.51e-4 | 4.51e-4 |
| slow symmetric, 0.8 s | BDF2 | 0.050 s | 1.471 s | 1.47x | 9,531 | 6.35e-4 | 6.34e-4 |
| slow symmetric, 0.8 s | BDF3 | 0.050 s | 1.530 s | 1.42x | 9,862 | 3.52e-4 | 3.52e-4 |
| slow symmetric, 0.8 s | BDF5 | 0.050 s | 1.607 s | 1.35x | 9,981 | 4.07e-4 | 4.08e-4 |
| fast symmetric, 0.08 s | BDF1 | 0.005 s | 1.428 s | 1.51x | 9,099 | 7.61e-4 | 7.59e-4 |
| fast symmetric, 0.08 s | BDF3 | 0.005 s | 1.374 s | 1.57x | 8,499 | 1.10e-3 | 1.09e-3 |
| fast symmetric, 0.08 s | BDF5 | 0.005 s | 1.319 s | 1.63x | 8,511 | 1.04e-3 | 1.04e-3 |
| fast asymmetric, 0.08 s | BDF1 | 0.005 s | 1.545 s | 1.32x | 10,668 | 8.89e-5 | 8.86e-5 |
| fast asymmetric, 0.08 s | BDF3 | 0.005 s | 1.530 s | 1.33x | 10,098 | 1.77e-4 | 1.77e-4 |
| fast asymmetric, 0.08 s | BDF5 | 0.005 s | 1.699 s | 1.20x | 9,990 | 1.70e-4 | 1.70e-4 |

No candidate produced negative/non-finite power or an in-step convergence
failure. BDF3 was modestly best in final slow-symmetric accuracy, but high
order did not consistently improve fast or asymmetric motion. This is not a
surprise: the forcing is only piecewise smooth, and every control-frame event
throws away the polynomial history by design. With eight candidate steps
(only two per control interval), BDF2, BDF3, and BDF5 become identical because
none can acquire history above order two before the next restart.

The maximum history error is dominated by linear interpolation across the
piecewise-constant control jumps (2.31% slow symmetric, 0.77% fast symmetric,
0.10% fast asymmetric at 16 steps), not by the endpoint BDF solve. A production
controller must align steps to known events and must not claim that polynomial
order recovers a control trajectory that was never sampled.

## Decision and next CPU phase

Keep backward Euler as the default and BDF2--6 opt-in. The implemented history
layer is suitable for deterministic studies and is stable on this first HP-MR
gate, but fixed high order alone is not an acceleration policy.

Next:

1. Add a Nordsieck predictor/corrector local-error estimate and conservative
   order/step controller, initially for uncoupled diffusion with no callbacks.
2. Limit step growth, reject/restore state transactionally, and land exactly on
   control events.
3. Verify work/error curves on point kinetics and TWIGL, then repeat this HP-MR
   gate with the same requested tolerance rather than the same step count.
4. Add rollback support for thermal state and coupling callbacks before enabling
   adaptive BDF in `coupled_transient`.
5. Reproduce the paper's LRA-2D case after the adaptive controller is pinned.

Reference: A. Cherezov, A. Vasiliev, H. Ferroukhi, “Application of Backward
Differential Formula and Anderson's method for multigroup diffusion transient
equation,” *Annals of Nuclear Energy* 210 (2025) 110837,
doi:10.1016/j.anucene.2024.110837 (local supplied PDF).
