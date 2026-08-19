# Step 05a: monolithic multigroup transient solve

Date: 2026-08-12  
Device: CPU / NumPy, FP64  
Decision: **CPU prototype accepted as an opt-in diffusion step solver; GPU gate pending**

## Method

Backward Euler already eliminates the end-of-step precursor fields
analytically. The remaining frozen-cross-section step is therefore one linear
multigroup system:

```text
[loss + time - off-group scatter - eliminated fission] phi = old-time + delayed
```

`MultigroupStepOperator` evaluates this block system matrix-free from the
existing finite-volume group operators. Real restarted FGMRES stores the
preconditioned Arnoldi basis, so its right preconditioner may be an inexact
energy-group Gauss--Seidel sweep with varying PCG iteration counts. No large
multigroup sparse matrix is assembled.

The production transient exposes the prototype through
`step_solver="monolithic"`; the validated fixed-point path remains the default.
The initial configuration is three scatter subsweeps, degree-1 group
preconditioning where requested, inner PCG `rtol=1e-3`, FGMRES restart 30, and
an outer true-residual target of `0.1 * tol_step`.

Including lagged fission inside the sweep was tested and rejected. It reduced
outer FGMRES applications (for example 28 to 26 with three sweeps) but raised
inner work from 1,402 to 2,298 iterations and wall time from about 0.18 to
0.25 s. FGMRES should resolve the near-critical fission coupling globally.

## Correctness gates

- real FGMRES solves a non-symmetric dense system under an alternating,
  variable right preconditioner;
- the matrix-free two-group operator matches independently assembled dense
  blocks to roundoff, including upscatter and fission orientation;
- one ordered sweep exactly inverts a downscatter-only triangular block when
  inner PCG is tight;
- FGMRES agrees with a dense direct solve;
- an end-to-end 11-group HP-MR step agrees with the production fixed point;
- homogeneous one-group transient histories agree across multiple steps;
- invalid method/options fail before an eigenvalue solve.

Focused Krylov/block/point-kinetics result: **50 passed in 27.50 s**. Broader
transient, coupled, and quasi-static result: **89 passed in 194.66 s**.

## 11-group HP-MR CPU gate

Configuration: 2-D HP-MR, uniform +0.5-dollar-like absorption step, 20 ms
backward-Euler step, real ENDF/B-VIII-derived 11-group upscattering data,
degree-1 polynomial group preconditioning. Fixed point used six group
subsweeps, whole-core rebalance, no Anderson mixing. Times exclude the common
initial eigenvalue solve. Each row is a single development run, so the work
counters drive the decision and timings are indicative.

| Refine | Active cells | Unknowns | Fixed time | Monolithic time | Speedup | Fixed PCG | Mono PCG | FGMRES | Power difference | Flux L2 difference |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 330 | 3,630 | 0.649 s | 0.194 s | 3.34x | 4,435 | 1,402 | 28 | 1.77e-7 | 1.27e-7 |
| 2 | 1,320 | 14,520 | 1.143 s | 0.446 s | 2.56x | 6,699 | 2,272 | 27 | 5.82e-8 | 4.16e-8 |
| 3 | 2,970 | 32,670 | 1.801 s | 0.732 s | 2.46x | 7,426 | 2,995 | 25 | 3.29e-7 | 2.35e-7 |

Refinement 1/2 used `tol_step=1e-9`, FGMRES `rtol=1e-10`; refinement 3 used
`tol_step=1e-8`, FGMRES `rtol=1e-9`. The power/flux differences track the
fixed-point stopping error: tightening the refine-2 fixed point from `1e-8` to
`1e-9` reduced the power difference from `6.97e-7` to `5.82e-8`.

Three scatter subsweeps were the CPU wall-time optimum on refine 1:

| Subsweeps | FGMRES applications | Inner PCG | Monolithic time |
|---:|---:|---:|---:|
| 1 | 76 | 2,421 | 0.287 s |
| 2 | 34 | 1,418 | 0.182 s |
| 3 | 28 | 1,402 | 0.184 s |
| 6 | 21 | 1,456 | 0.202 s |

The 2-versus-3 timing difference is noise-sized; three is retained because it
requires fewer global FGMRES reductions and is therefore the better initial
GPU candidate.

### Four-step history and tolerance ladder

On refinement 2, four production-style steps at `tol_step=1e-6` took 2.861 s
and 10,455 PCG iterations with the fixed point, versus 2.246 s and 7,164 PCG
iterations with conservative monolithic `rtol=1e-7` (1.27x). Their final powers
were 1.8413555 and 1.8416682 because the fixed-point source-change tolerance is
not a full-system residual bound. Tightening the fixed point demonstrates that
the monolithic answer is the converged one:

| Method/tolerance | Wall | Inner PCG | Final P/P0 |
|---|---:|---:|---:|
| Fixed point, `tol_step=1e-6` | 2.861 s | 10,455 | 1.84135550 |
| Fixed point, `tol_step=1e-7` | 3.846 s | 14,896 | 1.84165396 |
| Fixed point, `tol_step=1e-8` | 4.378 s | 19,771 | 1.84166672 |
| Monolithic, `rtol=1e-7` | 2.246 s | 7,164 | 1.84166815 |
| Monolithic, `rtol=1e-9` | 2.544 s | 8,226 | 1.84166816 |

At comparable converged accuracy, the direct path is therefore about 1.7x
faster on this four-step CPU case. Even `rtol=1e-5` gave 1.84166689 in 1.939 s,
but the public opt-in retains the more conservative `0.1*tol_step` default
until more heterogeneous histories calibrate the residual/error relation.

### Moving-control stress gate

A four-frame 11-group drum withdrawal from 150 to 154 degrees exercises a new
volume-mix geometry and operator block at each step. With a tightly converged
common eigenstate, refinement 1 fixed point took 1.233 s and 8,325 PCG
iterations; monolithic took 0.639 s and 4,131 (1.93x). Maximum power-history
difference was `1.40e-7` and final normalized-flux L2 difference `1.48e-7`.
This is now a regression test, so cached-block invalidation and scatter/fission
field orientation are covered under real control motion rather than only a
stationary bulk insertion.

## Adjoint-weighted rank-one correction

An optional adapted-deflation correction makes the preconditioned operator
exact on the latest forward amplitude mode using its adjoint importance. It
adds one calibration group sweep per step but no extra block apply thereafter.
Including calibration, refinement 1 changed 28 to 24 FGMRES applies and 1,402
to 1,267 PCG iterations (0.180 to 0.158 s); refinement 2 changed 27 to 23 and
2,272 to 2,031 (0.376 to 0.374 s, wall-neutral). A fresh adjoint cost 0.65 s
and 1.68 s respectively, so automatic use would lose on short runs. It remains
an opt-in `coarse_correction=True`; a compatible existing adjoint can be passed
as `coarse_adjoint` to avoid the startup eigen solve.

## Limitations and next gate

- Diffusion only. SP3/SDPN block rows need a separate moment-aware formulation.
- BDF1--6 now use this same production block, including explicit nonuniform
  widths and event restart (Phase 07a). Automatic BDF error control remains
  pending.
- Restart memory is `2 * restart` multigroup vectors. At restart 30 this is
  roughly 60 state vectors and must be checked against 3-D GPU memory.
- The current source coupling uses Python group loops and temporary arrays. A
  stacked/fused GPU block apply and allocation-stable FGMRES workspace are
  required before expecting the CPU speedup to transfer.
- The adjoint rank-one correction reduces group work but has not yet cleared a
  general wall-time gate; test it mainly where an IQS/coupled driver can reuse
  an existing adjoint or where GPU reductions are dominant.

GPU acceptance requires matching the fixed-point history/shape, lower total
stencil work, no memory exhaustion on 580k and 1.16M unknown cases, and lower
end-to-end transient time after graph/kernel warm-up. Until then the option is
experimental and not selected automatically.
