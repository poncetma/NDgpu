# Step 08a: adaptive-BDF GPU readiness

Date: 2026-08-18

## Decision

The adaptive monolithic path was not ready for a defensible 3-D GPU benchmark.
The spatial stencils were GPU-fused, but recently added time-control and
monolithic algebra reintroduced host synchronization, temporary full-state
arrays, unbatched energy coupling, and per-solve FGMRES allocation. The changes
below are retained behind the existing NumPy/CuPy common API. CPU correctness
is verified. The quick 3-D performance gate has now run on a Tesla T4; the
larger production gate and reference-convergence ladder remain pending.

## Audit and implementation

| Path | Before | Current implementation |
|---|---|---|
| Full-state BDF defect | Two Python-float reductions per field. An 11-group + 6-precursor state caused 34 host synchronizations per estimate, up to 102 for q-1/q/q+1 selection. Residual and scale lists retained two full states. | Reductions stay as device scalars, fields are streamed, and the final numerator/denominator pair is copied once. One synchronization per estimate and no full-state residual/scale lists. |
| BDF history extrapolation | Each coefficient formed a new `c*x` temporary and a new sum. | Output is allocated once; subsequent terms use a fused in-place AXPY (`adaptive` kernel family). |
| Monolithic energy coupling | Fission and scattering used Python group loops, giving O(G^2) launches and temporaries even when the transient had already built its guarded group stack. | The block operator and GS preconditioner reuse the existing VRAM-guarded stack. Fission is one group contraction; each row is a fused contraction/product accumulation. Sparse CPU and low-memory GPU fallbacks remain. |
| Flexible GMRES | V/Z lists and work vectors were allocated every solve. Modified Gram--Schmidt converted every dot product to a host float. The preconditioner copied each completed block vector. | A bounded persistent workspace owns V, Z, x, r, and w and survives coefficient/step rebuilds. GPU CGS2 batches projections on device, with one packed host transfer per Arnoldi step. Basis updates/reconstruction are fused. The GS preconditioner writes into a supplied Z slot. CPU retains MGS. |

For restart `r`, FGMRES now reports the explicit `(2r + 4)` full-state storage
in `fgmres_workspace_bytes`. Coupled profiles also report
`group_batch_active`, so a timing cannot silently be attributed to batching
when the free-VRAM guard selected the sparse path.

## Validation completed locally

- 92 focused time-scheme, fused-kernel, and monolithic tests: 90 passed, 2
  CUDA-dependent skips.
- Coupling and transient regression subsets pass after hot coupled-steady reuse
  and new counters were added.
- The Colab notebook is valid nbformat 4 JSON and every code cell parses.
- No CUDA device is available in the local environment, so this report does
  not independently reproduce the submitted Colab timings.

## 3-D acceptance notebook

`notebooks/colab_hpmr3d_adaptive_bdf_gpu.ipynb` runs a motion-resolving,
11-group, 3-D HP-MR coupled case and emits JSON, CSV, and a diagnostic figure.
It contains:

1. old/new error-norm and predictor A/B microbenchmarks;
2. a fine fixed-step backward-Euler accuracy reference;
3. accuracy-matched adaptive backward Euler and adaptive BDF5;
4. maximum power, thermal-exchange temperature, and final normalized-flux
   errors;
5. accepted/rejected steps, order occupancy, inner/outer Krylov work, phase
   timings, batching status, and FGMRES memory;
6. the matched adaptive-BE/adaptive-BDF march-time ratio, kept distinct from
   speedup against the finer reference.

The quick gate is refinement 2 x 10 axial layers (about 145k active flux
unknowns). The production switch selects refinement 4 x 20 layers. Before
quoting results, double the reference step count to establish time-reference
convergence and repeat timed legs at least three times.

## Submitted Tesla T4 result

Artifact:
`notebooks/notebook-results-json/hpmr3d_adaptive_bdf_gpu.json`.
Environment: Tesla T4, CuPy 14.0.1, Python 3.12.13. The quick configuration is
refinement 2 x 10 axial layers, 13,200 active cells, 145,200 active flux
unknowns, 11 groups, a 90--94 degree symmetric drum motion over 0.08 s, four
control/thermal intervals, and FGMRES restart 10.

The implementation gates passed:

- energy-group batching was active;
- the persistent FGMRES allocation was 61.27 MiB;
- the hot coupled equilibrium was reused by every timed leg;
- telemetry transfers equalled the four thermal windows;
- there were no mixed-precision fallbacks or CUDA graph errors.

The adaptive-state microbenchmark validates the intended GPU changes. The
one-transfer error norm is 3.298 ms against 3.685 ms for the 34-synchronization
form, a 1.117x gain. Its modest wall effect is expected: the reductions still
stream all 17 fields, and synchronization was only part of their cost. Fused
BDF extrapolation is the larger local win, 0.987 ms against 1.950 ms, or
1.975x.

### Accuracy and matched performance

| method | accepted / rejected | order occupancy | march (s) | inner work | outer work incl. rejects | max power error | final flux-shape error |
|---|---:|---:|---:|---:|---:|---:|---:|
| fine fixed BE, 32 steps | 32 / 0 | q1: 32 | 62.434 | 85,416 | 1,008 | reference | reference |
| adaptive BE, RTOL 1e-3 | 24 / 7 | q1: 24 | 62.752 | 95,522 | 1,126 | 0.0301% | 1.38e-6 |
| adaptive BDF5, RTOL 1e-3 | 22 / 9 | q1: 15, q2: 7 | 62.380 | 98,051 | 1,155 | 0.0260% | 2.74e-7 |

Adaptive BDF5 is **1.006x** faster than accuracy-matched adaptive BE. That is a
tie at single-sample Colab precision, not evidence of a method speedup. BDF
uses 8.3% fewer accepted steps and has 13.4% lower maximum power error and
5.0x lower final flux-shape error, but two additional rejected attempts make
both methods perform exactly 31 total attempts. Rejected work is 34.5% of BDF
outer work versus 25.3% for BE. Consequently BDF performs 2.58% more total
outer work and 2.65% more inner work; its 0.6% timing advantage is readily
explained by run noise and different step widths.

Mean and peak temperature errors are only 1.65e-5 K and 3.72e-5 K. They are
too small to discriminate the time integrators on this 80 ms physical-mass
case, but they verify the adaptive thermal-alignment and elapsed-time power
averaging paths.

Neutronics accounts for 94.9% of BDF march time and operator rebuilds another
4.7%; all thermal, telemetry, feedback, and result-transfer phases together
are below 0.1%. Further end-to-end GPU acceleration therefore requires fewer
neutron/Krylov attempts or a faster monolithic preconditioner. Optimizing the
remaining adaptive bookkeeping cannot materially move this case.

## Next measurement

1. Repeat the quick run with 64 and 128 fixed-BE reference steps. The current
   32-step reference establishes small differences but not a converged error
   denominator.
2. Run at least three timed samples after warm-up and compare medians; a 0.6%
   difference is below a credible Colab timing resolution.
3. Reduce BDF rejected work before pursuing more kernels: test a stricter
   post-restart order hold and rejection-controller tuning while retaining the
   same error gates.
4. Then run the refinement 4 x 20-layer configuration. Its larger spatial work
   will show whether the 1.117x norm and 1.975x predictor micro-wins remain
   visible or vanish behind Krylov work.

## 2026-08-19 energy-acceleration rerun

The updated artifact uses the same T4 quick geometry and adds the Step 10a
energy-preconditioner gates. The adaptive microbenchmarks remain positive:
the one-transfer error norm is now 1.819x faster and fused prediction 1.798x
faster in this run. More importantly, the short neutron solve distinguishes
backend behavior:

- tolerance-based depth-one energy Anderson is rejected on T4: 3.265 s versus
  2.139 s plain, with outer work increasing from 63 to 71;
- the reduction-free fixed-polynomial path takes 1.457 s, a 1.468x speedup
  over plain and 2.241x over tolerance-based Anderson.

The full adaptive legs in this artifact still used tolerance-based Anderson.
They remain accurate, but adaptive BDF5 is 0.936x versus adaptive BE
(42.164 s versus 39.450 s). The regenerated notebook now promotes the
fixed-polynomial configuration to all full legs; that rerun supersedes the
present performance number while retaining it as a numerical regression.

### Full fixed-polynomial replacement

The replacement artifact confirms the full path. Fine BE takes 32.958 s,
adaptive BE 34.029 s, and adaptive BDF5 35.977 s. These are respectively
1.115x, 1.159x, and 1.172x faster than the preceding tolerance-Anderson run,
with indistinguishable power, temperature, and flux errors. The fixed path
uses about twice as many outer iterations but cuts neutronics time per outer by
roughly 2.2x, demonstrating that removing the inner reductions is the relevant
T4 optimization.

The replacement microbenchmarks also show run-to-run timing sensitivity: the
error-norm speedup is 1.061x while predictor fusion is 3.320x, versus 1.819x
and 1.798x in the preceding session. Their numerical agreement and transfer
counts remain the stable gates; performance claims require medians.

Matched adaptive BDF5 remains below adaptive BE at 0.946x because BDF performs
9.6% more total outer work after its two additional rejected attempts and
reaches only orders one and two. At this checkpoint the one-step
fixed-polynomial speedup over plain PCG varied from 1.468x to 1.028x across the
two submitted runs, so the notebook was changed to repeat the short gates three
times in interleaved order and export medians. The following rerun settles that
comparison.

### Repeated short-gate decision

The third artifact supplies the requested three interleaved samples. Median
times are 2.310 s for plain tolerance PCG, 2.620 s for tolerance-based
Anderson, and 2.098 s for the fixed polynomial. The final T4 decisions are:

- reject tolerance-based Anderson at 0.882x versus plain;
- accept the fixed-polynomial path opt-in at 1.101x versus plain and 1.249x
  versus Anderson.

The full fixed-polynomial run again passes every reference gate. Adaptive BE
takes 36.056 s and adaptive BDF5 39.971 s, giving 0.902x. Together with the
preceding fixed-polynomial session's 0.946x, this establishes that adaptive
BDF does not beat adaptive BE on this mild drum trajectory. Its two additional
rejections leave 2,260 total outer iterations versus 2,062 for BE.
