# Transient performance implementation plan

## Objective and working method

Reduce end-to-end time for GPU-resident, SPH-corrected HP-MR coupled
transients without weakening the existing accuracy gates. Work proceeds one
accepted change at a time. Every change gets:

1. a CPU correctness regression;
2. an A/B benchmark against the immediately preceding implementation;
3. work counters as well as wall time;
4. an intermediate report in `benchmark-results/transient-acceleration`;
5. a GPU benchmark before a GPU-specific optimization is accepted.

An optimization is retained only when its result is physically equivalent at
the configured convergence tolerance, or when an explicitly approximate method
meets its documented power, temperature, and shape-error budget. Negative or
neutral experiments remain documented so they are not repeated later.

The current coupled profiles put 96--98% of transient time in the neutron
solve. Thermal conduction, feedback bookkeeping, telemetry, and result
transfers are already sub-percent. Consequently this plan targets the spatial
and multigroup solve first; coupling work is changed only when it removes
duplicated neutron work or permits fewer neutron solves.

## Progress log

- **Step 01, chronological raw-flux prediction: rejected (2026-08-11).** A
  damped two-state predictor reduced inner work by 5.4% on one asymmetric
  two-group ramp, but increased work on the slow ramp and by at least 23% on
  the stiff 11-group insertion (some damping choices were roughly 3x worse).
  The production edits were removed. See the
  [intermediate report](../benchmark-results/transient-acceleration/step-01-chronological-initial-guess.md).
  Any later history reuse should predict an amplitude-free shape or recycle a
  subspace inside the actual linear solve.
- **Step 02, guard-aware adjoint reuse: accepted on CPU (2026-08-11).** The
  fallback audit found no IQS-corrector duplication; adjoint eigen refreshes
  dominated instead. A projected transposed-eigenproblem residual now refreshes
  adjoints before a hard maximum age, while fallback always refreshes. The
  calibrated guard reduced adjoint solves from 9/10/11 to 6/7/9 on the
  slow/fast/asymmetric cases and brought the asymmetric run from 0.83x to 0.96x
  of full-diffusion speed while remaining below 2% maximum power error. See the
  [intermediate report](../benchmark-results/transient-acceleration/step-02-adjoint-residual-reuse.md).
- **Step 03a, persistent PCG workspaces: accepted on a large GPU case
  (2026-08-11/12).** The diffusion transient now retains one five-vector
  PCG workspace per group, and the stencil plus native polynomial
  preconditioner use persistent output/scratch arrays. The 11-group insertion
  retained exactly 17,786 inner iterations, 23 sweeps/step, and the same power;
  an allocation probe reduced traced peak allocation by 70%; Colab later
  measured a 1.035x gain at 1.16M unknowns. See the
  [intermediate report](../benchmark-results/transient-acceleration/step-03a-persistent-pcg-workspaces.md).
- **Step 04a, FP32 polynomial preconditioning: CPU accuracy accepted; initial
  GPU form rejected (2026-08-11/12).** A shadow FP32 operator evaluates only the
  Jacobi/Neumann correction inside FP64 PCG. True-residual replacement and
  automatic FP64 restart guard the outer solve. The 11-group CPU gate retained
  exactly 10,455 inner iterations and agreed in final power to `4.4e-13`; its
  expected 14.8% CPU slowdown was followed by a 4.7% loss against equivalent
  degree-1 FP64 on GPU after the fused-cast retest. See the
  [intermediate report](../benchmark-results/transient-acceleration/step-04a-mixed-precision-preconditioning.md).
- **Step 03 GPU measurements received (2026-08-12).** The CPU/GPU crossover is
  between 58k and 131k unknowns; the GPU is 1.45x faster at 131k. Persistent
  workspaces improve the 1.16M-unknown case by 1.035x, `check_every=3` by
  1.08x, and degree-1 Neumann by 1.20x. The 33k case remains 1.48x slower on
  GPU and therefore motivates fixed-block graph capture.
- **Step 04 GPU measurement: rejected (2026-08-12).** Degree-1 mixed FP32
  preserved the exact degree-1 work/power but remained 4.7% slower than
  degree-1 FP64 after fusing the residual cast and Jacobi start. Full FP32 was
  also slower and required roughly twice the Krylov work.
- **Step 03b, fixed-block CUDA-graph PCG: implementation complete, GPU gate
  pending (2026-08-12).** Allocation-free blocks now capture persistent vector
  and scalar recurrence state, with convergence checks unchanged between graph
  launches. CPU fallback preserved a real 11-group insertion exactly. The
  Colab notebook tests blocks 1 and 3 on the launch-bound 33k-unknown case. See
  the [intermediate report](../benchmark-results/transient-acceleration/step-03b-cuda-graph-pcg.md).
- **Step 05a, monolithic multigroup Krylov: accepted as an opt-in CPU
  prototype (2026-08-12).** A real flexible GMRES and matrix-free
  precursor-eliminated block operator now use three inexact group sweeps as a
  right preconditioner. On one tightly converged 11-group HP-MR step, refinement
  1/2/3 reduced CPU time by 3.34x/2.56x/2.46x and PCG work by roughly 2.2--3.0x,
  with power and flux differences following the reference fixed-point stopping
  error. The default remains fixed point until GPU memory/throughput gates pass.
  See the [intermediate report](../benchmark-results/transient-acceleration/step-05a-monolithic-multigroup.md).
- **Step 08a, adaptive/monolithic GPU readiness: T4 quick gate complete;
  production gate pending (2026-08-18).** The audit found 34 host
  synchronizations per 17-field BDF error norm, allocationful predictor
  algebra, an O(G^2) monolithic coupling fallback, and rebuilt FGMRES storage.
  Error estimates now transfer one scalar pair, BDF AXPYs and group
  contractions are fused, and a bounded persistent FGMRES workspace batches
  GPU Arnoldi projections. Runs expose batching status and workspace bytes.
  On the submitted Tesla T4 refinement-2 x 10-layer result, the latest error
  norm and fused-predictor microbenchmarks are 1.569x and 1.388x faster. Both
  adaptive methods meet the 32-step BE reference gates; batching is active and
  the bounded FGMRES workspace is 61.27 MiB. Repeated short gates reject
  tolerance-based energy Anderson (0.882x plain-PCG speed) and accept the
  reduction-free polynomial path as an opt-in T4 configuration: its median is
  2.098 s versus 2.310 s for plain PCG, a 1.101x gain. In two complete
  polynomial sessions, however, adaptive BDF is 5.7--10.9% slower than matched
  adaptive BE. The latest run takes 22/9 accepted/rejected attempts versus
  24/7, and the rejected BDF attempts consume 772 outer applications versus
  486 for BE. Reference doubling and the 580k/1.16M-unknown production cases
  remain. See the
  [intermediate report](../benchmark-results/transient-acceleration/step-08a-adaptive-bdf-gpu-readiness.md).
- **Step 09a, conservative spatial CMFD: algebra accepted, HP-MR path
  rejected (2026-08-18).** Exact full-energy `P.T A P` aggregation reduces
  FGMRES iterations from 136 to 37 on a 128-cell two-group line problem, so the
  two-level correction works when spatial low modes are present. On 11-group
  HP-MR refinements 1/2/3, however, the existing group-sweep outer count is
  already mesh-independent at 26/25/25. The best CMFD case gives 28/24/24,
  nearly doubles fine block applies, and is slower. The prototype remains as a
  diagnostic/test path but will not be GPU-ported or exposed in production.
  Phase 6 pivots to energy/source-expansion or recycled-subspace corrections.
  See the [intermediate report](../benchmark-results/transient-acceleration/step-09a-galerkin-cmfd-prototype.md).
- **Step 10a, dynamic energy-mode acceleration: backend-specific paths
  selected (2026-08-19).** Static adjoint rank-one, group-amplitude, and
  regional source spaces save at most four outer iterations and lose their
  gain to setup. Depth-one Anderson acceleration between three forward energy
  subsweeps instead follows the live coupling-error mode with no adjoint,
  coarse factorization, extra fine operator application, or GPU host sync.
  Combined with a safely inexact group-PCG tolerance of 0.1, HP-MR refinements
  1/2/3 change from 26/25/25 outer and 1,304/2,112/2,995 inner iterations to
  22/22/24 and 440/649/949. Three-sample median CPU speedups are
  1.59/1.51/1.60x. Sweep relaxation, depth-two Anderson, and higher polynomial
  degrees were rejected. On the submitted 145,200-unknown Tesla T4 case,
  tolerance-based Anderson is also rejected: its repeated median is 2.620 s
  versus 2.310 s for plain PCG. The reduction-free four-sweep fixed-polynomial
  path is accepted as an opt-in GPU configuration. The CPU path is
  exposed as `{"energy_anderson": 1, "inner_rtol": 0.1}`; the GPU candidate
  is `{"scatter_sweeps": 4, "energy_anderson": 1,
  "inner_fixed_relaxations": 1}`. Three interleaved short samples give a
  2.098 s median, 1.101x faster than plain PCG and 1.249x faster than Anderson.
  The complete adaptive run passes all accuracy and implementation gates. See
  the
  [intermediate report](../benchmark-results/transient-acceleration/step-10a-energy-sweep-anderson.md).

## Benchmark protocol

### Tier A: correctness

- focused transient, coupling, and quasi-static verification tests after each
  edit;
- full CPU suite before accepting a phase;
- identical time grid and material/control trajectory for every A/B pair;
- deterministic converged methods: maximum relative power-history difference
  <= `5e-8` and final normalized-flux L2 difference <= `5e-8`, unless a tighter
  existing regression applies;
- approximate QS/modal methods: compare against full backward-Euler diffusion
  and report maximum power error, temperature error, final shape error, and
  fallback count.

### Tier B: fast CPU development

- 11-group HP-MR bulk insertion, 2-D refinement 2, four 20 ms steps;
- slow, fast, and asymmetric 2-D coupled drum ramps from
  `examples/hpmr_quasistatic_benchmark.py`;
- record outer sweeps, inner iterations, operator rebuilds, shape solves,
  fallback intervals, and dynamic wall time;
- use at least three timed repetitions for changes whose expected wall-time
  gain is below 10%; compare medians and retain raw samples.

### Tier C: GPU throughput

- 11-group 2-D refinement 4 and 3-D refinement 4 x 10-layer cases;
- warm kernels and the memory pool before timing;
- record milliseconds/step, microseconds/inner iteration, iterations/step,
  launch/synchronization-sensitive `check_every`, memory high-water mark, and
  full coupled phase profile;
- verify GPU and CPU work counters agree for the same numerical controls.

### Tier D: production acceptance

- one-minute 3-D HP-MR simultaneous drum rotation with conduction and Doppler
  feedback;
- an asymmetric/local-drum case severe enough to exercise the IQS guard;
- compare full diffusion where affordable and otherwise use interval and mesh
  convergence plus selected full-diffusion replay windows;
- report startup separately from transient march time. A short calculation can
  otherwise hide an expensive initial forward/adjoint solve.

## Ordered implementation phases

### Phase 1: chronological initial guesses

Add an opt-in, positivity-preserving linear predictor from the last two
accepted flux states. Initialize both the group flux and the consistent fission
source from the prediction. Do not change equations or convergence tests.

Acceptance: no meaningful history/shape difference; fewer inner iterations or
fixed-point sweeps on at least two smooth transient legs; no material slowdown
on a step insertion. If linear extrapolation is unstable, test a damped
coefficient before considering a larger history basis.

### Phase 2: guard-aware adjoint reuse and fallback audit

The audit found that hard residual fallback is already decided before an IQS
corrector, so the presumed corrector/fallback duplication does not exist. The
measured asymmetric cost is instead dominated by adjoint eigen refreshes. Add
an inexpensive projected residual for the transposed eigenproblem, use it to
refresh a reused adjoint before a hard maximum age, and continue to force an
adjoint refresh on full-diffusion fallback.

Acceptance: unchanged fallback decisions; maximum power error remains within
the calibrated guarded envelope; reduced adjoint solves and dynamic time on
slow, fast, and asymmetric ramps.

### Phase 3: GPU workspaces and graph-ready Krylov batches

Make stencil, preconditioner, and Krylov operations accept persistent output
buffers. Capture a fixed block of iterations matching `check_every`, retaining
the residual test between graph launches. Autotune a small set of block sizes
per device/problem scale and cache the choice.

Status: workspace subphase accepted with a 1.035x gain on the 1.16M-unknown
GPU case. CUDA graph capture, block-size tuning, and memory-pool measurement
are implemented/pending measurement. The Colab refinement-3 result performed equivalent work but was
1.48x slower on GPU than CPU; source batching recovered only 1.08x, so the
small launch-bound case currently fails this gate.

Acceptance: identical iteration counts; lower microseconds/iteration on both a
launch-bound 2-D case and a larger 3-D case; graph construction amortized in
the reported transient duration.

### Phase 4: mixed-precision preconditioning

Keep physical state, residual evaluation, point kinetics, adjoint projection,
and convergence decisions in FP64. Evaluate FP32 stencil/preconditioner work
inside an FP64 outer correction or flexible Krylov iteration. Add residual
replacement and automatic FP64 fallback.

Acceptance: FP64 accuracy gate passes; lower GPU time and memory traffic; no
silent convergence at an FP32 residual floor.

Status: FP32 shadow preconditioner, FP64 residual replacement, and automatic
FP64 fallback are implemented and CPU-verified. The fused-cast GPU form failed
the relative performance gate by 4.7%. Mixed precision remains opt-in.

### Phase 5: monolithic multigroup Krylov solve

After analytically eliminating end-of-step precursor unknowns, express the
entire frozen-cross-section backward-Euler step as one matrix-free multigroup
linear system. Prototype restarted FGMRES with the existing energy-group
Gauss-Seidel sweep as a right preconditioner. Add an adjoint-weighted rank-one
coarse correction for the near-critical amplitude mode.

Acceptance: agreement with the existing converged fixed point; fewer total
stencil applications and reductions on the 11-group upscattering HP-MR case;
bounded restart memory for the 3-D problem.

Status: the CPU prototype, end-to-end opt-in diffusion path, dense-reference
tests, and 11-group HP-MR refinement gates are complete. Three scatter sweeps
beat one and six on total work/time; lagged fission in the preconditioner was
rejected. An adjoint-weighted adapted-deflation correction reduces group work
by 10--14%, but its fresh-adjoint startup is worthwhile only for long runs and
its CPU wall result is mixed; it remains separately opt-in and can reuse a
supplied adjoint. Fused stacked block coupling, a persistent FGMRES basis/work
space, batched GPU Arnoldi projection, and restart-memory reporting are now
implemented and CPU-verified. The Tesla T4 145k-unknown quick gate passed its
accuracy and implementation checks but showed no matched BDF-over-BE speedup.
The repeated short preconditioner gate selects a reduction-free polynomial GPU
path at 1.101x over plain PCG. Its full adaptive run passes; reference doubling
and the 580k/1.16M-unknown performance gates remain.

### Phase 6: structured multilevel/CMFD acceleration

Build a conservative tri-grid restriction/prolongation hierarchy. Start with a
geometric V-cycle for each diffusion block, then test a few-group coarse
correction for energy coupling. Preserve SPH/discontinuity-factor interface
currents when condensing coefficients. Reuse hierarchy topology and update only
state-dependent coefficients.

Acceptance: mesh-independent or substantially flatter iteration growth;
preserved fine-grid balance and interface currents; improvement in both the
initial eigen solve and transient march.

Status (2026-08-19): the conservative full-energy Galerkin/CMFD prototype is
algebraically verified but rejected for the monolithic HP-MR transient. HP-MR
already has flat outer counts, and spatial correction nearly doubles fine
block applications. Static adjoint rank-one, group-amplitude, and regional
source modes were also too weak for their setup cost. Depth-one Anderson
acceleration inside three group subsweeps, combined with `inner_rtol=0.1`, is
accepted as the dynamic CPU energy-mode alternative: it lowers HP-MR CPU time
by 34--37% with no extra fine block apply or host synchronization. The T4 gate
rejects tolerance-based Anderson and accepts the reduction-free polynomial
configuration as an explicit opt-in: three interleaved samples show a 1.101x
median gain over plain PCG and 1.249x over Anderson. This is backend-specific,
not a new universal default. With the energy path settled, the next transient
target is rejection-aware BDF work: in the latest matched run BDF saves 88
outer applications on accepted attempts but spends 286 extra on rejected
attempts. Retain recycled Krylov modes only if longer histories expose a
remaining accepted-step plateau.

### Phase 7: LRA reproduction and adaptive high-order BDF

Status (2026-08-12): **07a complete.** Diffusion fixed-point and monolithic
steps now share BDF1--6 flux/precursor history, constant or explicit nonuniform
widths, startup order ramping, and event-aligned history restart. CPU algebra,
equilibrium, coupled forwarding, and slow/fast/asymmetric 11-group HP-MR gates
are recorded in
[`step-07a-bdf-foundation.md`](../benchmark-results/transient-acceleration/step-07a-bdf-foundation.md).
No instability appeared, but high order was not consistently more accurate on
fast/local motion and cannot help when frequent piecewise-constant control
events repeatedly discard history. Backward Euler remains the default. The
next subphase is the Nordsieck local-error estimator, rejection/rollback, and
automatic order/step controller.

Status (2026-08-12): **07b LRA definition and predictor checkpoint complete.**
The original material map, corrected reflector data, rod law, two-family
kinetics, physical power edit, adiabatic heat equation, Doppler feedback, and a
CPU benchmark driver are implemented. A multilevel nonuniform BDF polynomial
predictor now feeds endpoint temperature to the cross-section build. It cuts
the assembly-mesh BDF5 first peak from 13,191 to 5,525 W/cm3 (reference 5,411)
without extra step solves. The monolithic multigroup path is required because
fixed-point group iteration fails through the prompt-supercritical peak.

Status (2026-08-12): **07c fully implicit endpoint feedback complete.** The
monolithic step can repeat neutron/feedback constituent solves without pushing
flux, precursor, or BDF history until convergence. LRA temperature Anderson
iteration averages 1.43 solves/step (maximum 4) on the assembly mesh and 1.31
at 3.75 cm. Its assembly-mesh first peak is 5,641 W/cm3, equal to Cherezov's
displayed FEMCORE result, at 1.30x the predictor-only runtime.

The numerical LRA values recorded in 07b/07c were later invalidated by the
reflector-data correction in 07f. The predictor and endpoint-coupling
conclusions remain useful, but their old peak and endpoint values must not be
used as benchmark results. See
[`lra2d-bdf-cpu/README.md`](../benchmark-results/lra2d-bdf-cpu/README.md).
The paper's normalized predictor/corrector defect and bounded Eq.-45 width
controller are now implemented and unit-tested. Constituent feedback uses a
provisional endpoint and commits through `on_step`, so a future rejected step
cannot advance external thermal history. The next numerical step is wiring
accept/reject into the monolithic march, followed by `q-1/q/q+1` order choice.

Backward Euler is the required production baseline for this phase. The first
equal-step implicit CPU comparison at 15 cm / 0.01 s gives BDF5 7.09 s and
12,878 FGMRES applications versus BE 13.51 s and 23,772 applications; BE also
damps the first peak from 5,641 to 2,810 W/cm3. At 0.005 s BE reaches only
4,054 W/cm3 in 22.96 s. Final claims must therefore use both equal-step and
matched-history-error comparisons, not infer speedup from equal `dt` alone.

The former 1.115114 region-R worth multiplier was based on the mistyped
reflector and is retired. It is neither a physical correction nor part of the
paper-faithful benchmark.

Status (2026-08-12): **07d adaptive width acceptance complete.** Monolithic
BDF2--6 can now reject a flux-defect error test without advancing neutron,
precursor, BDF, or external accepted-step state; accepted widths grow with the
bounded controller and align to events/final time. The 3.75 cm LRA case uses
250 accepted/17 rejected steps at `rtol=1e-3`, runs in 28.29 s, and matches the
fixed-grid history closely. At the nominal FEMCORE `1e-5`, the 15 cm case uses
569/26 versus Cherezov's roughly 399--403/5--7, but the norms are not yet
equivalent. Full details and the default-readiness gate are in
[`step-07d-adaptive-bdf-cpu.md`](../benchmark-results/transient-acceleration/step-07d-adaptive-bdf-cpu.md).
Those LRA timings and histories predate 07f; they exercise the controller but
are not current reference values.

Status (2026-08-18): **07e full-state defect and honest paper comparison
complete.** Adaptive acceptance now evaluates the concatenated flux,
precursor, and coupled-temperature state rather than flux alone. The LRA CLI
has a reproducible `--cherezov-controls` preset (`h0=1e-6`, `RTOL=1e-5`,
implicit feedback, BDF5 cap), publishes relative differences against both the
ANL reference and Cherezov's FEM-order ladder, and labels the ndgpu space as
cell-centred FV. On the 15 cm mesh it takes 521 accepted/24 rejected steps;
the paper reports 403/5 for first-order FEM and 399/7 for fourth-order FEM.
Its comparison contract remains current, but its numerical LRA rows also
predate the reflector correction and are superseded. See
[`step-07e-lra-full-state-benchmark.md`](../benchmark-results/transient-acceleration/step-07e-lra-full-state-benchmark.md).

Status (2026-08-18): **07f rods/control discrepancy resolved.** Cherezov Table
4 prints both reflector absorptions a factor of ten above the original
ANL/DIF3D input. The earlier model corrected only the fast value and omitted
axial buckling, creating a cancellation in rods-in and a false 377 pcm
rods-out deficit. With both original reflector values and `Bz^2=1e-4 cm-2`,
the 1 cm ndgpu rods-in result is `0.99632537` versus archived DIF3D
`0.996325`. At 3.75 cm, raw endpoint errors are -10/-64 pcm and the corrected
transient's power/temperature metrics are within 3.5% of the original
reference. See
[`step-07f-lra-control-discrepancy.md`](../benchmark-results/transient-acceleration/step-07f-lra-control-discrepancy.md).

Status (2026-08-18): **07g automatic order and matched-error baseline
complete on CPU.** Candidate full-state `q-1/q/q+1` defects now select the
largest safe next width without extra diffusion solves. At Cherezov controls,
automatic order reduces accepted LRA steps from 461 to 414 while preserving
the corrected history. Against a `1e-6` temporal reference, BDF5 at `1e-3`
and backward Euler at `1e-4` have comparable first-peak errors (-3.42% and
-2.92%); BDF is 5.18x faster with 5.02x fewer inner iterations. See
[`step-07g-automatic-order-matched-be.md`](../benchmark-results/transient-acceleration/step-07g-automatic-order-matched-be.md).

Status (2026-08-18): **07h rejection-overhead checkpoint complete.** Rejected
attempt time/width/order telemetry is now available. An error-scaled retry
with a factor-0.5 maximum removes repeated retry streaks at practical LRA
tolerance and reduces FGMRES/inner work by 5.6%; its three-sample median CPU
gain is 5.1%. A factor-0.8 retry was rejected after producing a BDF1
reject/accept oscillation. The optimized policy remains opt-in and the
paper-faithful preset retains factor-two rejection. See
[`step-07h-rejection-overhead.md`](../benchmark-results/transient-acceleration/step-07h-rejection-overhead.md).

FV spatial refinement remains useful for LRA convergence measurement, but
there is no evidence requiring a bespoke control/interface correction.

Status (2026-08-18): **07i two-dimensional HP-MR stability gate complete.**
The moving-drum harness now rejects unresolved trajectories, interpolates
pre-rasterized polar mixture weights rather than imposing artificial frame
jumps, and uses fine backward Euler as its default reference. On the
11-group refinement-2 slow-symmetric, fast-symmetric, and fast-asymmetric
maneuvers, adaptive BDF stays at orders one and two, holds maximum power error
below 0.055% and final flux error below 0.012%, and runs 1.49--2.00x faster
than the 64-step BE reference. Error-scaled rejection also reduces slow-case
inner work by 7.7% relative to factor-two retries. See
[`step-07i-hpmr-adaptive-bdf.md`](../benchmark-results/transient-acceleration/step-07i-hpmr-adaptive-bdf.md).

The next gate adds coupled thermal feedback to the moving 2-D cases, then
progresses to 3-D motion and GPU measurement.

Status (2026-08-18): **07j HP-MR speedup diagnosis and coupled gate
complete.** The previous 1.49--2.00x values compare adaptive BDF with a fine
64-step BE reference, not with accuracy-matched adaptive BE. On the 11-group
slow maneuver at equal `RTOL=1e-3`, BE and BDF both take 29 accepted steps;
BDF spends 23 at order one and has 2.9% more inner work. Forced high order and
removing knot restarts do not improve it. The mild, nearly linear maneuver
does not benefit from high order as LRA's prompt peaks do. The generic thermal
coupler now aligns adaptive steps to thermal exchanges and uses time-weighted
power accumulation. Two-group feedback-stress and physical-mass 11-group
coupled gates pass, but BDF is 5--6% slower than matched adaptive BE. See
[`step-07j-hpmr-speedup-and-coupling.md`](../benchmark-results/transient-acceleration/step-07j-hpmr-speedup-and-coupling.md).

The next CPU experiment should increase maneuver worth/duration enough to
exercise nonlinear thermal feedback. Production-scale work then proceeds to
3-D motion and GPU measurement through
[`colab_hpmr3d_adaptive_bdf_gpu.ipynb`](../notebooks/colab_hpmr3d_adaptive_bdf_gpu.ipynb).

Start by reviewing the local Cherezov--Vasiliev--Ferroukhi article and auditing
the repository for existing BDF coefficients, history state, or variable-step
machinery. Implement only the missing constant/variable-order BDF pieces, with
explicit history restart and order reduction at material/control discontinuities.

Reproduce the paper's LRA transient benchmark before using HP-MR as a tuning
case. Record its exact geometry, cross sections, thermal feedback model,
perturbation, reference time grid, power/temperature edits, BDF orders, and
reported error/cost measures from the supplied PDF. Preserve a paper-faithful
case plus a mesh/time convergence study in the repository.

Then test high-order BDF stability on a progression of HP-MR problems: uniform
shape-preserving insertion, slow symmetric drums, fast symmetric drums,
asymmetric/local drums, and finally coupled 3-D motion. Compare against a
fine-step backward-Euler reference and investigate positivity, prompt-mode
resolution, precursor accuracy, event-induced order reduction, feedback
splitting error, and whether variable order selects misleadingly large steps.
Neutron, thermal, and IQS shape grids remain independently adaptive. Compare
power-, residual-, and cross-section-based controllers rather than calibrating
only one maneuver.

Acceptance: reproduce the LRA reference within its reported tolerance; retain
stable convergence with order/time-step refinement; obtain the same HP-MR error
envelope with fewer full spatial solves; automatically reduce order/step around
rapid or discontinuous events. If high orders are not robust, retain the LRA
implementation and expose the empirically safe order limit rather than making
high-order BDF a production default.

### Phase 8: multi-mode IQS

Extend the scalar amplitude/shape split to a small biorthogonal basis containing
the fundamental mode and drum-localized/asymmetric corrections. Seed the basis
from accepted correctors/fallback snapshots, evolve reduced amplitudes, and
trigger enrichment from the projected residual outside the basis.

Acceptance: asymmetric guard error comparable with present guarded IQS, fewer
full-diffusion fallbacks, and reduced dynamic time. Retain single-mode IQS as
the low-memory/default path until this gate passes.

### Phase 9: restartable startup state

Serialize compatible forward/adjoint shapes, precursor inventory, temperature,
material/control signature, and reusable coarse-solver metadata. Reject rather
than silently reuse a state with incompatible mesh, cross sections, kinetics,
or boundary conditions.

Acceptance: restarted and fresh histories agree; short parameter studies no
longer pay an avoidable steady-state solve.

## Literature basis and access notes

- Saad, *A Flexible Inner-Outer Preconditioned GMRES Algorithm*, SIAM J. Sci.
  Comput. 14 (1993), [doi:10.1137/0914028](https://doi.org/10.1137/0914028).
  This is the basis for storing the preconditioned basis explicitly so the
  inexact energy-group PCG sweep may change from one Arnoldi vector to another.
- Tang, Nabben, Vuik, and Erlangga, *Comparison of Two-Level Preconditioners
  Derived from Deflation, Domain Decomposition and Multigrid Methods*, J. Sci.
  Comput. 39 (2009),
  [doi:10.1007/s10915-009-9272-6](https://doi.org/10.1007/s10915-009-9272-6).
  Its adapted-deflation/two-level framework motivates the rank-one coarse-mode
  correction; ndgpu uses the reactor adjoint as an oblique physical weighting.
- Austin, Chalmers, and Warburton, *Initial Guesses for Sequences of Linear
  Systems in a GPU-Accelerated Incompressible Flow Solver*, SIAM J. Sci.
  Comput. 43 (2021), [doi:10.1137/20M1368677](https://doi.org/10.1137/20M1368677),
  [open preprint](https://arxiv.org/abs/2009.10863). This motivates testing a
  small stabilized polynomial history before a communication-heavy projection
  basis.
- Parks, de Sturler, Mackey, Johnson, and Maiti, *Recycling Krylov Subspaces for
  Sequences of Linear Systems*, SIAM J. Sci. Comput. 28 (2006),
  [doi:10.1137/040607277](https://doi.org/10.1137/040607277),
  [repository record](http://hdl.handle.net/10919/48161). This is the basis for
  a later GCRO-DR/recycling experiment if simple extrapolation is beneficial.
- Ekelund, Markidis, and Peng, *Boosting Performance of Iterative Applications
  on GPUs: Kernel Batching with CUDA Graphs* (2025),
  [open preprint](https://arxiv.org/abs/2501.09398). It supports benchmarking
  captured iteration batches rather than assuming that a whole dynamic solver
  can be captured profitably.
- Carson and Higham, *A New Analysis of Iterative Refinement and its
  Application to Accurate Solution of Ill-Conditioned Sparse Linear Systems*,
  SIAM J. Sci. Comput. 39 (2017),
  [open accepted manuscript](https://eprints.maths.manchester.ac.uk/2604/).
  Mixed precision will therefore be used as an inexact correction or
  preconditioner with high-precision residuals, not as an unchecked dtype swap.
- Yoon and Joo, *Two-Level Coarse Mesh Finite Difference Formulation with
  Multigroup Source Expansion Nodal Kernels*, JNST 45 (2008),
  [doi:10.1080/18811248.2008.9711467](https://doi.org/10.1080/18811248.2008.9711467).
  A user-supplied full article is available locally in `literature/` and must
  be reviewed before implementing the energy-condensation stage.
- Cherezov, Vasiliev, and Ferroukhi, *Application of Backward Differential
  Formula and Anderson's Method for Multigroup Diffusion Transient Equation*,
  Annals of Nuclear Energy 210 (2025),
  [doi:10.1016/j.anucene.2024.110837](https://doi.org/10.1016/j.anucene.2024.110837).
  A user-supplied full article is available locally in `literature/`; it is
  relevant to the BDF and block-Gauss-Seidel comparison.
- Carreño, Vidal-Ferràndiz, Ginestar, and Verdú, *Adaptive Time-Step Control
  for Modal Methods to Integrate the Neutron Diffusion Equation*, Nuclear
  Engineering and Technology 53 (2021),
  [doi:10.1016/j.net.2020.07.004](https://doi.org/10.1016/j.net.2020.07.004),
  [open manuscript](https://riunet.upv.es/bitstream/handle/10251/182189/CarrenoVidal-FerrandizGinestar%20-%20Adaptive%20time-step%20control%20for%20modal%20methods%20to%20integrate%20the%20ne....pdf?isAllowed=y&sequence=1).
  It directly supports residual- and cross-section-based adaptive modal updates.
- Devooght and Mund, *Generalized Quasi-Static Method for Nuclear Reactor
  Space-Time Kinetics*, Nuclear Science and Engineering 76 (1980),
  [doi:10.13182/NSE80-A19288](https://doi.org/10.13182/NSE80-A19288). A
  user-supplied full article is available locally in `literature/`; its
  multiple-amplitude formulation is relevant before finalizing multi-mode IQS.
