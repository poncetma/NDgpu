# Transient acceleration reports

This directory records each independently benchmarked optimization from the
[transient performance implementation plan](../../docs/transient_performance_plan.md).
Rejected experiments are retained here so that their numerical behavior and
cost are not rediscovered later.

The consolidated, rendered LRA validation/performance document is
[Adaptive, Variable-Order BDF for Coupled Reactor Transients](../lra2d-adaptive-bdf-report/adaptive-bdf-lra-benchmark.pdf).

| Step | Experiment | Decision |
|---|---|---|
| 01 | [Raw-flux chronological initial guess](step-01-chronological-initial-guess.md) | Rejected |
| 02 | [Guard-aware adjoint reuse](step-02-adjoint-residual-reuse.md) | Accepted on CPU |
| 03a | [Persistent PCG workspaces](step-03a-persistent-pcg-workspaces.md) | CPU prerequisite accepted; GPU gate pending |
| 03b | [Fixed-block CUDA-graph PCG](step-03b-cuda-graph-pcg.md) | Implemented; CPU fallback verified, GPU gate pending |
| 04a | [FP32 polynomial preconditioning](step-04a-mixed-precision-preconditioning.md) | Accuracy accepted; revised GPU gate pending |
| 05a | [Monolithic multigroup transient solve](step-05a-monolithic-multigroup.md) | CPU prototype accepted; GPU gate pending |
| 07a | [BDF1--6 history, nonuniform steps, and event restart](step-07a-bdf-foundation.md) | CPU foundation accepted opt-in; adaptive controller pending |
| 07d | [Adaptive BDF width acceptance](step-07d-adaptive-bdf-cpu.md) | CPU checkpoint; fixed maximum order |
| 07e | [Full-state LRA defect and comparison contract](step-07e-lra-full-state-benchmark.md) | CPU accepted; automatic order and spatial gate pending |
| 07f | [LRA reflector erratum and control-worth resolution](step-07f-lra-control-discrepancy.md) | Discrepancy closed; corrected benchmark in reference envelope |
| 07g | [Automatic BDF order and matched-error backward Euler](step-07g-automatic-order-matched-be.md) | CPU accepted; 5.18x matched-error speedup |
| 07h | [Error-scaled rejected-step policy](step-07h-rejection-overhead.md) | Opt-in accepted; 5.6% practical-tolerance work reduction |
| 07i | [Adaptive BDF on moving HP-MR drums](step-07i-hpmr-adaptive-bdf.md) | 2-D 11-group stability gate passed; 1.49--2.00x vs fine BE |
| 07j | [HP-MR speedup diagnosis and adaptive thermal coupling](step-07j-hpmr-speedup-and-coupling.md) | Coupled gate passed; no BDF-over-BE win on mild motion |
| 08a | [Adaptive/monolithic GPU readiness](step-08a-adaptive-bdf-gpu-readiness.md) | T4 quick gate complete; larger production gate pending |
| 09a | [Conservative Galerkin/CMFD prototype](step-09a-galerkin-cmfd-prototype.md) | Algebra accepted; spatial HP-MR path rejected |
| 10a | [Dynamic energy-mode acceleration](step-10a-energy-sweep-anderson.md) | CPU 1.51--1.60x; T4 polynomial 1.101x vs plain accepted opt-in |
