# Multi-GPU diffusion bring-up

Status: development concluded on 2026-09-04. This branch records the accepted
correctness implementation, performance evidence, and production limits; no
additional multi-GPU development is planned here.

The accepted Cartesian and triangular implementations provide MPI rank
placement, global reductions, one-axis spatial decomposition, blocking halo
exchange, rank-local cross-section fields and operators, distributed
PCG/power iteration, fixed-point and monolithic transient stepping, and
explicit gathered output. Track subsequent optimization and physics extensions in
`multi_gpu_diffusion_plan.md`.

## Cartesian solver API

The explicit distributed solver accepts size-one and multi-rank CPU/GPU MPI
communicators:

```python
from mpi4py import MPI
from ndgpu import DistributedDiffusionEigenSolver

solver = DistributedDiffusionEigenSolver(
    problem.grid,
    problem.materials,
    problem.material_map,
    active=problem.active,
    mask_bc=problem.mask_bc,
    communicator=MPI.COMM_WORLD,
    decomposition="slab",
    device="gpu",
)
result = solver.solve(verbose=True)
flux = result.gather_flux(root=0)
```

`result.local_flux` and all solve fields are rank-local. Global flux
construction is an explicit collective `gather_flux()` call that returns a
host array on the selected root and `None` elsewhere. The solver accepts
global setup maps during bring-up or already-sliced maps with the local
partition shape. Only the global grid metadata and small material tables are
replicated during the solve.

The Cartesian gate is `examples/distributed_cartesian_eigen_gate.py`.
Reproducible launchers are:

- `slurm/stage_and_submit_phase2_cpu_eigen_gates.sh` for two and four CPU
  ranks.
- `slurm/stage_and_submit_phase2_gh_eigen_gate.sh` for two GH200s.

The accepted GH launcher uses `gh-hourly`: `gh-interactive` limits a job to
one GPU. Its `srun` step includes both `--gpus-per-task=1` and
`--gpu-bind=single:1`; omitting explicit binding can make rank placement
ambiguous. Device uniqueness is checked through PCI bus identity before any
cross-section fields are allocated.

## Triangular and transient APIs

`DistributedTriDiffusionEigenSolver` partitions triangular-grid rows and
supports both 2-D triangles and extruded 3-D prisms. It retains active masks,
mask boundary conditions, and volume-mixed drum cells. The corresponding
transient API localizes every material/mixing map returned by
`problem_at(t)`:

```python
from mpi4py import MPI
from ndgpu import DistributedTriTransientSolver

solver = DistributedTriTransientSolver(
    problem.grid,
    problem_at,
    kinetics,
    active=problem.active,
    mask_bc=problem.mask_bc,
    communicator=MPI.COMM_WORLD,
    decomposition="rows",
    device="gpu",
)
result = solver.solve(
    t_end=1.0, dt=0.01, verbose=MPI.COMM_WORLD.Get_rank() == 0)
flux = result.gather_flux(root=0)
precursors = result.gather_precursors(root=0)
```

`result.local_flux` and `result.local_precursors` remain rank-local. Both
gather methods are collective and return host arrays only on the selected
root.

Current limitations are explicit: distributed transients support fixed-step
fixed-point stepping and matrix-free monolithic FGMRES with global kinetics.
Adaptive BDF, feedback callbacks, and `xs_update_at` are rejected. A distributed
`initial_steady` can be reused when it has the same context and partition as
the transient solver. Multi-rank triangular discontinuity factors are also
deferred because their asymmetric partition-interface coefficients require a
separate exchange.
Whole-core rebalance uses plain Picard automatically; Anderson acceleration is
disabled for that rational source map.

Reproducible HPMR gates are:

- `slurm/stage_and_submit_phase3_gh_tri_gate.sh` for triangular eigenvalue
  decomposition on two GH200s.
- `slurm/stage_and_submit_phase4_cpu_transient_gates.sh` for two and four CPU
  transient ranks.
- `slurm/stage_and_submit_phase4_gh_transient_gate.sh` for a two-GH200 moving
  drum transient.
- `slurm/stage_and_submit_phase5_gh_3d_transient_gate.sh` for the 11-group,
  extruded 3-D eigenvalue and moving-drum transient gates. Pass `--eigen-only`
  or `--skip-eigen` when repeating one half of the combined diagnostic.

The retained Phase 1 size-one acceptance executable is
`examples/distributed_size_one_gate.py`. The corresponding reproducible Slurm
launchers are `slurm/stage_and_submit_phase1_cpu_gate.sh` and
`slurm/stage_and_submit_phase1_gpu_gate.sh`.

Accepted stacks on 2026-08-20:

- Merlin CPU: OpenMPI 4.1.6 with `mpi4py 4.1.2` in the isolated
  `/data/scratch/shared/poncet_m/ndgpu-mpi4py-x86-py312` target.
- Grace-Hopper: OpenMPI 5.0.7 with `mpi4py 4.1.2` installed offline in
  `/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1`.

Accepted multi-GPU Cartesian solve on 2026-09-01:

- Two GH200s on `gpu003`, OpenMPI 5.0.7, CuPy 13.6.0, `mpi4py 4.1.2`, explicit
  host-staged communication, job `202426`.

Accepted triangular solves on 2026-09-01:

- Polar volume-mixed r32 HPMR eigenvalue on two GH200s, job `202433`.
- Polar volume-mixed r16 HPMR moving-drum transient on two GH200s, job
  `202439`.
- Physical r4/nz10 11-group HPMR prism eigenvalue on two GH200s, job `202569`.
- Physical r4/nz10 11-group HPMR prism moving-drum transient on two GH200s,
  job `202532`.

The 3-D host-staged transient is a correctness baseline, not a performance
result: it took 117.4 s on two GPUs versus 27.9 s for the single-GPU reference.
No speedup was claimed at this stage. The later CUDA-aware, batched-halo, and
domain block-Jacobi results below supersede this early performance baseline.

Accepted GH200 communication probes on 2026-09-01:

- Jobs `202735` (host-staged) and `202736` (CUDA-aware) both completed `0:0`
  on two GH200s with OpenMPI 5.0.7.
- An 8 MiB one-way exchange reached 1.10 GB/s with host staging and
  14.69 GB/s with direct device buffers, a 13.4x bandwidth improvement.
- Device-scalar all-reduce latency was 75.8 us host-staged and 72.1 us
  CUDA-aware. The small change confirms that collective count, rather than
  scalar payload bandwidth, is the next PCG bottleneck.

The Phase 6 performance path adds communication counters, distributed
steady-state reuse, and an opt-in Chronopoulos-Gear PCG recurrence through
`linsolve_kwargs={"single_reduction": True}`. The recurrence combines its
scalar products into one all-reduce per iteration and retains one additional
persistent vector per energy-group workspace. Keep it opt-in until the HPMR
throughput and solution-history gates are accepted.

`DistributedContext.from_mpi(..., batched_halos=True)` is a second experimental
control. It posts both slab-face receives and sends nonblocking, completes them
together, overlaps transfer with the owned-domain stencil, and reduces the
CUDA-aware path from two synchronization pairs per operator application to
one. Unit tests cover both physical-boundary ranks;
production use still requires the real-MPI GH probe and transient timing gate.

## Refinement-informed final gates

The 2026-09-02 11-group drum-worth study established that 0.241 cm drum cells
(effective r64) are required for monotonic all-drum worth around 90 degrees.
The preferred local mesh is global r8 plus three drum-band levels: 169,674 2-D
cells versus 1,351,680 for uniform r64, with a 2.91 pcm error in the 90-to-95
degree worth. Full evidence and raw results are in
`docs/hpmr_drum_refinement.md` and
`benchmark-results/hpmr-drum-refinement/`.

The earlier structured fallback was uniform r64/nz10: 13,516,800 spatial cells
and 148,684,800 11-group unknowns. Phase 7 removes the need to pay that cost:
the production target is the local r8+3 mesh at nz10/nz20
(18,664,140/37,328,280 unknowns), with r16+2 as the accuracy check
(24,024,000/48,048,000 unknowns).

Phase 7 implements the target mesh without forcing its nonconforming radial
faces through the general 3-D face assembler. `ExtrudedMeshGrid` retains the
validated 2-D unstructured connectivity and tensor-products it with uniform
axial layers. `DistributedExtrudedMeshTransientSolver` divides those layers
among MPI ranks, exchanges complete radial planes, and reuses the common
distributed reductions, batched halo path, and single-reduction PCG.

The two-rank CPU correctness gate passed as Merlin job `8769714` on 2026-09-02:

- nonconforming global r1 plus one local level, `nz=10`, two groups;
- eigenvalue difference from serial `4.4e-16`;
- maximum transient power difference `1.5e-12`;
- final flux relative L2 difference `2.4e-12`;
- identical 126 fixed-point sweeps for the moving-drum step.

The gate is `examples/distributed_extruded_hpmr_gate.py`, launched by
`slurm/stage_and_submit_phase7_cpu_extruded_gate.sh`. The two-GH200 11-group
gate passed as gmerlin7 job `203392` on 2026-09-03. It used distinct GH200s,
CUDA-aware MPI, global r2 plus one local drum level, `nz=10`, and 192,720
unknowns. Relative initial/final flux differences from the serial GPU solve
were `1.83e-8`/`5.59e-8`; maximum power difference was `6.12e-8`, and both
paths took 95 fixed-point sweeps. Strict `1e-10` source convergence is required
for the independently computed initial shapes; looser steady tolerances were
the source of the earlier `2.6e-7` comparison floor.

The accepted gate is a correctness result, not a speedup result. Its transient
step took 31.9 s and issued 31,739 all-reduces plus 31,628 halo exchanges.
Distributed operators therefore expose a communication-free principal-block
apply for polynomial preconditioning. Higher Neumann degrees can reduce global
PCG iterations without adding halo traffic for the preconditioner stencils.
On the two-GH200 r2+1, one-step tuning case, degree 1 reduced inner iterations
from 10,449 to 6,011 and wall time from 16.57 s to 10.79 s relative to Jacobi,
a 35% improvement. Degree 2 reached 5,445 iterations but took 12.11 s because
the extra local stencil outweighed the saved collectives. The Phase 7
performance launcher therefore defaults to degree 1 and accepts rank count,
degree, refinement, local levels, step count, and axial layers as positional
controls. Production conclusions use the r8+3 1/2/4-rank timings below rather
than this small tuning case.

The fused extruded ELL kernel subsequently reduced the r8+3/nz10 one-GPU
fixed-point step from 83.4 s to 25.6 s. That improvement exposed the MPI cost:
the same fused step took 53.5 s on two GPUs and 49.3 s on four. A matrix-free
monolithic multigroup solve reduced the one-GPU step again to 15.6 s, but its
first distributed implementation still ran every inner group PCG against the
global operator. It took 28.6 s on two GPUs and 32.3 s on four while issuing
about 28,000 collectives and halo exchanges per rank.

The current monolithic preconditioner is therefore explicitly a
non-overlapping domain block-Jacobi (additive Schwarz) right preconditioner.
Each GPU approximately solves the principal block of its axial subdomain;
interface diagonal terms are retained, remote off-diagonal terms are omitted,
and no MPI operation occurs in an inner group PCG. Flexible GMRES applies the
fully coupled multigroup operator with all group halo planes stacked into one
message and resolves the interface error globally. Merlin CPU job `8783384`
passed the two-rank transient comparison with a `7.58e-13` final-flux relative
error. Halo calls fell from 585 to 36 and solve time from 0.120 s to 0.073 s.
The two-GH200 strict gate, job `203493`, then passed with a `3.07e-8`
final-flux relative error and only 78 halo calls, down from 5,562 for the first
monolithic distributed implementation.

The r8+3/nz10 strong-scaling result is nevertheless negative on GH200. Jobs
`203496`, `203497`, and `203498` took 12.65, 15.60, and 16.86 s on one, two,
and four GPUs. The decomposition increased outer FGMRES work from 29 to
30/43 applications, and two-to-five axial layers per GPU do not provide enough
local stencil work to amortize GPU launch and synchronization costs. Treat one
GH200 as the default for `nz=10`.

The realistic r8+3/nz20 mesh does have a narrow multi-GPU crossover. The
accepted performance configuration uses three energy-scatter sweeps,
depth-one energy Anderson acceleration, and a `0.1` local-PCG relative
tolerance; outer FGMRES still enforces the requested true global residual.
One-step jobs `203539`, `203540`, and `203541` took 12.56, 8.79, and 8.85 s on
one, two, and four GH200s. Two GPUs therefore delivered 1.43x, while four
provided no further gain. Five-step jobs `203549` and `203550` confirmed the
result: transient time fell from 38.72 to 25.69 s, or from 7.743 to 5.137 s per
step (1.51x), and final powers differed by `9.94e-8`.

The sustained 50-step gate, jobs `203653` and `203654`, reduced transient time
from 295.54 to 210.29 s (5.911 to 4.206 s per step, 1.41x). Setup, steady
state, and transient together fell from 346.51 to 288.99 s (1.20x), so this
case passes the end-to-end rather than only the timed-kernel criterion. Final
powers differed by `1.51e-7`. The jobs ran on different GH nodes and should
not be read as a precision hardware benchmark, but the sustained result is
consistent with both earlier short gates.

The distributed eigenvalue solve is slower, so setup plus steady state plus
five transient steps took 89.77 s on one GPU and 92.81 s on two. The measured
per-step savings amortize that startup penalty at approximately seven steps.
For a fixed perturbed operator, the production rule is consequently one GH200
for `nz=10`, two GH200s for a long `nz=20` pure-diffusion transient, and four
only as an unproven capacity option for substantially larger meshes. This is
useful two-GPU partitioning, not general strong scaling. At the sustained
solve rates, 200 s at fixed `dt=0.01 s` projects to approximately 32.8 wall
hours on one GH200 and 23.4 hours on two. This is a solver-only upper bound on
the benefit: it excludes changing geometry and Krylov work over a full drum
trajectory. The two-GPU solve is about 70% parallel-efficient, so it reduces
elapsed time while consuming more total GPU-hours.

A continuously rotating-drum cross-check exposes that excluded cost. The
performance driver now supports `--drum-motion linear-ramp`, caches repeated
callback times, and reports exact polar volume-mixing time per rank. Ten-step
jobs `203657` and `203658` advanced all drums from 90.0 to 93.88 degrees on the
same GH200 node. Each rank recomputed ten radial maps; this took 37.77 s on one
rank and a maximum 40.31 s on two. Total transient time improved only from
124.17 to 111.22 s (1.12x), while setup plus steady state plus transient
regressed from 170.81 to 186.95 s. Final powers differed by `1.13e-7`.

Therefore, default continuously rotating, exact-volume-mixed HPMR transients
to one GH200. Two GPUs are justified only when the operator is reused long
enough to amortize startup and map construction, or after the radial fraction
update is vectorized, tabulated, or otherwise removed from every time step.
Four GH200s are not practical for either tested production mesh. More MPI
preconditioner work is lower priority than eliminating the replicated host
geometry bottleneck.

An A100 host-staged cross-check illustrates that this conclusion is hardware
and subdomain-size dependent. At `nz=10`, two A100s improved 29.11 s to
17.67 s while four regressed to 39.57 s. At `nz=20`, one/two/four A100s took
38.93/25.78/22.94 s with the original inner tolerance. With the accepted
configuration, five-step transient times were 77.64/49.65/35.32 s, giving
1.56x and 2.20x speedups. The slower local A100 solve leaves more computation
to amortize communication. Direct CuPy buffers are not usable with the tested
A100 OpenMPI 4.1.6 stack (`cxil_map`/`MPI_ERR_OTHER`), so these are deliberately
reported as conservative host-staged results rather than merged with the
CUDA-aware GH200 curve.

The real two-rank CPU/OpenMPI 4.1.6 protocol gate passed as Merlin job
`8759584` (`0:0`, 39 s). It completed 200 nonblocking 32 KiB halo exchanges
at 16.9 us per exchange and verifies request/tag correctness outside the unit
test mock. The later GH200 gates provide the CUDA-aware acceptance evidence.

The generated GH module `openmpi/5.0.7-iw2c-GH200-gpu` currently references
stale dependency module names. The gate therefore validates and uses the
intact site installation prefix directly; its libraries retain full dependency
RPATHs. Remove this workaround once PSI regenerates the aarch64 module tree.

## Environment rules

- Run one MPI rank per allocated GPU.
- Build or install `mpi4py` against the same MPI loaded when the job runs.
- Keep CuPy matched to the CUDA toolkit and node architecture.
- Use `cuda-aware` with the accepted GH200/OpenMPI 5.0.7 stack. Retain
  `host-staged` as the explicit fallback for any other MPI installation.
- Do not install an unrelated MPI wheel into the Grace-Hopper environment.

The optional Python dependency is available as `ndgpu[mpi]`, but the site MPI
must already be loaded and discoverable while `mpi4py` is built.

## CPU smoke test

Inside a two-task CPU allocation:

```bash
export NDGPU_REPO="$PWD"
export NDGPU_PYTHON_BIN=/path/to/mpi-enabled/python
export NDGPU_DISTRIBUTED_DEVICE=cpu
bash slurm/run_ndgpu_multi_gpu.sh --elements 65536 --iterations 10
```

This exercises direct NumPy-buffer point-to-point traffic and all-reduce. It
is the fastest way to verify MPI process launch independently of CUDA.

## GPU host-staged probe

Request at least two tasks and one GPU per task, then run:

```bash
export NDGPU_REPO="$PWD"
export NDGPU_PYTHON_BIN=/path/to/mpi-cupy-python
export NDGPU_DISTRIBUTED_DEVICE=gpu
export NDGPU_MPI_COMMUNICATION=host-staged
bash slurm/run_ndgpu_multi_gpu.sh --elements 1048576 --iterations 20
```

This is the safe correctness fallback. CuPy send buffers are copied to host
before MPI and receive buffers are copied back explicitly. The summary must
report `"status": "passed"` and `"communication_mode": "host-staged"`.

## CUDA-aware probe

Only after confirming that the loaded MPI advertises CUDA support, repeat the
same allocation with:

```bash
export NDGPU_MPI_COMMUNICATION=cuda-aware
bash slurm/run_ndgpu_multi_gpu.sh --elements 1048576 --iterations 20
```

This passes contiguous CuPy buffers directly to `MPI.Sendrecv` and
`MPI.Allreduce`. A Python exception, MPI error, incorrect value, or process
crash means the stack has not passed and production runs must remain
host-staged. Selection is never automatic because handing a device pointer to
an incompatible MPI can terminate the process rather than raise safely.

The reproducible paired probe is
`slurm/stage_and_submit_phase6_gh_communication_probes.sh`. Long-transient
throughput comparisons use
`slurm/stage_and_submit_phase6_gh_transient_performance.sh`; their timer starts
after the initial distributed eigenstate is solved and reused, resets all
communication counters, and does not gather global flux.

## Output

Each rank emits one `NDGPU_MPI_RANK` JSON record with host, local rank, GPU
identity, MPI version, backend, and communication mode. Rank zero emits one
`NDGPU_MPI_PROBE` record with correctness status and measured ring bandwidth.
These prefixes are stable so Slurm logs can be parsed without depending on
human-readable messages.
