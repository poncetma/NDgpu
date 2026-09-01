# Multi-GPU diffusion bring-up

The accepted Cartesian and triangular implementations provide MPI rank
placement, global reductions, one-axis spatial decomposition, blocking halo
exchange, rank-local cross-section fields and operators, distributed
PCG/power iteration, fixed-point transient stepping, and explicit gathered
output. Track subsequent optimization and physics extensions in
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
fixed-point transient API localizes every material/mixing map returned by
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

Current limitations are explicit: distributed transients support fixed-step,
fixed-point stepping with global kinetics. Adaptive BDF, monolithic stepping,
feedback callbacks, `xs_update_at`, and `initial_steady` handoff are rejected.
Multi-rank triangular discontinuity factors are also deferred because their
asymmetric partition-interface coefficients require a separate exchange.
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
Do not claim multi-GPU speedup until collective/halo profiling and CUDA-aware
communication improve this ratio.

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
