# Multi-GPU diffusion bring-up

The accepted Cartesian implementation provides MPI rank placement, global
reductions, one-axis spatial decomposition, blocking halo exchange, rank-local
cross-section fields and operators, distributed PCG/power iteration, and
explicit gathered output. The triangular HP-MR row decomposition remains
Phase 3; track it in `multi_gpu_diffusion_plan.md`.

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

The generated GH module `openmpi/5.0.7-iw2c-GH200-gpu` currently references
stale dependency module names. The gate therefore validates and uses the
intact site installation prefix directly; its libraries retain full dependency
RPATHs. Remove this workaround once PSI regenerates the aarch64 module tree.

## Environment rules

- Run one MPI rank per allocated GPU.
- Build or install `mpi4py` against the same MPI loaded when the job runs.
- Keep CuPy matched to the CUDA toolkit and node architecture.
- Use `host-staged` until the site MPI passes the direct device-buffer probe.
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

## Output

Each rank emits one `NDGPU_MPI_RANK` JSON record with host, local rank, GPU
identity, MPI version, backend, and communication mode. Rank zero emits one
`NDGPU_MPI_PROBE` record with correctness status and measured ring bandwidth.
These prefixes are stable so Slurm logs can be parsed without depending on
human-readable messages.
