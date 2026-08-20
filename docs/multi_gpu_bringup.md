# Multi-GPU diffusion bring-up

The first implementation slice provides MPI rank placement, communication
probes, global reduction providers, and spatial partition metadata. It does
not yet provide a multi-rank diffusion operator. Track the remaining phases in
`multi_gpu_diffusion_plan.md`.

## Phase 1 solver API

The explicit distributed solver API is available for size-one CPU and GPU
communicators. It exercises the distributed reductions and result ownership
without claiming that spatial decomposition is ready:

```python
from mpi4py import MPI
from ndgpu import DistributedTriDiffusionEigenSolver

solver = DistributedTriDiffusionEigenSolver(
    problem.grid,
    problem.materials,
    problem.material_map,
    active=problem.active,
    mask_bc=problem.mask_bc,
    communicator=MPI.COMM_WORLD,
    decomposition="rows",
    device="gpu",
)
result = solver.solve(verbose=True)
flux = result.gather_flux(root=0)
```

`result.local_flux` is always the owned field. Global flux construction is an
explicit `gather_flux()` call. For now, construction with more than one rank
fails before cross-section fields are allocated; Phase 2 and Phase 3 will lift
that guard after their halo operators pass correctness gates.

The Phase 1 size-one acceptance executable is
`examples/distributed_size_one_gate.py`. The corresponding reproducible Slurm
launchers are `slurm/stage_and_submit_phase1_cpu_gate.sh` and
`slurm/stage_and_submit_phase1_gpu_gate.sh`.

Accepted stacks on 2026-08-20:

- Merlin CPU: OpenMPI 4.1.6 with `mpi4py 4.1.2` in the isolated
  `/data/scratch/shared/poncet_m/ndgpu-mpi4py-x86-py312` target.
- Grace-Hopper: OpenMPI 5.0.7 with `mpi4py 4.1.2` installed offline in
  `/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1`.

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
