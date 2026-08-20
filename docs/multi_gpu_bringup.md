# Multi-GPU diffusion bring-up

The first implementation slice provides MPI rank placement, communication
probes, global reduction providers, and spatial partition metadata. It does
not yet provide a distributed diffusion operator or eigenvalue solver. Track
the remaining phases in `multi_gpu_diffusion_plan.md`.

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
