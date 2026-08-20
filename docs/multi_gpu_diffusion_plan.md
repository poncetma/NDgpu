# Multi-GPU standalone diffusion development plan

Status: Phase 1 accepted on Merlin CPU and Grace-Hopper GPU

Date: 2026-08-20

Phase 1 acceptance evidence (2026-08-20):

- Merlin CPU job `8544402`, OpenMPI 4.1.6, one real Slurm/MPI rank: passed.
- GH200 job `196908`, OpenMPI 5.0.7, CuPy FP64, one real Slurm/MPI rank:
  passed in explicit `host-staged` mode.
- Cartesian serial/distributed histories were identical at 21 outer and 92
  inner iterations, with `k_eff = 0.7347185328944782`.
- Triangular serial/distributed histories were identical at 9 outer and 114
  inner iterations, with `k_eff = 1.268919614519496` on CPU and the expected
  last-bit backend difference (`1.2689196145194954`) on GH200.
- Both gates required array-equal serial/distributed flux on the same backend.

## Objective

Add a production-quality multi-GPU execution path for standalone multigroup
diffusion eigenvalue calculations. The first production target is the
body-fitted 2-D HP-MR problem solved by `TriDiffusionEigenSolver`, followed by
extruded 3-D triangular meshes and the Cartesian `DiffusionEigenSolver`.

The implementation must preserve the current discretization, material mixing,
boundary conditions, group sweep, convergence criteria, and eigenvalue. It
must distribute both memory and work. Merely replicating the full problem on
each GPU, or assigning independent drum angles to different GPUs, is useful
throughput parallelism but is not a multi-GPU solve and is outside this plan.

## Scope

Included in the first production release:

- Forward k-eigenvalue diffusion.
- Adjoint k-eigenvalue diffusion after the forward path is stable.
- Two-group and general multigroup data.
- Cartesian finite-volume meshes.
- Body-fitted triangular 2-D meshes and extruded triangular-prism 3-D meshes.
- Active masks, vacuum/reflective/zero-flux boundaries, and volume-mixed cells.
- FP64 first, with the existing FP32 option restored after FP64 validation.
- Single-node and multi-node NVIDIA GPU execution under Slurm.
- A host-staging communication fallback for installations without CUDA-aware
  MPI, clearly reported as a fallback rather than silently selected.

Explicitly deferred:

- Transients, IQS, thermal feedback, noise, S_N, SP3, SDPN, and hybrid methods.
- Distributed mesh adaptation or dynamic repartitioning.
- Multi-GPU execution from one Python process.
- Fault tolerance and checkpoint/restart during an eigenvalue solve.
- Energy decomposition across GPUs.

The abstractions should not prevent later transient or SPN support, but those
paths must not be changed merely to land the first diffusion implementation.

## Current execution path

The relevant serial/single-GPU path is:

1. `Fields` in `ndgpu/solver.py` maps material and mixing data to full-domain
   per-cell arrays on a NumPy or CuPy backend.
2. `DiffusionEigenSolver` or `TriDiffusionEigenSolver` constructs one
   matrix-free within-group operator per energy group.
3. `_PowerIterationSolver.solve()` performs the multigroup power iteration.
4. Each energy group is swept in Gauss-Seidel order and solved with PCG by
   default.
5. `GroupOperator.apply()` or `TriGroupOperator.apply()` applies a fused local
   stencil on GPU.
6. `ndgpu.linalg.pcg()` computes all Krylov reductions through local CuPy or
   NumPy reductions.
7. Fission normalization, source convergence, and Anderson acceleration also
   reduce full-domain arrays directly through the local array backend.

This division is favorable for domain decomposition. Energy coupling is local
to a cell, while spatial coupling is nearest-neighbor. The changes are
concentrated in operator application, global reductions, distributed field
construction, and result ownership.

## Why MPI with one rank per GPU

Use one MPI rank per GPU and one spatial subdomain per rank.

This model is preferred because:

- It works across nodes, which is required when a node exposes one GH200.
- MPI supplies nearest-neighbor point-to-point transfers and global
  collectives in one portable interface.
- CUDA-aware MPI can consume contiguous CuPy buffers through their CUDA array
  interface, avoiding device-to-host copies when the MPI build supports it.
- Each process has one active CuPy device and one allocator, avoiding complex
  cross-device stream and memory ownership inside Python.
- The same distributed algorithm can run under CPU MPI for fast CI and
  correctness testing.

NCCL is not the primary control plane. It may later accelerate device
collectives, but it does not replace the decomposition and point-to-point
protocol needed across arbitrary Slurm allocations. NVSHMEM, Dask, and a
single-process multi-device implementation are not first-stage dependencies.

The optional package dependency should be exposed as an `mpi` extra containing
`mpi4py`. The system MPI and its CUDA support remain an environment concern;
they must not be replaced by an unrelated MPI wheel at runtime.

## Decomposition strategy

### Energy ownership

Every rank owns all energy groups for its local spatial cells. The current
Gauss-Seidel group sweep therefore remains unchanged in order and physics.
Energy decomposition would perform poorly for the two-group HP-MR case and
would require full spatial field transfers between every group solve.

### Initial triangular decomposition

Partition the first triangular-grid axis into contiguous row slabs. For a
global shape `(nrows, ncols, 2)` each rank owns
`[row_start:row_stop, :, :]`. For an extruded shape
`(nrows, ncols, 2, nz)`, each rank owns the same complete axial extent.

This is a particularly clean cut for `TriGroupOperator`:

- Hypotenuse couplings remain inside one row.
- Horizontal face-family couplings remain inside one row.
- Only the vertical face family crosses a row partition.
- One row of scalar flux per neighboring rank is sufficient as a halo.
- The 3-D axial stencil remains local when the complete axial extent is owned.

Use equal stored-row counts initially because the fused operator currently
executes across the full rectangular backing array, including inactive cells.
Balancing only active cells would not balance the current kernel. If a later
compacted active-cell layout is added, repartition by measured cell weights.

Each rank must own at least one row. Reject allocations with more ranks than
decomposable rows rather than creating empty ranks.

### Cartesian decomposition

Implement a one-axis slab decomposition first, using the longest grid axis.
It needs one halo plane on either side for the seven-point stencil. This path
provides an analytic bare-box gate and exercises the communication layer before
the masked triangular HPMR case.

A 2-D Cartesian process grid can be added after the slab implementation. It is
needed when a large GPU count makes slab surface-to-volume ratio or row count
limiting. Do not add pencils before strong-scaling data demonstrates the need.

### Setup ownership

Use two setup stages:

1. Bring-up stage: build the host mesh once on rank zero, scatter local maps and
   mixing data, and construct local device fields and operators. This avoids
   duplicating device fields and solve memory while minimizing initial code
   change.
2. Production stage: generate deterministic HPMR geometry and material maps by
   local row range. Only small material tables and global metadata are
   broadcast. This removes rank-zero memory and scatter limits at very large
   refinement.

The solve must not retain a full-domain device array on any rank in either
stage. Temporary full-domain host setup on rank zero is acceptable only during
the explicitly labeled bring-up stage.

## Distributed data model

Add `ndgpu/distributed.py` with small, explicit objects rather than scattering
MPI conditionals through numerical kernels.

### `DistributedContext`

Responsibilities:

- Own the MPI communicator, rank, size, local rank, and selected device.
- Verify one rank per GPU unless an explicit test-only override is supplied.
- Report MPI library version, CUDA-aware status, host-staging status, and rank
  placement.
- Supply global sum/max operations and neighbor exchange operations.
- Synchronize the correct CUDA stream before communication in the safe
  baseline.
- Keep all MPI imports optional. Serial use must not import `mpi4py`.

Suggested construction:

```python
from mpi4py import MPI
from ndgpu.distributed import DistributedContext

dist = DistributedContext.from_mpi(MPI.COMM_WORLD, device="gpu")
```

### `SpatialPartition`

Store immutable decomposition metadata:

- Global shape and global cell count.
- Owned global index range.
- Local owned shape.
- Lower and upper neighbor ranks.
- Whether each local face is a physical boundary or an MPI interface.
- Counts/displacements needed for gather and scatter.
- A mapping between local owned indices and global indices for diagnostics.

Provide `TriRowPartition` and `CartesianSlabPartition` implementations behind a
small common interface.

### Local vectors and halos

Krylov and flux vectors should contain owned cells only. The distributed
operator owns persistent receive halos and any extended operator workspace.
This prevents ghost values from entering dot products, source normalization,
or result output accidentally.

For the triangular row decomposition, each halo is contiguous with shape
`(ncols, 2[, nz])`. The first implementation may copy owned boundary rows into
persistent contiguous send buffers. A later optimization may send a contiguous
view directly when MPI and CuPy handle it without hidden packing.

### Distributed results

Do not make `Result.flux` silently gather a full field. Add a separate
`DistributedResult` with:

- `local_flux` for the owned rank-local field.
- `partition` and global shape metadata.
- Identical scalar eigenvalue and iteration telemetry on every rank.
- `gather_flux(root=0)` as an explicit, potentially expensive operation.
- Parallel rank-local output as the production large-problem path.

The root-only gathered representation should use the same global array layout
as the serial result so existing plotting and comparison utilities can consume
it.

## Communication required by the algorithms

### Matrix-free operator application

Every stencil application needs one nearest-neighbor halo exchange.

The safe first implementation is:

1. Synchronize the producing CUDA stream.
2. Post receives from lower and upper neighbors.
3. Send the first and last owned planes.
4. Wait for communication.
5. Apply the local operator using owned data and received halos.

Once correct, split the fused operator into interior and boundary work:

1. Post nonblocking halo receives and sends.
2. Launch the interior stencil, which does not consume halo data.
3. Wait for the halo transfers.
4. Launch boundary-row kernels.

The overlap path needs persistent buffers, explicit streams/events, and timing
evidence. It must not be enabled merely because nonblocking MPI calls exist;
some MPI builds make no progress until `Wait`.

The local diagonal must include coupling across an MPI interface exactly once
for each owned row. An MPI interface is not a vacuum or reflective boundary.
This distinction is the most important operator-construction correctness gate.

### PCG collectives

The current PCG uses local dot products for:

- Initial right-hand-side and residual norms.
- `r dot z`.
- `p dot A p`.
- Checked residual norms.
- Updated `r dot z`.

Each must become a global sum over owned cells. Add a reduction provider to the
linear solver API rather than making PCG import MPI:

```python
pcg(..., reductions=dist.reductions)
```

The serial reduction provider delegates to the current fused NumPy/CuPy dot.
The MPI provider computes a local device scalar and performs an all-reduce.
This keeps `ndgpu.linalg` independently testable and lets other linear solvers
adopt the same interface later.

The first distributed PCG should use standard mathematically equivalent
reductions. After correctness and scaling are measured, evaluate a one-reduction
Chronopoulos-Gear or pipelined CG variant. Such recurrences alter roundoff and
residual reliability, so they require residual replacement and independent
accuracy gates; they must not be part of initial bring-up.

CUDA graph capture must be disabled for a distributed PCG block unless only a
purely local interior region is captured. MPI calls are not part of the current
CuPy graph contract.

### Power iteration collectives

Replace full-array local reductions with global reductions for:

- Initial and updated total fission source.
- Source-error numerator and denominator.
- Fission-source normalization.
- Any eigenvalue and convergence diagnostics derived from global fields.

Pack the source-error numerator and denominator into one two-scalar all-reduce.
All ranks then compute the same `k`, source error, inner tolerance, and stop
decision without broadcasting separate control messages.

Normalize with the existing global `grid.n_cells`, including inactive backing
cells, to reproduce current behavior exactly. Do not substitute a local or
active-cell count during the distributed refactor.

### Anderson acceleration

`_anderson_source()` currently forms an `m x m` Gram matrix through many
independent full-array sums. In distributed mode:

1. Compute all local upper-triangle Gram entries and right-hand-side entries.
2. Pack them into one small device or host vector.
3. Perform one all-reduce per outer iteration.
4. Reconstruct the symmetric matrix identically on every rank.
5. Solve the small dense system on each rank.

Do not perform one MPI collective per Gram entry. With depth eight that would
turn source acceleration into a collective-latency bottleneck.

## CUDA-aware MPI and fallback behavior

The environment probe must test behavior, not infer it from package names.

Required probe:

- Initialize one rank per allocated GPU.
- Exchange a nontrivial contiguous CuPy halo with neighboring ranks.
- All-reduce a CuPy scalar and vector.
- Verify values on device without first converting the send buffer to NumPy.
- Report MPI library/version, selected CUDA device, hostnames, and transfer
  mode.
- Repeat enough times to report latency and bandwidth, because a path can be
  correct while silently staging through host memory.

Communication modes:

1. `cuda-aware`: MPI receives CuPy device buffers directly.
2. `host-staged`: explicit persistent pinned host buffers and asynchronous
   device copies surround MPI. This is a correctness fallback and must be
   visible in logs/results.
3. `serial`: no MPI and no communication.

Never silently pass a device pointer to an MPI implementation that has not
passed the runtime probe. A failure there can be a crash rather than a Python
exception.

For the baseline, synchronize before MPI accesses a CuPy buffer and before a
kernel consumes a receive buffer. Later stream-aware optimization may replace
global synchronization with CUDA events if the selected MPI stack documents
and demonstrates correct stream semantics.

## Proposed public API

Keep the existing solver classes and constructors unchanged. Introduce an
explicit distributed solver first:

```python
from mpi4py import MPI
from ndgpu import DistributedTriDiffusionEigenSolver

solver = DistributedTriDiffusionEigenSolver(
    problem.grid,
    problem.materials,
    problem.material_map,
    active=problem.active,
    mask_bc=problem.mask_bc,
    mix_material=problem.mix_material,
    mix_weight=problem.mix_weight,
    communicator=MPI.COMM_WORLD,
    decomposition="rows",
    device="gpu",
)
result = solver.solve(tol_k=1e-8, tol_source=1e-7, verbose=True)

flux = result.gather_flux(root=0)  # collective; non-root ranks receive None
if result.rank == 0:
    print(flux.shape)
```

An explicit class makes ownership and collective behavior visible and avoids
turning an accidental `mpi4py` import into a collective operation in existing
serial programs. Once the implementation is mature, a `communicator=` argument
may be folded into the common solver if it demonstrably simplifies rather than
obscures the API.

The CLI benchmark should launch under `srun` and emit one machine-readable
summary on rank zero plus rank-level communication telemetry files.

## File-level change map

### New files

- `ndgpu/distributed.py`: communicator, reductions, partition metadata, device
  selection, host-staging fallback, and distributed result helpers.
- `ndgpu/distributed_stencil.py`: Cartesian and triangular distributed operator
  wrappers and halo workspaces.
- `ndgpu/distributed_solver.py`: distributed power iteration and public
  diffusion solver classes.
- `examples/hpmr_eigen_multi_gpu.py`: standalone HP-MR benchmark/production
  entry point.
- `slurm/run_ndgpu_multi_gpu.sh`: one-rank-per-GPU Slurm launcher with strict
  preflight and verbose rank placement.
- `tests/verification/test_distributed_diffusion.py`: subprocess MPI correctness
  gates for CPU and optional GPU markers.

### Existing files

- `ndgpu/linalg.py`: inject global reduction operations into PCG without
  changing serial defaults.
- `ndgpu/solver.py`: factor local power-iteration work so serial and distributed
  drivers share group source assembly and convergence policy.
- `ndgpu/stencil.py`: expose partition-safe coefficient construction and local
  interior/boundary apply hooks.
- `ndgpu/tri.py`: expose triangular face coefficients by owned row range and
  distinguish physical boundaries from partition interfaces.
- `ndgpu/backend.py`: select a GPU from local rank and report device/rank
  placement; no MPI import in the serial path.
- `ndgpu/profiling.py`: add halo, collective, wait, interior, and boundary
  timing regions.
- `ndgpu/__init__.py`: export distributed classes only through lazy or
  dependency-safe imports.
- `pyproject.toml`: add the optional `mpi` dependency group.
- `docs/user_guide.md`: document launch, ownership, output, and limitations
  after the production gate passes.

Do not copy and permanently fork the full power-iteration implementation.
Extract local numerical operations from collective orchestration so serial and
distributed paths cannot drift in physics or convergence logic.

## Development phases and acceptance gates

### Phase 0: environment and topology probe

Deliverables:

- A two-rank CPU MPI smoke test.
- A two-GPU CuPy point-to-point and all-reduce probe on Merlin/Grace-Hopper.
- Explicit detection of CUDA-aware and host-staged modes.
- Rank, hostname, local rank, GPU UUID/name, and peer topology in logs.
- A documented Slurm launch command.

Acceptance:

- Every rank selects a unique allocated GPU.
- Device-buffer exchange and reduction return correct values.
- Failure is early and actionable when ranks outnumber GPUs or CUDA-aware MPI
  is requested but unavailable.

### Phase 1: reduction abstraction with no decomposition

Deliverables:

- Serial and MPI reduction providers.
- PCG accepts an injected reduction provider.
- Fission/source/Anderson reductions use packable reduction helpers.
- A size-one communicator path through the distributed solver.

Acceptance:

- Existing serial tests are unchanged.
- Size-one distributed CPU and GPU results agree with existing solvers to the
  current backend tolerances.
- Iteration counts remain identical in FP64 where reduction order is unchanged.

### Phase 2: Cartesian slab prototype

Deliverables:

- One-axis Cartesian partition.
- Blocking halo exchange and partition-aware seven-point operator.
- Explicit gather for result comparison.
- Analytic bare-box multi-rank benchmark.

Acceptance:

- Applying the distributed operator and gathering its result agrees with the
  serial operator for heterogeneous material fields and all boundary laws.
- Two- and four-rank eigenvalues match the serial analytic benchmark within
  solver tolerance.
- Flux L2 error after common normalization is bounded by reduction-order
  roundoff and the requested solve tolerance.
- No rank owns a full-domain device field.

### Phase 3: triangular HP-MR row decomposition

Deliverables:

- `TriRowPartition` and vertical-face halo exchange.
- Partition-aware triangular coefficient/diagonal construction.
- Active-mask and polar volume-mixing support.
- 2-D HP-MR distributed eigenvalue example.

Acceptance:

- A gathered distributed triangular operator apply matches serial for cuts
  through fuel, reflector, drum, mixed absorber, and void regions.
- r4 and r16 HP-MR eigenvalues, fission rates, leakage, and normalized flux
  agree between one, two, and four ranks.
- Forward outer and total inner iteration counts do not materially change.
- Deliberately treating an MPI cut as a physical boundary is caught by a
  regression test.

### Phase 4: distributed setup and output

Deliverables:

- Rank-zero scatter setup followed by local deterministic HPMR row generation.
- Local material, mixing, and active arrays only on each device.
- `DistributedResult`, explicit gather, and parallel rank-local output.
- Per-rank memory high-water telemetry.

Acceptance:

- The r200 scaling case starts without any rank allocating a full-domain device
  flux or cross-section field.
- Gathered small-case output retains the serial shape and ordering.
- Large cases can complete without gathering the flux.

### Phase 5: CUDA-aware communication and overlap

Deliverables:

- Direct device-buffer halo and collective path.
- Persistent halo buffers and nonblocking exchange.
- Interior/boundary split kernels with measured overlap.
- Explicit pinned-host fallback.

Acceptance:

- Direct and staged modes produce equivalent answers.
- Profiling proves whether communication overlaps computation on the target MPI
  stack.
- The optimized path improves wall time over the blocking baseline; otherwise
  retain the simpler baseline.
- Communication mode and bytes transferred appear in the result telemetry.

### Phase 6: collective and preconditioner scalability

Start this phase only after a profile identifies the limiting operation.

Candidates in priority order:

1. Pack independent scalar reductions where mathematically possible.
2. Increase polynomial preconditioning only if saved all-reduces repay added
   halo exchanges.
3. Evaluate one-reduction or pipelined CG with periodic true-residual
   replacement.
4. Add overlapping additive Schwarz as a rank-local preconditioner.
5. Add a geometric multigrid or coarse-mesh correction for the regular
   triangular lattice if PCG iteration count prevents weak scaling.
6. Compare against PETSc/HYPRE GPU AMG as an external reference before writing
   a complex in-house multilevel solver.

Acceptance:

- Any alternative solver reaches the same true global residual.
- Eigenvalue, reaction-rate, and flux-shape gates remain satisfied.
- Reported speedup includes setup, communication, and rejected/fallback work.
- A more complex method is accepted only with a repeatable end-to-end win.

### Phase 7: 3-D prisms, adjoint, and production hardening

Deliverables:

- Extruded triangular-prism support using the same row decomposition.
- Forward and adjoint diffusion.
- Uneven partitions and non-power-of-two rank counts.
- Stable Slurm examples, user documentation, and failure diagnostics.

Acceptance:

- 3-D forward/adjoint eigenvalues match serial and each other as expected.
- Multi-node runs pass on at least two GPU counts.
- Repeated runs have stable work counts and bounded timing variation.
- Serial CPU, serial GPU, MPI CPU, and MPI GPU regression suites pass.

## Test strategy

### Unit tests

- Partition coverage, no overlap of owned cells, and correct neighbor ranks.
- Halo packing/unpacking for lower, upper, physical, and periodic-free edges.
- Global reductions using a fake communicator.
- Device-selection logic from explicit local rank and Slurm environment.
- Packed Anderson Gram entries against the current serial implementation.
- Result gather ordering for uneven row counts.

### Operator tests

- Compare gathered distributed `A*x` with serial `A*x` for random fields.
- Exercise Cartesian 2-D/3-D and triangular 2-D/prism layouts.
- Cut partitions through material interfaces and active-mask boundaries.
- Include one-cell local slabs and zero-length physical coupling arrays.
- Test reflective, vacuum, zero-flux, and numeric albedo boundaries.
- Test volume-mixed drum cells on both sides of a rank boundary.

### Eigenvalue tests

- Analytic homogeneous bare box.
- Reflected slab with asymmetric materials.
- Reduced two-group HP-MR.
- Eleven-group HP-MR after two-group convergence is stable.
- Forward and adjoint equality.

Compare:

- `k_eff` and convergence history.
- Outer and inner iteration counts.
- Global fission-source norm.
- Normalized flux L2 and maximum error.
- Region reaction rates and boundary leakage.
- Positivity and inactive-cell behavior.

Bitwise agreement is not required after MPI changes reduction order. Tolerances
must be derived from the requested solver tolerance and serial CPU/GPU spread,
not widened until a test happens to pass.

### Failure tests

- More ranks than rows.
- Multiple ranks selecting one GPU.
- Missing `mpi4py` with a distributed API request.
- CUDA-aware mode forced on an unsupported MPI.
- One rank failing setup before a collective.
- Noncontiguous or wrong-dtype communication buffers.
- Insufficient device memory during local setup.

Avoid tests that let one rank raise while peers remain blocked indefinitely.
Collectively validate arguments before entering the iterative solve.

## Performance campaign

Use the current standalone 2-D HP-MR eigenvalue scaling ladder as the first
strong-scaling workload. The r200 case has approximately 13.2 million active
cells and 26.4 million two-group flux unknowns and is expected to exceed five
hours on one CPU thread.

For each size, record:

- Backend, host, GPU, MPI library, and communication mode.
- Rank count and row ownership.
- Build, field-map, operator-build, warm-up, and solve time.
- Outer and inner iterations.
- Stencil applications and halo exchanges.
- Bytes sent/received.
- Time in interior compute, boundary compute, packing, MPI wait, and all-reduce.
- Peak device and host memory per rank.
- `k_eff`, global residual, and flux/reaction-rate differences.

Strong-scaling matrix:

| Problem | GPUs | Purpose |
|---|---:|---|
| r32 | 1, 2 | correctness and crossover sanity |
| r64 | 1, 2, 4 | communication baseline |
| r128 | 1, 2, 4, 8 | primary strong-scaling curve |
| r200 | 1, 2, 4, 8 | production memory and scaling gate |

Weak scaling should hold owned rows per rank approximately constant and use
synthetic rectangular triangular fields first. The fixed physical HP-MR core
changes resolution rather than physical extent, so it is a strong-scaling and
mesh-refinement workload, not a pure weak-scaling case.

Report speedup against the same numerical algorithm and convergence tolerance.
Do not compare a different preconditioner or convergence check cadence without
also reporting that as a separate algorithmic result.

Initial performance targets are deliberately gates, not promises:

- Two GPUs should improve end-to-end solve time for r64 and larger.
- Four GPUs should improve r128 and r200 over two GPUs.
- Parallel efficiency should be reported both for the entire solve and for the
  stencil-only region.
- No production acceptance if iteration count changes enough to explain the
  apparent speedup.

## Slurm launch shape

The intended launcher is one task and one GPU per rank, for example:

```bash
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
     --cpus-per-task=1 --gpus-per-task=1 \
     python -u examples/hpmr_eigen_multi_gpu.py 200 gpu
```

The final script must also:

- Load the same MPI used to build `mpi4py`.
- Load CUDA before importing CuPy.
- select the GPU from the local rank and allocated visible-device list.
- set `OMP_NUM_THREADS=1` unless a measured CPU helper path needs more.
- run the device communication probe before the expensive mesh build.
- use shared scratch rather than AFS for runtime files.
- write a rank-zero summary and rank-specific diagnostic logs.

Exact module names and MPI environment variables must come from the Merlin
probe in Phase 0 rather than being assumed in source code.

## Risks and mitigations

### Global reductions dominate

PCG performs global dot products every iteration. At high rank count,
all-reduce latency can dominate even when halo exchange is small.

Mitigation: measure first, pack reductions, use polynomial preconditioning only
when profitable, then evaluate pipelined CG and a coarse preconditioner.

### Jacobi-PCG iteration growth

The current preconditioner is local and iteration count grows with refinement.
Adding GPUs does not cure that algorithmic scaling.

Mitigation: separate distributed correctness from preconditioner research, then
evaluate additive Schwarz and multilevel methods with true-residual gates.

### Partition interfaces become false boundaries

Reusing a local serial operator naively can add vacuum leakage at every rank
cut or omit the cross-rank diagonal contribution.

Mitigation: make physical/interface face type explicit in partition metadata
and require operator-apply equivalence tests before eigenvalue tests.

### CUDA-aware MPI is environment-dependent

An MPI package can import successfully while device-buffer communication is
unsupported or host-staged.

Mitigation: runtime probe, explicit mode selection, pinned fallback, and mode
telemetry in every result.

### Setup remains replicated

A solver can distribute flux while still constructing full-domain material and
operator arrays on every rank, hiding a memory scalability failure.

Mitigation: instrument memory, prohibit full-domain device fields, and complete
local deterministic HPMR construction before calling the path production-ready.

### Python overhead and synchronization

Fine-grained MPI calls and accidental `float(device_scalar)` conversions can
serialize the GPU path.

Mitigation: persistent buffers, packed collectives, explicit synchronization
points, NVTX ranges, and no per-cell Python communication.

### Numerical drift from reduction order

MPI changes summation order and therefore convergence crossing by an iteration
in marginal cases.

Mitigation: compare true residuals and physics edits, keep deterministic local
ordering, and define tolerance-based rather than bitwise distributed gates.

## Decision points

The following decisions require measured evidence during development:

1. Whether CUDA-aware MPI on Merlin provides direct GPU transfers with the
   installed CuPy and MPI stack.
2. Whether row-slab decomposition remains efficient beyond four or eight GPUs.
3. Whether communication overlap is real under the selected MPI progress
   model.
4. Whether reduction latency or halo bandwidth limits the r128/r200 cases.
5. Whether pipelined CG is sufficient or a multilevel preconditioner is needed.
6. Whether PETSc/HYPRE should become an optional production linear-solver
   backend rather than only a reference.

These should be recorded in this document as dated decisions with benchmark
artifacts, following the existing performance-plan practice.

## Definition of done

The first multi-GPU diffusion release is complete when:

- A user can launch a 2-D HP-MR eigenvalue solve with `srun`, one MPI rank per
  GPU, without changing physics inputs.
- No rank owns full-domain device fields.
- Two- and four-GPU answers pass eigenvalue, flux, reaction-rate, and leakage
  comparisons against the serial solver.
- r64 or larger demonstrates a repeatable end-to-end two-GPU speedup.
- r128 and r200 have documented 1/2/4 GPU scaling and memory use.
- CUDA-aware and pinned-host modes are explicit and tested.
- Failures in placement, MPI/CUDA compatibility, or decomposition are early and
  actionable.
- The serial CPU and single-GPU APIs and regression suite remain unchanged.
- The implementation, Slurm launcher, benchmark results, and user guide are
  committed together.
