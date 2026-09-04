# HP-MR drum-worth mesh refinement

## Objective

Resolve differential control-drum worth without globally refining the complete
core to the scale of the 1 cm B4C layer. The mesh keeps a moderate global
assembly lattice and recursively splits only the complete annular band swept by
each drum absorber. Polar volume mixing remains active on every local cell and
conserves each drum's annular-sector area independently.

This separates two errors that were previously conflated:

- global core flux-shape error, controlled by the assembly-scale refinement;
- local absorber/flux-depression error, controlled by drum-band refinement.

The mesh is independent of drum angle. A transient can therefore rotate the
volume fractions without remeshing or projecting its flux between meshes.

## Implementation

`build_hpmr2d_local()` returns an unstructured triangular problem for
`UnstructuredDiffusionSolver`. Nonconforming interfaces can span arbitrary
dyadic depth; every fine leaf face is coupled conservatively to its coarse
neighbor. The unstructured solver applies the same material-blending rules as
the structured solver: linear reaction and scattering data, harmonic diffusion
coefficients, and fission-weighted spectra.

This controlled midpoint-refinement workflow is the primary intended use of
NDgpu's nonconforming unstructured path. It does not imply support for arbitrary
Gmsh/CAD meshes; see [unstructured_mesh_scope.md](unstructured_mesh_scope.md)
for the discretisation, parser, and 3-D limitations.

Use `absorber="polar", samples=0`. This computes the exact intersection of
each triangular cell with the curved annular absorber sector. Positive
`samples` retain equal-area subcell quadrature only for historical comparison.
Both paths correct each drum independently to the analytic absorber area,
240.331837999619 cm2 over all 12 drums.

## Preliminary CPU diagnostic

The two-group `90 -> 95 degree` diagnostic at fixed global `r4` is monotonic:

| Local levels | Effective drum refine | Cells | Worth (pcm) |
|---:|---:|---:|---:|
| 0 | r4 | 5,280 | 52.00 |
| 1 | r8 | 7,116 | 98.12 |
| 2 | r16 | 14,460 | 109.69 |
| 3 | r32 | 43,836 | 111.50 |

This two-group result was useful for bringing up recursive interfaces, but the
11-group study below supersedes it. In particular, effective `r32` still leaves
an orientation-induced reversal over small angle increments.

## 11-group A100 acceptance

`examples/hpmr_drum_refinement.py` runs the given 11-group ENDF/B-VIII constants
on one GPU. The final runs used exact volume mixing at angles 85, 87, 89, 90,
90.25, 90.5, 91, 93, and 95 degrees. All values below are from jobs `203276`,
`203279`, `203282`, and `203278`.

| Mesh | Drum side (cm) | Cells | Unknowns | Monotonic | 90 -> 95 worth (pcm) | Mean solve (s) |
|---|---:|---:|---:|:---:|---:|---:|
| local r4 + 4 levels | 0.241 | 161,340 | 1,774,740 | yes | 256.61 | 50.23 |
| **local r8 + 3 levels** | **0.241** | **169,674** | **1,866,414** | **yes** | **261.27** | **16.97** |
| local r16 + 2 levels | 0.241 | 218,400 | 2,402,400 | yes | 263.50 | 21.57 |
| uniform r64 reference | 0.241 | 1,351,680 | 14,868,480 | yes | 264.18 | 90.17 |

Uniform and local effective `r32` meshes, with 0.483 cm drum cells, are not
accepted: both reverse between 89 and 90.25 degrees. Four cells through the
1 cm absorber, effective `r64`, are required for a monotonic curve in this
study.

The preferred working mesh is **global r8 plus three local levels**. It differs
from the uniform r64 reference by 2.91 pcm over the 90-to-95-degree interval,
uses 7.97 times fewer cells, and solves 5.31 times faster. Use global r16 plus
two local levels as the higher-accuracy check; its worth differs by 0.68 pcm.
The r4 plus four-level mesh is smaller but its deeper hanging-node hierarchy is
poorly conditioned and makes it three times slower than the r8 candidate.

The resolved uniform curve gives 264.18 pcm from 90 to 95 degrees. Linear
interpolation of the resolved points places a 200 pcm all-drum movement at
about 90 to 93.88 degrees. The old coarse-r4 estimate of roughly 200 pcm for a
0.5-degree movement was a mesh artifact.

Raw JSON is tracked in `benchmark-results/hpmr-drum-refinement/`.

## Multi-GPU consequences

The accepted local mesh now has a dedicated tensor-product extrusion path.
`ExtrudedMeshGrid` keeps the 2-D nonconforming radial connectivity unchanged
and adds uniform axial layers; `DistributedExtrudedMeshTransientSolver`
partitions those layers across MPI ranks. This avoids asking the general 3-D
mesh assembler to infer coarse-to-fine prism faces and avoids a 3-D graph
partitioner for the prismatic HP-MR geometry.

Use the following staged gates:

1. **Existing structured-path crossover:** uniform r32/nz10, exact mixing,
   37,171,200 unknowns, on 1/2/4 GH200s. This checks scaling mechanics but is
   not a drum-worth-accepted production mesh.
2. **Existing structured physical reference:** uniform r64/nz10, exact mixing,
   148,684,800 unknowns, on 2/4 GH200s, adding one GPU if it fits inside the
   one-hour queue. Run only one or two transient steps after steady-state reuse.
3. **Target local production gate:** r8 plus three
   local levels gives 18,664,140 unknowns at nz10 and 37,328,280 at nz20. Run
   1/2/4 GPUs over enough steps to amortize setup and report seconds per step,
   global reductions, halo time, and identical power/iteration histories.
4. **Accuracy gate:** repeat the target test with r16 plus two local levels,
   24,024,000 unknowns at nz10 and 48,048,000 at nz20.

The two-rank CPU correctness gate passed as Merlin job `8769714`, matching the
serial eigenvalue, power, flux, precursors, and iteration history. The GH200
correctness and throughput gates remain required before claiming acceleration.

The target is transient throughput, not memory capacity: both local meshes may
fit on one GH200, but a 200 s transient can still benefit if communication is a
small fraction of step time. Exact moving-drum maps must be precomputed along
the trajectory and reused; recomputing curved intersections every time step
would dominate the solve.
