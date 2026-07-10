# NDgpu — GPU-native neutron diffusion & SP3 solver

Steady-state multigroup **k-eigenvalue** reactor physics (criticality) on 3D
structured grids, running natively on CUDA GPUs via CuPy, with a NumPy CPU
fallback that shares 100% of the code path.

```python
from ndgpu import DiffusionEigenSolver, Grid, PWR_TWO_GROUP

grid = Grid(shape=(128, 128, 128), size=(150.0, 150.0, 150.0))  # cm
result = DiffusionEigenSolver(grid, PWR_TWO_GROUP, device="auto").solve()
print(result)          # k_eff, iterations, time, device
flux = result.flux     # (groups, nx, ny, nz), on the solve device
```

## Physics

Solves the multigroup eigenvalue problem  `M φ = (1/k) F φ` with two
selectable angular approximations:

- **`DiffusionEigenSolver`** — classic multigroup diffusion:
  `-∇·(D_g ∇φ_g) + Σ_r,g φ_g = Σ_{g'≠g} Σ_s,g'→g φ_g' + (χ_g/k) Σ_g' νΣ_f,g' φ_g'`
- **`SP3EigenSolver`** — simplified P3 (Brantley–Larsen form): two coupled
  diffusion-type equations per group in the moments `(φ0+2φ2, φ2)`,
  symmetrized into an SPD block system. Captures leading transport effects
  (steep gradients, strong absorbers, small cores) at ~2–3× diffusion cost.

Features: arbitrary group count with up/downscatter, heterogeneous cores via
per-cell material maps (harmonic-mean face diffusion coefficients), per-face
zero-flux/reflective boundary conditions (quarter-core symmetry, exact 2D),
float64 or float32.

## Why it's fast on GPU (architecture)

- **Matrix-free**: the 7-point finite-volume stencil is applied as six shifted
  fused array multiply-adds — no sparse-matrix assembly, no CSR indirection,
  perfectly coalesced memory access. Face couplings and diagonals are
  precomputed once per group.
- **Device-resident**: fluxes, sources and coefficients live on the GPU for
  the entire solve. The only host↔device traffic is one scalar convergence
  check per CG iteration.
- **CG-friendly formulation**: every within-group operator (diffusion and
  symmetrized SP3) is symmetric positive definite by construction, so the
  inner solves are matrix-free Jacobi-preconditioned conjugate gradients —
  dots, axpys and stencils, all bandwidth-bound GPU kernels.
- **Cheap where it can be**: inner CG tolerances track the outer
  power-iteration residual and every solve is warm-started, so late outer
  iterations cost only a handful of stencil applications.
- **Polynomial preconditioning for GPUs**: `precond_degree=m` swaps the
  Jacobi preconditioner for a truncated Neumann series (I+N+…+Nᵐ)D⁻¹ — m
  extra stencil applies per CG iteration (pure streaming, no reductions) in
  exchange for ~2–3× fewer iterations, i.e. 2–3× fewer of the global dot
  products that are the GPU's only synchronization points (E et al., NED 320
  (2017) found degree-3 Neumann-PCG the fastest GPU solver for 2·10⁴–3·10⁶
  cells). Default 0; on CPU it is roughly cost-neutral.
- **One code path**: written against the NumPy API surface that CuPy mirrors;
  `device="cpu"|"gpu"|"auto"` picks the backend. All physics is validated on
  CPU and runs unchanged on GPU.

## Install

```bash
pip install -e .                 # CPU (NumPy)
pip install -e .[cuda12]        # + CuPy for CUDA 12.x GPUs
```

## Repository map

| Directory | Contains |
|---|---|
| `ndgpu/` | the solver library (operators, solvers, grids, readers) |
| `ndgpu/benchmarks/` | benchmark **problem definitions + published reference values** (importable: `build_twigl`, `twigl.P_REFERENCE`, …) |
| `tests/verification/` | exact-mathematics checks: analytic solutions, convergence order, invariants, reader transcription |
| `tests/validation/` | published reactor problems solved end-to-end vs their references (which live with the builders above) |
| `examples/` | runnable demos; `speed_benchmark.py` is the CPU-vs-GPU performance harness |
| `docs/` | theory & benchmarks report |
| `dev-refs/` | third-party reference inputs used to derive data; never imported |

See `tests/README.md` for the verification/validation taxonomy.

## Verification & validation (run `pytest`)

Checked against **exact analytic solutions**, not just self-consistency:

| Case | Check |
|---|---|
| Bare homogeneous box, 1 & 2 groups | k_eff matches the exact buckling solution; error falls 4× per mesh doubling (2nd order) |
| Same, SP3 | matches the exact analytic **SP3** eigenvalue (2G×2G moment system), 2nd order |
| Reflective boundaries | k = k_∞ exactly, flat flux; SP3 reduces exactly to diffusion |
| Flux shape | fundamental sin×sin×sin mode, cosine similarity > 0.99999 |
| Symmetry plane | half-core with reflective face reproduces full-core k to 1e-7 |
| Reflected core | k between bare-core k and k_∞ |

At 64³ × 2 groups, k_eff is reproduced to ~1 pcm of the analytic value.

### OECD/NEA C5G7 MOX benchmark (2D)

`python examples/c5g7_2d.py [cells_per_pin] [device]` — the full quarter core
(UO2/MOX checkerboard + reflector, 7 groups with upscatter), pin-cell
homogenized (C5G7 is a transport benchmark with explicit cylindrical pins;
volume-weighted homogenization is the standard diffusion-level treatment).
Cross sections auto-extracted from the published benchmark set.

| Solver (204×204×7, converged mesh) | k_eff | Δ vs transport reference 1.18655 |
|---|---|---|
| diffusion | 1.18695 | +40 pcm |
| SP3 | 1.18830 | +175 pcm |

The residual is the homogenization + angular physics gap, not solver error
(the solver's own discretization converges 2nd order). Max fuel-pin power
2.57–2.59 vs ≈2.50 transport reference.

### HP-MR heat-pipe microreactor (2D and 3D)

`python examples/hpmr_2d.py [refine] [absorber] [device]` — assembly-level
radial model of the ANL/INL HP-MR reference microreactor (NEAMS VTB design: 30
TRISO fuel assemblies, central shutdown cell, Be reflector ring, 12 rotating
B4C control drums) on the body-fitted triangular mesh, geometry decoded from
the VTB Serpent model. Sweeps the drum angle and prints the worth curve
(k 1.030 → 1.002, ≈ −2700 pcm fully inserted, at the placeholder two-group
cross sections — swap in SPH-corrected sets via `build_hpmr2d(materials=…)`).

The curved B4C absorber arc is the one non-hex feature, and
`build_hpmr2d(..., absorber=…)` offers two treatments. `"raster"` stamps whole
cells by centroid — a staircase whose worth-vs-angle curve has dead steps and
486-pcm cliffs, and whose drum worth needs refine ≥ 6 to settle. `"polar"`
volume-mixes each cell's **exact polar area fraction** of the arc (harmonic for
D, linear for the reaction cross sections) via the solver's `mix_material` /
`mix_weight` hook — the same partial-volume idea as control-rod-tip mixing.
It gives a smooth worth-vs-angle curve (max step 145 vs 486 pcm), converges
from a much better coarse-mesh value, and runs below the raster's refine floor
(the 1 cm annulus is represented even when it is thinner than a triangle). Both
agree at fine mesh (~2720 pcm), so the polar treatment is unbiased.

`python examples/hpmr_3d.py [refine] [nz] [absorber] [device]` — the same core
extruded to full height as triangular *prisms* (160 cm fueled + 20 cm Be axial
reflectors, drums running the full height, vacuum z faces): the tri lattice
gains a trailing z axis (`TriGrid(shape=(rows, cols, 2, nz))`) and the
operator two extra shifted multiply-adds, so the solve stays matrix-free on
both backends. The tri-z scheme is validated against exact references in
`tests/verification/test_tri_prisms.py`: k_∞ reproduced to 1e-9 with reflective faces, the
analytic 1D-slab eigenvalue approached at exactly 2nd order in dz, and the
extruded VVER-440 core with reflective z faces matching the 2D k to < 0.01 pcm.
`build_hpmr3d(..., absorber="polar")` carries the drum-arc volume-mixing (below)
into 3D — the arc runs the full height, so the 2D `mix_material`/`mix_weight`
simply extrude over the z-layers; at refine 4 the polar drum worth is ~2650 pcm
vs the raster's under-resolved ~1950.

**Real cross sections.** `build_hpmr2d`/`build_hpmr3d` default to two-group
placeholders, but `hpmr_endfb8_materials(xs_path)` builds the material set from
the VTB's actual **11-group ENDF/B-8** Griffin library
(`fullcore_xml_G11_endfb8_ss_tr.xml`): the pin-level fuel/moderator/graphite/
heat-pipe cross sections are flat-flux-homogenized into the fuel assembly (from
the Serpent pin lattice's volume fractions), with the Be reflector, drum body
and B4C arc taken from the library directly. At this data the 3D core gives
k ≈ 1.120 drums-out, 1.083 drums-in (≈ 3040 pcm drum worth). The ~9000 pcm
excess over the transport reference (k ≈ 1.03) is the *homogenization bias* —
the gap a superhomogenization (SPH) correction removes; pass per-group SPH
factors (from a transport/FEMFFUSION reference) via `hpmr_endfb8_materials(...,
sph_fuel=…)` or `volume_homogenize(..., sph_factors=…)`.

### Griffin and FEMFFUSION cross-section files

- `ndgpu.read_griffin_library` / `read_griffin_material` parse Griffin/YakXs
  (ISOXML) multigroup libraries — the format the NEAMS Virtual Test Bed ships
  reactor cross sections in — into `Material` lists, with `volume_homogenize`
  for flat-flux region mixing and optional SPH factors. Conventions verified by
  the `Total = Absorption + scatter-out` balance and the sink/source scattering
  transpose.
- `ndgpu.read_xsec` / `ndgpu.read_material_xml` parse FEMFFUSION's two-group
  `.xsec` and multigroup XML material files (conventions verified against
  FEMFFUSION's own parsers; round-trip validated on its VVER-440 and C5G7
  examples).

## Transient (time-dependent) solver

`TransientSolver` marches the multigroup diffusion equations in time with
delayed-neutron precursors (backward Euler, precursors integrated
analytically per step; the initial steady state is critically adjusted by
`1/k0` so a bare unperturbed core stays flat). Each step is a fixed-source
multigroup problem with a positive diagonal shift — SPD and *better*
conditioned than the eigenvalue solve, so the same matrix-free Jacobi-CG
machinery applies; the fission/scatter coupling is closed by an
Anderson-accelerated source iteration.

```python
from ndgpu import TransientSolver
from ndgpu.benchmarks import build_twigl

prob = build_twigl(perturbation="step", cells_per_8cm=4)
res = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                      bc=prob.bc, device="auto").solve(t_end=0.5, dt=1e-3)
print(res.power)   # P(t)/P(0)
```

Validated against two standard kinetics benchmarks from the
[FEMFFUSION](https://github.com/Zonni/FEMFFUSION) repository, plus an exact
point-kinetics reference:

| Case | Check | Result |
|---|---|---|
| Uniform +$0.50 absorption step, bare core | flux shape fixed ⇒ exact point-kinetics ODE (prompt jump to β/(β−ρ) resolved) | < 5e-4 over the transient |
| **2D TWIGL** step (Σ_a2 drop) | P(0.1), P(0.5) vs literature 2.06, 2.13 | 2.061, 2.130 |
| **2D TWIGL** ramp | P(0.1), P(0.5) vs literature 1.31, 2.11 | 1.308, 2.109 |
| **3D Langenbuch (LMW)** rod-bank transient | peak power / time vs reference ≈1.6 @ ≈21 s | 1.61 @ 21 s |

(TWIGL at the converged `cells_per_8cm=4` mesh, `dt=1e-3`.)

## Benchmarks

```bash
python examples/speed_benchmark.py 64 128 192   # CPU vs GPU, verified against analytic k
```

On a GPU, `notebooks/colab_cpu_gpu_benchmarks.ipynb` times the identical solve
CPU vs GPU across four reactor problems that span the solver's regimes — C5G7
(2D Cartesian, 7 groups), IAEA-3D (masked 3D), VVER-440 (2D triangular) and the
HP-MR microreactor (3D triangular prisms) — asserting the two backends return
the same `k_eff` and charting the speed-up (which widens with mesh size and
again with `dtype=float32`).

## Testing the GPU path without a local GPU

There is no practical CUDA emulator for CuPy (GPU Ocelot is dead, ZLUDA
targets AMD). This repo's answer is architectural: the NumPy backend **is**
the virtual GPU — CuPy mirrors NumPy semantics operation-for-operation, so
the physics, indexing and convergence logic exercised by the CPU test suite
is byte-for-byte the code that runs on the GPU. For real-hardware validation,
run `notebooks/colab_gpu_benchmark.ipynb` or
`notebooks/colab_cpu_gpu_benchmarks.ipynb` on Google Colab's free T4 GPU.

## Roadmap

- Marshak (vacuum/albedo) boundary conditions for SP3 and diffusion
- Wielandt shift / Chebyshev acceleration of the power iteration
- Geometric multigrid preconditioning for the inner solves
- Fused RawKernel stencil (single kernel launch per apply)
- Multi-GPU domain decomposition

## References

- OECD/NEA, *Benchmark on Deterministic Transport Calculations Without
  Spatial Homogenisation* (C5G7), NEA/NSC/DOC(2001)4 — cross sections
  extracted via [OpenMOC](https://github.com/mit-crpg/OpenMOC)'s
  `sample-input/c5g7-mgxs-hdf5.py`.
- P. S. Brantley, E. W. Larsen, *The Simplified P3 Approximation*,
  Nucl. Sci. Eng. 134 (2000).
