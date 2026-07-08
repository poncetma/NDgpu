# ndgpu — GPU-native neutron diffusion & SP3 solver

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
- **One code path**: written against the NumPy API surface that CuPy mirrors;
  `device="cpu"|"gpu"|"auto"` picks the backend. All physics is validated on
  CPU and runs unchanged on GPU.

## Install

```bash
pip install -e .                 # CPU (NumPy)
pip install -e .[cuda12]        # + CuPy for CUDA 12.x GPUs
```

## Validation (all in `tests/`, run `pytest`)

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
| Uniform fissile step, bare core | flux shape fixed ⇒ exact point-kinetics ODE | < 0.2% over the transient |
| **2D TWIGL** step (Σ_a2 drop) | P(0.1), P(0.5) vs literature 2.06, 2.13 | 2.061, 2.130 |
| **2D TWIGL** ramp | P(0.1), P(0.5) vs literature 1.31, 2.11 | 1.308, 2.109 |
| **3D Langenbuch (LMW)** rod-bank transient | peak power / time vs reference ≈1.6 @ ≈21 s | 1.61 @ 21 s |

(TWIGL at the converged `cells_per_8cm=4` mesh, `dt=1e-3`.)

## Benchmarks

```bash
python examples/benchmark.py 64 128 192   # CPU vs GPU, verified against analytic k
```

## Testing the GPU path without a local GPU

There is no practical CUDA emulator for CuPy (GPU Ocelot is dead, ZLUDA
targets AMD). This repo's answer is architectural: the NumPy backend **is**
the virtual GPU — CuPy mirrors NumPy semantics operation-for-operation, so
the physics, indexing and convergence logic exercised by the CPU test suite
is byte-for-byte the code that runs on the GPU. For real-hardware validation,
run `notebooks/colab_gpu_benchmark.ipynb` on Google Colab's free T4 GPU.

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
