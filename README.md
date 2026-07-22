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
float64 or float32. Besides Cartesian boxes, the structured grid solves
**cylindrical (r-z) bodies of revolution** (`Grid(..., geometry="cylindrical")`,
x-axis = radius): the stencil is volume-weighted so it stays SPD, the r = 0
axis is a natural boundary, and the same eigenvalue/transient machinery runs
unchanged (verified 2nd-order against the exact Bessel-mode bare-cylinder k).
Alternative Krylov solvers (`linear_solver="gmres"` / `"bicgstab"`) exist for
operators CG can't touch; `symmetric_operator=False` exercises that path today
by solving the cylindrical stencil in its natural non-symmetric divergence
form instead of the SPD weighting (validated on ANL-7416 8-A1: identical k
and power trace).

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

## Quickstart

The `Model` front end defines a reactor in centimetres, paints regions with your
own materials, and prints a human-readable solution. Define materials, geometry,
and boundary conditions, then `run()`:

```python
import ndgpu
from ndgpu import Material

fuel = Material(name="fuel", diffusion=[1.26, 0.35], sigma_a=[0.012, 0.121],
                nu_sigma_f=[0.0085, 0.185], sigma_s=[[0, 0.026], [0, 0]], chi=[1, 0])
reflector = Material(name="reflector", diffusion=[1.15, 0.90], sigma_a=[0.0002, 0.005],
                     nu_sigma_f=[0, 0], sigma_s=[[0, 0.045], [0, 0]])

model = (ndgpu.Model(size=(120, 120, 120), cells=(40, 40, 40))   # cm
         .fill(reflector)                                        # background
         .add_box(fuel, x=(30, 90), y=(30, 90), z=(30, 90))      # central fuel block
         .set_boundary("vacuum"))                                # leak from the surface

print(model.run())          # solves on GPU if available, else CPU
```

```
NDgpu reactor solution
======================
  geometry    : 120 x 120 x 120 cm,  40 x 40 x 40 cells (64,000)  [3D]
  groups      : 2     method: diffusion     device: cpu (numpy)
  boundary    : x: vacuum | y: vacuum | z: vacuum

  k_eff       : 1.109007
  reactivity  : +9829 pcm   (rho = (k-1)/k)
  status      : converged in 29 outer / 681 inner iterations, 0.96 s

  where the fission neutrons go (per neutron produced / k):
    absorbed  :  96.3 %
    leaked    :   3.7 %

  flux peaking (thermal group): peak / average = 4.17

  material           volume   fission
    reflector        87.5%      0.0%
    fuel             12.5%    100.0%
```

`Model` accepts 1-D, 2-D or 3-D `size`/`cells`, `method="diffusion"` or `"sp3"`,
`adjoint=True` for the importance solve, and per-face boundary names (`"vacuum"`,
`"reflective"`, `"zero-flux"`, or an albedo). Two sibling builders reach the other
geometry backends with the same report:

| Builder | Geometry | Examples |
|---|---|---|
| `ndgpu.Model` | structured Cartesian (1/2/3-D), diffusion/SP3, adjoint | `bare_reactor.py`, `reflected_core.py`, `adjoint_importance.py` |
| `ndgpu.MeshModel` | arbitrary unstructured mesh (Gmsh or assembled), 2/3-D | `unstructured_mesh.py` |
| `ndgpu.HexLattice` | hexagonal assembly lattice on the triangular solver, diffusion/SP3 | `hex_lattice.py` |

```python
# unstructured mesh: paint by tag, centroid box, or a predicate on the centroid
ndgpu.MeshModel("core.msh").fill(reflector).assign(fuel, tag=1).set_boundary("vacuum").run()

# hexagonal lattice on the body-fitted triangular solver
(ndgpu.HexLattice(pitch=20, refine=4)
 .set_site((0, 0), fuel).set_site((1, 0), reflector)
 .set_boundary("vacuum").run(method="sp3"))
```

A `Model` also runs **transients**. It solves the steady state first (its
eigenvalue `k0` normalises the fission source, so an unperturbed run stays at
`P/P0 = 1`), then marches; the perturbation is a `materials_at(t)` callback and
the result carries both the power history and the initial `.steady` solution:

```python
(model.set_kinetics(velocities=[1e7, 3e5], beta=[0.0065], decay=[0.08])
      .transient(t_end=3.0, dt=0.02,
                 materials_at=lambda t: [fuel, rod if t >= 0.5 else coolant]))
# -> report with k0, a power-vs-time sparkline, peak/final P/P0; see examples/transient_rod_drop.py
```

All builders return raw `k_eff`, `flux`, and balance fractions as plain values.
For SPH or the lower-level knobs, use the solver classes directly.

Full API reference and a literature-validated worked example (the TWIGL
benchmark, static + transient) are in **[docs/model_api.md](docs/model_api.md)**;
runnable as `examples/twigl_benchmark.py`.

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
| **Reflected slab** (Lamarsh Ch. 7) | k matches the exact transcendental eigenvalue, 2nd order; reflector savings > bare; solved via both the low-level solver and the Model API (`examples/reflected_slab.py`) |

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

`python examples/hpmr_hybrid.py [refine] [device]` — a **hybrid SP3/diffusion**
solve: transport (the SP3 second moment) runs *only* in the rotating drum
bodies (~1/5 of the core) and plain diffusion everywhere else, via the
`hybrid_mask=` argument on any SP3/SDPN eigen-solver (build the mask with
`ndgpu.benchmarks.hpmr_transport_mask`). The mask zeroes the higher-moment
source and moment coupling outside itself, so the transport correction is
*generated* only at the near-black B4C arcs — where the steep angular flux
lives — and decays into the surrounding graphite, while the net-current-carrying
moment stays one global operator (continuous across the interface). An all-True
mask reproduces full SP3 bit-for-bit; an empty mask reproduces diffusion. Since
control-drum worth is a reactivity *difference*, the angle-independent global
transport correction (outer-boundary/reflector leakage) cancels and the
drum-local self-shielding the hybrid captures dominates: it closes ~55 % of the
diffusion→SP3 worth error with transport in 22 % of the cells. (Default
"faithful" mode keeps every moment global and closure-free; `hybrid_confine=True`
instead pins the higher moments to exactly zero outside the mask — bit-exact
diffusion there — at the cost of an interface closure.)

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

### Discrete-ordinates (Sₙ) transport reference and hybrid Sₙ/diffusion

`ndgpu.SNTransportSolver` is a 2D Gauss-Legendre discrete-ordinates (Sₙ)
k-eigenvalue solver — the 2D extension of the 1D Sₙ reference that scores the
SDPN family (`examples/sdpn_benchmark_1d.py`). It uses a product quadrature
(Gauss-Legendre in the polar cosine × uniform azimuthal), diamond differencing
swept one ordinate at a time (each 2D sweep factored into a per-row 1D
bidiagonal solve), GMRES on the within-group scattering fixed point, and an
Anderson-accelerated reflective-boundary fixed point. Cross sections come from
ndgpu `Material`s reconstructed into the transport problem whose P1 limit is
exactly the diffusion data (Σ_t = 1/3D, within-group scatter Σ_t−Σ_a−Σ_out), so
Sₙ, diffusion and SDPN score the same physics. It reproduces k∞ exactly on a
reflective homogeneous box, matches diffusion in the scattering-dominated
limit, and gives the transport k that SP3/SDPN bracket toward (`test_sn.py`).
It's a CPU/numpy reference, not the GPU path.

`ndgpu.HybridSNDiffusionSolver` (`examples/hybrid_sn_hpmr.py`) is the Sₙ
counterpart of the hybrid SP3/diffusion solver: full transport (Sₙ) runs only
in a masked subdomain — the control-drum absorber — and diffusion in the bulk.
Because Sₙ and diffusion are genuinely different discretizations (angular
unknowns + sweep vs a scalar stencil), this is a real domain decomposition: the
drum is excised from the diffusion domain and the two are coupled by the
interface **net current** (an Sₙ fixed-source solve on the drum box with
incoming from the neighbouring diffusion flux; its outgoing current becomes a
source on the ring of bulk diffusion cells). The limits are exact — an empty
mask reproduces the diffusion solver bit-for-bit, a full mask reproduces Sₙ. On
a 2D-Cartesian HP-MR-motivated drum core it closes ~80 % of the diffusion→Sₙ
control-drum-worth error with transport in a few percent of the cells: diffusion
over-predicts the near-black drum's worth (~+30 %) for want of the transport
self-shielding, and the hybrid recovers most of it by resolving only the drum
with transport. (The coupling is accurate for a well-resolved isolated drum;
tightly-packed multiple drums need a better interface reconstruction than the
present isotropic-incoming/net-current model.)

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
| **ANL-7416 Problem 8-A1** (2D r-z, 6 precursor families) | initial k vs book 0.86690/0.86705; Exhibit A power trace | k to 75 pcm (coarse) / ~40 pcm (refined); ramp phase < 3%, tail within the documented discretization band |

(TWIGL at the converged `cells_per_8cm=4` mesh, `dt=1e-3`.)

The 8-A1 problem (`ndgpu.benchmarks.build_anl8a1`, from the 1977 Argonne
Benchmark Problem Book) is the r-z geometry's validation case: a delayed
supercritical Σ_a ramp in three regions of a 240 × 525 cm thermal reactor.
Its published power trace is tied to the reference codes' vertex-centered
coarse-mesh discretization (which *under*-estimates the mesh-converged ramp
worth by ~5%, as re-deriving that scheme shows — see
`ndgpu/benchmarks/anl_bss8.py` for the analysis and for a cross-section
erratum in the book), so the eigenvalue is validated tightly and the
excursion tail within a quantified band.

## Neutron noise (frequency domain)

`NoiseSolver` computes the stationary flux fluctuations δφ(**r**, ω) that a
fluctuating cross section δΣ(**r**, ω) induces on top of a critical mean flux —
neutron noise analysis, the basis of core-monitoring diagnostics (detecting
vibrating fuel/absorbers, coolant-density waves, unseated assemblies). Writing
each quantity as mean + fluctuation, linearizing, and Fourier transforming in
time (∂/∂t → iω) turns the time-dependent diffusion + precursor equations into
a **fixed-source complex** problem, one linear solve per frequency:

```
[ -∇·D_g∇ + Σ_r,g + iω/v_g ] δφ_g − Σ_{g'≠g} Σ_s,g'→g δφ_g'
    − (χ_eff,g(ω)/k) Σ_g' νΣ_f,g' δφ_g'  =  S_noise,g
```

with the frequency-dependent effective spectrum `χ_eff,g(ω) = χ_g − Σ_i χ_d,i,g
β_i·iω/(iω+λ_i)` — the exact analog of the transient's backward-Euler weight
(`1/(1+λΔt) → iω/(iω+λ)`) — and the noise source `S_noise,g = −δΣ_r,g φ_0,g + …`
collecting every cross-section fluctuation multiplying the static flux. This is
structurally the transient within-step fixed point with the real shift `1/(vΔt)`
replaced by the imaginary shift `iω/v`, so the machinery is reused wholesale:
the within-group operator is the same matrix-free `GroupOperator` built with a
*complex* removal (complex-symmetric ⇒ solved by **COCG**), and the group/fission
coupling is closed by the same Anderson-accelerated Gauss-Seidel sweep.

The same solver runs the **SPN transport approximations** (`angular="sp3"`,
`"sp5"`, `"sp7"`): `iω/v` enters the even-moment U-form block as its time term
`θ`, the block stays complex-symmetric (COCG again), and the scalar flux drives
the source/fission/scatter through the block weights. Vacuum boundaries use the
exact **moment-coupled Marshak** condition (`marshak_vacuum=True`; diffusion's
`α=½` vacuum already *is* the Marshak condition), so the SPN noise matches
FEMFFUSION's `Full_SPN` vacuum treatment.

```python
from ndgpu import NoiseSolver, NoiseSource
import numpy as np

ns = NoiseSolver(grid, materials, material_map, kinetics=kin, bc=bc)
absorber = np.zeros(grid.shape, dtype=complex); absorber[i, j, 0] = 5e-4
src = NoiseSource(d_sigma_a=[np.zeros(grid.shape), absorber])   # vibrating thermal absorber
res = ns.solve(src, omega=2*np.pi*1.0)     # 1 Hz
dphi = res.relative()                       # δφ_g / φ_0,g  (complex, per group)
```

Validated against point-kinetics theory: for a homogeneous, fully reflected
(leakage-free) one-group reactor a uniform absorption fluctuation drives a flat
response whose complex amplitude is *exactly* the zero-power reactor transfer
function `G(ω) = 1/[iω(Λ + Σ_i β_i/(iω+λ_i))]` times the perturbation reactivity
— reproduced to **< 1e-6** (magnitude and phase) across five frequency decades
with six delayed families, response flat to machine precision. In a heterogeneous
near-critical core (2D TWIGL) a localized absorber shows the hallmark
**global-to-local transition**: the response follows the fundamental mode at low
frequency and localizes around the perturbation as ω rises. See
`examples/noise_transfer_function.py` and
`tests/validation/test_noise_point_kinetics.py`.

**Cross-checked against [FEMFFUSION](https://github.com/Zonni/FEMFFUSION)'s noise
module** on its own 1D two-group regression case (`test/1D_noise_SPN`: a 1 Hz
absorption fluctuation in one cell of a 300 cm slab), for both diffusion (vs
FEMFFUSION diffusion) and SP3 with the coupled Marshak vacuum (vs FEMFFUSION
`Full_SPN`, N=3). The two codes share the formulation — critical adjustment
(νΣ_f/k), `ω = 2πf`, the identical effective spectrum `χ_eff,g(ω)`, and the
Marshak vacuum boundary — so they differ only in spatial discretization (ndgpu's
cell-centred finite volume vs FEMFFUSION's continuous-Galerkin FE):

| case | mesh | Δk_eff | δφ rel-L2 (g1 / g2) | peak phase diff |
|---|---|---|---|---|
| diffusion | 60 cells | 0.8 pcm | 4.1e-3 / 5.5e-3 | 0.03° |
| diffusion | 120 cells | 0.2 pcm | 1.0e-3 / 1.4e-3 | 0.008° |
| SP3 Marshak | 60 cells | 0.5 pcm | 3.3e-3 / 5.7e-3 | 0.02° |
| SP3 Marshak | 120 cells | 0.9 pcm | 0.8e-3 / 1.4e-3 | 0.01° |

The field difference falls ~4× when the mesh halves — clean second-order
convergence to the same continuum solution, confirming the residual is purely
discretization. (Cost: FEMFFUSION assembles the coupled complex system and
solves it monolithically with preconditioned GMRES, ~13 ms / 47 iters; ndgpu's
matrix-free source iteration is slower on this near-critical, low-frequency case
— its worst regime, where the fission fixed point is near-neutral — at ~90 ms
diffusion, ~210 ms SP3.) See `examples/noise_femffusion.py`,
`tests/validation/test_noise_femffusion.py`, and
`ndgpu.benchmarks.build_femffusion_1d_noise`.

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

- Neutron noise: per-material kinetics (2D velocities/beta), δD leakage
  fluctuations, and a monolithic complex-Krylov solve for stiff near-critical /
  low-frequency cases (the source iteration's worst regime)
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
