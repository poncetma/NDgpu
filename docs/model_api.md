# The NDgpu Model API

The `Model` family is a high-level front end for defining and running a reactor.
The low-level solver classes (`DiffusionEigenSolver`, `SP3EigenSolver`, the
triangular and unstructured solvers) are fully general but ask you to build a
grid, hand-assemble a `material_map` array, and read a terse `Result`. The
builders here wrap the common cases: define a reactor in centimetres, paint it
with named materials, set boundary conditions by name, and get a transparent,
human-readable report.

Three builders share one report type (`ReactorResult`):

| Builder | Geometry | Methods |
|---|---|---|
| [`Model`](#model--structured-cartesian) | structured Cartesian, 1-D / 2-D / 3-D | diffusion, SP3, adjoint, **transient** |
| [`MeshModel`](#meshmodel--unstructured) | arbitrary unstructured mesh (Gmsh or assembled), 2-D / 3-D | diffusion |
| [`HexLattice`](#hexlattice--triangular) | hexagonal assembly lattice on the triangular solver | diffusion, SP3 |

Materials are ordinary `ndgpu.Material` objects throughout, so defining cross
sections is unchanged — only the grid/array bookkeeping is removed.

---

## Materials

A `Material` is a homogenized multigroup cross-section set (macroscopic, 1/cm; D
in cm). Group order is fast → thermal.

```python
from ndgpu import Material

fuel = Material(
    name="fuel",
    diffusion=[1.26, 0.35],          # D_g
    sigma_a=[0.012, 0.121],          # absorption
    nu_sigma_f=[0.0085, 0.185],      # nu * fission production
    sigma_s=[[0.0, 0.026],           # scatter[g_from][g_to]; here fast -> thermal
             [0.0, 0.0]],
    chi=[1.0, 0.0],                  # fission spectrum (defaults to all in group 0)
)
```

`sigma_s`, `chi`, and `total` are optional. A material with `nu_sigma_f = 0` is a
non-fissile reflector/absorber. `total` (Σ_t) is only used by SP3.

---

## `Model` — structured Cartesian

### Building the geometry

`Model(size, cells)` defines a rectangular box in centimetres; the length of
`size`/`cells` (1, 2 or 3) sets the dimensionality. Paint it with `fill` (the
background) and `add_box` (rectangular overrides, ranges in cm; a `None` axis
spans the whole core). Methods return `self`, so they chain.

```python
import ndgpu

model = (
    ndgpu.Model(size=(120, 120, 120), cells=(40, 40, 40))   # 3-D, cm
    .fill(reflector)                                        # background material
    .add_box(fuel, x=(30, 90), y=(30, 90), z=(30, 90))      # central fuel block
    .set_boundary("vacuum")
)
```

Later `add_box` calls overwrite earlier ones. Inspect what you built with
`model.material_map` (an integer array, squeezed to the model's dimension) and
`model.materials` (index 0 is the background).

> **Distinct regions need distinct `Material` objects.** Regions are keyed by
> object identity: painting two boxes with the *same* `Material` gives them the
> same index. If two regions must be controlled independently (e.g. one is
> perturbed in a transient), give them separate `Material` instances — even if
> their cross sections are identical.

### Boundary conditions

`set_boundary` takes a face spec by name: `"vacuum"`, `"reflective"`,
`"zero-flux"`, or a non-negative albedo `α` (the law `J_net = α · φ_surface`).

```python
model.set_boundary("vacuum")                                    # every face
model.set_boundary(x=("reflective", "zero-flux"), y="vacuum")   # per axis / per face
```

A single string applies to all real axes; `x`/`y`/`z` override one axis (a face
spec, or a `(lo, hi)` pair for the axis's two faces). Collapsed axes of a 1-D/2-D
model are always reflective. Use `"zero-flux"` (Dirichlet) to compare against the
analytic bare-reactor buckling; `"vacuum"` (a Robin/Marshak condition) is the
physical leakage boundary.

### Running

```python
result = model.run(method="diffusion", device="auto")
print(result)                    # the human-readable report
result.k_eff                     # float
result.flux                      # (G, nx, ny, nz) NumPy array
result.reactivity_pcm            # (1 - 1/k) * 1e5
result.leakage_fraction          # fraction of produced neutrons that leak out
```

`method` is `"diffusion"` or `"sp3"`; `device` is `"auto"` / `"cpu"` / `"gpu"`.
Tolerances default to `tol_k=1e-6`, `tol_source=1e-5`.

The report:

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

The balance (`absorbed + leaked = 100%`) comes from the reaction rates
`Σ_a·φ` / `νΣ_f·φ` and `leakage = production/k − absorption`, so it is correct
for diffusion and SP3 alike.

### Adjoint (importance)

`run(adjoint=True)` solves the adjoint k-eigenproblem: the same eigenvalue, but
the "flux" is the neutron *importance* (how much a neutron born at each point
contributes to the chain reaction). It is the weighting function for
perturbation theory and adjoint-weighted kinetics.

```python
adj = model.run(adjoint=True)
adj.k_eff                        # equals the forward eigenvalue
```

The report labels the importance solution and omits the physical balance (an
importance function is not a neutron population).

### Transients

Every transient starts from a steady state. `Model.transient` makes this
explicit: it solves the t = 0 equilibrium first, uses its eigenvalue `k0` to
normalise the fission source (the **critical adjustment** — so an *unperturbed*
run stays at `P/P0 = 1`), then marches in time. Both the power history and the
initial steady solution come back on the result.

```python
model.set_kinetics(velocities=[1e7, 3e5], beta=[0.0065], decay=[0.08])

result = model.transient(
    t_end=3.0, dt=0.02,
    materials_at=lambda t: [fuel, rod if t >= 0.5 else coolant],   # index order!
)

result.k0            # initial eigenvalue (fission source normalised by it)
result.steady        # a full ModelResult for the t = 0 state
result.times         # (n_steps + 1,)
result.power         # P/P0 at each time
result.peak_power, result.peak_time, result.final_power
```

`kinetics` is `Kinetics(velocities, beta, decay)` — one velocity per group, one
`beta`/`decay` per delayed-neutron family — set via `set_kinetics` or passed to
`transient(kinetics=...)`. The perturbation is `materials_at(t) → list[Material]`
returning the materials **in the model's index order** at time `t`; return the
*same* objects while nothing changes so the solver reuses its operators. Omit it
for an unperturbed run. The `at=` escape hatch takes a full
`t → (materials, material_map)` callback when the geometry itself must move.

The report includes a power-vs-time sparkline:

```
  initial steady state : k0 = 0.913100  (fission source normalised by k0; unperturbed -> P/P0 stays 1)
  time span            : 0 -> 0.5 s, dt = 0.005 s, 100 steps, ...
  power P/P0:
    peak      : 2.1310 at t = 0.5 s
    final     : 2.1310 at t = 0.5 s
    trace     : ▁▁▂▃▃▄▄▅▅▆▆▆▇▇▇▇████...
```

---

## `MeshModel` — unstructured

Runs the matrix-free finite-volume solver on an arbitrary mesh — a Gmsh `.msh`
file, or a `Mesh` assembled with `assemble_mesh` / `assemble_mesh_3d` — in 2-D or
3-D (triangles, quads, tets, hexes, prisms). Materials are assigned by physical
region rather than by box only:

```python
mm = (
    ndgpu.MeshModel("core.msh")                  # or MeshModel(mesh_object)
    .fill(reflector)
    .assign(fuel, tag=1)                          # by Gmsh physical tag
    .assign(absorber, where=lambda c: c[2] > 100) # by a predicate on the cell centroid
    .add_box(control, x=(40, 60), y=(40, 60))     # by centroid box
    .set_boundary("vacuum")                       # one albedo on all boundary faces
)
print(mm.run(device="auto"))
```

`assign` takes either `tag=` (match the mesh's per-cell tag) or `where=` (a
boolean mask over cells, or a callable receiving each cell centroid). The report
is volume-weighted, so material fractions and the balance are correct on
non-uniform meshes. `MeshModel` is diffusion-only (there is no SP3 mesh solver).

---

## `HexLattice` — triangular

A hexagonal lattice of assemblies solved on the body-fitted triangular solver
(the HP-MR geometry family). Place assemblies at axial hex coordinates `(R, C)`:

```python
lat = ndgpu.HexLattice(pitch=20.0, refine=4).set_boundary("vacuum")
lat.set_site((0, 0), fuel)
for rc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
    lat.set_site(rc, reflector)

diffusion = lat.run(method="diffusion")
transport = lat.run(method="sp3")          # SP3 captures the transport correction
```

`refine` sets `6·refine²` triangles per hex. `method` is `"diffusion"` or
`"sp3"`, and `adjoint=True` is available. A void placeholder fills the lattice
padding automatically; the report counts only the active triangles.

### Control drums

A hex site can instead be a **control drum** — an assembly body carrying a thin
absorber *arc* that rotates to change reactivity. `set_drum` volume-mixes the arc
into the drum cells by area fraction (so it is represented smoothly even when
thinner than a triangle, and varies continuously with rotation):

```python
lat.set_drum((0, 3), body=beryllium, absorber=b4c,
             inner_radius=12.25, outer_radius=13.25,   # the annular arc, cm from the hex centre
             arc_deg=90,                                # angular span of the arc
             angle_deg=180)                             # 0 = arc outward (withdrawn), 180 = toward core
```

`angle_deg` is measured from the outward radial direction, so sweeping it from 0
to 180 for every drum traces the **drum-worth curve**. Any number of drums may be
placed and rotated independently. `run(..., samples=8)` controls the arc
sub-sampling resolution. The report's neutron balance accounts for the mixed
absorber. This reproduces `ndgpu.benchmarks.build_hpmr2d`'s polar drum model
exactly; see `examples/hpmr_hexlattice.py`, which builds the full HP-MR
microreactor — 55 assemblies and 12 drums — and sweeps the worth curve.

---

## Worked example: the TWIGL benchmark (validated against literature)

The TWIGL seed–blanket benchmark (Hageman & Yasinsky, 1969) is a two-group,
80 × 80 cm quarter core with three rectangular regions — a blanket, an L-shaped
seed, and a central perturbable seed — reflective on the inner faces and
zero-flux on the outer ones. It has published values for both the **static
eigenvalue** and the **transient power**, so it validates the whole Model API.

```python
import numpy as np
import ndgpu
from ndgpu import Material

seed_xs = dict(diffusion=1 / (3 * np.array([0.238095, 0.83333])),
               sigma_a=[0.010, 0.150], nu_sigma_f=[0.007, 0.200],
               sigma_s=[[0.0, 0.01], [0.0, 0.0]], chi=[1, 0])
blanket_xs = dict(diffusion=1 / (3 * np.array([0.25641, 0.66667])),
                  sigma_a=[0.008, 0.050], nu_sigma_f=[0.003, 0.060],
                  sigma_s=[[0.0, 0.01], [0.0, 0.0]], chi=[1, 0])

blanket = Material(name="blanket", **blanket_xs)
seed = Material(name="seed", **seed_xs)
seed_c = Material(name="seed (perturbable)", **seed_xs)   # distinct object -> its own region

model = (
    ndgpu.Model(size=(80, 80), cells=(40, 40))
    .fill(blanket)
    .add_box(seed, x=(0, 24), y=(24, 56)).add_box(seed, x=(24, 56), y=(0, 24))  # L-shaped seed
    .add_box(seed_c, x=(24, 56), y=(24, 56))                                    # central seed
    .set_boundary(x=("reflective", "zero-flux"), y=("reflective", "zero-flux"))
    .set_kinetics(velocities=[1.0e7, 2.0e5], beta=[0.0075], decay=[0.08])
)

print("static k_eff:", model.run(tol_k=1e-8, tol_source=1e-7).k_eff, " (published 0.91321)")

# Ramp transient: the central seed's thermal absorption falls over 0.2 s.
def ramp(t):
    f = 1.0 - 0.11667 * min(max(t, 0.0), 0.2)
    pert = Material(name="pert", **{**seed_xs, "sigma_a": [0.010, 0.150 * f]})
    return [blanket, seed, pert]

res = model.transient(t_end=0.5, dt=0.005, materials_at=ramp)
at = lambda tt: res.power[np.argmin(np.abs(res.times - tt))]
print(f"ramp P(0.1) = {at(0.1):.2f} (published 1.31),  P(0.5) = {at(0.5):.2f} (published 2.11)")
```

Result (40 × 40 cells):

| quantity | Model API | published | notes |
|---|---|---|---|
| static k_eff | 0.91310 | 0.91321 | −11 pcm; → −3 pcm at 80 × 80 (2nd-order) |
| ramp P(0.1 s) | 1.309 | 1.31 | |
| ramp P(0.5 s) | 2.110 | 2.11 | |
| step P(0.1 s) | 2.061 | 2.06 | |
| step P(0.5 s) | 2.131 | 2.13 | |

The runnable version is `examples/twigl_benchmark.py`, and
`tests/validation/test_model_twigl.py` guards these numbers against the published
references.

---

## Reference

- The builders and `ReactorResult` live in `ndgpu/model.py`.
- For capabilities the Model family does not wrap (SPH homogenization, the raw
  transient `problem_at` with moving geometry, custom preconditioners), use the
  solver classes directly — see the other `examples/`.
- TWIGL: R. A. Hageman & J. B. Yasinsky, "Comparison of alternating-direction
  time-differencing methods … ", *Nucl. Sci. Eng.* (1969); cross sections here
  match the FEMFFUSION `2D_TWIGL` example.
