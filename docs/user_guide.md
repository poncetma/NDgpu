# NDgpu user guide

NDgpu is a finite-volume multigroup reactor solver designed for fast,
device-resident calculations on structured grids. Its main application is a
homogenized full-core model on a body-fitted triangular grid, including
SPH-corrected diffusion, delayed-neutron transients, heat conduction, and
temperature feedback. The same calculation runs through NumPy on a CPU or
CuPy on an NVIDIA GPU.

This guide describes the public, human-oriented Python API. For the complete
builder reference and a validated TWIGL example, see
[model_api.md](model_api.md). For coupling internals and external thermal-code
integration, see [coupling.md](coupling.md).

## Main capabilities

| Capability | Public entry point | Current scope |
|---|---|---|
| Hexagonal/prismatic full-core models | `HexLattice.build()` → `TriReactor` | 2-D triangular and extruded 3-D triangular-prism grids |
| Cartesian models | `Model` | 1-D, 2-D, or 3-D structured grids |
| Restricted unstructured meshes | `MeshModel` | TPFA-compatible Gmsh 2.2 ASCII or assembled 2-D/3-D meshes; steady diffusion only |
| Criticality | `steady()` / `run()` | Diffusion; SP1/3/5/7 and SDP1/2/3 on the tri-grid |
| Importance | `steady(adjoint=True)` | Adjoint k-eigenvalue calculation |
| Neutron kinetics | `transient()` | Multigroup diffusion with delayed-neutron families |
| SPH hand-off | `TriReactor.with_materials()` | Reuse the same geometry with corrected constants |
| Thermal feedback | `configure_thermal()` | Material conductivity, heat capacity, sinks, Doppler and density feedback |
| Coupled calculations | `coupled_steady()`, `coupled_transient()`, `quasistatic_transient()` | In-process, device-resident diffusion/conduction coupling |
| Moving controls | `state_at(t)` | Cached same-shaped reactor states, including rotating drum frames |
| CPU/GPU portability | `device="auto"` | NumPy CPU fallback or CuPy/CUDA GPU execution |

Frequency-domain noise, discrete-ordinates transport, SPH-factor generation,
and custom coupling adapters are available through the advanced solver
interfaces. They are deliberately not hidden behind a single catch-all model
method.

## Which model builder should I use?

For a new hexagonal or prismatic reactor design, start with `HexLattice` and
compile it to a `TriReactor`. This is the most complete user path and the one
intended for SPH-corrected coupled transients.

| Design | Builder | Use it when |
|---|---|---|
| Hexagonal assembly lattice or microreactor | `HexLattice` | The design maps naturally to hex sites and triangular cells |
| Rectangular benchmark or Cartesian core | `Model` | Regions can be painted as axis-aligned boxes |
| Controlled unstructured/local-refinement mesh | `MeshModel` | The mesh meets the documented TPFA geometry and importer restrictions |
| Custom discretization or research method | Solver classes | You need direct access to operators, arrays, or an experimental method |

NDgpu is a homogenized diffusion-family solver, not a general CAD or
pin-resolved transport system. A design should be represented by macroscopic
cross sections over cells or assemblies. The tri-grid is especially suitable
for hexagonal cores, control drums, and axially extruded prismatic regions.

## Installation and device selection

From a checkout:

```bash
pip install -e .             # NumPy/SciPy CPU installation
pip install -e .[cuda12]     # add CuPy for a CUDA 12.x runtime
```

Every public solve accepts `device="cpu"`, `"gpu"`, or `"auto"`. Use `auto` in
portable scripts; it chooses CUDA when CuPy and a working GPU are available.
Requesting `gpu` explicitly is useful in production because a missing CUDA
installation then fails instead of silently running on the CPU.

```python
solution = core.steady(device="auto")
print(solution.device)
```

For a hosted GPU check, run
[`colab_coupled_transient_gpu_latest.ipynb`](../notebooks/colab_coupled_transient_gpu_latest.ipynb).
It exercises the current tri-grid API, steady solve, transient, coupled
transient, thermal subcycling, profiler counters, and an optional one-minute
3-D HP-MR workload. The proposed acceleration path for that long calculation
is documented in [quasistatic_acceleration_plan.md](quasistatic_acceleration_plan.md).

## 1. Define homogenized materials

All builders use `Material`. Lengths are in centimetres, macroscopic cross
sections are in cm⁻¹, and diffusion coefficients are in centimetres. Energy
groups are ordered fast to thermal.

```python
import ndgpu

fuel = ndgpu.Material(
    name="fuel",
    diffusion=[1.25, 0.35],
    sigma_a=[0.012, 0.120],
    nu_sigma_f=[0.0085, 0.185],
    sigma_s=[
        [0.0, 0.026],   # from fast to [fast, thermal]
        [0.0, 0.0],     # from thermal to [fast, thermal]
    ],
    chi=[1.0, 0.0],
)

reflector = ndgpu.Material(
    name="reflector",
    diffusion=[1.15, 0.25],
    sigma_a=[0.001, 0.020],
    nu_sigma_f=[0.0, 0.0],
    sigma_s=[[0.0, 0.035], [0.0, 0.0]],
)

absorber = ndgpu.Material(
    name="absorber",
    diffusion=[0.50, 0.12],
    sigma_a=[0.040, 0.80],
    nu_sigma_f=[0.0, 0.0],
    sigma_s=[[0.0, 0.010], [0.0, 0.0]],
)
```

`sigma_s[g_from][g_to]` is the scattering convention. `chi` defaults to all
prompt fission neutrons in group zero. `total` is needed by transport-corrected
SPN/SDPN formulations. Optional `kappa_fission` values are the groupwise
energy-release cross sections κΣf used to shape thermal power, not κ alone.
They are retained through SPH scaling. If every material omits them, power
shaping falls back to `nu_sigma_f` and the requested `total_power` still fixes
the absolute normalization.

Give thermally coupled materials unique, non-empty names. Name-keyed thermal
and feedback dictionaries avoid exposing the integer material ordering.

## 2. Build a 2-D or 3-D tri-grid reactor

Hex sites use axial `(R, C)` coordinates. Paint broad regions first: later
calls overwrite earlier ones.

```python
axial_reflector = ndgpu.Material(
    name="axial reflector",
    diffusion=[1.10, 0.23],
    sigma_a=[0.0015, 0.023],
    nu_sigma_f=[0.0, 0.0],
    sigma_s=[[0.0, 0.033], [0.0, 0.0]],
)

core = (
    ndgpu.HexLattice(pitch=18.0, refine=3)
    .set_disk(2, reflector)                  # 19-site background
    .set_disk(1, fuel)                       # central seven sites
    .set_site((0, 0), fuel)                  # individual override
    .set_boundary("vacuum")                 # radial edge
    .extrude(height=100.0, nz=10, boundary="vacuum")
    .add_axial_region(axial_reflector, z=(0.0, 10.0), replace=fuel)
    .add_axial_region(axial_reflector, z=(90.0, 100.0), replace=fuel)
    .set_kinetics(
        velocities=[1.0e7, 3.0e5],
        beta=[0.0065],
        decay=[0.08],
    )
    .build(name="demonstration prismatic core")
)
```

The main geometry operations are:

| Operation | Meaning |
|---|---|
| `set_site((R, C), material)` | Paint one assembly site |
| `set_sites(sites, material)` | Paint several sites |
| `set_ring(radius, material)` | Paint the `6 × radius` sites on one ring |
| `set_disk(radius, material)` | Paint all sites through a ring |
| `set_boundary(spec)` | Set the outer radial neutronic boundary |
| `extrude(height, nz, boundary=...)` | Turn the 2-D mesh into `nz` prism layers |
| `add_axial_region(material, z=(lo, hi), replace=...)` | Override selected layers |
| `set_drum(...)` | Add a rotating annular absorber arc |
| `build(samples=8, name=...)` | Rasterize once and return a reusable reactor |

Each hex contains `6 × refine²` triangles. Increase `refine` until important
integral results and control worths are insensitive to further refinement.
Choose axial layer boundaries that coincide with physical interfaces when
exact region heights matter; axial painting selects cells by their centres.

`core` is a `TriReactor`. Its `grid`, `materials`, `material_map`, `active`,
`mix_material`, and `mix_weight` remain accessible when an advanced workflow
needs the compiled arrays.

### Control drums

An absorber drum is represented as a volume-mixed annular arc, so a thin arc
can move smoothly across triangular cells:

```python
lattice = ndgpu.HexLattice(pitch=30.0, refine=4)
lattice.set_disk(2, reflector)
lattice.set_drum(
    (0, 2),
    body=reflector,
    absorber=absorber,
    inner_radius=12.25,
    outer_radius=13.25,
    arc_deg=90.0,
    angle_deg=180.0,
)
lattice.set_drum_angle((0, 2), 90.0)
drum_state = lattice.build(samples=8)
```

An angle of 0° points the absorber toward the core; 180° points it outward.
`samples` controls sub-cell area sampling for the absorber fraction.

## 3. Run steady and adjoint calculations

```python
diffusion = core.steady(
    method="diffusion",
    device="auto",
    tol_k=1e-8,
    tol_source=1e-7,
)
print(diffusion)

importance = core.steady(method="diffusion", adjoint=True)
sp3 = core.steady(method="sp3", device="gpu")
```

On a `TriReactor`, `method` can be `diffusion`, `sp1`, `sp3`, `sp5`, `sp7`,
`sdp1`, `sdp2`, or `sdp3`. `run` is an alias for `steady`.

The returned `ReactorResult` is deliberately ordinary Python/NumPy data:

```python
diffusion.k_eff
diffusion.reactivity_pcm
diffusion.flux                 # NumPy array, group first
diffusion.leakage_fraction
diffusion.absorbed_fraction
diffusion.raw                  # underlying solver result, when needed
```

Printing it gives convergence, neutron balance, leakage, flux peaking, and
volume/fission shares by material. Inactive raster padding and volume-mixed
drum absorber are accounted for in those diagnostics.

## 4. Apply SPH-corrected cross sections

SPH-factor generation is an explicit reference/coarse calculation because the
reference solution and region collapse are problem-specific. Once a
`SphResult` exists, corrected constants enter the user API without rebuilding
the mesh:

```python
sph_core = core.with_materials(
    sph_result.corrected_materials,
    name="SPH-corrected demonstration core",
)
sph_solution = sph_core.steady(method="diffusion", device="gpu")
```

The replacement list must retain the original material count and order.
Geometry, volume mixtures, kinetics, and thermal configuration are preserved.
See `examples/hpmr_sph_reference_families.py` for generation from SP3, SDP1,
and SDP2 reference solutions.

## 5. Run a neutron-only transient

Kinetics can be attached while building the lattice or later with
`core.set_kinetics(...)`. One velocity is required per energy group; `beta` and
`decay` contain one entry per delayed-neutron family.

```python
rodded_fuel = ndgpu.Material(
    name="fuel",
    diffusion=fuel.diffusion,
    sigma_a=fuel.sigma_a * [1.0, 1.002],
    nu_sigma_f=fuel.nu_sigma_f,
    sigma_s=fuel.sigma_s,
    chi=fuel.chi,
)

result = core.transient(
    t_end=1.0,
    dt=0.01,
    device="gpu",
    materials_at=lambda t: [
        rodded_fuel if (t > 0.2 and m.name == "fuel") else m
        for m in core.materials
    ],
)

result.times
result.power                 # relative power P(t)/P(0)
result.k0                    # initial critical adjustment
result.flux_numpy            # final flux transferred to NumPy
result.step_iterations
```

The solver first obtains the initial eigenstate and uses `k0` to make it
critical. Consequently an unperturbed transient remains at `P/P0 = 1`, even if
the physical model's initial `k_eff` is not exactly one.

`materials_at(t)` must preserve material count and ordering. Return cached
material objects while the state is unchanged so operator data can be reused.
For moving geometry, use `state_at(t)` and return a cached `TriReactor` with the
same grid shape.

## 6. Configure thermal feedback

The built-in conduction model is both a useful coupled solver and the reference
adapter for integration with a larger thermal core code.

```python
core.configure_thermal(
    {
        "fuel": ndgpu.ThermalMaterial(
            conductivity=0.25,
            heat_capacity=2.5,
            sink_coeff=0.015,
            sink_temperature=650.0,
        ),
        "reflector": ndgpu.ThermalMaterial(
            conductivity=0.35,
            heat_capacity=2.0,
        ),
        "axial reflector": ndgpu.ThermalMaterial(
            conductivity=0.32,
            heat_capacity=2.1,
        ),
    },
    total_power=50_000.0,
    feedback={
        "fuel": ndgpu.FeedbackSpec(
            reference_temperature=700.0,
            doppler=[2.0e-4, 0.0],
            expansion=0.0,
        ),
    },
    thermal_mask_bc=0.01,
    ambient_temperature=400.0,
)
```

`ThermalMaterial` describes cell conductivity, volumetric heat capacity, and
an optional linear heat sink. `FeedbackSpec.doppler` may be a scalar or one
coefficient per group; `expansion` is the optional frozen-geometry density
coefficient. Missing feedback entries mean zero feedback.

`total_power` is the power represented by the modeled domain. For a full 3-D
core, use the core power. For a 2-D model, use the power assigned to its modeled
unit thickness, not the entire plant rating.

## 7. Run coupled steady and transient calculations

```python
hot = core.coupled_steady(
    device="gpu",
    tol=1e-6,
    max_iter=100,
    anderson_depth=4,
)

coupled = core.coupled_transient(
    t_end=10.0,
    dt=0.01,                 # neutron step
    dt_thermal=0.10,         # thermal step; integer multiple of dt
    device="gpu",
    state_at=control_state,     # cached-state callback; see below
    profile=True,
)

# Slow control motion / long thermal follow-through: advance amplitude at dt,
# but correct the full spatial shape every 2 s and the adjoint every 5 shapes.
qs = core.quasistatic_transient(
    t_end=60.0, dt=0.2, dt_thermal=1.0, shape_dt=2.0,
    adjoint_every=5, device="gpu", state_at=control_state, profile=True,
    residual_tol=2e-3, fallback_residual=1e-2,
)
```

The coupled transient keeps flux, cross-section fields, precursors, power, and
temperature on the selected device. `dt_thermal` can be larger than the
neutron step because conduction evolves more slowly; power is accumulated
between thermal advances. This avoids forcing a GPU neutronics calculation
through a CPU-speed exchange on every small kinetics step.

The result exposes both physics and performance information:

```python
coupled.times
coupled.power                    # P(t)/P(0)
coupled.peak_temperature         # K at each output time
coupled.mean_temperature
coupled.temperature              # final host temperature field
coupled.flux                     # final device flux
coupled.steady                   # initial coupled equilibrium
coupled.phase_seconds            # populated by profile=True
coupled.counters                 # steps, iterations, rebuilds, transfers
```

`quasistatic_transient` defaults to time-dependent IQS treatment. It carries
the accepted fine-step spatial precursor history through each macro shape
solve, removes the corrector's independent amplitude component, maintains
the adjoint-weighted neutron-population coordinate, and corrects physical
fission power with the new shape's time importance. Its result distinguishes
`.amplitude` from `.power` and records `.power_shape_factor` plus the
shape/residual/fallback histories; for a fixed shape the factor is one and
amplitude equals power.
`residual_tol` forces an early correction; `fallback_residual` advances an
unsafe fine interval with the full diffusion equations. `iqs_predictor_tol`
halves future shape intervals when the coarse predictor disagrees with accepted
power and restores them after the event settles. Use
`shape_method="adiabatic"` to select instantaneous eigen shapes.
For production HP-MR drum transients, guarded adaptive IQS is preferred over
fixed-cadence IQS. The r4, 11-group GH200 calibration uses
`residual_tol=2e-3`, `fallback_residual=1e-2`,
`iqs_predictor_tol=2e-2`, and `adjoint_every=5`: it reduced maximum sampled
power error against the available full-diffusion reference from 5.708% to
0.173% at a 25.5% transient-time cost. The HPMR example applies this guard
profile automatically when IQS is selected; pass `--unguarded-iqs` only for
fixed-cadence benchmark comparisons. See
[`benchmark-results/hpmr-iqs-gh200`](../benchmark-results/hpmr-iqs-gh200/README.md).
For expensive adjoint solves, combine a larger `adjoint_every` with
`adjoint_residual_tol`: the former limits maximum adjoint age and the latter
refreshes early when the amplitude-free transposed-eigenproblem residual is too
large. A hard full-diffusion fallback always refreshes the adjoint.
Thresholds are model- and mesh-dependent: establish them against a shorter
full-diffusion reference before relying on them for a long production run.

Useful performance controls are `precond_degree`, `check_every`,
`thermal_precond_degree`, `thermal_check_every`, and
`thermal_diagnostics_every`. Start with the defaults. Profile a representative
full-core case before tuning them; small meshes often run faster on the CPU
because GPU launch overhead dominates.

Diffusion transients also reuse one persistent PCG workspace per energy group
by default, avoiding repeated solver-vector allocation. This does not change
iterations or tolerances. `reuse_krylov_workspaces=False` exists for profiling
and regression A/B runs; ordinary simulations should leave reuse enabled.

An experimental CPU-validated direct multigroup step is also available for
strongly upscattering diffusion problems:

```python
direct = core.transient(
    t_end=1.0, dt=0.02, device="cpu",
    step_solver="monolithic",
    multigroup_kwargs={"scatter_sweeps": 3, "energy_anderson": 1,
                       "inner_rtol": 0.1},
)

coupled_direct = core.coupled_transient(
    t_end=10.0, dt=0.02, dt_thermal=0.2, device="cpu",
    transient_kwargs={
        "step_solver": "monolithic",
        "multigroup_kwargs": {"scatter_sweeps": 3, "energy_anderson": 1,
                              "inner_rtol": 0.1},
    },
)
```

This solves the same analytically precursor-eliminated implicit BDF step as one
matrix-free block system using flexible GMRES. On 11-group 2-D HP-MR CPU
gates it reduced a tightly converged step by 2.5--3.3x. The default remains
`step_solver="fixed-point"`: the direct method still needs GPU memory and
throughput validation, is diffusion-only, and should be compared with the
default on a short representative interval before production use. Its
`step_iterations` and coupled `neutron_outer_iterations` count FGMRES applies,
not fission-source sweeps.

`energy_anderson=1` applies a depth-one dynamic correction between the group
subsweeps. On the 11-group 2-D HP-MR gate it reduced inner work by 17--21% and
CPU solve time by 13--18% at the conservative `inner_rtol=1e-3`. Because
FGMRES tests the true outer residual, an inexact `inner_rtol=0.1` is both safe
and substantially faster on that CPU gate. The T4 result has the opposite
ranking. A three-sample interleaved T4 gate gives median times of 2.310 s for
plain group PCG, 2.620 s for tolerance-based Anderson, and 2.098 s for the
reduction-free polynomial configuration. For the current GPU experiment use
four sweeps with `energy_anderson=1` and `inner_fixed_relaxations=1`; its 1.101x
gain over plain PCG is accepted as an opt-in result. Keep the CPU and GPU
configurations explicit until other devices and reactor libraries are measured.

The diffusion solver also supports constant- or explicitly nonuniform-step
BDF1--BDF6 on both in-step solver paths:

```python
bdf = core.transient(
    t_end=1.0, dt=0.01, device="cpu", time_scheme="bdf3",
    step_solver="monolithic",
    # A cached control frame changes abruptly at these step boundaries.
    bdf_restart_times=[0.25, 0.50, 0.75],
)

# Nonuniform widths must be positive and sum exactly to t_end.
event_aligned = core.transient(
    t_end=0.10, dt=[0.01, 0.01, 0.005, 0.005, 0.02, 0.05],
    time_scheme="bdf2",
)
print(event_aligned.time_orders)
```

BDF order ramps up as accepted history becomes available. A restart discards
flux and precursor history before the indicated step and forces BDF1 there.
Use it at discontinuous material/control frames and feedback-law corners.
BDF3--6 are A(alpha)-stable, not A-stable, so they require a stability and
time-step convergence comparison for the intended transient. Explicit step
schedules remain the deterministic production path. An experimental adaptive
width/order controller is available on monolithic BDF1--BDF6 diffusion
transients:

```python
adaptive = core.transient(
    t_end=3.0, dt=1e-4, time_scheme="bdf5",
    step_solver="monolithic", bdf_restart_times=[2.0],
    adaptive_bdf={"rtol": 1e-3, "min_dt": 1e-6, "max_dt": 0.1,
                  "automatic_order": True,
                  # Optional practical-tolerance optimization. Failed steps
                  # still shrink by at least a factor of two.
                  "rejection_strategy": "error",
                  "reject_max_factor": 0.5},
)
print(adaptive.step_widths, adaptive.local_errors,
      adaptive.rejected_steps)
```

Rejected attempts roll back flux, precursor, and BDF history and do not invoke
the accepted `on_step` hook. The controller measures flux and precursor
defects; coupled drivers may append temperature or other external accepted
state through `adaptive_error_state`. The LRA driver uses that seam for its
fully implicit temperature. With `automatic_order=True`, full-state
`q-1/q/q+1` predictor defects compete by proposed safe width; event restarts
still force order one. The corrected LRA CPU gate shows a 5.18x matched-error
speedup over adaptive backward Euler. Error-scaled rejection with a 0.5 cap
reduced practical-tolerance LRA work by 5.6%; the default factor-two policy is
retained for paper reproduction. Rejected-attempt time, width, and order are
available in `rejected_times`, `rejected_step_widths`, and
`rejected_time_orders`. Do not treat `rtol` as equivalent to
FEMCORE/LSODE tolerances or make this the default until the HP-MR and GPU gates
also pass.
`ndgpu.BDF.error_norm(...)` and `ndgpu.BDFStepController` remain public for
benchmark and custom-driver development.

`coupled_transient` supports the same adaptive controls through
`transient_kwargs`. Thermal exchange times are exact non-restarting adaptive
endpoints, and unequal neutron-step powers are averaged by elapsed time rather
than step count. The coupled HP-MR gate shows why method speedups must be
matched by error: its mild drum motion stays almost entirely at BDF order one,
so adaptive BDF and adaptive backward Euler have essentially the same cost.

On GPU, the adaptive error estimator performs one two-scalar host transfer per
complete multigroup/precursor norm; it does not synchronize once per field.
BDF history extrapolation uses the `adaptive` fused-kernel family. Monolithic
steps reuse bounded FGMRES basis/work storage and, when the free-VRAM guard
allows it, batched energy-group contractions. Coupled profiles expose
`group_batch_active` and `fgmres_workspace_bytes`; check both before comparing
timings. The executable 3-D, 11-group accuracy and performance gate is
[`notebooks/colab_hpmr3d_adaptive_bdf_gpu.ipynb`](../notebooks/colab_hpmr3d_adaptive_bdf_gpu.ipynb).

For a long run, `multigroup_kwargs={"coarse_correction": True}` adds an
adjoint-weighted rank-one amplitude correction. It reduces group work but first
computes an adjoint eigenstate, so it is normally a loss on short histories. An
advanced IQS/coupling driver that already has a compatible device-resident
adjoint may pass it as `coarse_adjoint` to avoid that startup solve.

CUDA-graph PCG is currently experimental. Set matching values through the
low-level solve controls, for example
`linsolve_kwargs={"check_every": 3, "graph_block": 3}`. Capture is used only
for a compatible fused CuPy operator with at most 65,536 cells per group;
otherwise the solve continues normally and records the reason in
`result.cuda_graph_errors`. Successful runs report capture/replay counts.

For an opt-in mixed-precision GPU experiment, pass
`precond_dtype="float32"` with `precond_degree=1` or higher. Only the
polynomial preconditioner uses FP32; physical state, true residuals, and
stopping decisions remain FP64. Unsafe or failed mixed solves automatically
retry with the native FP64 preconditioner, reported as
`result.mixed_precision_fallbacks`. Keep this option disabled until it wins on
the target GPU and problem size.

### Cached moving-control states

Build control positions ahead of a transient and select among them in the
callback. Do not rebuild or rasterize geometry inside every time step.

```python
frames = {
    0.0: inserted_core,
    90.0: half_core,
    180.0: withdrawn_core,
}

def control_state(t):
    if t < 1.0:
        return frames[180.0]
    if t < 2.0:
        return frames[90.0]
    return frames[0.0]
```

All frames must have the same grid shape. The base reactor owns the thermal
configuration; a state supplies the time-dependent material and volume-mix
arrays.

## Cartesian and unstructured alternatives

For a rectangular model, paint boxes with `Model`:

```python
box = (
    ndgpu.Model(size=(120.0, 120.0, 200.0), cells=(48, 48, 80))
    .fill(reflector)
    .add_box(fuel, x=(20, 100), y=(20, 100), z=(20, 180))
    .set_boundary("vacuum")
)
print(box.run(method="diffusion", device="auto"))
```

For a compatible Gmsh 2.2 ASCII model, assign material by retained cell tag or
centroid:

```python
mesh_model = (
    ndgpu.MeshModel("core.msh")
    .fill(reflector)
    .assign(fuel, tag=1)
    .assign(absorber, where=lambda centre: centre[2] > 180.0)
    .set_boundary("vacuum")
)
mesh_solution = mesh_model.run(device="auto")
```

`MeshModel` is not a general CAD-mesh interface. Its cell-centred two-point-flux
operator requires orthogonal or near-orthogonal cell-centre/face geometry. The
reader supports only first-order triangles/quads or tets/hexes/prisms, discards
boundary tags, and applies one boundary condition to all exterior faces. Only
recursive midpoint nonconformity in 2-D is supported; 3-D meshes must conform.
The intended advanced use is controlled local refinement, such as the HP-MR
drum bands, rather than arbitrary Gmsh output. Read the complete
[unstructured mesh scope and limitations](unstructured_mesh_scope.md) and run a
mesh-convergence comparison before adopting a new mesh family.

`MeshModel` currently provides steady forward diffusion on one CPU process or
one GPU. Use `TriReactor` for the complete SPN/SDPN, transient, and coupled
workflow. The developing MPI path targets the structured tri-grid solvers, not
`MeshModel`.

## Accuracy and validation checklist

A runnable result is not automatically a predictive reactor model. Before
using a new design for engineering conclusions:

1. Confirm units, group order, scattering orientation, fission spectrum, and
   delayed-neutron data against the cross-section source.
2. Check the printed material volume shares and plot or inspect the material
   map before interpreting `k_eff`.
3. Repeat the steady solve with increased radial `refine` and axial `nz`.
4. Repeat transients with a smaller neutron `dt` and, for coupled cases, a
   smaller `dt_thermal`.
5. Check the neutron balance and solver convergence status.
6. Validate control worth, temperature coefficients, and integral power
   against a higher-fidelity calculation or experiment.
7. Generate SPH factors using a representative transport reference, region
   definition, leakage environment, and control state.
8. Compare CPU and GPU results for one reduced case before a long production
   campaign.

The repository separates exact verification tests from published benchmark
validation in `tests/verification/` and `tests/validation/`.

## Complete examples

| Example | Demonstrates |
|---|---|
| `examples/custom_tri_reactor.py` | New 3-D design, named thermal data, coupled transient |
| `examples/hpmr_hexlattice.py` | Full hex lattice, 12 control drums, worth sweep |
| `examples/hpmr_coupled_transient.py` | Full-diffusion or quasi-static coupled transient and profiling |
| `examples/hpmr_sph_reference_families.py` | SPH generation and reference-method comparison |
| `examples/twigl_benchmark.py` | Validated static and transient Cartesian workflow |
| `examples/unstructured_mesh.py` | Gmsh/finite-volume `MeshModel` workflow |

## Public API at a glance

```text
Material(...)

HexLattice(pitch, refine)
  .set_site(...) / .set_sites(...) / .set_ring(...) / .set_disk(...)
  .set_drum(...) / .set_drum_angle(...)
  .set_boundary(...)
  .extrude(height, nz, boundary=...)
  .add_axial_region(material, z, replace=...)
  .set_kinetics(...)
  .build(samples=8, name=...) -> TriReactor

TriReactor
  .steady(method=..., device=..., adjoint=...) -> ReactorResult
  .transient(t_end, dt, materials_at=... | state_at=...)
  .with_materials(corrected_materials) -> TriReactor
  .configure_thermal(thermal_materials, total_power=..., feedback=...)
  .coupled_steady(...)
  .coupled_transient(t_end, dt, dt_thermal=..., state_at=..., profile=...)
  .quasistatic_transient(t_end, dt, shape_dt=..., adjoint_every=...)

Model(size, cells)
  .fill(...) / .add_box(...) / .set_boundary(...) / .set_kinetics(...)
  .run(...) / .transient(...)

MeshModel(mesh)
  .fill(...) / .assign(...) / .add_box(...) / .set_boundary(...)
  .run(...)
```

When the public objects do not expose a research-specific option,
`ReactorResult.raw`, `TriReactor.coupling_context()`, and the compiled grid/map
attributes provide a controlled path to the low-level solver APIs without
duplicating geometry construction.
