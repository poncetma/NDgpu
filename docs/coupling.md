# Coupled neutronics / thermal calculations

`ndgpu` solves the neutronics of a reactor core. This document covers the
thermal side added alongside it — a conduction solver with a volumetric
heat-pipe sink — and the two ways the two physics are coupled: **internally**,
by a fixed-point driver inside the package, and **externally**, as two separate
processes exchanging fields through [preCICE](https://precice.org).

The external coupling exists to be checked against the internal one. They run
the *same* two half-steps, so anything they disagree about is coupling
machinery rather than physics.

---

## 1. The thermal model

```
-div(k(r) grad T)  +  h(r) (T - T_sink(r))  =  q'''(r)
```

with a Robin surface law `-k dT/dn = alpha (T_s - T_inf)`.

A heat-pipe microreactor has no coolant. Each fuel assembly is pierced by
alkali-metal heat pipes whose evaporator sits at a nearly uniform temperature
and which draw power in proportion to the local solid-to-pipe temperature
difference. Homogenized over the assembly, that is the volumetric conductance
`h` to a fixed `T_sink`, nonzero only in the fuel. Everything else — reflector,
drum bodies, absorber — merely conducts, losing a little heat to the vessel
through a weak surface coefficient.

The sink is also what makes the problem well posed. With an adiabatic outer
boundary and no sink, a steady state with an internal source does not exist;
`ConductionSolver` detects that case and says so rather than handing a singular
system to CG.

### No new discretization

This is the operator the diffusion solver already builds,
`-div(D grad .) + Sigma_r`, read with `D -> k` and `Sigma_r -> h`. So the
conduction solve inherits, unchanged and bit-for-bit:

- harmonic-mean face coefficients (exact for piecewise-constant `k`),
- the `active` mask for non-rectangular cores,
- the Robin boundary machinery,
- Cartesian, cylindrical r-z, triangular and extruded-prism meshes,
- the matrix-free CPU/GPU stencil and the Jacobi-CG solve.

`ConductionSolver` picks `GroupOperator` or `TriGroupOperator` from the grid
type; both take arbitrary per-cell `D` and `removal` arrays and carry no
neutronics semantics.

### The ambient boundary term

A Robin surface at temperature `T_inf` contributes a source that has to match
the operator's own boundary coefficients exactly — including the cylindrical
metric weights and the core-surface faces of a masked mesh. Rather than
re-deriving `robin_face_term` (and getting a weight subtly wrong), the solver
reads the term back out of the operator:

```python
w   = op.rhs_weight if op.rhs_weight is not None else 1.0
bnd = op.apply(ones) - h * w                 # the Robin surface terms, alone
rhs = w * (q + h * T_sink) + T_inf * bnd
```

On a **constant** field the interior face couplings telescope to zero, so
`A.1 = h*w + (surface terms)` exactly. Subtracting the sink leaves the surface
terms whatever the geometry.

Two consequences worth knowing:

- Excised cells fall out for free. The operators give them a unit diagonal and
  no couplings, so `A.1 = 1` there and they solve to exactly `T_inf` — inert,
  and they drop out of the energy balance identically. No masking needed.
- **Discontinuity factors break it.** With `df` set the two sides of a face
  carry different weights, a constant is no longer in the leakage null space,
  and the identity picks up a spurious interior source. `ConductionSolver`
  raises rather than mis-solving.

### The energy balance is exact

Summing the discrete equations over all cells makes interior face couplings
cancel in pairs (the operator is symmetric), leaving

```
Sum_c V_c q_c  ==  Sum_c V_c h_c (T_c - T_sink)  +  Sum_c V_c bnd_c (T_c - T_inf)
   fission heat            heat-pipe removal              vessel loss
```

as an **identity of the discretization**, not an approximation that improves
with mesh. That makes `ThermalResult.balance_residual` a real check: a sign
error, a dropped metric weight or a mis-scaled boundary term all break it while
CG converges perfectly happily. It runs at ~1e-13 on the HP-MR core.

---

## 2. Power density

A k-eigenvalue flux carries an arbitrary normalization, so criticality gives a
power *shape* and nothing else. The absolute source comes from the shape plus
one imposed number, the rated thermal power:

```
q'''(r) = P_rated * kappa_Sigma_f . phi(r) / sum_cells kappa_Sigma_f . phi V
```

Normalizing this way means the absolute units of the library's `kappaFission`
never enter — whatever they are, they cancel — and only its *shape* is used.
That is the honest reading, because multigroup libraries disagree on those
units while the reactor's power is a design specification.

Power is proportional to `kappa*Sigma_f`, **not** `nu*Sigma_f`. Both vanish
outside the fuel, so either puts heat in the right place, but `nu` varies by
group (on the HP-MR's 11-group set `kappaSigma_f/nuSigma_f` spans 1.12x), so
nu-weighting tilts the distribution by the local spectrum. `power_density`
falls back to `nu_sigma_f` only when no material carries `kappaFission`, and
`fission_energy_xs` reports which was used.

> Two bugs fixed while wiring this up: `griffin_xs.volume_homogenize` dropped
> `kappa_fission`, so every homogenized HP-MR core silently fell back to
> nu-weighting; and `Material.kappa_fission` was never converted to an array in
> `__post_init__`, unlike every other cross section.

---

## 3. Temperature feedback

Two linearized effects about the temperature at which the cross sections were
tabulated:

**Doppler** — resonance broadening exposes more of each self-shielded resonance
to the flux:

```
dSigma_a,g(T) = c_D * Sigma_a,g(T_ref) * (sqrt(T) - sqrt(T_ref))
```

applied **additively to absorption**. Never as a scale on `removal`: removal is
`sigma_a + out-scatter`, so scaling it would drag the scattering along and
change the spectrum as well as the absorption. This is why `Fields` now keeps
`sigma_a` separately.

**Density / expansion** — `f(T) = 1 - beta_exp (T - T_ref)`, with every
macroscopic cross section scaled by `f` and `D` by `1/f`. Unlike Doppler this
one *is* legitimately multiplicative on removal, since it scales absorption and
the whole scattering matrix equally. `chi` is untouched: it is a normalized
emission spectrum, not a cross section.

*Honest limitation:* the mesh does not move, so `beta_exp` is a density effect
on frozen geometry — a standard reduction of a fuel-density coefficient, but
not core expansion, and it should not be reported as one.

### Where it plugs in

`Fields.__init__` takes an optional `xs_update(fields)` callable, run as the
last step of construction and free to modify the assembled per-cell arrays.
`ThermalFeedback.hook(T)` builds one. Every eigen solver accepts `xs_update=`
and forwards it.

`Fields` is the right seam because `TriDiffusionEigenSolver`, the SP3/SPN/SDPN
families and all their `Tri*` variants override only `_build_operators` — so
they inherit the hook for free — and because the transient and noise solvers
build `Fields` too, should they ever want it.

Two things to know:

- Operators **copy** their coefficients at construction, so mutating a `Fields`
  after its solver exists changes nothing. A driver that varies the state must
  rebuild the solver — measured at 12 ms against a 530 ms solve on the 2-group
  HP-MR at refine 6, and a smaller share again on the 11-group core. Not worth
  a second, incremental code path to keep bit-consistent with the first.
- `TransientSolver` rebuilds its fields only when `problem_at(t)` returns
  different *objects*, so a state-dependent hook would silently freeze at
  `t = 0`. Coupled kinetics needs that trigger widened first.

---

## 4. The coupled steady state

```
T -> cross sections -> k-eigenvalue solve -> phi -> q''' -> conduction -> T'
```

**What is computed, and what is not.** The eigenvalue solve renormalizes its
flux, so the power *level* is imposed and not computed. Feedback cannot change
how much heat the core makes; it changes where the heat is made, and it changes
`k`. That is the standard steady-state coupled formulation, and it is why the
iteration is strongly contractive — measured contraction factor ~8e-3 per
iteration on the HP-MR, converging in 4–6 Picard steps with no acceleration
needed. `test_coupling.py` asserts that ratio stays below 0.5; if it ever fails
the physics is wrong, not the accelerator.

The reportable result is `k` at hot full power against the cold unfed core —
the **temperature defect** — and, through `criticality_search`, the drum angle
that holds the core critical once that defect is paid for.

### Convergence tolerance has a floor

The inner eigen solve converges only to `tol_source`, and that noise reappears
as jitter in the temperature. On the 11-group HP-MR at `tol_source = 1e-9` the
coupled residual settles around 1e-6 K on a ~430 K span. Measured:

| coupled `tol` | iterations | wall | k_eff | peak T |
|---|---|---|---|---|
| 1e-8 | 4 | 23 s | 1.1541105 | 829.2817 K |
| 1e-10 | 31 | 164 s | 1.1541105 | 829.2817 K |

Same answer to 7 digits; the tighter tolerance just chases noise. Keep `tol` at
least ~50x above `tol_source / span`.

### Warm starting: the flux, not `k`

Reusing the previous flux is safe and saves time. Reusing the previous `k` as
well is **not**: the power iteration stops on the change between successive
outers, so seeding `k_guess` makes the first `|dk|` vanish and the solve exits
before the flux has responded to the new cross sections. Measured, that froze
the coupling after one iteration and reported a `k` 0.9 pcm off, with a
residual of exactly zero that looked exactly like convergence.

---

## 5. The preCICE coupling

Two processes — `examples/precice/neutronics.py` and `thermal.py` — exchanging
`Power` (W/cm³) and `Temperature` (K).

**Volume coupling.** The meshes are point clouds of cell centroids over the
whole core, not surfaces. preCICE needs connectivity only for projection-based
mappings; `nearest-neighbor` wants coordinates alone.

**The mapping is exact.** Both participants build their vertex list from the
single function `ndgpu.coupling.coupling_vertices`, on identically-constructed
problems, so every vertex's nearest neighbour is itself at distance zero: the
mapping is the identity permutation and data crosses preCICE bit-exact. That is
what makes the comparison against the internal driver a test of coupling
machinery alone. If the two sides ever drift, preCICE will *not* complain — it
will map each vertex to a nearby wrong cell and the coupling quietly becomes a
smoother, which is why there is an explicit exactness test.

**Neither participant contains physics.** Both call `neutronics_step` and
`thermal_step` from `ndgpu.coupling`, the same functions `CoupledSolver` calls.
If each script re-implemented its half, the two couplings agreeing would only
show that the same assumptions were made twice.

**Steady state as pseudo-time.** One time window, `serial-implicit`, with the
iterations inside it as the Picard steps.

Config points that bite:

- `waveform-degree="0"` requires `substeps="false"` on the exchanges — with
  constant interpolation there is nothing to sample.
- `initialize="true"` goes on the **Temperature** exchange, not Power.
  `Neutronics` is `first`, so it reads before `Thermal` has run in the window;
  `Thermal` is therefore the participant whose `requires_initial_data()`
  returns true. Power needs none — it is written before it is read, inside the
  iteration.
- In `serial-implicit`, acceleration acts on the **second** participant's data,
  i.e. `Temperature`. This is why the internal driver relaxes `T` and not `q` —
  otherwise the two loops are different iterations and lockstep is meaningless.
- XML comments cannot contain `--`.

### Running it

The system `libprecice` on this machine is an Ubuntu 24.04 build on a 22.04
box; it needs GLIBC_2.38 and cannot load, so `pip install pyprecice` (source-only
on PyPI) would link against it and fail at import. Use conda-forge:

```bash
conda create -n ndgpu-precice --override-channels -c conda-forge \
    python=3.13 pyprecice numpy scipy pytest
conda run -n ndgpu-precice python -m pip install -e . --no-deps
conda run -n ndgpu-precice bash examples/precice/run.sh --refine 4 --groups 11
```

`examples/precice/precice-config.xml` uses constant relaxation (for the
lockstep verification), `precice-config-iqnils.xml` uses IQN-ILS (for real
runs), and `precice-config-3d.xml` is the `dimensions="3"` variant for the
extruded core (`--nz > 0`). The participants assert that the config's
dimensionality matches the centroids they built.

---

## 6. Verification

Run:

```bash
pytest tests/verification/test_conduction.py tests/verification/test_feedback.py \
       tests/verification/test_power_density.py tests/verification/test_coupling.py
conda run -n ndgpu-precice python -m pytest tests/validation/test_coupled_precice.py
```

**Conduction against closed form.** The sinked equation has an exact hyperbolic
solution on a slab, `T(x) = T_sink + q/h + A cosh(x/L_d)` with
`L_d = sqrt(k/h)` — genuinely curved, not a polynomial the scheme reproduces by
accident. Both the Dirichlet and the Robin branch converge at **second order**
(measured 1.96 → 2.00 over four mesh doublings), and the triangular-prism path
reproduces the Cartesian errors digit-for-digit, confirming the tri-z closure.

**Exact invariants.** `T = T_sink + q/h` with no conduction; `T ≡ T_inf` for a
sourceless problem with a Robin surface (this is what pins the ambient boundary
term); energy balance closure < 1e-10 on the real masked HP-MR core.

**Feedback.** Sign is negative; reactivity is linear in the coefficient
amplitude (which is what justifies single-probe calibration); Doppler and
expansion are separable and additive to within the genuine second-order cross
term; a zeroed feedback is **bit-identical** to no hook at all.

**preCICE vs internal**, three tiers:

| tier | what it tests | measured |
|---|---|---|
| mapping is the identity | data crossing preCICE is bit-exact | equal as strings, every iteration |
| same fixed-point iteration | constant relaxation 0.5 both sides, lockstep | `max\|Δk\| = 5.0e-13` over all 33 iterations, identical iteration count |
| same fixed point, different accelerator | IQN-ILS vs Anderson | same `k` to < 2e-8; **6 iterations vs 33** |

The lockstep tier is the sharp one. Two different iterations can share a fixed
point, so agreeing only at convergence proves little; agreeing at *every* step
localizes any discrepancy to the step where it appears. It requires four things
the test enforces: warm start off (so `G(T)` is a pure function of `T`),
identical inner tolerances, the same initial field, and the literal expression
`omega*G(T) + (1-omega)*T` on the internal side — not the algebraically
identical `T + omega*(G(T) - T)`, whose last bits differ and propagate through
CG's stopping decision.

---

## 7. Coupled transients and GPU cost controls

`coupled_transient` uses `TransientSolver.on_step` as the exchange seam. Flux,
precursors, cross sections, temperature and accumulated power remain on the
solve device; the temperature advances every `dt_thermal`, which must be an
integer multiple of the neutron step `dt`. A final partial thermal window is
advanced with its actual width.

The production-oriented coupled defaults differ deliberately from the
verification defaults of the individual solvers:

```python
r = coupled_transient(
    ctx, t_end=60.0, dt=0.05, dt_thermal=0.5,
    precond_degree=1, check_every=4,
    thermal_rtol=1e-8, thermal_check_every=4,
    thermal_diagnostics_every=0,
)
```

Degree-1 Neumann-PCG and spaced residual checks reduce global synchronization
in the repeated neutron solves. The thermal tolerance is set by the accuracy
needed by feedback rather than by the `1e-12` balance-verification default.
The exact thermal energy balance costs three steady or four transient global
reductions, so production runs disable it or request it every N thermal steps
with `thermal_diagnostics_every=N`.

The power normalization integral remains a zero-dimensional device value. Peak
and mean temperature are copied together only when temperature actually moves,
not twice per neutron step. That same three-scalar transfer checks that every
normalization in the thermal window had finite positive fission power.

Pass `profile=True` to collect `result.phase_seconds` and `result.counters`.
GPU timings use CUDA events resolved after one final synchronization and emit
NVTX ranges; phase measurement therefore does not itself serialize the device.
Reported phases distinguish the neutron solve, feedback/operator rebuild,
power edit, thermal solve, telemetry transfer and final result transfer.
`examples/hpmr_coupled_transient.py --profile` prints both dictionaries.

The first quasi-static acceleration stage is available as an explicit advanced
API:

```python
from ndgpu import (fixed_shape_coupled_transient,
                   quasistatic_coupled_transient)

qs = fixed_shape_coupled_transient(
    ctx, t_end=60.0, dt=0.2, dt_thermal=1.0,
    problem_at=drum_frames, profile=True,
)
print(qs.power, qs.rho, qs.counters)

adiabatic = quasistatic_coupled_transient(
    ctx, t_end=60.0, dt=0.2, dt_thermal=1.0,
    shape_dt=2.0, adjoint_every=5,
    problem_at=drum_frames, profile=True,
)
```

It reuses the converged coupled forward flux, solves one adjoint, projects each
changed control/temperature operator, marches only the small effective
amplitude/precursor system at `dt`, and drives conduction with the
amplitude-scaled fixed power shape. This eliminates full-core fixed-source
solves from the time loop. `problem_at(0)` must be physically identical to the
base `ctx`, and callers should return cached objects between real state changes
to avoid redundant operator rebuilds.

`fixed_shape_coupled_transient` performs no shape correction and is not an
appropriate approximation for a large drum or rod movement.
`quasistatic_coupled_transient` adds periodic warm-started forward shape solves,
configurable adjoint refresh, normalized power-shape replacement, and complete
shape timing/counters. It is the intended path for slow HP-MR drum ramps and
long thermal holds. The remaining production work is an IQS transient shape
equation, residual-triggered updates, and automatic fallback for rapid or large
localized changes. See
[the quasi-static acceleration plan](quasistatic_acceleration_plan.md).
