"""LRA-2D BWR rod-ejection benchmark (ANL-7416, Benchmark 14).

The model is the 165 x 165 cm quarter core used by Cherezov et al. (2025):
an 11 x 11 lattice of 15 cm assemblies, reflective at ``x=0`` and ``y=0``
and zero flux at the two outer faces.  Material 5 is the reflector; the
remaining 78 assemblies are the core.

The specified ``B_z**2 = 1e-4 cm^-2`` is included by default, matching the
original DIF3D input. An earlier checkpoint appeared to match the rods-in
value only by omitting this leakage while simultaneously using the paper's
factor-ten typo in reflector thermal absorption. With the original reflector
value restored, the specified buckling reproduces the archived fine-mesh
static calculation. ``axial_buckling=False`` remains an explicit diagnostic.

The transient control region is the 2 x 2 block labelled R in the published
layout.  It has material-3 data at t=0 and its thermal absorption decreases
linearly to the material-4 value over two seconds.  ``problem_at`` describes
that prescribed motion only.  The benchmark's local adiabatic heat equation
and sqrt(T) Doppler feedback are state dependent and are exposed separately
through :class:`LRAAdiabaticState` so benchmark drivers cannot accidentally
confuse a prescribed perturbation with feedback.

One specification wart is intentionally visible: the map contains 78 core
assemblies, hence a geometric core area of 17,550 cm2.  Cherezov et al. print
17,750 cm2 in Eq. (51), which is incompatible with the map and assembly pitch.
The geometric value is used for spatial averages; the paper value is retained
as ``PAPER_CORE_AREA_CM2`` for reproducible reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Kinetics, Material

ASSEMBLY_PITCH_CM = 15.0
AXIAL_BUCKLING2 = 1.0e-4
NU = 2.43
INITIAL_TEMPERATURE_K = 300.0
INITIAL_POWER_DENSITY_W_CM3 = 1.0e-6
FISSION_ENERGY_J = 3.204e-11
ADIABATIC_ALPHA = 3.83e-11       # K cm3 / fission
DOPPLER_GAMMA = 3.034e-3         # K^-1/2, fast-group absorption only
GEOMETRIC_CORE_AREA_CM2 = 78 * ASSEMBLY_PITCH_CM**2
PAPER_CORE_AREA_CM2 = 17750.0

LRA_KINETICS = Kinetics(
    velocities=[3.0e7, 3.0e5],
    beta=[0.0054, 0.001087],
    decay=[0.0654, 1.35],
)

# Cherezov et al., Table 4, with the reflector erratum from the original
# ANL/DIF3D data. Columns are D1, D2, Sigma_a1, Sigma_a2,
# Sigma_1->2, nuSigma_f1, nuSigma_f2.  Region 4 differs from region 3 only in
# thermal absorption; region R interpolates precisely that entry.
_XS = {
    1: (1.255, 0.2110, 8.252e-3, 1.0030e-1, 2.533e-2, 4.602e-3, 1.091e-1),
    2: (1.268, 0.1902, 7.181e-3, 7.0470e-2, 2.767e-2, 4.609e-3, 8.675e-2),
    3: (1.259, 0.2091, 8.002e-3, 8.3440e-2, 2.617e-2, 4.663e-3, 1.021e-1),
    4: (1.259, 0.2091, 8.002e-3, 7.3324e-2, 2.617e-2, 4.663e-3, 1.021e-1),
    # Cherezov Table 4 has two factor-ten reflector typos. The original DIF3D
    # input and independent LRA specifications use Sigma_a1=6.034e-4 and
    # Sigma_a2=1.911e-2. Its fast removal is therefore 4.81434e-2 after adding
    # downscatter. The printed values do not reproduce the static endpoints.
    5: (1.257, 0.1592, 6.034e-4, 1.9110e-2, 4.754e-2, 0.0, 0.0),
}

# Rows are written top-to-bottom exactly as in the original DIF3D input and
# Fig. 9 of Cherezov.  They are flipped below because ndarray axis 1 grows
# from the physical bottom symmetry plane upward.
_ASSEMBLY_MAP_TOP_DOWN = np.asarray([
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    [3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
    [3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5],
    [2, 1, 1, 1, 1, 2, 2, 3, 3, 5, 5],
    [2, 1, 1, 1, 1, 2, 2, 3, 3, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5],
    [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5],
    [2, 1, 1, 1, 1, 2, 2, 3, 3, 5, 5],
], dtype=np.int64)

# R replaces material 3 in columns 7:9 and physical rows y=75..105 cm.
_CONTROL_ASSEMBLIES = ((7, 5), (8, 5), (7, 6), (8, 6))

K_REFERENCE = {"rods_in": 0.99633, "rods_out": 1.01546}
TRANSIENT_REFERENCE = {
    "first_peak_time_s": 1.436,
    "first_peak_power_w_cm3": 5411.0,
    "second_peak_power_w_cm3": 784.0,
    "power_at_3s_w_cm3": 96.2,
    "mean_temperature_at_3s_k": 1087.0,
    "peak_temperature_at_3s_k": 2948.0,
}
CHEREZOV_FEMCORE = {
    "rods_in": 0.99639,
    "rods_out": 1.01550,
    "first_peak_time_s": 1.439,
    "first_peak_power_w_cm3": 5641.0,
    "second_peak_power_w_cm3": 820.0,
    "power_at_3s_w_cm3": 98.7,
    "mean_temperature_at_3s_k": 1122.0,
    "peak_temperature_at_3s_k": 3068.0,
}

# Cherezov et al. Table 5. These are spatial-order results, not a time-order
# sweep: every row uses adaptive BDF (maximum order five) and the stated FEM
# basis degree. A cell-centred FV run must be compared to this ladder rather
# than presented as if it used the paper's converged fourth-order space.
CHEREZOV_BY_FEM_ORDER = {
    1: {"first_peak_time_s": 1.352, "first_peak_power_w_cm3": 5820.0,
        "second_peak_power_w_cm3": 854.0, "power_at_3s_w_cm3": 109.9},
    2: {"first_peak_time_s": 1.387, "first_peak_power_w_cm3": 5741.0,
        "second_peak_power_w_cm3": 852.0, "power_at_3s_w_cm3": 104.0},
    3: {"first_peak_time_s": 1.429, "first_peak_power_w_cm3": 5660.0,
        "second_peak_power_w_cm3": 829.0, "power_at_3s_w_cm3": 101.5},
    4: {"first_peak_time_s": 1.440, "first_peak_power_w_cm3": 5644.0,
        "second_peak_power_w_cm3": 824.0, "power_at_3s_w_cm3": 99.0},
    5: {"first_peak_time_s": 1.440, "first_peak_power_w_cm3": 5638.0,
        "second_peak_power_w_cm3": 824.0, "power_at_3s_w_cm3": 99.1},
    6: {"first_peak_time_s": 1.441, "first_peak_power_w_cm3": 5642.0,
        "second_peak_power_w_cm3": 824.0, "power_at_3s_w_cm3": 98.5},
    7: {"first_peak_time_s": 1.441, "first_peak_power_w_cm3": 5639.0,
        "second_peak_power_w_cm3": 823.0, "power_at_3s_w_cm3": 99.3},
    8: {"first_peak_time_s": 1.441, "first_peak_power_w_cm3": 5643.0,
        "second_peak_power_w_cm3": 823.0, "power_at_3s_w_cm3": 97.7},
}


def _material(region: int, thermal_absorption: float | None = None,
              buckling2: float = 0.0) -> Material:
    D1, D2, sa1, sa2, s12, nf1, nf2 = _XS[region]
    if thermal_absorption is not None:
        sa2 = float(thermal_absorption)
    # The axial-leakage model appears as an additional diagonal loss.
    sa = np.asarray([sa1, sa2]) + np.asarray([D1, D2]) * float(buckling2)
    return Material(
        name=f"LRA region {region}", diffusion=[D1, D2], sigma_a=sa,
        nu_sigma_f=[nf1, nf2], sigma_s=[[0.0, s12], [0.0, 0.0]],
        chi=[1.0, 0.0],
    )


@dataclass
class LRAProblem:
    grid: Grid
    material_map: np.ndarray
    core_mask: np.ndarray
    control_mask: np.ndarray
    kinetics: Kinetics
    bc: tuple
    problem_at: object


def build_lra2d(refine: int = 1, control: str = "transient",
                axial_buckling: bool = True,
                control_worth_scale: float = 1.0) -> LRAProblem:
    """Build the LRA quarter core.

    ``refine`` is the number of finite-volume cells per 15 cm assembly edge.
    ``control`` is ``"in"``, ``"out"``, or ``"transient"`` (the 0--2 s
    linear withdrawal).  Material indices 0..4 correspond to regions 1..5;
    the time-dependent R material is an additional index 5.

    ``axial_buckling=True`` is the published/original case and adds
    ``D_g * 1e-4`` leakage. False omits axial leakage as a diagnostic.

    ``control_worth_scale`` multiplies the region-R absorption perturbation.
    Its default of one is the paper-faithful model. Other values are exposed
    only for spatial-discretization diagnostics (for example, holding the raw
    endpoint eigenvalue fixed while comparing time schemes); they do not
    constitute a reproduction of the published material definition.
    """
    refine = int(refine)
    if refine < 1:
        raise ValueError("refine must be a positive integer")
    if control not in ("in", "out", "transient"):
        raise ValueError("control must be 'in', 'out', or 'transient'")
    control_worth_scale = float(control_worth_scale)
    if not np.isfinite(control_worth_scale) or control_worth_scale <= 0.0:
        raise ValueError("control_worth_scale must be finite and positive")
    n = 11 * refine
    grid = Grid(shape=(n, n, 1), size=(165.0, 165.0, 1.0))
    assembly = np.flip(_ASSEMBLY_MAP_TOP_DOWN, axis=0) - 1
    mmap2 = np.repeat(np.repeat(assembly, refine, axis=0), refine, axis=1)
    mmap = mmap2[:, :, None].copy()
    control_mask = np.zeros_like(mmap, dtype=bool)
    for i, j in _CONTROL_ASSEMBLIES:
        control_mask[i * refine:(i + 1) * refine,
                     j * refine:(j + 1) * refine, 0] = True
    # R receives a dedicated time-varying material slot.
    mmap[control_mask] = 5
    core_mask = mmap != 4

    buckling2 = AXIAL_BUCKLING2 if axial_buckling else 0.0
    base = [_material(i, buckling2=buckling2) for i in range(1, 6)]
    sa3, sa4 = _XS[3][3], _XS[4][3]
    cache: dict[float, list[Material]] = {}

    def fraction(t: float) -> float:
        if control == "in":
            return 0.0
        if control == "out":
            return 1.0
        return min(max(float(t), 0.0) / 2.0, 1.0)

    def problem_at(t: float):
        f = round(fraction(t), 14)
        if f not in cache:
            sa_r = sa3 + control_worth_scale * f * (sa4 - sa3)
            cache[f] = base + [_material(
                3, thermal_absorption=sa_r, buckling2=buckling2)]
        return cache[f], mmap

    bc = (("reflective", "zero-flux"),
          ("reflective", "zero-flux"), "reflective")
    return LRAProblem(grid, mmap, core_mask, control_mask, LRA_KINETICS, bc,
                      problem_at)


def lra2d_static_keff(*, refine: int = 1, control: str = "in",
                      axial_buckling: bool = True,
                      control_worth_scale: float = 1.0,
                      steady_kwargs: dict | None = None) -> float:
    """Solve one raw LRA static endpoint on CPU.

    ``control`` must be ``"in"`` or ``"out"``.  In particular, the returned
    rods-out value is the raw eigenvalue used by ANL-7416 and Cherezov et al.,
    not ``k_out / k_in``.  Keeping this operation public gives benchmark
    drivers one unambiguous endpoint comparison path.
    """
    from ..solver import DiffusionEigenSolver

    if control not in ("in", "out"):
        raise ValueError("static control must be 'in' or 'out'")
    problem = build_lra2d(
        refine=refine, control=control, axial_buckling=axial_buckling,
        control_worth_scale=control_worth_scale,
    )
    materials, material_map = problem.problem_at(0.0)
    result = DiffusionEigenSolver(
        problem.grid, materials, material_map, bc=problem.bc, device="cpu",
    ).solve(**(steady_kwargs or {"tol_k": 2e-9, "tol_source": 2e-8}))
    if not result.converged:
        raise RuntimeError(f"LRA {control} eigenvalue solve did not converge")
    return float(result.k_eff)


@dataclass
class LRAAdiabaticState:
    """Local temperature state and the paper's Doppler cross-section hook.

    The update is deliberately small and backend-agnostic.  A coupled driver
    supplies the *physical* fission power density in W/cm3 and an accepted
    step width.  Temperature advances exactly according to the adiabatic
    model; ``hook`` then applies only the fast-group absorption multiplier
    specified by Cherezov et al. (diffusion is not modified).
    """

    temperature: object

    @classmethod
    def uniform(cls, shape, temperature=INITIAL_TEMPERATURE_K):
        return cls(np.full(shape, float(temperature)))

    def advance(self, power_density_w_cm3, dt: float):
        self.temperature[...] = (self.temperature
                                 + float(dt) * (ADIABATIC_ALPHA / FISSION_ENERGY_J)
                                 * power_density_w_cm3)

    def hook(self, temperature=None, axial_buckling2=0.0):
        temperature = self.temperature if temperature is None else temperature
        axial_buckling2 = float(axial_buckling2)

        def update(fields):
            xp = fields.blend.xp
            T = xp.asarray(temperature, dtype=fields.blend.dtype)
            factor = 1.0 + DOPPLER_GAMMA * (
                xp.sqrt(xp.maximum(T, 0.0)) - np.sqrt(INITIAL_TEMPERATURE_K))
            # The published model scales Sigma_a1 only.  Removal contains the
            # same absorption plus unchanged downscatter.
            old = fields.sigma_a[0]
            # D*Bz^2 occupies the generic diagonal-loss slot, but Eq. (50)
            # scales only absorption. Keep axial leakage outside the Doppler
            # multiplier.
            axial_leakage = fields.diffusion[0] * axial_buckling2
            new = (old - axial_leakage) * factor + axial_leakage
            fields.sigma_a[0] = new
            fields.removal[0] = fields.removal[0] + (new - old)

        return update


@dataclass
class LRATransientResult:
    """CPU benchmark history with physical LRA power and temperatures."""

    transient: object
    average_power_w_cm3: np.ndarray
    average_temperature_k: np.ndarray
    peak_assembly_temperature_k: np.ndarray
    peak_temperature_k: np.ndarray
    temperature: np.ndarray
    coupling: str

    def metrics(self):
        p = self.average_power_w_cm3
        # The first peak occurs before the rod stops at 2 s.  The value at the
        # schedule point nearest 2 s is the standard second-peak metric.
        before = np.flatnonzero(self.transient.times < 2.0)
        i1 = int(before[np.argmax(p[before])])
        i2 = int(np.argmin(abs(self.transient.times - 2.0)))
        return {
            "first_peak_time_s": float(self.transient.times[i1]),
            "first_peak_power_w_cm3": float(p[i1]),
            "second_peak_power_w_cm3": float(p[i2]),
            "power_at_3s_w_cm3": float(p[-1]),
            "mean_temperature_at_3s_k": float(self.average_temperature_k[-1]),
            # Finnemann's comparison field, reproduced in Cherezov's
            # Appendix B, is assembly averaged.  Keep the pointwise/cell
            # maximum as a separate diagnostic when local feedback is used.
            "peak_temperature_at_3s_k": float(
                self.peak_assembly_temperature_k[-1]),
            "peak_cell_temperature_at_3s_k": float(
                self.peak_temperature_k[-1]),
        }


def run_lra2d_cpu(*, refine: int = 1, dt=0.01, bdf_order: int = 5,
                  axial_buckling: bool = True, tol_step: float = 2e-7,
                  max_sweeps: int = 300, rebalance: bool = True,
                  step_solver: str = "monolithic",
                  predict_feedback: bool = True,
                  implicit_feedback: bool = False,
                  feedback_rtol: float = 1e-6,
                  max_feedback_iterations: int = 12,
                  thermal_zones: str = "assembly",
                  control_worth_scale: float = 1.0,
                  adaptive_bdf: dict | None = None,
                  steady_kwargs: dict | None = None) -> LRATransientResult:
    """Run the coupled three-second LRA transient on CPU.

    This is the first reproducible ndgpu comparison path, deliberately built
    from the public solver rather than a benchmark-only kinetics engine.  The
    neutronics and adiabatic heat equation both use the requested BDF order.
    By default Doppler cross sections use a polynomial endpoint-temperature
    predictor from the accepted multilevel BDF history; the accepted thermal
    BDF correction then consumes the new fission rate. Set
    ``implicit_feedback=True`` to repeat the monolithic neutron solve and BDF
    heat correction at the same endpoint until their relative temperature
    defect is below ``feedback_rtol``. Set ``predict_feedback=False`` for the
    older one-step-lagged coupling.

    ``dt`` may be a scalar or an explicit positive schedule summing to 3 s.
    With ``adaptive_bdf`` it is the initial width and the monolithic transient
    controls subsequent accepted widths; the rod-law corner at 2 s is aligned
    exactly and restarts both neutron and thermal BDF histories.
    Returned power is the physical core-average W/cm3, not merely P/P0.
    ``thermal_zones="assembly"`` matches Cherezov's stated element-wise
    temperature approximation even when neutronics is refined inside each
    assembly. ``"cell"`` follows the benchmark's space-dependent heat and
    Doppler equations on the full FV mesh, approaching the original
    fine-mesh reference model as the spatial grid is refined.
    """
    from ..solver import DiffusionEigenSolver
    from ..timescheme import BDF
    from ..transient import TransientSolver, _step_schedule

    bdf_order = int(bdf_order)
    if bdf_order < 1 or bdf_order > 6:
        raise ValueError("bdf_order must be between 1 and 6")
    if thermal_zones not in ("assembly", "cell"):
        raise ValueError("thermal_zones must be 'assembly' or 'cell'")
    if implicit_feedback and step_solver != "monolithic":
        raise ValueError("implicit LRA feedback requires the monolithic step solver")
    if adaptive_bdf is not None and not implicit_feedback:
        raise ValueError("adaptive LRA currently requires implicit_feedback=True")
    if feedback_rtol <= 0.0:
        raise ValueError("feedback_rtol must be positive")
    if adaptive_bdf is None:
        target_times, widths = _step_schedule(3.0, dt)
    else:
        if np.ndim(dt) != 0:
            raise ValueError("adaptive LRA requires a scalar initial dt")
        target_times, widths = None, None
    problem = build_lra2d(refine=refine, control="transient",
                          axial_buckling=axial_buckling,
                          control_worth_scale=control_worth_scale)
    buckling2 = AXIAL_BUCKLING2 if axial_buckling else 0.0
    mats0, mmap = problem.problem_at(0.0)
    eig = DiffusionEigenSolver(problem.grid, mats0, mmap, bc=problem.bc,
                               device="cpu")
    steady = eig.solve(**(steady_kwargs or
                          {"tol_k": 2e-9, "tol_source": 2e-8}))
    if not steady.converged:
        raise RuntimeError("LRA initial eigenvalue solve did not converge")

    state = LRAAdiabaticState.uniform(problem.grid.shape)
    thermal_bdf = BDF(bdf_order)
    automatic_order = bool(
        adaptive_bdf is not None
        and adaptive_bdf.get("automatic_order", False))
    if automatic_order:
        thermal_bdf.enable_order_selection()
    thermal_bdf.start([state.temperature.copy()])
    core = problem.core_mask
    n_core = int(np.count_nonzero(core))
    # Production does not change during this benchmark; only absorption does.
    nf_table = np.asarray([m.nu_sigma_f for m in mats0])
    nf = [nf_table[mmap, g] for g in range(2)]
    k0 = float(steady.k_eff)
    mean_T = [INITIAL_TEMPERATURE_K]
    peak_assembly_T = [INITIAL_TEMPERATURE_K]
    peak_T = [INITIAL_TEMPERATURE_K]
    callback_step = 0
    accepted_time = 0.0
    thermal_restart_pending = False
    trial_time = None
    trial_temperature = state.temperature.copy()
    feedback_history = []
    feedback_change = np.inf

    def assembly_average(field):
        if refine == 1:
            return field
        coarse = field.reshape(11, refine, 11, refine, 1).mean(axis=(1, 3))
        return np.repeat(np.repeat(coarse, refine, axis=0), refine, axis=1)

    def thermal_zone_average(field):
        if thermal_zones == "cell":
            return field
        return assembly_average(field)

    def power_density(flux):
        # TransientSolver normalizes sum(nuSigma_f*phi/k0) to one initially.
        # Multiplication by n_core therefore makes the initial core mean q0.
        fission = (nf[0] * np.asarray(flux[0])
                   + nf[1] * np.asarray(flux[1])) / k0
        return thermal_zone_average(
            INITIAL_POWER_DENSITY_W_CM3 * n_core * fission)

    def prepare_thermal_step(step, width):
        thermal_bdf.prepare_step(step, width)
        return (thermal_bdf.carried(step)[0], thermal_bdf.a0(step))

    def thermal_correction(step, width, flux):
        carried, a0 = prepare_thermal_step(step, width)
        return (carried + (width / a0)
                * (ADIABATIC_ALPHA / FISSION_ENERGY_J) * power_density(flux))

    def accept_temperature(T_new, t):
        nonlocal callback_step, accepted_time, thermal_restart_pending
        callback_step += 1
        state.temperature[...] = T_new
        thermal_bdf.push([T_new.copy()])
        accepted_time = float(t)
        if adaptive_bdf is not None and np.isclose(
                accepted_time, 2.0, rtol=1e-12, atol=2e-14):
            thermal_restart_pending = True
        mean_T.append(float(np.mean(T_new[core])))
        peak_assembly_T.append(float(np.max(assembly_average(T_new)[core])))
        peak_T.append(float(np.max(T_new[core])))

    def attempted_width(t):
        if adaptive_bdf is not None:
            return float(t) - accepted_time
        index = int(np.searchsorted(target_times, float(t)))
        return float(widths[index])

    def xs_update_at(t):
        nonlocal trial_time, trial_temperature, feedback_history, feedback_change
        nonlocal thermal_restart_pending
        if implicit_feedback and t > 0.0:
            if trial_time != float(t):
                if thermal_restart_pending:
                    thermal_bdf.start([state.temperature.copy()])
                    thermal_restart_pending = False
                trial_temperature = thermal_bdf.predict(
                    attempted_width(t))[0].copy()
                trial_time = float(t)
                feedback_history = []
                feedback_change = np.inf
            return state.hook(trial_temperature, buckling2)
        if (not predict_feedback) or t <= 0.0:
            return state.hook(axial_buckling2=buckling2)
        return state.hook(thermal_bdf.predict(attempted_width(t))[0],
                          buckling2)

    def on_step(_t, flux, _relative_power):
        step = callback_step + 1
        width = attempted_width(_t)
        if implicit_feedback:
            # The constituent callback below only updates this provisional
            # endpoint. Commit thermal/BDF history here, after the neutron
            # solver has accepted the complete step. This distinction is what
            # makes future local-error rejection rollback-safe.
            accept_temperature(trial_temperature.copy(), _t)
        else:
            accept_temperature(thermal_correction(step, width, flux), _t)

    def feedback_iteration(_t, flux, _relative_power, _iteration):
        nonlocal trial_temperature, feedback_history, feedback_change
        step = callback_step + 1
        width = attempted_width(_t)
        corrected = thermal_correction(step, width, flux)
        delta = corrected - trial_temperature
        denom = max(float(np.linalg.norm(corrected.ravel())), 1.0)
        change = float(np.linalg.norm(delta.ravel()) / denom)
        if change <= feedback_rtol:
            trial_temperature = corrected
            feedback_change = change
            return True

        # Anderson acceleration of the coupled temperature fixed point. Reset
        # if its residual grows materially; a plain corrected iterate is the
        # safe fallback and all accepted histories remain untouched here.
        if change > 1.5 * feedback_change:
            feedback_history = []
        feedback_change = change
        feedback_history.append((trial_temperature.copy(), corrected.copy()))
        feedback_history = feedback_history[-5:]
        trial_temperature = corrected
        if len(feedback_history) >= 2:
            from ..transient import _anderson_mix
            mixed = _anderson_mix(feedback_history, corrected, np)
            if np.all(np.isfinite(mixed)) and np.all(mixed > 0.0):
                trial_temperature = mixed
        return False

    def adaptive_error_state(_t, width):
        """Add the paper's coupled thermal unknown to the BDF defect norm."""
        predicted = thermal_bdf.predict(
            width,
            max_degree=(thermal_bdf.order_at(callback_step + 1)
                        if automatic_order else None))
        return [trial_temperature], predicted

    def adaptive_order_error_state(_t, width, order):
        predicted = thermal_bdf.predict(width, max_degree=order)
        return [trial_temperature], predicted

    def adaptive_order_callback(order):
        thermal_bdf.select_order(order)

    transient = TransientSolver(
        problem.grid, problem.problem_at, problem.kinetics, bc=problem.bc,
        device="cpu", xs_update_at=xs_update_at,
        on_step=on_step,
        feedback_iteration=feedback_iteration if implicit_feedback else None,
        adaptive_error_state=(adaptive_error_state
                              if adaptive_bdf is not None else None),
        adaptive_order_error_state=(adaptive_order_error_state
                                    if automatic_order else None),
        adaptive_order_callback=(adaptive_order_callback
                                 if automatic_order else None),
    ).solve(
        t_end=3.0, dt=(dt if adaptive_bdf is not None else widths),
        tol_step=tol_step, max_sweeps=max_sweeps,
        initial_steady=steady, rebalance=rebalance,
        time_scheme=f"bdf{bdf_order}", step_solver=step_solver,
        adaptive_bdf=adaptive_bdf,
        bdf_restart_times=([2.0] if adaptive_bdf is not None else None),
        max_feedback_iterations=max_feedback_iterations,
    )
    if callback_step != len(transient.times) - 1:
        raise RuntimeError("LRA coupling callback did not advance every step")
    return LRATransientResult(
        transient=transient,
        average_power_w_cm3=(INITIAL_POWER_DENSITY_W_CM3
                             * transient.power),
        average_temperature_k=np.asarray(mean_T),
        peak_assembly_temperature_k=np.asarray(peak_assembly_T),
        peak_temperature_k=np.asarray(peak_T),
        temperature=np.asarray(state.temperature).copy(),
        coupling=(("implicit" if implicit_feedback else
                   ("predicted" if predict_feedback else "lagged"))
                  + ("-backward-euler" if bdf_order == 1 else
                     f"-bdf{bdf_order}")
                  + f"-{thermal_zones}"),
    )
