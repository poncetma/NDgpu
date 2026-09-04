"""Thermal constants and temperature feedback for the HP-MR microreactor.

Companion to :mod:`ndgpu.benchmarks.hpmr`: the same 55-site core, described to
the conduction solver and the feedback law instead of to the neutronics. Every
number below is a homogenized, assembly-level constant chosen to be defensible
and auditable rather than proprietary-accurate -- the VTB reference design
publishes geometry and cross sections, not a thermal property table.

**The heat path.** An HP-MR has no coolant. Each fuel assembly is pierced by
alkali-metal heat pipes whose evaporator sits at a nearly uniform temperature;
they draw power in proportion to the local solid-to-pipe temperature
difference, and carry it out of the core entirely. Homogenized over the
assembly that is a volumetric conductance ``h`` to a fixed ``T_sink``, active
in the fuel only. Everything else -- reflector, drums, absorber -- merely
conducts, and loses a little heat to the vessel through a weak surface
coefficient.

**The sink is calibrated, not measured** (arithmetic kept here so it can be
checked): the hex area at the 26.752 cm flat-to-flat pitch is 619.8 cm^2, so
30 fuel assemblies over the 160 cm fueled length hold 2.975e6 cm^3. At 2 MWt
that is a mean power density of 0.672 W/cm^3. Choosing a 50 K mean fuel-to-pipe
temperature rise fixes ``h = 0.672 / 50 = 0.0134 W/cm^3/K``, and with
``T_sink = 750 K`` the mean fuel temperature lands near **800 K**.

That last coincidence is the point, not decoration: 800 K is the Griffin
library node (``grid_index="3 3"``) the cross sections were read at. It keeps
the feedback a small perturbation about the state where the data is actually
valid, which is the only regime in which an analytic law layered on
single-temperature tabulated cross sections means anything.
"""

from __future__ import annotations

import numpy as np

from ..feedback import ThermalFeedback
from ..thermal import ThermalMaterial
from .hpmr import (AXIAL_REFLECTOR, BE_REFLECTOR, CENTRAL, CORE_HEIGHT,
                   DRUM_ABSORBER, DRUM_BE, FUEL, MATERIAL_NAMES,
                   MATERIAL_NAMES_3D, PITCH, VOID)

#: Rated thermal power of the VTB reference HP-MR.
RATED_POWER_W = 2.0e6

#: Number of fuel assemblies (see hpmr._FUEL_SITES).
N_FUEL_ASSEMBLIES = 30

#: Heat-pipe evaporator temperature, K. A sodium heat pipe operating well
#: inside its wick limit; the design point sits in the 900-1100 K range and
#: 750 K is the conservative end, chosen here so the resulting mean fuel
#: temperature matches the cross-section evaluation temperature.
SINK_TEMPERATURE_K = 750.0

#: Reference temperature of the cross sections: the Griffin Tfuel/Tmod node
#: that ndgpu.griffin_xs reads with grid_index="3 3".
XS_REFERENCE_K = 800.0

#: Design mean fuel-to-heat-pipe temperature rise, K, used to set `h`.
DESIGN_FILM_DROP_K = 50.0

#: Weak surface loss to the vessel, W/(cm^2 K), with the vessel at
#: AMBIENT_K. Small compared with the heat pipes -- it exists so the outer
#: reflector has a defined temperature rather than floating, and it carries a
#: percent-level share of the power.
VESSEL_HTC = 1.0e-3
AMBIENT_K = 400.0

# Conductivities in W/(cm K) at ~800 K. Nuclear graphite is 0.25-0.35 in this
# range (it falls with irradiation and temperature); beryllium ~0.90; boron
# carbide ~0.20. The fuel assembly is a graphite monolith carrying TRISO
# compacts, moderator pins and heat-pipe channels, so its homogenized value is
# a little below bulk graphite.
K_FUEL_ASSEMBLY = 0.25
K_GRAPHITE = 0.30
K_BERYLLIUM = 0.90
K_B4C = 0.20
#: Void is outside the core; a small positive value only keeps the harmonic
#: face mean finite. It conducts nothing because it is masked inactive.
K_VOID = 1.0e-6

# Volumetric heat capacities rho*cp in J/(cm^3 K) at ~800 K. These set the
# thermal time constant, so they decide whether a reactivity insertion shows
# up as a power spike that the fuel rides out or as a temperature excursion.
# Graphite: 1.8 g/cm^3 x ~1.9 J/(g K) at 800 K. Beryllium: 1.85 g/cm^3 x
# ~2.4 J/(g K). B4C: 2.5 g/cm^3 x ~1.9 J/(g K). The fuel assembly is mostly
# graphite matrix, taken slightly higher for the TRISO and heat-pipe mass.
RHOCP_FUEL_ASSEMBLY = 3.6
RHOCP_GRAPHITE = 3.4
RHOCP_BERYLLIUM = 4.4
RHOCP_B4C = 4.8
RHOCP_VOID = 1.0e-6

#: Fuel thermal time constant rho*cp/h ~ 3.6/0.0134 = 268 s. That is the number
#: that matters for a transient: a drum rotation over a few seconds is fast
#: compared with it, so the fuel barely warms during the insertion and the
#: feedback that arrests the excursion arrives late.


def hex_area(pitch: float = PITCH) -> float:
    """Area of a hexagon given its flat-to-flat pitch, cm^2."""
    return float(np.sqrt(3.0) / 2.0 * pitch * pitch)


def fueled_volume(pitch: float = PITCH, height: float = CORE_HEIGHT,
                  n_assemblies: int = N_FUEL_ASSEMBLIES) -> float:
    """Total fuel-assembly volume, cm^3."""
    return hex_area(pitch) * height * n_assemblies


def mean_power_density(power_w: float = RATED_POWER_W, **kw) -> float:
    """Core-average volumetric heat rate in the fuel, W/cm^3."""
    return power_w / fueled_volume(**kw)


def sink_coefficient(power_w: float = RATED_POWER_W,
                     film_drop_k: float = DESIGN_FILM_DROP_K, **kw) -> float:
    """Volumetric heat-pipe conductance h, W/(cm^3 K), from a design film drop."""
    return mean_power_density(power_w, **kw) / film_drop_k


def hpmr_thermal_materials(three_d: bool = False, power_w: float = RATED_POWER_W,
                           film_drop_k: float = DESIGN_FILM_DROP_K,
                           sink_temperature: float = SINK_TEMPERATURE_K):
    """Thermal materials in MATERIAL_NAMES (or MATERIAL_NAMES_3D) order.

    Pass straight to :class:`ndgpu.ConductionSolver` alongside the SAME
    ``material_map`` and mix arrays the neutronics uses.
    """
    h = sink_coefficient(power_w, film_drop_k)
    names = MATERIAL_NAMES_3D if three_d else MATERIAL_NAMES
    by_index = {
        VOID: ThermalMaterial(K_VOID, heat_capacity=RHOCP_VOID, name="void"),
        FUEL: ThermalMaterial(K_FUEL_ASSEMBLY, sink_coeff=h,
                              sink_temperature=sink_temperature,
                              heat_capacity=RHOCP_FUEL_ASSEMBLY, name="fuel"),
        CENTRAL: ThermalMaterial(K_GRAPHITE, heat_capacity=RHOCP_GRAPHITE,
                                 name="central"),
        BE_REFLECTOR: ThermalMaterial(K_BERYLLIUM, heat_capacity=RHOCP_BERYLLIUM,
                                      name="be_reflector"),
        DRUM_BE: ThermalMaterial(K_BERYLLIUM, heat_capacity=RHOCP_BERYLLIUM,
                                 name="drum_be"),
        DRUM_ABSORBER: ThermalMaterial(K_B4C, heat_capacity=RHOCP_B4C,
                                       name="drum_absorber"),
        AXIAL_REFLECTOR: ThermalMaterial(K_BERYLLIUM, heat_capacity=RHOCP_BERYLLIUM,
                                         name="axial_reflector"),
    }
    return [by_index[i] for i in range(len(names))]


#: Relative Doppler response per material. Only the fuel resonates: the
#: reflector, drums and absorber are given zero, so a temperature change there
#: has no reactivity effect. The AMPLITUDE is meaningless until calibrated --
#: only the pattern is physics. See hpmr_feedback().
_DOPPLER_SHAPE = {FUEL: 1.0}


def hpmr_feedback(n_materials: int, target_pcm_per_K: float = -2.5,
                  t_ref: float = XS_REFERENCE_K, doppler_amplitude: float = 1e-3,
                  expansion_pcm_per_K: float | None = None) -> ThermalFeedback:
    """Fuel-only Doppler feedback for the HP-MR core.

    ``target_pcm_per_K`` is documentation of intent, not a fitted result: this
    returns the SHAPE with a nominal amplitude, and
    :func:`ndgpu.feedback.calibrate` scales it to hit the target against the
    actual core and cross sections. A graphite-moderated HEU microreactor sits
    around -2 to -3 pcm/K, dominated by U-238 Doppler in the TRISO kernels.
    """
    if target_pcm_per_K >= 0:
        raise ValueError("a fuel temperature coefficient must be negative")
    doppler = np.zeros(n_materials)
    for idx, weight in _DOPPLER_SHAPE.items():
        if idx < n_materials:
            doppler[idx] = doppler_amplitude * weight
    expansion = None
    if expansion_pcm_per_K is not None:
        expansion = np.zeros(n_materials)
        for idx in _DOPPLER_SHAPE:
            if idx < n_materials:
                expansion[idx] = 1e-5
    return ThermalFeedback(t_ref=np.full(n_materials, t_ref), doppler=doppler,
                           expansion=expansion)


def hpmr_endfb8_builtin(three_d: bool = False):
    """The real 11-group ENDF/B-8 core material list, self-contained.

    Flat-flux volume homogenization of the pin lattice into the fuel assembly
    (:data:`ndgpu.benchmarks.hpmr._ASSEMBLY_VOLUME_FRACTIONS`), with the
    reflector, drum body and B4C arc taken from the vendored core library. No
    external XS file and no transport solve -- the flux-weighted alternative
    (``hpmr_sn_homogenization``) is better physics but costs ~90 s.

    Use this rather than the 2-group placeholders for anything thermal: the
    placeholders carry no ``kappa_fission``, so the power density silently
    falls back to nu-weighting and the temperature shape is tilted by the
    local spectrum.
    """
    from ..griffin_xs import volume_homogenize
    from .hpmr import _ASSEMBLY_VOLUME_FRACTIONS, _XS_FUEL_COMPACT, hpmr_materials_builtin
    from .hpmr_assembly import PIN_XS_IDS, pin_materials_builtin

    lib = dict(zip(PIN_XS_IDS, pin_materials_builtin()))
    fuel = volume_homogenize(lib, _ASSEMBLY_VOLUME_FRACTIONS,
                             chi_from=_XS_FUEL_COMPACT, name="hpmr-fuel-asm")
    if fuel.kappa_fission is None:                       # pragma: no cover
        raise RuntimeError("homogenized fuel lost its kappaFission")
    return hpmr_materials_builtin(fuel, three_d=three_d)


#: Fallback 11-group boundaries in eV, high to low, used ONLY when no library
#: speeds are available. The G11 structure is a property of the VTB Griffin
#: library the cross sections came from, and that library carries its own
#: ``Velocity`` block -- but the vendored ``.npz`` extracts in this repo predate
#: :func:`ndgpu.griffin_xs.read_velocity` and stored only the six cross-section
#: arrays, so nothing here can recover the real numbers. Prefer
#: :func:`hpmr_velocities_from_library` whenever the XML is to hand.
_G11_BOUNDS_EV = (2.0e7, 1.353e6, 4.979e5, 6.738e4, 9.119e3, 1.301e2,
                  3.928e0, 1.855e0, 6.250e-1, 1.800e-1, 5.000e-2, 1.000e-5)

#: npz key the speeds land under once an extract carries them.
_VELOCITY_KEY = "velocity"


def hpmr_velocities_from_library(xs_path, mid: int = 801,
                                 grid_index: str = "3 3"):
    """The library's own per-group speeds, cm/s -- the numbers to use.

    ``mid`` defaults to the fuel compact, whose spectrum the speeds are
    tabulated against. Raises if the library has no velocity block, rather than
    silently substituting an assumption.
    """
    from ..griffin_xs import read_velocity

    v = read_velocity(xs_path, mid, grid_index)
    if v is None:
        raise KeyError(f"{xs_path} carries no Velocity/InverseVelocity for "
                       f"material {mid}")
    return v


def _vendored(*keys):
    """Arrays from the vendored extract, or None if it predates them."""
    import os

    path = os.path.join(os.path.dirname(__file__), "data",
                        "hpmr_core_xs_g11.npz")
    with np.load(path, allow_pickle=False) as d:
        if all(k in d.files for k in keys):
            return tuple(np.array(d[k], dtype=float) for k in keys)
    return None


def _vendored_velocities():
    got = _vendored(_VELOCITY_KEY)
    return None if got is None else got[0]


def hpmr_velocities_11g(thermal_temperature: float = XS_REFERENCE_K,
                        xs_path=None, strict: bool = False):
    """Per-group neutron speeds v_g in cm/s for the 11-group core.

    Resolution order, best first:

    1. ``xs_path`` -- read the library's own ``Velocity`` block. Use this.
    2. the vendored ``.npz``, if it was extracted with speeds in it.
    3. a **reconstruction**: ``v = sqrt(2E/m)`` at each group's
       lethargy-midpoint energy for the conventional structure in
       :data:`_G11_BOUNDS_EV`, with the thermal group taken at ``kT`` of the
       moderator instead (the geometric mean of a bin running down to 1e-5 eV
       gives 37 km/s, an order below the ~2.2 km/s a thermal neutron actually
       travels at).

    Pass ``strict=True`` to refuse the reconstruction.

    **What rides on it:** v_g sets the prompt neutron generation time, so it
    governs the prompt jump. The drum manoeuvre modelled in these examples is a
    fraction of a dollar, where the power history is carried by delayed
    neutrons and is almost independent of Lambda -- so a reconstructed
    structure is tolerable *there*, and would not be for a prompt excursion.
    """
    if xs_path is not None:
        return hpmr_velocities_from_library(xs_path,
                                            grid_index="3 3")
    v = _vendored_velocities()
    if v is not None:
        return v
    if strict:
        raise KeyError(
            "no library speeds available: the vendored extract has no "
            f"'{_VELOCITY_KEY}' array. Pass xs_path=<Griffin XML>, or "
            "re-extract the npz now that griffin_xs.read_velocity exists")
    e = np.asarray(_G11_BOUNDS_EV, dtype=float)
    e_mid = np.sqrt(e[:-1] * e[1:])
    v = 1.3831e9 * np.sqrt(e_mid * 1e-6)          # cm/s, E in MeV
    v[-1] = 1.3831e9 * np.sqrt(8.617e-5 * thermal_temperature * 1e-6)
    return v


def hpmr_kinetics_11g(thermal_temperature: float = XS_REFERENCE_K,
                      xs_path=None, strict: bool = False):
    """:class:`~ndgpu.Kinetics` for the 11-group core -- the LIBRARY's own.

    The VTB Griffin library tabulates the whole kinetics set beside the cross
    sections: 11 group speeds, six precursor families with their decay
    constants and fractions (beta_total = 680 pcm), and a delayed emission
    spectrum. All of it now comes from the vendored extract, so a transient
    runs on the same data as the statics.

    That replaces the two-group placeholder in
    :data:`ndgpu.benchmarks.hpmr.HPMR_KINETICS` (one family, beta = 650 pcm),
    which was only ever a stand-in -- and it matters, because beta and lambda
    set the entire time scale of a sub-prompt transient.
    """
    from ..materials import Kinetics

    if xs_path is not None:
        from ..griffin_xs import read_kinetics
        return read_kinetics(xs_path, 801, "3 3")

    got = _vendored("velocity", "dnp_beta", "dnp_lambda")
    if got is not None:
        v, beta, lam = got
        chi_d = _vendored("dnp_chi")
        return Kinetics(velocities=v, beta=beta, decay=lam,
                        chi_delayed=None if chi_d is None else chi_d[0])

    if strict:
        raise KeyError("the vendored extract carries no kinetics; re-run "
                       "tools/extract_hpmr_xs.py against the Griffin library")
    # Last resort: reconstructed speeds on the placeholder delayed data.
    from .hpmr import HPMR_KINETICS
    return Kinetics(
        velocities=hpmr_velocities_11g(thermal_temperature, xs_path, strict),
        beta=HPMR_KINETICS.beta, decay=HPMR_KINETICS.decay)


def hpmr_tabulated_feedback(strict: bool = True):
    """The library's own temperature branches as a feedback -- the real thing.

    The VTB library tabulates every fuel cross section at Tfuel = Tmod =
    {600, 700, 800, 1000, 1200} K. ``tools/extract_hpmr_xs.py`` re-homogenizes
    the assembly at each node along that diagonal (one solid temperature, which
    is what the conduction model carries) and stores the result, so the coupled
    solve interpolates evaluated data instead of scaling one node.

    Measured on the refine-3 core, this gives a Doppler coefficient of
    **-3.5 pcm/K** averaged over 600-1200 K, and it is not constant:
    -3.98 pcm/K across 600-700 K easing to -3.30 across 1000-1200 K, the
    familiar flattening of resonance broadening with temperature. A single
    fitted coefficient cannot represent that, which is the argument for using
    the branches.
    """
    from ..feedback import TabulatedFeedback
    from .hpmr import FUEL

    keys = {"diffusion": "fuel_branch.D", "sigma_a": "fuel_branch.sa",
            "nu_sigma_f": "fuel_branch.nsf", "sigma_s": "fuel_branch.ss",
            "chi": "fuel_branch.chi", "sigma_t": "fuel_branch.total"}
    got = _vendored("grid.Tfuel", *keys.values())
    if got is None:
        if strict:
            raise KeyError(
                "the vendored extract carries no fuel_branch tables; run "
                "tools/extract_hpmr_xs.py against the Griffin library")
        return None
    temps, *tabs = got
    return TabulatedFeedback(material=FUEL, temperatures=temps,
                             tables=dict(zip(keys, tabs)))


def hpmr_drum_worth(angle_from, angles, *, refine: int = 4, nz: int = 0,
                    materials=None, device: str = "cpu", samples: int = 10):
    """Reactivity in pcm of each angle relative to ``angle_from``.

    Measured, not assumed: two static eigenvalue solves per point. Worth this
    cost because drum worth is strongly non-linear in angle, so a manoeuvre
    written in degrees is not a fixed reactivity.

    **DIFFERENTIAL worth needs a converged mesh -- more than k_eff does.**
    The old globally refined ladder remains non-monotonic even after exact
    per-drum area conservation (16 equal-area subcells, 90 -> 95 deg,
    2-group set):

        refine    2      3      4      6      8
        pcm    +42.4 +169.5  +52.0 +141.9 +100.4

    while the absolute k over the same meshes climbs smoothly (1.014214 ->
    1.017409). Each k is converging; their *difference* is not, because the two
    angles place the 1 cm B4C arc differently against triangles that are still
    ~2.5 cm across at refine 6, and that discretization error does not cancel
    between them. Volume mixing (``absorber="polar"``, used throughout) gets the
    arc's AREA right at any refinement, but not the flux depression across an
    absorber thinner than a cell.

    So treat values from a coarse mesh as internally consistent -- fine for
    checking kinetics against theory, where the same number appears on both
    sides -- but do not quote them as the reactor's drum worth or use them to
    choose a realistic transient insertion. The accepted 11-group study uses
    ``build_hpmr2d_local(refine=8, drum_refine_levels=3, samples=0)`` for the
    working mesh and ``refine=16, drum_refine_levels=2`` for verification. Both
    give effective-r64 resolution in the drum band; see
    ``docs/hpmr_drum_refinement.md``.
    """
    from ..tri import TriDiffusionEigenSolver
    from .hpmr import build_hpmr2d, build_hpmr3d

    def k_of(a):
        p = (build_hpmr3d(refine=refine, nz=nz, drum_angle_deg=float(a),
                          absorber="polar", materials=materials, samples=samples)
             if nz else
             build_hpmr2d(refine=refine, drum_angle_deg=float(a),
                          absorber="polar", materials=materials, samples=samples))
        r = TriDiffusionEigenSolver(
            p.grid, p.materials, p.material_map, bc=p.bc, active=p.active,
            mask_bc=p.mask_bc, mix_material=p.mix_material,
            mix_weight=p.mix_weight, device=device).solve(tol_k=1e-10,
                                                          tol_source=1e-9)
        return r.k_eff

    k0 = k_of(angle_from)
    return {float(a): 1e5 * (1.0 / k0 - 1.0 / k_of(a)) for a in angles}


def hpmr_angle_for_dollars(angle_from, dollars, *, beta=None, scan=None,
                           with_worth=False, **kw):
    """The drum angle that inserts ``dollars`` of reactivity from ``angle_from``.

    Specify manoeuvres this way rather than in degrees. The drum has almost no
    worth left above ~150 deg -- the whole travel from 150 to 180 is ~0.2 $ --
    so an innocuous-looking "150 to 153 deg" is 0.05 $ and moves nothing, while
    the same three degrees from 90 deg is worth ten times more. Reactivity is
    the quantity the physics responds to; degrees are not.

    Raises if the target is outside the scanned range, rather than
    extrapolating a curve that flattens.

    **The target may not be exactly attainable.** On a coarse mesh the polar
    absorber's area fraction only changes when the arc crosses a cell boundary,
    so the worth curve is a staircase -- at refine 2, 91 and 92 deg give the
    same 17.4 pcm. This returns the closest angle it can find; pass
    ``with_worth=True`` to get ``(angle, rho_pcm)`` and report the reactivity
    actually inserted rather than the one requested.
    """
    from .hpmr import HPMR_KINETICS

    beta = float(HPMR_KINETICS.beta.sum()) if beta is None else float(beta)
    if scan is None:
        # Geometric spacing: the worth curve is steep close to the base and
        # flattens far from it, so a uniform scan either misses small targets
        # (the first 5 deg from 90 is already 0.3 $) or wastes solves on the
        # flat tail.
        # Starts fine: at refine 3 even one degree from 90 is worth ~0.8 $,
        # so a scan beginning at 5 deg cannot bracket a small target.
        offsets = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0,
                   18.0, 27.0, 40.0, 60.0)
        sign = 1.0 if dollars >= 0 else -1.0
        scan = [angle_from + sign * d for d in offsets]
        scan = [a for a in scan if 0.0 <= a <= 180.0]
    worth = hpmr_drum_worth(angle_from, scan, **kw)
    angles = sorted(worth)
    rhos = [worth[a] for a in angles]
    target = dollars * beta * 1e5
    lo, hi = min(rhos), max(rhos)
    if not (lo <= target <= hi):
        raise ValueError(
            f"{dollars:+.3f} $ ({target:+.0f} pcm) is outside the worth "
            f"available from {angle_from:g} deg over {angles[0]:g}-{angles[-1]:g} "
            f"deg, which spans {lo:+.0f} to {hi:+.0f} pcm. The drum is nearly "
            f"withdrawn above ~150 deg; start from a lower angle.")
    order = np.argsort(rhos)
    angle = float(np.interp(target, np.asarray(rhos)[order],
                            np.asarray(angles)[order]))

    # The scan only brackets the target; the curve between points is convex, so
    # interpolating it straight overshoots badly (0.25 $ asked, 0.31 $ given).
    # Secant-refine against real solves until the delivered reactivity is within
    # 2% of what was requested -- which is the whole promise of this function.
    known = dict(worth)
    lo_a = max([a for a in angles if worth[a] <= target], default=angle_from)
    hi_a = min([a for a in angles if worth[a] >= target], default=angle)
    lo_r = known.get(lo_a, 0.0)
    for _ in range(6):
        r = hpmr_drum_worth(angle_from, [angle], **kw)[angle]
        known[angle] = r
        if abs(r - target) <= 0.02 * abs(target):
            return (angle, r) if with_worth else angle
        # Keep a bracket so a bad secant step cannot walk off the curve.
        if (r - target) * (lo_r - target) > 0:
            lo_a, lo_r = angle, r
        else:
            hi_a = angle
        span = known.get(hi_a, r) - lo_r
        angle = (lo_a + (hi_a - lo_a) * (target - lo_r) / span
                 if span else 0.5 * (lo_a + hi_a))
        angle = min(max(angle, min(lo_a, hi_a)), max(lo_a, hi_a))
    # Quantized curve: hand back the best of everything actually evaluated.
    best = min(known, key=lambda a: abs(known[a] - target))
    return (best, known[best]) if with_worth else best


def hpmr_drum_ramp(problem, *, angle_from: float, angle_to: float,
                   t_start: float = 0.0, t_ramp: float = 1.0,
                   n_angles: int = 25, refine: int = 4, nz: int = 0,
                   materials=None, samples: int = 10):
    """A ``problem_at(t)`` that rotates the control drums during a transient.

    Rotating a drum is a *geometric* perturbation: the B4C arc sweeps a
    different area fraction of each triangle at every angle, which lives in the
    volume-blend arrays rather than in the cross sections. So this returns the
    4-element form of ``problem_at`` and lets the mix arrays move in time.

    The arrays are **precomputed** on ``n_angles`` steps across the sweep and
    the nearest one is returned, for two reasons. Rasterizing the exact polar
    area fractions costs ~0.5 s at refine 6, which no time loop can afford per
    step; and because the transient rebuilds its operators only when handed a
    *different object*, quantizing the angle also quantizes the rebuilds --
    the run does ``n_angles`` operator builds instead of one per step. Take
    ``n_angles`` large enough that the reactivity ramp is smooth on the scale
    the kinetics care about, not large enough to be exact.

    Note the total absorber area is conserved as the drum turns; what changes
    is where it sits relative to the core, which is the whole mechanism.
    """
    from .hpmr import build_hpmr2d, build_hpmr3d

    if n_angles < 2:
        raise ValueError("n_angles must be at least 2")
    angles = np.linspace(angle_from, angle_to, n_angles)
    build = ((lambda a: build_hpmr3d(refine=refine, nz=nz, drum_angle_deg=a,
                                     absorber="polar", materials=materials,
                                     samples=samples))
             if nz else
             (lambda a: build_hpmr2d(refine=refine, drum_angle_deg=a,
                                     absorber="polar", materials=materials,
                                     samples=samples)))
    frames = []
    for a in angles:
        q = build(float(a))
        frames.append((q.mix_material, q.mix_weight))

    mats, mmap = problem.materials, problem.material_map

    def problem_at(t):
        if t <= t_start:
            i = 0
        elif t >= t_start + t_ramp:
            i = n_angles - 1
        else:
            i = int(round((n_angles - 1) * (t - t_start) / t_ramp))
        mixm, mixw = frames[i]
        return mats, mmap, mixm, mixw

    problem_at.angles = angles
    return problem_at


def build_hpmr_coupling(problem, *, power_w: float = RATED_POWER_W,
                        design_power_w: float = RATED_POWER_W,
                        feedback=None, target_pcm_per_K: float = -2.5,
                        device: str = "cpu", warm_start: bool = True,
                        eigen_kwargs=None):
    """A :class:`ndgpu.coupling.CouplingContext` for an ``HpmrProblem``.

    Carries the neutronics problem across unchanged -- same mesh, same material
    map, same drum-arc volume mixing -- and attaches the thermal materials and
    the feedback law.

    ``power_w`` is the power the core is actually making; ``design_power_w``
    is the rating the heat pipes were sized for, and it alone sets the sink
    conductance ``h``. They are separate on purpose: a heat pipe's conductance
    is a property of the hardware, so running at half power must halve the
    temperature rise, not halve the conductance. (Tying them together made a
    zero-power case come out at ambient with the heat pipes switched off.)
    """
    from ..coupling import CouplingContext

    three_d = len(problem.grid.shape) == 4
    n_mats = len(problem.materials)
    n_groups = problem.materials[1].n_groups
    # The 2D model is a SLICE of the core, of thickness grid.height (1 cm by
    # default), not the whole thing -- so it carries its share of the rated
    # power, not all of it. Skipping this puts 160x the heat into a
    # 1 cm slab and the fuel comes out at 11,000 K. The 3D grid spans the full
    # 200 cm (160 fueled + 2 x 20 reflector) and takes the rated power as is.
    slice_power = power_w if three_d else power_w * problem.grid.height / CORE_HEIGHT
    return CouplingContext(
        grid=problem.grid, materials=problem.materials,
        material_map=problem.material_map,
        thermal_materials=hpmr_thermal_materials(three_d=three_d,
                                                 power_w=design_power_w),
        # Default to the library's own temperature branches when the extract
        # carries them (11-group), falling back to the fitted analytic law for
        # the 2-group placeholder set, which has no branches to interpolate.
        feedback=(feedback if feedback is not None
                  else (hpmr_tabulated_feedback(strict=False) if n_groups > 2
                        else None) or hpmr_feedback(n_mats, target_pcm_per_K)),
        # The problem's kinetics are the 2-group placeholder; a wider group set
        # needs velocities to match or the transient refuses to start.
        total_power=slice_power,
        kinetics=(problem.kinetics if n_groups == 2 else hpmr_kinetics_11g()),
        active=problem.active, mask_bc=problem.mask_bc, bc=problem.bc,
        mix_material=problem.mix_material, mix_weight=problem.mix_weight,
        thermal_bc=("adiabatic", "adiabatic", VESSEL_HTC) if three_d else "adiabatic",
        thermal_mask_bc=VESSEL_HTC, ambient_temperature=AMBIENT_K,
        device=device, warm_start=warm_start,
        eigen_kwargs=dict(eigen_kwargs or {"tol_k": 1e-10, "tol_source": 1e-9}))
