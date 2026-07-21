"""OECD/NEA C5G7-TD: time-dependent extension of the C5G7 MOX benchmark (2D).

"Benchmark for Deterministic Time-Dependent Neutron Transport Calculations
without Spatial Homogenization (C5G7-TD)", V. Boyarinov, P. Fomichenko,
J. Hou, K. Ivanov et al., NEA/NSC/DOC(2016); see also Hou, Ivanov, Boyarinov,
Fomichenko, Nucl. Eng. Des. 317 (2017) 177-189. Kinetics data (8 delayed
families) transcribed from the benchmark tables via the FEMFFUSION repository
(examples/2D_C5G7-TD/c5g7.prec, https://github.com/Zonni/FEMFFUSION).

The 2D exercises perturb the steady C5G7 quarter core (see c5g7.py):

  TD0 (5 cases)  step control-rod insertion: the guide-tube material of the
                 affected assemblies becomes (1-f) GT + f CR with f = 0.10 for
                 0 < t <= 1 s, 0.05 for 1 < t <= 2 s, 0 afterwards.
  TD1 (5 cases)  ramp: f = 0.01 t up to 1 s, 0.01 (2 - t) back down, 0 after.
  TD2 (3 cases)  same ramp shape with a 10x stroke: f = 0.10 t / 0.10 (2 - t).
  TD3 (4 cases)  moderator density: all in-assembly moderator cross sections
                 (Zone 2 of every pin cell; the reflector is untouched) scale
                 by g(t), linear from 1 to omega over the first second and
                 back to 1 over the next; omega = 0.95/0.90/0.85/0.80.

Rod banks (one per assembly of the quarter core) follow the benchmark's
numbering: bank 1 = inner UO2 (at the reflective corner), banks 2/3 = the two
MOX assemblies, bank 4 = outer UO2. The core's diagonal mirror symmetry makes
banks 2 and 3 equivalent for core-integral results; the benchmark moves bank 3
in the single-MOX cases. Control rods enter the 24 guide tubes only -- the
central fission chamber is never rodded. Cases: TD0-1/TD1-1 move bank 1,
TD0-2/TD1-2 bank 3, TD0-3/TD1-3 bank 4, TD0-4/TD1-4 banks 1+3+4, TD0-5/TD1-5
all four; TD2-1/2/3 move banks 1/3/4; TD3-1..4 set omega = 0.95..0.80.
Simulations run to 10 s.

As in the steady benchmark, the geometry is either pin-cell volume-homogenized
(default) or pin-resolved (exact-area rasterization of the 0.54 cm cylinders,
cells_per_pin >= 8-10). Kinetics are the benchmark's material-dependent
tables: per-material group velocities (1/v volume-averaged over the pin cell
in the homogenized model), per-fuel delayed fractions, and per-family delayed
spectra (the spectra differ across fuels only in the third digit; the UO2 set
is used core-wide). Velocities are held at their unrodded values during rod
movement, the simplest treatment explicitly permitted by the specification
(Sec 3.1). The fission chamber's (negligible) fission source is treated as
all-prompt: the benchmark assigns it no delayed data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Kinetics
from ._c5g7_data import C5G7_XS
from .c5g7 import (CORE_LAYOUT, FUEL_FRACTION, MOX_LATTICE, N_PIN, PIN_PITCH,
                   UO2_LATTICE, _material_from_xs, _pin_coverage)

# --- kinetics data (benchmark Tables 9-13) --------------------------------

C5G7TD_DECAY = np.array([1.247e-02, 2.829e-02, 4.252e-02, 1.330e-01,
                         2.925e-01, 6.665e-01, 1.635e+00, 3.555e+00])

# Delayed fractions per family, one row per fuel.
_BETA = {
    "UO2":      [2.13333e-04, 1.04514e-03, 6.03969e-04, 1.33963e-03,
                 2.29386e-03, 7.05174e-04, 6.00381e-04, 2.07736e-04],
    "MOX-4.3%": [7.82484e-05, 6.40534e-04, 2.27884e-04, 5.78624e-04,
                 9.97539e-04, 4.33265e-04, 3.22355e-04, 1.23882e-04],
    "MOX-7%":   [7.65120e-05, 6.34833e-04, 2.23483e-04, 5.68882e-04,
                 9.81163e-04, 4.29227e-04, 3.18971e-04, 1.21830e-04],
    "MOX-8.7%": [7.58799e-05, 6.33750e-04, 2.22271e-04, 5.66810e-04,
                 9.77854e-04, 4.29965e-04, 3.19265e-04, 1.21188e-04],
}

# Group velocities (cm/s) per material zone.
_VELOCITY = {
    "Moderator":       [2.23517e+09, 4.98880e+08, 3.84974e+07, 5.12639e+06,
                        1.67542e+06, 7.26031e+05, 2.81629e+05],
    "UO2":             [2.23466e+09, 5.07347e+08, 3.86595e+07, 5.13931e+06,
                        1.67734e+06, 7.28603e+05, 2.92902e+05],
    "MOX-4.3%":        [2.23473e+09, 5.07114e+08, 3.88385e+07, 5.16295e+06,
                        1.75719e+06, 7.68973e+05, 2.94764e+05],
    "MOX-7%":          [2.23479e+09, 5.07355e+08, 3.91436e+07, 5.18647e+06,
                        1.78072e+06, 7.84470e+05, 3.02310e+05],
    "MOX-8.7%":        [2.23483e+09, 5.07520e+08, 3.93259e+07, 5.20109e+06,
                        1.79321e+06, 7.91377e+05, 3.05435e+05],
    "Fission Chamber": [2.24885e+09, 5.12300e+08, 3.75477e+07, 5.02783e+06,
                        1.66563e+06, 6.70396e+05, 2.51392e+05],
    "Guide Tube":      [2.21473e+09, 4.54712e+08, 4.22099e+07, 5.36964e+06,
                        1.71422e+06, 7.63783e+05, 2.93629e+05],
}

# Delayed emission spectra per family (UO2 table; groups 4-7 are zero).
C5G7TD_CHI_DELAYED = np.array([
    [0.00075, 0.98512, 0.01413, 0.0, 0.0, 0.0, 0.0],
    [0.03049, 0.96907, 0.00044, 0.0, 0.0, 0.0, 0.0],
    [0.00457, 0.97401, 0.02142, 0.0, 0.0, 0.0, 0.0],
    [0.02002, 0.97271, 0.00727, 0.0, 0.0, 0.0, 0.0],
    [0.05601, 0.93818, 0.00581, 0.0, 0.0, 0.0, 0.0],
    [0.06098, 0.93444, 0.00458, 0.0, 0.0, 0.0, 0.0],
    [0.10635, 0.88298, 0.01067, 0.0, 0.0, 0.0, 0.0],
    [0.09346, 0.90260, 0.00394, 0.0, 0.0, 0.0, 0.0],
])

# --- case table ------------------------------------------------------------

# case -> (exercise, rod banks moved) for the rod exercises, omega for TD3.
CASES = {
    "TD0-1": (0, (1,)), "TD0-2": (0, (3,)), "TD0-3": (0, (4,)),
    "TD0-4": (0, (1, 3, 4)), "TD0-5": (0, (1, 2, 3, 4)),
    "TD1-1": (1, (1,)), "TD1-2": (1, (3,)), "TD1-3": (1, (4,)),
    "TD1-4": (1, (1, 3, 4)), "TD1-5": (1, (1, 2, 3, 4)),
    "TD2-1": (2, (1,)), "TD2-2": (2, (3,)), "TD2-3": (2, (4,)),
    "TD3-1": (3, 0.95), "TD3-2": (3, 0.90), "TD3-3": (3, 0.85),
    "TD3-4": (3, 0.80),
}

# Assembly block (row, col) of CORE_LAYOUT -> rod bank number.
_BANK_OF_BLOCK = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}

# Materials-list layout (fixed indices, stable in time -- required by the
# per-material kinetics tables and the transient operator-rebuild cache):
# 0-3 fuels U/MOX4.3/MOX7/MOX8.7, 4-7 guide tubes of banks 1-4, 8 fission
# chamber, 9 in-assembly moderator, 10 reflector water.
_ZONE1_NAME = ["UO2", "MOX-4.3%", "MOX-7%", "MOX-8.7%",
               "Guide Tube", "Guide Tube", "Guide Tube", "Guide Tube",
               "Fission Chamber"]
_CORE_WATER, _REFLECTOR = 9, 10
N_MATERIALS = 11


def _rod_fraction(exercise: int, t: float) -> float:
    """Control-rod share f of the guide-tube mixture at time t (end-of-step
    convention: the value holds for the step *ending* at t, so the TD0 steps
    at t = 1, 2 s take effect on the first step after)."""
    if t <= 0.0 or t > 2.0:
        return 0.0
    if exercise == 0:
        return 0.10 if t <= 1.0 else 0.05
    rate = 0.01 if exercise == 1 else 0.10
    return rate * t if t <= 1.0 else rate * (2.0 - t)


def _water_factor(omega: float, t: float) -> float:
    """Moderator density fraction g(t) for TD3."""
    if t <= 0.0 or t >= 2.0:
        return 1.0
    return 1.0 - (1.0 - omega) * (t if t <= 1.0 else 2.0 - t)


def _xs(name: str) -> dict:
    return {k: np.asarray(v, dtype=np.float64) for k, v in C5G7_XS[name].items()}


def _blend_xs(a: dict, b: dict, f: float) -> dict:
    """(1-f) a + f b for every table (chi from a: b is the non-fissile rod)."""
    out = {k: (1.0 - f) * a[k] + f * b[k] for k in ("total", "fission",
                                                    "nu_fission", "scatter")}
    out["chi"] = a["chi"]
    return out


def _scale_xs(a: dict, w: float) -> dict:
    """Density scaling: every macroscopic cross section is proportional to
    the nuclide density (chi is a spectrum and does not scale)."""
    out = {k: w * a[k] for k in ("total", "fission", "nu_fission", "scatter")}
    out["chi"] = a["chi"]
    return out


def _pin_cell(name: str, zone1: dict, water: dict):
    """Volume-homogenized pin cell (Zone 1 cylinder + Zone 2 moderator).
    Returns (Material, fission xs)."""
    f = FUEL_FRACTION
    mix = lambda key: f * zone1[key] + (1.0 - f) * water[key]
    mat = _material_from_xs(name=name, total=mix("total"),
                            nu_fission=mix("nu_fission"), chi=zone1["chi"],
                            scatter=mix("scatter"))
    return mat, mix("fission")


def _pure(name: str, xs: dict):
    mat = _material_from_xs(name=name, total=xs["total"],
                            nu_fission=xs["nu_fission"], chi=xs["chi"],
                            scatter=xs["scatter"])
    return mat, xs["fission"]


@dataclass
class C5G7TDProblem:
    case: str
    grid: Grid
    bc: tuple
    kinetics: Kinetics
    problem_at: object       # callable t -> (materials, material_map)
    material_map: np.ndarray
    fission_xs: np.ndarray   # (N_MATERIALS, G) at t=0, for pin powers
    pin_map: np.ndarray      # (51, 51) material indices at pin-cell level
    cells_per_pin: int
    mix_material: np.ndarray = None   # pin-resolved only
    mix_weight: np.ndarray = None


def build_c5g7_td(case: str = "TD1-1", cells_per_pin: int = 2,
                  pin_resolved: bool = False) -> C5G7TDProblem:
    """Assemble a 2D C5G7-TD transient case.

    case          : one of CASES ("TD0-1" ... "TD3-4").
    cells_per_pin : spatial cells per 1.26 cm pin cell (as build_c5g7_2d).
    pin_resolved  : heterogeneous pin geometry via exact-area mixing
                    (needs cells_per_pin >= 8-10) instead of volume-
                    homogenized pin cells.

    The returned problem_at(t) yields a *stable* materials-list layout (see
    module comment) whose entries change value -- never identity order -- as
    rods move or the moderator density changes; results are cached per
    perturbation state, so operators are only rebuilt when cross sections
    actually change. Pass mix_material/mix_weight (pin-resolved) and kinetics
    straight to TransientSolver.
    """
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; expected one of {sorted(CASES)}")
    exercise, spec = CASES[case]
    banks = spec if exercise != 3 else ()
    omega = spec if exercise == 3 else 1.0

    # Pin-cell material map with per-bank guide-tube indices.
    n = N_PIN * len(CORE_LAYOUT)
    pin_map = np.full((n, n), _REFLECTOR, dtype=np.int64)
    lattices = {"U": UO2_LATTICE, "M": MOX_LATTICE}
    fuel_idx = {"U": 0, "m": 1, "o": 2, "x": 3, "F": 8}
    for bj, row in enumerate(CORE_LAYOUT):
        for bi, block in enumerate(row):
            if block == ".":
                continue
            bank = _BANK_OF_BLOCK[(bj, bi)]
            for pj, prow in enumerate(lattices[block]):
                for pi, c in enumerate(prow):
                    idx = 3 + bank if c == "G" else fuel_idx[c]
                    pin_map[bj * N_PIN + pj, bi * N_PIN + pi] = idx

    s = cells_per_pin
    expanded = np.kron(pin_map, np.ones((s, s), dtype=np.int64))
    mix_material = mix_weight = None
    if pin_resolved:
        # Exact-area rasterization as in build_c5g7_2d, but partial cells
        # blend against the *in-assembly* moderator (index 9), which TD3
        # scales while the reflector stays nominal.
        frac = np.tile(_pin_coverage(s), (n, n))
        frac[expanded >= _CORE_WATER] = 0.0
        eps = 1e-9
        full = frac >= 1.0 - eps
        part = (frac > eps) & ~full
        in_core = expanded != _REFLECTOR
        base = np.where(full, expanded,
                        np.where(in_core, _CORE_WATER, _REFLECTOR))
        mm = np.where(part, expanded, -1)
        wt = np.where(part, frac, 0.0)
        material_map = base.T[:, :, None].copy()
        mix_material = mm.T[:, :, None].copy()
        mix_weight = wt.T[:, :, None].copy()
    else:
        material_map = expanded.T[:, :, None].copy()

    # Materials list for a perturbation state (rod fractions per bank, water
    # density factor). Zone-1 tables of the rodded guide tubes blend GT->CR;
    # in the homogenized model every pin cell re-homogenizes against the
    # scaled Zone-2 water.
    zone1_static = [_xs(nm) for nm in _ZONE1_NAME]
    gt, cr, water = _xs("Guide Tube"), _xs("Control Rod"), _xs("Water")

    def materials_for(f_banks: tuple, w: float):
        wat = water if w == 1.0 else _scale_xs(water, w)
        mats, fis = [], []
        for k in range(9):
            zone1 = zone1_static[k]
            if 4 <= k <= 7 and f_banks[k - 4] > 0.0:
                zone1 = _blend_xs(gt, cr, f_banks[k - 4])
            name = (f"{_ZONE1_NAME[k]}" + (f" bank {k - 3}" if 4 <= k <= 7 else "")
                    + (f" f={f_banks[k - 4]:.6f}" if 4 <= k <= 7 else ""))
            if pin_resolved:
                mat, f_xs = _pure(name, zone1)
            else:
                mat, f_xs = _pin_cell(name + " pin cell", zone1, wat)
            mats.append(mat)
            fis.append(f_xs)
        for name, xs in (("Moderator", wat), ("Water reflector", water)):
            mat, f_xs = _pure(name, xs)
            mats.append(mat)
            fis.append(f_xs)
        return mats, np.array(fis)

    cache: dict[tuple, list] = {}

    def problem_at(t: float):
        f = _rod_fraction(exercise, t)
        f_banks = tuple(round(f, 12) if b in banks else 0.0
                        for b in range(1, 5))
        w = round(_water_factor(omega, t), 12)
        key = (f_banks, w)
        if key not in cache:
            cache[key] = materials_for(f_banks, w)[0]
        return cache[key], material_map

    mats0, fission_xs = materials_for((0.0,) * 4, 1.0)
    cache[((0.0,) * 4, 1.0)] = mats0

    # Kinetics: velocities per material (1/v volume-averaged over the pin
    # cell in the homogenized model), delayed fractions per fuel, common
    # decay constants, per-family delayed spectra.
    v_mod = np.array(_VELOCITY["Moderator"])
    V = np.empty((N_MATERIALS, 7))
    for k in range(9):
        v1 = np.array(_VELOCITY[_ZONE1_NAME[k]])
        if pin_resolved:
            V[k] = v1
        else:
            V[k] = 1.0 / (FUEL_FRACTION / v1 + (1.0 - FUEL_FRACTION) / v_mod)
    V[_CORE_WATER] = V[_REFLECTOR] = v_mod
    B = np.zeros((N_MATERIALS, 8))
    for k, nm in enumerate(("UO2", "MOX-4.3%", "MOX-7%", "MOX-8.7%")):
        B[k] = _BETA[nm]
    kinetics = Kinetics(velocities=V, beta=B, decay=C5G7TD_DECAY,
                        chi_delayed=C5G7TD_CHI_DELAYED)

    L = n * PIN_PITCH
    grid = Grid(shape=(n * s, n * s, 1), size=(L, L, PIN_PITCH))
    bc = (("reflective", "zero-flux"),
          ("reflective", "zero-flux"),
          "reflective")
    return C5G7TDProblem(case=case, grid=grid, bc=bc, kinetics=kinetics,
                         problem_at=problem_at, material_map=material_map,
                         fission_xs=fission_xs, pin_map=pin_map,
                         cells_per_pin=s, mix_material=mix_material,
                         mix_weight=mix_weight)
