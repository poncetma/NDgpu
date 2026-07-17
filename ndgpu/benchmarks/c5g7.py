"""OECD/NEA C5G7 MOX benchmark (2D), NEA/NSC/DOC(2001)4.

Quarter core, 7 energy groups: a 2x2 checkerboard of UO2 and MOX 17x17 fuel
assemblies against the two reflective symmetry planes, surrounded by a
21.42 cm water reflector, vacuum on the outer boundary.

C5G7 is specified as a *transport* benchmark with explicit cylindrical fuel
pins. For a Cartesian diffusion solver we apply the standard first-order
treatment: each 1.26 cm pin cell is homogenized by volume-weighting the pin
and moderator cross sections (fuel volume is conserved exactly and the
quarter-core symmetry is preserved). Heterogeneity remains at the pin-cell
level: six distinct homogenized pin types plus the pure water reflector. The
outer vacuum boundary is modeled as zero flux, which is accurate here because
a full assembly-width of water separates it from the fuel.

Expect k_eff within a few hundred pcm of the transport reference — the
residual is the physics gap (diffusion + volume homogenization vs. transport),
not solver error; the solver's own discretization converges with cells_per_pin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Material
from ._c5g7_data import C5G7_XS

# MCNP transport reference for the 2D configuration (benchmark report).
K_REFERENCE_2D = 1.18655

PIN_PITCH = 1.26          # cm
PIN_RADIUS = 0.54         # cm
N_PIN = 17                # pins per assembly side
ASSEMBLY_PITCH = N_PIN * PIN_PITCH  # 21.42 cm
FUEL_FRACTION = np.pi * PIN_RADIUS**2 / PIN_PITCH**2

# 17x17 assembly maps (from the benchmark spec, transcribed via OpenMOC's
# sample-input/benchmarks/c5g7/lattices.py). G = guide tube, F = fission
# chamber; U = UO2; m/o/x = 4.3% / 7.0% / 8.7% MOX.
UO2_LATTICE = [
    "UUUUUUUUUUUUUUUUU",
    "UUUUUUUUUUUUUUUUU",
    "UUUUUGUUGUUGUUUUU",
    "UUUGUUUUUUUUUGUUU",
    "UUUUUUUUUUUUUUUUU",
    "UUGUUGUUGUUGUUGUU",
    "UUUUUUUUUUUUUUUUU",
    "UUUUUUUUUUUUUUUUU",
    "UUGUUGUUFUUGUUGUU",
    "UUUUUUUUUUUUUUUUU",
    "UUUUUUUUUUUUUUUUU",
    "UUGUUGUUGUUGUUGUU",
    "UUUUUUUUUUUUUUUUU",
    "UUUGUUUUUUUUUGUUU",
    "UUUUUGUUGUUGUUUUU",
    "UUUUUUUUUUUUUUUUU",
    "UUUUUUUUUUUUUUUUU",
]
MOX_LATTICE = [
    "mmmmmmmmmmmmmmmmm",
    "mooooooooooooooom",
    "mooooGooGooGoooom",
    "mooGoxxxxxxxoGoom",
    "moooxxxxxxxxxooom",
    "moGxxGxxGxxGxxGom",
    "mooxxxxxxxxxxxoom",
    "mooxxxxxxxxxxxoom",
    "moGxxGxxFxxGxxGom",
    "mooxxxxxxxxxxxoom",
    "mooxxxxxxxxxxxoom",
    "moGxxGxxGxxGxxGom",
    "moooxxxxxxxxxooom",
    "mooGoxxxxxxxoGoom",
    "mooooGooGooGoooom",
    "mooooooooooooooom",
    "mmmmmmmmmmmmmmmmm",
]
# Quarter core, row 0 against one symmetry plane, column 0 against the other.
CORE_LAYOUT = ["UM.", "MU.", "..."]  # U/M = assembly, '.' = water reflector

PIN_XS_NAME = {"U": "UO2", "m": "MOX-4.3%", "o": "MOX-7%", "x": "MOX-8.7%",
               "G": "Guide Tube", "F": "Fission Chamber"}


def _material_from_xs(name: str, total, nu_fission, chi, scatter) -> Material:
    total = np.asarray(total)
    scatter = np.asarray(scatter)
    chi = np.asarray(chi)
    if chi.sum() > 0:
        chi = chi / chi.sum()
    return Material(
        name=name,
        diffusion=1.0 / (3.0 * total),  # transport-corrected total -> D
        sigma_a=total - scatter.sum(axis=1),
        nu_sigma_f=nu_fission,
        sigma_s=scatter,
        chi=chi,
        total=total,
    )


def _homogenized_pin(pin_char: str) -> tuple[Material, np.ndarray]:
    """Volume-weighted mix of a pin with its surrounding moderator.

    Returns (Material, fission cross section) for the homogenized pin cell.
    """
    pin = {k: np.asarray(v) for k, v in C5G7_XS[PIN_XS_NAME[pin_char]].items()}
    wat = {k: np.asarray(v) for k, v in C5G7_XS["Water"].items()}
    f = FUEL_FRACTION
    mix = lambda key: f * pin[key] + (1.0 - f) * wat[key]
    mat = _material_from_xs(
        name=f"{PIN_XS_NAME[pin_char]} pin cell",
        total=mix("total"),
        nu_fission=mix("nu_fission"),
        chi=pin["chi"],  # water is non-fissile; emission spectrum is the pin's
        scatter=mix("scatter"),
    )
    return mat, mix("fission")


def _pure_pin(pin_char: str) -> tuple[Material, np.ndarray]:
    """The pin's own (un-homogenized) material -- the cylinder interior, used by
    the pin-resolved geometry. Returns (Material, fission cross section)."""
    xs = C5G7_XS[PIN_XS_NAME[pin_char]]
    mat = _material_from_xs(name=PIN_XS_NAME[pin_char], total=xs["total"],
                            nu_fission=xs["nu_fission"], chi=xs["chi"],
                            scatter=xs["scatter"])
    return mat, np.asarray(xs["fission"])


@dataclass
class C5G7Problem:
    grid: Grid
    materials: list
    material_map: np.ndarray
    bc: tuple
    fission_xs: np.ndarray   # (n_materials, G), for pin-power post-processing
    pin_map: np.ndarray      # (51, 51) material indices at pin-cell level
    cells_per_pin: int
    mix_material: np.ndarray = None   # pin-resolved: per-cell blend partner
    mix_weight: np.ndarray = None     # pin-resolved: fuel-covered area fraction


def _pin_coverage(s: int, sub: int = 16) -> np.ndarray:
    """Fraction of each of the s x s sub-cells of a pin covered by the centred
    r = 0.54 cm fuel cylinder (identical for every pin). Sub-sampled sub x sub."""
    R = PIN_RADIUS / PIN_PITCH
    off = (np.arange(s) + 0.5) / s - 0.5                    # sub-cell centres
    soff = (np.arange(sub) + 0.5) / sub - 0.5               # points within a sub-cell (in cell units)
    frac = np.empty((s, s))
    for i, cx in enumerate(off):
        for j, cy in enumerate(off):
            px = cx + soff / s
            py = cy + soff / s
            d2 = px[:, None]**2 + py[None, :]**2
            frac[i, j] = np.mean(d2 < R * R)
    return frac


def build_c5g7_2d(cells_per_pin: int = 2,
                  pin_resolved: bool = False) -> C5G7Problem:
    """Assemble the 2D quarter-core problem on a Cartesian grid.

    cells_per_pin: spatial cells per pin-cell side; the 64.26 cm quarter core
    becomes a (51 * cells_per_pin)^2 x 1 grid. z is reflective (exact 2D).

    pin_resolved: when True, follow the benchmark's heterogeneous geometry -- the
    r = 0.54 cm fuel cylinder is rasterized onto the fine mesh (interior cells
    take the pin's own material, the rest water) instead of volume-homogenizing
    each pin cell. The staircase fuel area converges to pi r^2 / pitch^2 as
    cells_per_pin grows, so this needs a fine mesh (cells_per_pin >= 8-10). This
    is the pin-resolved treatment used by pin-transport codes and by the FE
    solutions of Carreno et al. (2024); the default (False) keeps the fast
    volume-homogenized pin cells.
    """
    chars = ["U", "m", "o", "x", "G", "F"]
    build_pin = _pure_pin if pin_resolved else _homogenized_pin
    materials, fission = [], []
    for c in chars:
        mat, fis = build_pin(c)
        materials.append(mat)
        fission.append(fis)
    water = C5G7_XS["Water"]
    materials.append(_material_from_xs("Water reflector", water["total"],
                                       water["nu_fission"], water["chi"],
                                       water["scatter"]))
    fission.append(np.asarray(water["fission"]))
    idx = {c: i for i, c in enumerate(chars)}
    water_idx = len(chars)

    # Pin-cell material map, [row, col] with row 0 / col 0 on the symmetry planes.
    n = N_PIN * len(CORE_LAYOUT)
    pin_map = np.full((n, n), water_idx, dtype=np.int64)
    lattices = {"U": UO2_LATTICE, "M": MOX_LATTICE}
    for bj, row in enumerate(CORE_LAYOUT):
        for bi, block in enumerate(row):
            if block == ".":
                continue
            for pj, prow in enumerate(lattices[block]):
                for pi, c in enumerate(prow):
                    pin_map[bj * N_PIN + pj, bi * N_PIN + pi] = idx[c]

    # Expand to the solve grid: [row, col] -> [x=col, y=row], z thickness 1 cell.
    s = cells_per_pin
    expanded = np.kron(pin_map, np.ones((s, s), dtype=np.int64))
    mix_material = mix_weight = None
    if pin_resolved:
        # Carve the r = 0.54 cm fuel cylinder out of each pin cell with exact
        # area weighting: cells fully inside the circle take the pin material,
        # fully outside cells are water, and boundary cells blend pin + water by
        # their covered area fraction (mix_material / mix_weight). This conserves
        # the pi r^2 / pitch^2 fuel loading at *any* resolution -- a plain
        # in/out raster does not, and its fuel area (hence k) swings wildly with
        # cells_per_pin.
        frac = np.tile(_pin_coverage(s), (n, n))           # covered fraction
        frac[expanded == water_idx] = 0.0                  # water pins: no fuel
        eps = 1e-9
        full = frac >= 1.0 - eps
        part = (frac > eps) & ~full
        base = np.where(full, expanded, water_idx)         # pure interior / water
        mm = np.where(part, expanded, -1)                  # blend partner
        wt = np.where(part, frac, 0.0)
        material_map = base.T[:, :, None].copy()
        mix_material = mm.T[:, :, None].copy()
        mix_weight = wt.T[:, :, None].copy()
    else:
        material_map = expanded.T[:, :, None].copy()

    L = n * PIN_PITCH  # 64.26 cm
    grid = Grid(shape=(n * s, n * s, 1), size=(L, L, PIN_PITCH))
    bc = (("reflective", "zero-flux"),   # x: symmetry plane, outer vacuum
          ("reflective", "zero-flux"),   # y: symmetry plane, outer vacuum
          "reflective")                  # z: exact 2D
    return C5G7Problem(grid=grid, materials=materials, material_map=material_map,
                       bc=bc, fission_xs=np.array(fission), pin_map=pin_map,
                       cells_per_pin=s, mix_material=mix_material,
                       mix_weight=mix_weight)
