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


@dataclass
class C5G7Problem:
    grid: Grid
    materials: list
    material_map: np.ndarray
    bc: tuple
    fission_xs: np.ndarray   # (n_materials, G), for pin-power post-processing
    pin_map: np.ndarray      # (51, 51) material indices at pin-cell level
    cells_per_pin: int


def build_c5g7_2d(cells_per_pin: int = 2) -> C5G7Problem:
    """Assemble the 2D quarter-core problem on a Cartesian grid.

    cells_per_pin: spatial cells per pin-cell side; the 64.26 cm quarter core
    becomes a (51 * cells_per_pin)^2 x 1 grid. z is reflective (exact 2D).
    """
    # Homogenized pin-cell materials + pure water, indexed 0..6.
    chars = ["U", "m", "o", "x", "G", "F"]
    materials, fission = [], []
    for c in chars:
        mat, fis = _homogenized_pin(c)
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
    material_map = expanded.T[:, :, None].copy()

    L = n * PIN_PITCH  # 64.26 cm
    grid = Grid(shape=(n * s, n * s, 1), size=(L, L, PIN_PITCH))
    bc = (("reflective", "zero-flux"),   # x: symmetry plane, outer vacuum
          ("reflective", "zero-flux"),   # y: symmetry plane, outer vacuum
          "reflective")                  # z: exact 2D
    return C5G7Problem(grid=grid, materials=materials, material_map=material_map,
                       bc=bc, fission_xs=np.array(fission), pin_map=pin_map,
                       cells_per_pin=s)
