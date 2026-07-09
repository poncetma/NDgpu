"""3D IAEA PWR k-eigenvalue benchmark.

The classic IAEA-3D two-group benchmark (Benchmark Problem Book, ANL-7416
Suppl. 2, 1977): a 17 x 17 x 19 Cartesian lattice of 20 cm nodes (340 x 340 x
380 cm), octagonal core embedded in the square with excised corners. Five
homogenized two-group compositions: radial/axial reflector (1), rodded
reflector (2), exterior fuel (3), interior fuel (4) and rodded fuel (5). The
control rods are partially inserted, so the upper axial region differs from the
core. Boundaries are an albedo condition (FEMFFUSION BC=3, factor 0.4695) on
all faces.

Geometry and cross sections transcribed from the FEMFFUSION repository
(examples/3D_IAEA, https://github.com/Zonni/FEMFFUSION): the per-plane maps use
that example's Geometry_Points row ranges, and the boundary uses its
Albedo_Factors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Material

NODE_PITCH = 20.0     # cm
ALBEDO = 0.4695       # FEMFFUSION Albedo_Factors (both groups)

# Four distinct axial planes; "." is outside the octagonal core.
_PLANES = {
    "reflector": [
        ".....1111111.....", "...11111111111...", "..1111111111111..",
        ".111111111111111.", ".111111111111111.", "11111111111111111",
        "11111111111111111", "11111111111111111", "11111111111111111",
        "11111111111111111", "11111111111111111", "11111111111111111",
        ".111111111111111.", ".111111111111111.", "..1111111111111..",
        "...11111111111...", ".....1111111.....",
    ],
    "core": [
        ".....1111111.....", "...11133333111...", "..1133344433311..",
        ".113344444443311.", ".133544454445331.", "11344444444444311",
        "13344444444444331", "13444444444444431", "13445444544454431",
        "13444444444444431", "13344444444444331", "11344444444444311",
        ".133544454445331.", ".113344444443311.", "..1133344433311..",
        "...11133333111...", ".....1111111.....",
    ],
    "core_rodded": [
        ".....1111111.....", "...11133333111...", "..1133344433311..",
        ".113344444443311.", ".133544454445331.", "11344444444444311",
        "13344454445444331", "13444444444444431", "13445444544454431",
        "13444444444444431", "13344454445444331", "11344444444444311",
        ".133544454445331.", ".113344444443311.", "..1133344433311..",
        "...11133333111...", ".....1111111.....",
    ],
    "top_rodded": [
        ".....1111111.....", "...11111111111...", "..1111111111111..",
        ".111111111111111.", ".111211121121111.", "11111111111111111",
        "11111121112111111", "11111111111111111", "11112111211121111",
        "11111111111111111", "11111121112111111", "11111111111111111",
        ".111211121112111.", ".111111111111111.", "..1111111111111..",
        "...11111111111...", ".....1111111.....",
    ],
}
# Bottom reflector, 13 core layers, 4 rodded-core layers, top rodded reflector.
_AXIAL = (["reflector"] + ["core"] * 13 + ["core_rodded"] * 4 + ["top_rodded"])

# Two-group data per material:
#   (Sigma_tr1, Sigma_a1, nuSigma_f1, Sigma_1->2, Sigma_tr2, Sigma_a2, nuSigma_f2)
_XS = {
    1: (0.166667, 0.000, 0.000, 0.040, 1.111111, 0.010, 0.000),  # reflector
    2: (0.166667, 0.000, 0.000, 0.040, 1.111111, 0.055, 0.000),  # rodded reflector
    3: (0.222222, 0.010, 0.000, 0.020, 0.833333, 0.080, 0.135),  # exterior fuel
    4: (0.222222, 0.010, 0.000, 0.020, 0.833333, 0.085, 0.135),  # interior fuel
    5: (0.222222, 0.010, 0.000, 0.020, 0.833333, 0.130, 0.135),  # rodded fuel
}

# Published 3D benchmark eigenvalue (IAEA Benchmark Problem 11-A2).
K_REFERENCE = 1.02903


def _material(mid: int) -> Material:
    tr1, a1, nsf1, s12, tr2, a2, nsf2 = _XS[mid]
    return Material(
        name=f"iaea-{mid}",
        diffusion=[1.0 / (3.0 * tr1), 1.0 / (3.0 * tr2)],
        sigma_a=[a1, a2],
        nu_sigma_f=[nsf1, nsf2],
        sigma_s=[[0.0, s12], [0.0, 0.0]],
        chi=[1.0, 0.0],
    )


def _plane_array(name: str) -> np.ndarray:
    rows = _PLANES[name]
    return np.array([[0 if ch == "." else int(ch) for ch in row] for row in rows],
                    dtype=np.int64)


@dataclass
class IaeaProblem:
    grid: Grid
    materials: list
    material_map: np.ndarray
    active: np.ndarray
    bc: object
    mask_bc: object


def build_iaea(cells_per_node: int = 1) -> IaeaProblem:
    """Assemble the 3D IAEA core.

    cells_per_node : mesh refinement; each 20 cm node is split into r x r x r
    cells (base lattice 17 x 17 x 19).
    """
    r = cells_per_node
    base = np.stack([_plane_array(name) for name in _AXIAL], axis=2)  # 17x17x19
    if base.shape != (17, 17, 19):
        raise ValueError("IAEA map must be 17 x 17 x 19")
    vol = np.kron(base, np.ones((r, r, r), dtype=np.int64))
    mmap = vol                                    # 0 = void, 1..5 = material
    active = mmap > 0

    nx, ny, nz = 17 * r, 17 * r, 19 * r
    grid = Grid(shape=(nx, ny, nz),
                size=(17 * NODE_PITCH, 17 * NODE_PITCH, 19 * NODE_PITCH))

    void = Material(name="void", diffusion=[1.0, 1.0], sigma_a=[0.0, 0.0],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.0], [0.0, 0.0]])
    materials = [void] + [_material(mid) for mid in range(1, 6)]

    # Albedo boundary (0.4695) on every outer and core-surface face.
    return IaeaProblem(grid=grid, materials=materials, material_map=mmap,
                       active=active, bc=ALBEDO, mask_bc=ALBEDO)
