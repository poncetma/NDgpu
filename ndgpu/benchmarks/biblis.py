"""2D BIBLIS two-group PWR k-eigenvalue benchmark.

Classic two-dimensional PWR quarter-symmetric core (Nakata & Martin, Nucl.
Sci. Eng. 85, 289-305 (1983)). A 17 x 17 lattice of 23.1226 cm assemblies on
a Cartesian grid; the octagonal core is embedded in the square with the corner
positions excised (vacuum boundary on the core surface). Eight homogenized
two-group compositions (material 3 is the radial reflector); one downscatter,
no upscatter, no fission spectrum in the thermal group.

Geometry and cross sections transcribed from the FEMFFUSION repository
(examples/2D_BIBLIS, https://github.com/Zonni/FEMFFUSION): the composition map
combines that example's Geometry_Matrix (active-cell pattern) with the .xsec
Materials layout, and boundaries are vacuum (FEMFFUSION BC=2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Material

ASSEMBLY_PITCH = 23.1226  # cm

# 17 x 17 composition map: digit = material id (1..8), "." = outside the core.
_MAP = [
    "....333333333....",
    "..3334444444333..",
    ".334481111184433.",
    ".344517171715443.",
    "33452828182825433",
    "34818282628281843",
    "34172818281827143",
    "34118281818281143",
    "34171628182617143",
    "34118281818281143",
    "34172818281827143",
    "34818282628281843",
    "33452828182825433",
    ".344517171715443.",
    ".334481111184433.",
    "..3334444444333..",
    "....333333333....",
]

# Two-group data per material, transcribed from biblis.xsec:
#   (Sigma_tr1, Sigma_a1, nuSigma_f1, Sigma_1->2, Sigma_tr2, Sigma_a2, nuSigma_f2)
# D_g = 1 / (3 Sigma_tr,g). Material 3 is the non-fissile reflector.
_XS = {
    1: (0.232126276695, 0.0095042, 0.0058708, 0.017754, 0.917010545621, 0.0750058, 0.0960670),
    2: (0.232029328507, 0.0096785, 0.0061908, 0.017621, 0.916758342501, 0.0784360, 0.1035800),
    3: (0.252525252525, 0.0026562, 0.0000000, 0.023106, 1.202501202500, 0.0715960, 0.0000000),
    4: (0.231658442792, 0.0103630, 0.0074527, 0.017101, 0.916254352208, 0.0914080, 0.1323600),
    5: (0.231787311963, 0.0100030, 0.0061908, 0.017290, 0.909504320146, 0.0848280, 0.1035800),
    6: (0.231722859460, 0.0101320, 0.0064285, 0.017192, 0.909504320146, 0.0873140, 0.1091100),
    7: (0.231658442792, 0.0101650, 0.0061908, 0.017125, 0.906043308870, 0.0880240, 0.1035800),
    8: (0.231594061928, 0.0102940, 0.0064285, 0.017027, 0.905797101449, 0.0905100, 0.1091100),
}

# Published 2D eigenvalue for the BIBLIS configuration.
K_REFERENCE = 1.0287


def _material(mid: int) -> Material:
    tr1, a1, nsf1, s12, tr2, a2, nsf2 = _XS[mid]
    return Material(
        name=f"biblis-{mid}",
        diffusion=[1.0 / (3.0 * tr1), 1.0 / (3.0 * tr2)],
        sigma_a=[a1, a2],
        nu_sigma_f=[nsf1, nsf2],
        sigma_s=[[0.0, s12], [0.0, 0.0]],
        chi=[1.0, 0.0],
    )


@dataclass
class BiblisProblem:
    grid: Grid
    materials: list
    material_map: np.ndarray
    active: np.ndarray
    bc: object
    mask_bc: object


def build_biblis(cells_per_assembly: int = 1) -> BiblisProblem:
    """Assemble the 2D BIBLIS core.

    cells_per_assembly : mesh refinement; each 23.1226 cm assembly is split into
    r x r cells (base lattice 17 x 17).
    """
    r = cells_per_assembly
    base = np.array([[0 if ch == "." else int(ch) for ch in row] for row in _MAP],
                    dtype=np.int64)
    if base.shape != (17, 17):
        raise ValueError("BIBLIS map must be 17 x 17")
    comp = np.kron(base, np.ones((r, r), dtype=np.int64))  # 17r x 17r
    mmap = comp[:, :, None]                                 # 0 = void, 1..8 = material
    active = mmap > 0

    n = 17 * r
    L = 17 * ASSEMBLY_PITCH
    grid = Grid(shape=(n, n, 1), size=(L, L, ASSEMBLY_PITCH))

    # Index 0 is the inert void filler for the excised corners.
    void = Material(name="void", diffusion=[1.0, 1.0], sigma_a=[0.0, 0.0],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.0], [0.0, 0.0]])
    materials = [void] + [_material(mid) for mid in range(1, 9)]

    # Vacuum on the in-plane (x, y) faces; reflective on z (2D = infinite slab).
    bc = ("vacuum", "vacuum", "reflective")
    return BiblisProblem(grid=grid, materials=materials, material_map=mmap,
                         active=active, bc=bc, mask_bc="vacuum")
