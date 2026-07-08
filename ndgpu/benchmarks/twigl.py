"""2D TWIGL seed-blanket kinetics benchmark.

Classic two-group transient benchmark (Hageman & Yasinsky, 1969). Quarter
core, 80 x 80 cm, reflective on the two inner faces and zero flux on the
outer ones. Three regions: a central perturbed seed, an L-shaped unperturbed
seed (identical cross sections), and a blanket. One delayed precursor family.

The transient is a reduction of the thermal absorption cross section of the
perturbed seed (region 3):
  - "step": Sigma_a2 -> 0.976667 * Sigma_a2 for t > 0
  - "ramp": Sigma_a2 * (1 - 0.11667 t) for t <= 0.2 s, constant afterwards

Geometry, cross sections and kinetics data transcribed from the FEMFFUSION
repository (examples/2D_TWIGL, https://github.com/Zonni/FEMFFUSION), which
matches the standard literature specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Kinetics, Material

TWIGL_KINETICS = Kinetics(velocities=[1.0e7, 2.0e5], beta=[0.0075], decay=[0.08])

_SEED = dict(
    diffusion=1.0 / (3.0 * np.array([0.238095, 0.83333])),
    sigma_a=[0.010, 0.150],
    nu_sigma_f=[0.007, 0.200],
    sigma_s=[[0.0, 0.01], [0.0, 0.0]],
    chi=[1.0, 0.0],
)
_BLANKET = dict(
    diffusion=1.0 / (3.0 * np.array([0.25641, 0.66667])),
    sigma_a=[0.008, 0.050],
    nu_sigma_f=[0.003, 0.060],
    sigma_s=[[0.0, 0.01], [0.0, 0.0]],
    chi=[1.0, 0.0],
)


def _perturbed_seed(factor: float) -> Material:
    xs = dict(_SEED)
    xs["sigma_a"] = [0.010, 0.150 * factor]
    return Material(name=f"seed (perturbed, f={factor:.6f})", **xs)


@dataclass
class TwiglProblem:
    grid: Grid
    material_map: np.ndarray
    kinetics: Kinetics
    bc: tuple
    problem_at: object  # callable t -> (materials, material_map)


def build_twigl(perturbation: str = "ramp", cells_per_8cm: int = 2) -> TwiglProblem:
    """Assemble the TWIGL quarter core.

    perturbation : "step", "ramp", or "none" (constant cross sections).
    cells_per_8cm: mesh refinement; the base lattice is 10 x 10 cells of 8 cm.
    """
    if perturbation not in ("step", "ramp", "none"):
        raise ValueError(f"unknown perturbation {perturbation!r}")
    r = cells_per_8cm
    n = 10 * r
    grid = Grid(shape=(n, n, 1), size=(80.0, 80.0, 8.0))

    # Region map from cell centers; symmetry planes at x = 0 and y = 0.
    x = grid.cell_centers(0)
    X, Y = np.meshgrid(x, x, indexing="ij")
    mmap = np.zeros((n, n, 1), dtype=np.int64)  # 0 = blanket
    seed = ((X < 24) & (Y > 24) & (Y < 56)) | ((Y < 24) & (X > 24) & (X < 56))
    pert = (X > 24) & (X < 56) & (Y > 24) & (Y < 56)
    mmap[seed, 0] = 1
    mmap[pert, 0] = 2

    blanket = Material(name="blanket", **_BLANKET)
    seed_mat = Material(name="seed", **_SEED)
    cache: dict[float, list] = {}

    def factor(t: float) -> float:
        if perturbation == "none" or t <= 0.0:
            return 1.0
        if perturbation == "step":
            return 0.976667
        return 1.0 - 0.11667 * min(t, 0.2)

    def problem_at(t: float):
        f = round(factor(t), 12)
        if f not in cache:
            cache[f] = [blanket, seed_mat, _perturbed_seed(f)]
        return cache[f], mmap

    bc = (("reflective", "zero-flux"), ("reflective", "zero-flux"), "reflective")
    return TwiglProblem(grid=grid, material_map=mmap, kinetics=TWIGL_KINETICS,
                        bc=bc, problem_at=problem_at)
