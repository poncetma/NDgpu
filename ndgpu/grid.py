"""Structured 3D grid for the finite-volume discretization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Uniform cell-centered 3D grid over a box of size (Lx, Ly, Lz) cm.

    Cells are indexed [i, j, k] along (x, y, z); cell centers sit at
    (i + 1/2) * dx etc., so the domain boundary lies half a cell beyond the
    outermost cell centers (where the zero-flux condition is imposed).
    """

    shape: tuple[int, int, int]
    size: tuple[float, float, float]

    def __post_init__(self):
        if len(self.shape) != 3 or len(self.size) != 3:
            raise ValueError("shape and size must be length-3 tuples")
        if any(n < 1 for n in self.shape):
            raise ValueError(f"grid shape must be positive, got {self.shape}")
        if any(L <= 0 for L in self.size):
            raise ValueError(f"grid size must be positive, got {self.size}")

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(L / n for L, n in zip(self.size, self.shape))

    @property
    def n_cells(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def cell_volume(self) -> float:
        dx, dy, dz = self.spacing
        return dx * dy * dz

    def cell_centers(self, axis: int) -> np.ndarray:
        n = self.shape[axis]
        d = self.spacing[axis]
        return (np.arange(n) + 0.5) * d
