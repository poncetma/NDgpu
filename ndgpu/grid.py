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

    geometry : "cartesian" (default) or "cylindrical". A cylindrical grid is
        an (r, z) revolution body: the x axis is the radius (size[0] = outer
        radius, measured from the symmetry axis at r = 0), the z axis is
        axial, and the y axis is unused (ny must be 1). The axis needs no
        boundary condition -- the r = 0 face has zero area -- so any bc given
        for x_lo is inert; use zero-flux/vacuum on x_hi for a bare surface.
    """

    shape: tuple[int, int, int]
    size: tuple[float, float, float]
    geometry: str = "cartesian"

    def __post_init__(self):
        if len(self.shape) != 3 or len(self.size) != 3:
            raise ValueError("shape and size must be length-3 tuples")
        if any(n < 1 for n in self.shape):
            raise ValueError(f"grid shape must be positive, got {self.shape}")
        if any(L <= 0 for L in self.size):
            raise ValueError(f"grid size must be positive, got {self.size}")
        if self.geometry not in ("cartesian", "cylindrical"):
            raise ValueError(f"geometry must be 'cartesian' or 'cylindrical', got {self.geometry!r}")
        if self.geometry == "cylindrical" and self.shape[1] != 1:
            raise ValueError("cylindrical (r-z) grids are 2D: ny must be 1")

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

    def cylindrical_metrics(self):
        """Radial metric factors for the finite-volume stencil, or None.

        For a cylindrical grid, returns (cell_w, face_w, r_lo, r_hi):
        cell_w  : (nx, 1, 1) cell-center radii r_i+1/2 -- the relative cell
                  volume factor (multiply a per-unit-volume equation by it to
                  make the radial two-point flux stencil symmetric),
        face_w  : (nx-1, 1, 1) interior x-face radii r_i+1 (relative face area),
        r_lo/r_hi : radii of the two radial boundary faces (r_lo = 0: the
                  symmetry axis carries no boundary term).
        Cartesian grids return None (all factors are 1).
        """
        if self.geometry != "cylindrical":
            return None
        dr = self.spacing[0]
        rc = ((np.arange(self.shape[0]) + 0.5) * dr).reshape(-1, 1, 1)
        rf = (np.arange(1, self.shape[0]) * dr).reshape(-1, 1, 1)
        return rc, rf, 0.0, self.size[0]
