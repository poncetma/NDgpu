"""Rasterizing hexagonal core maps onto the body-fitted triangular lattice.

Reactor cores built from hexagonal assemblies (VVER, prismatic microreactors)
are described most naturally as a map from hex lattice *sites* to materials.
This module converts such a map into the (nrows, ncols, 2) triangular
``material_map`` the :mod:`ndgpu.tri` solver consumes, splitting every
assembly into exactly 6 refine^2 equilateral triangles.

Coordinates. Sites use axial coordinates (R, C): site centres sit at

    x = pitch * (C + R/2),      y = pitch * sqrt(3)/2 * R,

so the six neighbours of a site are (R, C+-1), (R+-1, C), (R+1, C-1) and
(R-1, C+1). The triangle lattice is indexed (row a, col b, t) with t = 0 the
"down" and t = 1 the "up" triangle of each rhombus; a cell belongs to the hex
site nearest its centroid (cube-coordinate rounding), which tiles the
hexagons exactly when refine divides the pitch evenly -- no staircase.

Sub-hex detail (control drums, partially inserted absorbers) is painted with
a per-cell ``paint`` callback that sees the centroid position and may
override the site's material.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_SQRT3 = math.sqrt(3.0)


def hex_site_xy(R: int, C: int, pitch: float):
    """Centre of hex site (R, C) in axial coordinates."""
    return pitch * (C + R / 2.0), pitch * (_SQRT3 / 2.0) * R


def hex_round(cf: float, rf: float):
    """Nearest hex site (C, R) to fractional axial coordinates (cf, rf).

    Standard cube-coordinate rounding: embed (C, R) as (x, y, z) with
    x + y + z = 0, round each, then fix the axis with the largest rounding
    error so the constraint still holds.
    """
    x, z = cf, rf
    y = -x - z
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


@dataclass
class TriRaster:
    """A rasterized core: the material map plus the physical frame.

    material_map : (nrows, ncols, 2) int map, 0 = void, padded with a
                   one-cell void border (required by the tri operator's
                   boundary handling).
    side         : triangle edge length, cm.
    origin       : (i0, j0) lattice offsets such that cell (a, b) of the
                   padded map has corner  O = (i0+a-1)*b_vec + (j0+b-1)*a_vec
                   with a_vec = side*(sqrt(3)/2, 1/2) and b_vec = side*(0, 1).
    """

    material_map: np.ndarray
    side: float
    origin: tuple

    def cell_vertices(self, a: int, b: int, t: int) -> np.ndarray:
        """Physical (x, y) vertices of triangle (a, b, t), shape (3, 2)."""
        h = self.side
        av = np.array([h * _SQRT3 / 2, h * 0.5])
        bv = np.array([0.0, h])
        O = (self.origin[0] + a - 1) * bv + (self.origin[1] + b - 1) * av
        if t == 0:                        # "down" triangle
            return np.array([O, O + av, O + bv])
        return np.array([O + av, O + bv, O + av + bv])

    def cell_centroid(self, a: int, b: int, t: int) -> np.ndarray:
        return self.cell_vertices(a, b, t).mean(axis=0)


def rasterize_hex_sites(site_material: dict, pitch: float, refine: int,
                        paint=None) -> TriRaster:
    """Rasterize {(R, C): material_id} onto the triangular lattice.

    site_material : hex sites to fill; ids must be positive (0 is void).
    pitch         : hex flat-to-flat / centre-to-centre spacing, cm.
    refine        : triangles per hex = 6 refine^2.
    paint         : optional callback  paint(x, y, site, material_id) -> id
                    called for every triangle centroid (x, y) whose nearest
                    site is in the map, to override the site's material with
                    sub-hex detail (e.g. a control-drum absorber arc).

    The raster covers all sites plus a two-hex margin, and carries the
    one-cell void border the tri operator requires.
    """
    r = int(refine)
    h = pitch / (_SQRT3 * r)
    av = np.array([h * _SQRT3 / 2, h * 0.5])
    bv = np.array([0.0, h])

    RC = list(site_material)
    imin = min(R - C for R, C in RC); imax = max(R - C for R, C in RC)
    jmin = min(R + 2 * C for R, C in RC); jmax = max(R + 2 * C for R, C in RC)
    i0 = r * imin - 2 * r; j0 = r * jmin - 2 * r
    ni = r * imax + 2 * r - i0 + 1; nj = r * jmax + 2 * r - j0 + 1

    out = np.zeros((ni, nj, 2), dtype=np.int64)
    for a in range(ni):
        for b in range(nj):
            O = (i0 + a) * bv + (j0 + b) * av
            for t, f in ((0, 1.0 / 3.0), (1, 2.0 / 3.0)):
                cx, cy = O + (av + bv) * f
                Rf = cy / (pitch * _SQRT3 / 2)
                C, R = hex_round(cx / pitch - Rf / 2, Rf)
                mid = site_material.get((R, C))
                if mid is None:
                    continue
                if paint is not None:
                    mid = paint(cx, cy, (R, C), mid)
                out[a, b, t] = mid
    return TriRaster(material_map=np.pad(out, ((1, 1), (1, 1), (0, 0))),
                     side=h, origin=(i0, j0))
