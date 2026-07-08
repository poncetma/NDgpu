"""2D VVER-440 hexagonal PWR k-eigenvalue benchmark.

Two-group hexagonal core (Chao & Shatilla, Nucl. Sci. Eng. 121, 210-225
(1995)): 349 fuel assemblies plus a reflector ring, 14.7 cm hex pitch, vacuum
boundary. Eight compositions (1-3 fuel, 4 absorber, 5/7 radial reflector,
6/8 axial reflector). This is the hexagonal-geometry case: it runs on the
structured *hex* solver (one finite-volume cell per assembly, axial
coordinates), not the Cartesian one.

Geometry and cross sections transcribed from the FEMFFUSION repository
(examples/2D_VVER440); the FEMFFUSION row-staggered map is converted to axial
coordinates by `offset_to_axial`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..materials import Kinetics, Material
from ..tri import TriGrid

HEX_PITCH = 14.7  # cm, flat-to-flat / centre-to-centre
_SQRT3 = math.sqrt(3.0)
VVER_KINETICS = Kinetics(velocities=[1.25e7, 2.5e5], beta=[0.0065], decay=[0.07841])

# Row-staggered (offset) assembly map; "." is outside the core.
_OFFSET = [
    "44444.................", "4443333444............", "4433332333344.........",
    "43633722733634........", "44333121212133344.....", "433312121121213334....",
    "4337218216125127334...", "43221221211212212234..", "433221121222121122334.",
    "43711612611621611734..", "433221121222121122334.", "4331122212112122211334",
    "463281162151261152364.", "4331122212112122211334", "433221121222121122334.",
    "43711612611621611734..", "433221121222121122334.", "43221221211212212234..",
    "4337215216125127334...", "433312121121213334....", "44333121212133344.....",
    "43633722733634........", "4433332333344.........", "4443333444............",
    "44444.................",
]

# (Sigma_tr1, Sigma_a1, nuSigma_f1, Sigma_1->2, Sigma_tr2, Sigma_a2, nuSigma_f2)
_XS = {
    1: (2.47545e-1, 0.008312, 0.004413, 0.016976, 9.00718e-1, 0.064282, 0.072784),
    2: (2.49179e-1, 0.008745, 0.005491, 0.016000, 9.07249e-1, 0.079145, 0.104256),
    3: (2.50201e-1, 0.009411, 0.006990, 0.014974, 9.17841e-1, 0.099536, 0.147261),
    4: (2.30279e-1, 0.000933, 0.000000, 0.032215, 1.324111e0, 0.033037, 0.000000),
    5: (2.70626e-1, 0.012120, 0.001345, 0.020782, 1.388732e0, 0.118846, 0.027352),
    6: (2.491789e-1, 0.008747, 0.005492, 0.015996, 9.070813e-1, 0.079153, 0.104316),
    7: (2.475442e-1, 0.008317, 0.004416, 0.016968, 9.00470e-1, 0.064282, 0.072846),
    8: (2.706419e-1, 0.012123, 0.001342, 0.020785, 1.389224e0, 0.118870, 0.027299),
}


def _material(mid: int) -> Material:
    tr1, a1, nsf1, s12, tr2, a2, nsf2 = _XS[mid]
    return Material(
        name=f"vver-{mid}",
        diffusion=[1.0 / (3.0 * tr1), 1.0 / (3.0 * tr2)],
        sigma_a=[a1, a2], nu_sigma_f=[nsf1, nsf2],
        sigma_s=[[0.0, s12], [0.0, 0.0]], chi=[1.0, 0.0],
    )


def _material8_ramp(sigma_a2_thermal: float) -> Material:
    """Material 8 (axial reflector) with a perturbed thermal absorption."""
    tr1, a1, nsf1, s12, tr2, _, nsf2 = _XS[8]
    return Material(
        name=f"vver-8 (Sa2={sigma_a2_thermal:.6f})",
        diffusion=[1.0 / (3.0 * tr1), 1.0 / (3.0 * tr2)],
        sigma_a=[a1, sigma_a2_thermal], nu_sigma_f=[nsf1, nsf2],
        sigma_s=[[0.0, s12], [0.0, 0.0]], chi=[1.0, 0.0])


def _ramp_hex_sigma_a2(t: float) -> float:
    """FEMFFUSION 'Ramp_hex': material-8 thermal Sigma_a ramps 0.118870 -> 0.016917
    over t in [0, 1] s, then back to 0.118870 over [1, 2] s (constant afterwards)."""
    hi, lo = 0.118870, 0.016917
    if t <= 1.0:
        return hi * (1.0 - t) + lo * t
    if t < 2.0:
        return hi * (t - 1.0) + lo * (2.0 - t)
    return hi


# Bulk +0.1$ insertion: material 1 (102 assemblies, spread across the core)
# thermal Sigma_a scaled by (1 - 0.0035); a distributed, well-resolved
# perturbation far below prompt-critical (FEMFFUSION Ramp, Slope=-3.5, Cut=1 ms).
_BULK_FRACTION = 0.0035


def _bulk_factor(t: float) -> float:
    return 1.0 + (-3.5) * min(t, 0.001)         # -> 1 - 0.0035 for t >= 1 ms


def _material1_bulk(factor: float) -> Material:
    tr1, a1, nsf1, s12, tr2, a2, nsf2 = _XS[1]
    return Material(
        name=f"vver-1 (Sa2 x{factor:.5f})",
        diffusion=[1.0 / (3.0 * tr1), 1.0 / (3.0 * tr2)],
        sigma_a=[a1, a2 * factor], nu_sigma_f=[nsf1, nsf2],
        sigma_s=[[0.0, s12], [0.0, 0.0]], chi=[1.0, 0.0])


def _coarse_hex_map() -> np.ndarray:
    """Correct hexagonal assembly map (R, C) -> material id (0 = void).

    Each .xsec row is centred AND counter-sheared by -R/2 so the core is a true
    hexagon (not the sheared parallelogram the naive placement produces).
    """
    raw = []
    for row in _OFFSET:
        vals = [0 if ch == "." else int(ch) for ch in row]
        nz = [i for i, v in enumerate(vals) if v > 0]
        raw.append([vals[i] for i in range(nz[0], nz[-1] + 1)])
    W = max(len(r) for r in raw)
    NR = len(raw)
    coarse = np.zeros((NR, W + 2 * NR), dtype=np.int64)
    for r, rowvals in enumerate(raw):
        start = (W - len(rowvals)) // 2 - r // 2 + NR
        coarse[r, start:start + len(rowvals)] = rowvals
    return coarse


def _hex_round(cf, rf):
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
    return int(rx), int(rz)                      # (C, R)


def _tri_material_map(coarse: np.ndarray, r: int):
    """Body-fitted triangular material map (nrows, ncols, 2) at refinement r,
    plus the triangle side length. Each hexagon -> 6 r^2 triangles."""
    CR, CC = coarse.shape
    RC = [(R, C) for R in range(CR) for C in range(CC) if coarse[R, C] > 0]
    imin = min(R - C for R, C in RC); imax = max(R - C for R, C in RC)
    jmin = min(R + 2 * C for R, C in RC); jmax = max(R + 2 * C for R, C in RC)
    p = HEX_PITCH
    h = p / (_SQRT3 * r)
    ax = np.array([h * _SQRT3 / 2, h * 0.5]); bx = np.array([0.0, h])
    i0 = r * imin - 2 * r; j0 = r * jmin - 2 * r
    ni = r * imax + 2 * r - i0 + 1; nj = r * jmax + 2 * r - j0 + 1
    out = np.zeros((ni, nj, 2), dtype=np.int64)
    for a in range(ni):
        for b in range(nj):
            O = (i0 + a) * bx + (j0 + b) * ax
            for k, f in ((0, 1.0 / 3.0), (1, 2.0 / 3.0)):
                cx, cy = O + (ax + bx) * f
                Rf = cy / (p * _SQRT3 / 2); Cf = cx / p - Rf / 2
                C, R = _hex_round(Cf, Rf)
                if 0 <= R < CR and 0 <= C < CC and coarse[R, C] > 0:
                    out[a, b, k] = coarse[R, C]
    return np.pad(out, ((1, 1), (1, 1), (0, 0))), h


@dataclass
class Vver440Problem:
    grid: TriGrid
    materials: list
    material_map: np.ndarray
    active: np.ndarray
    mask_bc: object
    kinetics: object = None
    problem_at: object = None  # callable t -> (materials, material_map)


def build_vver440(refine: int = 1, perturbation: str = "none") -> Vver440Problem:
    """Assemble the 2D VVER-440 core on a body-fitted triangular mesh.

    refine       : triangular refinement; each assembly is split into 6 r^2
                   triangles (r=1 already gives the correct hexagonal geometry).
    perturbation : "none" (static k), "ramp_hex" (material-8 thermal ramp), or
                   "bulk" (distributed +0.1$ step on material 1).

    Use with TriDiffusionEigenSolver, or TransientSolver(..., group_operator=
    TriGroupOperator, eig_solver=TriDiffusionEigenSolver).
    """
    coarse = _coarse_hex_map()
    mmap, side = _tri_material_map(coarse, refine)
    active = mmap > 0
    grid = TriGrid(shape=mmap.shape, side=side)
    void = Material(name="void", diffusion=[1.0, 1.0], sigma_a=[0.0, 0.0],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.0], [0.0, 0.0]])
    materials = [void] + [_material(mid) for mid in range(1, 9)]

    problem_at = None
    if perturbation != "none":
        cache: dict[float, list] = {}
        if perturbation == "ramp_hex":
            key_fn, swap = (lambda t: round(_ramp_hex_sigma_a2(t), 9),
                            lambda key: ("m8", _material8_ramp(key)))
        elif perturbation == "bulk":
            key_fn, swap = (lambda t: round(_bulk_factor(t), 9),
                            lambda key: ("m1", _material1_bulk(key)))
        else:
            raise ValueError(f"unknown perturbation {perturbation!r}")

        def problem_at(t: float):
            key = key_fn(t)
            if key not in cache:
                mats = list(materials)
                which, m = swap(key)
                mats[8 if which == "m8" else 1] = m
                cache[key] = mats
            return cache[key], mmap

    return Vver440Problem(grid=grid, materials=materials, material_map=mmap,
                          active=active, mask_bc="vacuum",
                          kinetics=VVER_KINETICS, problem_at=problem_at)


def build_vver440_msh(msh_path):
    """Read the FEMFFUSION VVER-440 Gmsh mesh and assign materials by the .xsec
    reading order (each assembly's 3 quads share a tag = its 1-based index).

    Returns (mesh, materials, cell_material, alpha_boundary) for
    UnstructuredDiffusionSolver. `msh_path` is examples/2D_VVER440/VVER440.msh
    in a FEMFFUSION checkout.
    """
    from ..mesh import read_gmsh
    mesh = read_gmsh(msh_path)
    void = Material(name="void", diffusion=[1.0, 1.0], sigma_a=[0.0, 0.0],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.0], [0.0, 0.0]])
    materials = [void] + [_material(mid) for mid in range(1, 9)]
    # material id per assembly, in reading order (assembly tag k -> _MATSEQ[k-1])
    cell_material = np.array([_MATSEQ[t - 1] for t in mesh.cell_tag])
    return mesh, materials, cell_material, 0.5


# Assembly materials in .xsec reading order (row by row), for the .msh loader.
_MATSEQ = [int(c) for row in _OFFSET for c in row if c != "."]
