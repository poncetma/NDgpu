"""HP-MR heat-pipe microreactor core on a body-fitted triangular mesh (2D
radial slice, or full 3D as extruded triangular prisms).

Assembly-level radial model of the ANL/INL HP-MR reference microreactor (the
NEAMS Virtual Test Bed model; Stauff et al., "High Fidelity Multiphysics
Modeling of a Heat-Pipe Microreactor using BlueCrab", NSE 2024): 2 MWt, 30
hexagonal TRISO fuel assemblies around a central shutdown-rod cell, ringed by
12 beryllium reflector hexes alternating with 12 rotating control drums, all
on a 26.752 cm flat-to-flat lattice. Geometry decoded from the VTB Serpent
model (microreactors/mrad/Serpent_Model/serpent_input.i): fuel fills hex
rings 1-2 plus the twelve ring-3 edge cells; the six ring-3 corner sites
(80.256 cm from centre) and six ring-4 mid-edge sites (92.67 cm) hold the
drums; twelve ring-4 sites hold the Be hexes. Each drum is a beryllium
cylinder of radius 13.25 cm carrying a 1 cm thick B4C absorber layer
(12.25-13.25 cm) spanning a 90 degree arc; rotating the arc toward the core
inserts negative reactivity. Vacuum boundary outside (the surrounding air is
treated as void).

The core is rasterized onto the structured triangular lattice exactly like
the VVER-440 benchmark (6 r^2 triangles per hex at refinement r); the drum
circle and its absorber arc are painted per-triangle inside the drum's hex
cell, so the arc resolution follows the mesh refinement (r >= 4 recommended,
r >= 6 for drum-worth studies).

Cross sections are two-group PLACEHOLDERS with plausible magnitudes for a
graphite/YH2-moderated TRISO core with Be reflector and B4C arcs -- good for
exercising geometry, symmetry and drum-worth behaviour, not for predicting
k_eff. Swap in SPH-corrected constants (e.g. read with ndgpu.femffusion) via
the ``materials`` argument, ordered as ``MATERIAL_NAMES``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..hexraster import TriRaster, hex_site_xy, rasterize_hex_sites
from ..materials import Kinetics, Material
from ..tri import TriGrid

PITCH = 26.752            # cm, assembly flat-to-flat / lattice spacing
DRUM_RADIUS = 13.25       # cm, drum outer radius
DRUM_ABSORBER_INNER = 12.25   # cm, B4C layer inner radius
DRUM_ARC_HALF_DEG = 45.0  # B4C arc spans +-45 deg about its facing direction
CORE_HEIGHT = 160.0       # cm, fueled length
AXIAL_REFLECTOR_HEIGHT = 20.0  # cm, Be slab above and below the fuel
TOTAL_HEIGHT = CORE_HEIGHT + 2 * AXIAL_REFLECTOR_HEIGHT

# Axial (R, C) hex sites, decoded from the VTB Serpent 13x13 core lattice.
# x = PITCH * (C + R/2), y = PITCH * sqrt(3)/2 * R.
_FUEL_SITES = [
    (-3, 1), (-3, 2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
    (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3),
    (0, -2), (0, -1), (0, 1), (0, 2),
    (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
    (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (3, -2), (3, -1),
]
_BE_SITES = [
    (-4, 1), (-4, 3), (-3, -1), (-3, 4), (-1, -3), (-1, 4),
    (1, -4), (1, 3), (3, -4), (3, 1), (4, -3), (4, -1),
]
# Drum sites: 6 ring-3 corners (outward azimuth 0, 60, ... deg), then 6
# ring-4 mid-edges (30, 90, ... deg). Order fixes the per-drum angle indexing.
_DRUM_SITES = [
    (0, 3), (3, 0), (3, -3), (0, -3), (-3, 0), (-3, 3),
    (2, 2), (4, -2), (2, -4), (-2, -2), (-4, 2), (-2, 4),
]

# material_map indices; keep in sync with _placeholder_materials().
VOID, FUEL, CENTRAL, BE_REFLECTOR, DRUM_BE, DRUM_ABSORBER, AXIAL_REFLECTOR = range(7)
MATERIAL_NAMES = ("void", "fuel", "central", "be_reflector", "drum_be",
                  "drum_absorber")
MATERIAL_NAMES_3D = MATERIAL_NAMES + ("axial_reflector",)

# Graphite-moderated thermal system, one delayed family (placeholder values).
HPMR_KINETICS = Kinetics(velocities=[1.1e7, 3.0e5], beta=[0.0065], decay=[0.078])

# Griffin/YakXs library material IDs for the HP-MR (VTB mrad Serpent model).
_XS_FUEL_COMPACT, _XS_MODERATOR, _XS_GRAPHITE = 801, 802, 803
_XS_BERYLLIUM, _XS_DRUM_BE, _XS_DRUM_B4C = 805, 810, 811
_XS_HEATPIPE, _XS_MOD_SHELL, _XS_HP_SHELL, _XS_CENTRAL = 815, 816, 817, 820

# Fuel-assembly volume fractions, from the Serpent 19x19 pin lattice (63 fuel,
# 27 moderator, 37 heat-pipe pins; pin radii 1.0 / 0.825(+0.92 shell) /
# 0.97(+1.07 shell) cm) over the 26.752 cm assembly hex; graphite monolith is
# the balance. Used to flat-flux-homogenize the pin materials into one
# assembly material (the pre-SPH homogenization).
_ASSEMBLY_VOLUME_FRACTIONS = {
    _XS_FUEL_COMPACT: 0.3193, _XS_MODERATOR: 0.0931, _XS_MOD_SHELL: 0.0227,
    _XS_HEATPIPE: 0.1765, _XS_HP_SHELL: 0.0383, _XS_GRAPHITE: 0.3501,
}


def hpmr_endfb8_materials(xs_path, grid_index="3 3", three_d=True,
                          sph_fuel=None) -> list:
    """Real multigroup HP-MR materials from a Griffin/YakXs library.

    Reads the VTB HP-MR cross-section library (e.g.
    ``fullcore_xml_G11_endfb8_ss_tr.xml``, 11-group ENDF/B-8, transport
    corrected) and returns the material list in ``MATERIAL_NAMES_3D`` order,
    ready for ``build_hpmr3d(materials=...)`` / ``build_hpmr2d``.

    The fuel assembly is flat-flux-homogenized from its pin constituents (fuel
    compact, YH2 moderator + shell, heat pipe + shell, graphite monolith) via
    :data:`_ASSEMBLY_VOLUME_FRACTIONS`; reflector, drum body, B4C arc and
    central cell are taken from the library directly. This is the *pre-SPH*
    homogenization -- pass ``sph_fuel`` (per-group multipliers from a transport
    reference) to apply a superhomogenization correction to the fuel material.

    grid_index : temperature node of the library to read ("3 3" ~ 800 K, the
                 node nearest the 873 K operating point).
    """
    from ..griffin_xs import read_library, volume_homogenize

    ids = (_XS_FUEL_COMPACT, _XS_MODERATOR, _XS_GRAPHITE, _XS_BERYLLIUM,
           _XS_DRUM_BE, _XS_DRUM_B4C, _XS_HEATPIPE, _XS_MOD_SHELL,
           _XS_HP_SHELL, _XS_CENTRAL)
    lib = read_library(xs_path, ids, grid_index)
    G = lib[_XS_FUEL_COMPACT].n_groups

    void = Material(name="void", diffusion=[1.0] * G, sigma_a=[0.0] * G,
                    nu_sigma_f=[0.0] * G, sigma_s=np.zeros((G, G)))
    fuel = volume_homogenize(lib, _ASSEMBLY_VOLUME_FRACTIONS,
                             chi_from=_XS_FUEL_COMPACT, name="hpmr-fuel-asm",
                             sph_factors=sph_fuel)
    mats = [void, fuel,
            lib[_XS_GRAPHITE],      # central: graphite block, rod withdrawn
            lib[_XS_BERYLLIUM],     # radial Be reflector
            lib[_XS_DRUM_BE],       # drum body (Be)
            lib[_XS_DRUM_B4C],      # B4C absorber arc
            ]
    if three_d:
        mats.append(lib[_XS_BERYLLIUM])   # axial Be reflector
    else:
        mats = mats  # 2D uses the 6-material MATERIAL_NAMES order
    return mats


def _placeholder_materials(three_d: bool = False) -> list:
    """Two-group placeholder set (see module docstring); index = map id."""
    def mat(name, D, sa, nsf=(0.0, 0.0), s12=0.0, chi=None):
        return Material(name=f"hpmr-{name} (placeholder)", diffusion=D,
                        sigma_a=sa, nu_sigma_f=nsf,
                        sigma_s=[[0.0, s12], [0.0, 0.0]],
                        chi=chi if chi is not None else [1.0, 0.0])

    mats = [
        mat("void", [1.0, 1.0], [0.0, 0.0]),
        # TRISO fuel + graphite monolith + YH2 pins + heat pipes, homogenized;
        # nu_sigma_f trimmed so k_eff(drums out) ~ 1.03, the Griffin reference
        mat("fuel", [1.4, 0.9], [0.010, 0.100], nsf=[0.005, 0.1265], s12=0.025),
        # central shutdown-rod cell, rod withdrawn: graphite + SS tube + air
        mat("central", [1.1, 0.7], [0.0004, 0.004], s12=0.030),
        # beryllium reflector hex / drum body
        mat("be_reflector", [0.60, 0.45], [0.0006, 0.0012], s12=0.055),
        mat("drum_be", [0.60, 0.45], [0.0006, 0.0012], s12=0.055),
        # B4C absorber arc (strong thermal absorber; diffusion-level values)
        mat("drum_absorber", [0.60, 0.30], [0.10, 2.5], s12=0.008),
    ]
    if three_d:
        # axial Be slab; a separate entry so measured data can differ from
        # the radial ring
        mats.append(mat("axial_reflector", [0.60, 0.45], [0.0006, 0.0012],
                        s12=0.055))
    return mats


def _drum_geometry(drum_angle_deg):
    """(drum centres, per-drum absorber-arc centre azimuths) at these angles."""
    drum_xy = [hex_site_xy(R, C, PITCH) for R, C in _DRUM_SITES]
    arc_az = [math.atan2(y, x) + math.radians(a)          # outward + rotation
              for (x, y), a in zip(drum_xy, drum_angle_deg)]
    return drum_xy, arc_az


def hpmr_raster(refine: int, drum_angle_deg, paint_absorber: bool = True) -> TriRaster:
    """Rasterize the radial core (see :mod:`ndgpu.hexraster`).

    drum_angle_deg : per-drum absorber rotation (12,); 0 = arc facing radially
    outward (withdrawn), 180 = facing the core centre (inserted).
    paint_absorber : if True (default) the B4C arc is stamped by centroid
    (staircase); if False the drum cells stay drum-body Be, for the polar
    volume-mixing path (see :func:`absorber_fraction_map`).
    """
    site_mat = {(0, 0): CENTRAL}
    site_mat.update({s: FUEL for s in _FUEL_SITES})
    site_mat.update({s: BE_REFLECTOR for s in _BE_SITES})
    site_mat.update({s: DRUM_BE for s in _DRUM_SITES})
    if not paint_absorber:
        return rasterize_hex_sites(site_mat, PITCH, refine)

    drum_index = {s: d for d, s in enumerate(_DRUM_SITES)}
    drum_xy, arc_az = _drum_geometry(drum_angle_deg)
    arc_half = math.radians(DRUM_ARC_HALF_DEG)

    def paint(cx, cy, site, mid):
        """B4C where the centroid falls inside a drum's absorber arc."""
        if mid != DRUM_BE:
            return mid
        d = drum_index[site]
        dx, dy = cx - drum_xy[d][0], cy - drum_xy[d][1]
        rr = math.hypot(dx, dy)
        # rr > DRUM_RADIUS: hex-corner slivers, kept as drum body.
        if DRUM_ABSORBER_INNER < rr <= DRUM_RADIUS:
            dphi = (math.atan2(dy, dx) - arc_az[d] + math.pi) \
                   % (2 * math.pi) - math.pi
            if abs(dphi) <= arc_half:
                return DRUM_ABSORBER
        return mid

    return rasterize_hex_sites(site_mat, PITCH, refine, paint=paint)


def absorber_fraction_map(raster: TriRaster, drum_angle_deg, samples: int = 10):
    """Per-cell B4C area fraction of each drum's absorber arc.

    The absorber is a polar region -- an annular sector r in
    [DRUM_ABSORBER_INNER, DRUM_RADIUS], azimuth within +-DRUM_ARC_HALF_DEG of
    the (rotated) arc centre. For every triangle within reach of a drum, the
    fraction of the cell area covered by that region is estimated by
    barycentric sub-sampling. Unlike the centroid raster this is *exact in the
    limit* and, crucially, non-zero for cells the thin (1 cm) annulus only
    partly crosses -- so the arc is represented (diluted) even when it is
    thinner than a triangle, and the fraction varies smoothly as the drum
    rotates. Feed the result as ``mix_weight`` with ``mix_material`` =
    DRUM_ABSORBER to volume-mix the absorber into the drum-body cells.

    samples : barycentric sub-sampling order; (samples+1)(samples+2)/2 points
    per cell (10 -> 66). Higher = finer area/rotation resolution.
    """
    mmap = raster.material_map
    ni, nj, _ = mmap.shape
    frac = np.zeros((ni, nj, 2), dtype=float)
    drum_xy, arc_az = _drum_geometry(drum_angle_deg)
    drum_xy = np.asarray(drum_xy)
    arc_az = np.asarray(arc_az)
    arc_half = math.radians(DRUM_ARC_HALF_DEG)
    reach2 = (DRUM_RADIUS + raster.side) ** 2

    n = samples
    bary = np.array([(i / n, j / n, (n - i - j) / n)
                     for i in range(n + 1) for j in range(n + 1 - i)])  # (S, 3)

    for a in range(ni):
        for b in range(nj):
            for t in (0, 1):
                if mmap[a, b, t] == 0:
                    continue
                V = raster.cell_vertices(a, b, t)          # (3, 2)
                cx, cy = V.mean(0)
                d2 = (drum_xy[:, 0] - cx) ** 2 + (drum_xy[:, 1] - cy) ** 2
                d = int(d2.argmin())
                if d2[d] > reach2:
                    continue
                pts = bary @ V                              # (S, 2)
                dx = pts[:, 0] - drum_xy[d, 0]
                dy = pts[:, 1] - drum_xy[d, 1]
                rr = np.hypot(dx, dy)
                dphi = (np.arctan2(dy, dx) - arc_az[d] + np.pi) % (2 * np.pi) - np.pi
                inside = ((rr > DRUM_ABSORBER_INNER) & (rr <= DRUM_RADIUS)
                          & (np.abs(dphi) <= arc_half))
                frac[a, b, t] = inside.mean()
    return frac


@dataclass
class HpmrProblem:
    grid: TriGrid
    materials: list
    material_map: np.ndarray
    active: np.ndarray
    mask_bc: object
    bc: object = "reflective"   # z faces only (2D grids ignore it)
    kinetics: object = None
    drum_angle_deg: np.ndarray = None
    mix_material: np.ndarray = None   # polar absorber volume-mixing (optional)
    mix_weight: np.ndarray = None


def build_hpmr2d(refine: int = 4, drum_angle_deg=0.0,
                 materials: list | None = None,
                 absorber: str = "raster", samples: int = 10) -> HpmrProblem:
    """Assemble the 2D HP-MR core on a body-fitted triangular mesh.

    refine         : triangles per hex = 6 refine^2; also sets the drum-arc
                     rasterization fidelity (>= 4 recommended for "raster").
    drum_angle_deg : absorber-arc rotation, scalar or one value per drum (12).
                     0 = arc outward (withdrawn), 180 = arc toward the core.
    materials      : optional replacement list ordered as MATERIAL_NAMES
                     (e.g. SPH-corrected sets read via ndgpu.femffusion).
    absorber       : "raster" (centroid staircase, one material per cell) or
                     "polar" (exact polar area fraction volume-mixed into the
                     drum cells; sets ``mix_material``/``mix_weight`` on the
                     problem -- pass them to the solver). The polar path
                     represents the arc's area and rotation smoothly and works
                     below the raster's refine>=4 floor.
    samples        : sub-sampling order for the polar area fractions.

    Use with TriDiffusionEigenSolver, e.g.::

        p = build_hpmr2d(refine=6, drum_angle_deg=120.0, absorber="polar")
        res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                      active=p.active, mask_bc=p.mask_bc,
                                      mix_material=p.mix_material,
                                      mix_weight=p.mix_weight).solve()
    """
    if absorber == "raster" and refine < 4:
        raise ValueError("refine >= 4 required for absorber='raster': below "
                         "that the 1 cm B4C annulus is thinner than a triangle "
                         "and no absorber is rasterized (use absorber='polar', "
                         "which volume-mixes the arc at any refinement)")
    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float),
                             (len(_DRUM_SITES),))
    mats = _placeholder_materials() if materials is None else list(materials)
    if len(mats) != len(MATERIAL_NAMES):
        raise ValueError(f"expected {len(MATERIAL_NAMES)} materials "
                         f"({', '.join(MATERIAL_NAMES)}), got {len(mats)}")

    mix_material = mix_weight = None
    if absorber == "raster":
        raster = hpmr_raster(refine, angles)
    elif absorber == "polar":
        raster = hpmr_raster(refine, angles, paint_absorber=False)
        frac = absorber_fraction_map(raster, angles, samples=samples)
        mix_weight = frac
        mix_material = np.where(frac > 0.0, DRUM_ABSORBER, -1).astype(np.int64)
    else:
        raise ValueError(f"absorber must be 'raster' or 'polar', got {absorber!r}")

    mmap, side = raster.material_map, raster.side
    return HpmrProblem(grid=TriGrid(shape=mmap.shape, side=side),
                       materials=mats, material_map=mmap, active=mmap > 0,
                       mask_bc="vacuum", kinetics=HPMR_KINETICS,
                       drum_angle_deg=np.array(angles),
                       mix_material=mix_material, mix_weight=mix_weight)


def build_hpmr3d(refine: int = 4, nz: int = 20, drum_angle_deg=0.0,
                 materials: list | None = None) -> HpmrProblem:
    """Assemble the 3D HP-MR: the 2D radial core extruded to 200 cm.

    The fueled length is 160 cm with 20 cm beryllium axial reflectors above
    and below (fuel and central cells become AXIAL_REFLECTOR there); the
    radial Be ring and the drums -- including their B4C arcs -- run the full
    height, as in the VTB Serpent model. Vacuum on both z faces.

    refine, drum_angle_deg : as in :func:`build_hpmr2d`.
    nz        : number of uniform axial layers over the 200 cm height; must be
                a multiple of 10 so layer boundaries fall on the 20/180 cm
                core-reflector interfaces (nz=20 -> dz=10 cm).
    materials : optional replacement list ordered as MATERIAL_NAMES_3D.

    Use with TriDiffusionEigenSolver, passing ``bc=p.bc`` for the vacuum z
    faces::

        p = build_hpmr3d(refine=4, nz=20)
        res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                      active=p.active, mask_bc=p.mask_bc,
                                      bc=p.bc).solve()
    """
    if nz % 10 != 0:
        raise ValueError("nz must be a multiple of 10 so layer boundaries "
                         "align with the 20 cm axial reflectors")
    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float),
                             (len(_DRUM_SITES),))
    mats = _placeholder_materials(three_d=True) if materials is None \
        else list(materials)
    if len(mats) != len(MATERIAL_NAMES_3D):
        raise ValueError(f"expected {len(MATERIAL_NAMES_3D)} materials "
                         f"({', '.join(MATERIAL_NAMES_3D)}), got {len(mats)}")
    p2d = build_hpmr2d(refine=refine, drum_angle_deg=angles)
    mmap = np.repeat(p2d.material_map[..., None], nz, axis=3)
    dz = TOTAL_HEIGHT / nz
    zc = (np.arange(nz) + 0.5) * dz
    refl = (zc < AXIAL_REFLECTOR_HEIGHT) | (zc > TOTAL_HEIGHT - AXIAL_REFLECTOR_HEIGHT)
    fuelish = (mmap == FUEL) | (mmap == CENTRAL)
    mmap[fuelish & refl[None, None, None, :]] = AXIAL_REFLECTOR
    grid = TriGrid(shape=mmap.shape, side=p2d.grid.side, height=TOTAL_HEIGHT)
    return HpmrProblem(grid=grid, materials=mats, material_map=mmap,
                       active=mmap > 0, mask_bc="vacuum",
                       bc=("reflective", "reflective", "vacuum"),
                       kinetics=HPMR_KINETICS, drum_angle_deg=np.array(angles))
