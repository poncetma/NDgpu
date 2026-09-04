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
from dataclasses import dataclass, replace

import numpy as np

from ..backend import asnumpy
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

# Stable public geometry metadata for examples and downstream diagnostics.
# Keep the private lists for compatibility with the internal raster builders,
# but expose immutable tuples so example code does not depend on internals.
HPMR_FUEL_SITES = tuple(_FUEL_SITES)
HPMR_BE_SITES = tuple(_BE_SITES)
HPMR_DRUM_SITES = tuple(_DRUM_SITES)

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


def hpmr_materials_builtin(fuel, three_d=False):
    """Core material list in MATERIAL_NAMES order, from VENDORED cross sections.

    The structural constants (graphite/central cell, Be reflector, drum body,
    B4C arc) are the same real 11-group ENDF/B-8 data
    :func:`hpmr_endfb8_materials` reads, extracted once into
    ``data/hpmr_core_xs_g11.npz`` -- so a full-core run needs no external
    library file. ``fuel`` is the assembly material, normally the FLUX-weighted
    homogenization of a heterogeneous assembly solve
    (:mod:`ndgpu.benchmarks.hpmr_assembly`) rather than a flat-flux volume mix.
    """
    import os
    path = os.path.join(os.path.dirname(__file__), "data",
                        "hpmr_core_xs_g11.npz")
    d = np.load(path, allow_pickle=False)

    def mat(mid, name):
        n = str(mid)
        chi = d[f"{n}.chi"]
        return Material(name=name, diffusion=d[f"{n}.D"], sigma_a=d[f"{n}.sa"],
                        nu_sigma_f=d[f"{n}.nsf"], sigma_s=d[f"{n}.ss"],
                        chi=chi if chi.sum() > 0 else None,
                        kappa_fission=(d[f"{n}.kf"]
                                       if d[f"{n}.kf"].sum() > 0 else None))

    G = int(d["G"])
    void = Material(name="void", diffusion=[1.0] * G, sigma_a=[0.0] * G,
                    nu_sigma_f=[0.0] * G, sigma_s=np.zeros((G, G)))
    mats = [void, fuel, mat(803, "central"), mat(805, "be_reflector"),
            mat(810, "drum_be"), mat(811, "drum_absorber")]
    if three_d:
        mats.append(mat(805, "axial_reflector"))
    return mats


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


def hpmr_placeholder_materials(three_d: bool = False) -> list:
    """Return the illustrative HP-MR material set used by simple examples.

    The constants exercise geometry and solver behavior; they are not a
    predictive reactor library. Prefer :func:`hpmr_materials_builtin` for the
    vendored 11-group benchmark data.
    """
    return _placeholder_materials(three_d=three_d)


def _drum_geometry(drum_angle_deg):
    """(drum centres, per-drum absorber-arc centre azimuths) at these angles.

    Convention: angle 0 points each drum's B4C arc squarely at the core centre
    (fully inserted), 180 squarely away (fully withdrawn). atan2(y, x) is the
    drum's outward radial azimuth, so the arc centre is that direction rotated by
    180 + angle -- i.e. toward the centre at angle 0, back outward at angle 180.
    Every drum is measured from its own radial line, so a given angle is the same
    physical insertion for all 12 drums.
    """
    drum_xy = [hex_site_xy(R, C, PITCH) for R, C in _DRUM_SITES]
    arc_az = [math.atan2(y, x) + math.radians(180.0 + a)   # 0 = toward centre
              for (x, y), a in zip(drum_xy, drum_angle_deg)]
    return drum_xy, arc_az


def hpmr_raster(refine: int, drum_angle_deg, paint_absorber: bool = True) -> TriRaster:
    """Rasterize the radial core (see :mod:`ndgpu.hexraster`).

    drum_angle_deg : per-drum absorber rotation (12,); 0 = arc facing the core
    centre (inserted), 180 = facing radially outward (withdrawn).
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

    samples : 0 selects exact triangle/annular-sector intersection. A positive
    value retains equal-area sub-cell quadrature with samples^2 centroids
    (10 -> 100) for comparison with historical results.
    """
    mmap = raster.material_map
    ni, nj, _ = mmap.shape
    frac = np.zeros((ni, nj, 2), dtype=float)
    drum_owner = np.full((ni, nj, 2), -1, dtype=np.int16)
    drum_xy, arc_az = _drum_geometry(drum_angle_deg)
    drum_xy = np.asarray(drum_xy)
    arc_az = np.asarray(arc_az)
    arc_half = math.radians(DRUM_ARC_HALF_DEG)
    reach2 = (DRUM_RADIUS + raster.side) ** 2

    exact = int(samples) == 0
    bary = None if exact else _triangle_subcell_barycenters(samples)

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
                drum_owner[a, b, t] = d
                if exact:
                    frac[a, b, t] = _absorber_fraction_triangle(
                        V, drum_xy[d], arc_az[d], arc_half)
                else:
                    pts = bary @ V                          # (S, 2)
                    dx = pts[:, 0] - drum_xy[d, 0]
                    dy = pts[:, 1] - drum_xy[d, 1]
                    rr = np.hypot(dx, dy)
                    dphi = ((np.arctan2(dy, dx) - arc_az[d] + np.pi)
                            % (2 * np.pi) - np.pi)
                    inside = ((rr > DRUM_ABSORBER_INNER) & (rr <= DRUM_RADIUS)
                              & (np.abs(dphi) <= arc_half))
                    frac[a, b, t] = inside.mean()
    cell_area = math.sqrt(3.0) / 4.0 * raster.side**2
    return _conserve_absorber_area(frac, drum_owner, cell_area)


def _triangle_subcell_barycenters(order: int) -> np.ndarray:
    """Barycentres of the ``order**2`` equal-area sub-triangles.

    Sampling equal-area sub-cells avoids the boundary bias of an equally
    weighted barycentric point lattice, which over-represents triangle edges
    and can make a thin rotating annulus' volume fraction oscillate with mesh
    orientation.
    """
    n = int(order)
    if n < 1:
        raise ValueError("samples must be >= 1")
    bary = []
    for i in range(n):
        for j in range(n - i):
            u, v = (i + 1.0 / 3.0) / n, (j + 1.0 / 3.0) / n
            bary.append((1.0 - u - v, u, v))
            if i + j < n - 1:
                u, v = (i + 2.0 / 3.0) / n, (j + 2.0 / 3.0) / n
                bary.append((1.0 - u - v, u, v))
    return np.asarray(bary)


def _clip_polygon_ray(poly, ray, keep_left):
    """Clip a polygon to one side of a line through the origin."""
    if len(poly) == 0:
        return poly

    def signed(point):
        value = ray[0] * point[1] - ray[1] * point[0]
        return value if keep_left else -value

    output = []
    start = poly[-1]
    f_start = signed(start)
    for end in poly:
        f_end = signed(end)
        inside_start, inside_end = f_start >= -1e-14, f_end >= -1e-14
        if inside_start != inside_end:
            t = f_start / (f_start - f_end)
            output.append(start + t * (end - start))
        if inside_end:
            output.append(end)
        start, f_start = end, f_end
    return np.asarray(output)


def _circle_polygon_area(poly, radius):
    """Exact area of a polygon intersected with an origin-centred disk."""
    if len(poly) < 3:
        return 0.0
    total = 0.0
    radius2 = radius * radius
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        delta = b - a
        aa = float(np.dot(delta, delta))
        bb = 2.0 * float(np.dot(a, delta))
        cc = float(np.dot(a, a)) - radius2
        cuts = [0.0, 1.0]
        disc = bb * bb - 4.0 * aa * cc
        if aa > 0.0 and disc > 0.0:
            root = math.sqrt(disc)
            for value in ((-bb - root) / (2.0 * aa),
                          (-bb + root) / (2.0 * aa)):
                if 1e-14 < value < 1.0 - 1e-14:
                    cuts.append(value)
        cuts.sort()
        for left, right in zip(cuts[:-1], cuts[1:]):
            p = a + left * delta
            q = a + right * delta
            middle = 0.5 * (p + q)
            cross = float(p[0] * q[1] - p[1] * q[0])
            if float(np.dot(middle, middle)) <= radius2 * (1.0 + 1e-14):
                total += 0.5 * cross
            else:
                total += 0.5 * radius2 * math.atan2(cross, float(np.dot(p, q)))
    return abs(total)


def _absorber_fraction_triangle(vertices, drum_centre, arc_az, arc_half):
    """Exact curved B4C area fraction inside one triangle."""
    poly = np.asarray(vertices, dtype=float) - np.asarray(drum_centre)
    lo = np.array([math.cos(arc_az - arc_half), math.sin(arc_az - arc_half)])
    hi = np.array([math.cos(arc_az + arc_half), math.sin(arc_az + arc_half)])
    poly = _clip_polygon_ray(poly, lo, keep_left=True)
    poly = _clip_polygon_ray(poly, hi, keep_left=False)
    if len(poly) < 3:
        return 0.0
    annular_area = (_circle_polygon_area(poly, DRUM_RADIUS)
                    - _circle_polygon_area(poly, DRUM_ABSORBER_INNER))
    vertices = np.asarray(vertices)
    edge_a = vertices[1] - vertices[0]
    edge_b = vertices[2] - vertices[0]
    triangle_area = 0.5 * abs(float(
        edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))
    return min(max(annular_area / triangle_area, 0.0), 1.0)


def _conserve_absorber_area(frac, drum_owner, cell_area):
    """Correct quadrature fractions to the exact annular-sector area.

    The correction is applied independently to every drum and only over cells
    where quadrature found absorber. It scales down excess area directly; for
    a deficit it fills available partial-cell capacity, preserving ``0 <= w <=
    1``. This removes angle-dependent absorber inventory without changing the
    mesh or manufacturing absorber in unrelated drum cells.
    """
    frac = np.asarray(frac, dtype=float).copy()
    owner = np.asarray(drum_owner)
    area = np.broadcast_to(np.asarray(cell_area, dtype=float), frac.shape)
    target = math.radians(DRUM_ARC_HALF_DEG) * (
        DRUM_RADIUS**2 - DRUM_ABSORBER_INNER**2)
    for drum in range(len(_DRUM_SITES)):
        mask = (owner == drum) & (frac > 0.0)
        current = float(np.sum(area[mask] * frac[mask]))
        if current <= 0.0:
            continue
        delta = target - current
        if delta <= 0.0:
            frac[mask] *= target / current
            continue
        capacity = float(np.sum(area[mask] * (1.0 - frac[mask])))
        if capacity > 0.0:
            frac[mask] += min(delta / capacity, 1.0) * (1.0 - frac[mask])
    return np.clip(frac, 0.0, 1.0)


def absorber_fraction_mesh(mesh, cell_material, drum_angle_deg, samples: int = 0):
    """B4C volume fraction on an arbitrary triangular HP-MR mesh."""
    cm = np.asarray(cell_material)
    if cm.shape != (mesh.n_cells,):
        raise ValueError("cell_material must have one entry per mesh cell")
    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float),
                             (len(_DRUM_SITES),))
    drum_xy, arc_az = _drum_geometry(angles)
    drum_xy = np.asarray(drum_xy)
    arc_az = np.asarray(arc_az)
    arc_half = math.radians(DRUM_ARC_HALF_DEG)
    exact = int(samples) == 0
    bary = None if exact else _triangle_subcell_barycenters(samples)
    frac = np.zeros(mesh.n_cells)
    drum_owner = np.full(mesh.n_cells, -1, dtype=np.int16)

    for c in np.flatnonzero(cm == DRUM_BE):
        vertices = mesh.coords[list(mesh.cells[c])]
        centre = vertices.mean(axis=0)
        d2 = ((drum_xy - centre) ** 2).sum(axis=1)
        drum = int(d2.argmin())
        drum_owner[c] = drum
        if exact:
            frac[c] = _absorber_fraction_triangle(
                vertices, drum_xy[drum], arc_az[drum], arc_half)
        else:
            pts = bary @ vertices
            dx = pts[:, 0] - drum_xy[drum, 0]
            dy = pts[:, 1] - drum_xy[drum, 1]
            radius = np.hypot(dx, dy)
            dphi = ((np.arctan2(dy, dx) - arc_az[drum] + np.pi)
                    % (2.0 * np.pi) - np.pi)
            inside = ((radius > DRUM_ABSORBER_INNER) & (radius <= DRUM_RADIUS)
                      & (np.abs(dphi) <= arc_half))
            frac[c] = inside.mean()
    return _conserve_absorber_area(frac, drum_owner, mesh.area)


@dataclass
class HpmrMeshProblem:
    """Unstructured 2D HP-MR problem with optional drum-local refinement."""

    mesh: object
    materials: list
    cell_material: np.ndarray
    alpha_boundary: float
    mix_material: np.ndarray = None
    mix_weight: np.ndarray = None
    drum_angle_deg: np.ndarray = None
    global_refine: int = 0
    drum_refine: int = 0


@dataclass
class HpmrExtrudedMeshProblem:
    """Axially extruded locally refined HP-MR diffusion problem."""

    grid: object
    materials: list
    material_map: np.ndarray
    active: np.ndarray
    mask_bc: object
    bc: object
    kinetics: object
    mix_material: np.ndarray = None
    mix_weight: np.ndarray = None
    drum_angle_deg: np.ndarray = None
    global_refine: int = 0
    drum_refine: int = 0


def build_hpmr2d_local(refine: int = 3, drum_angle_deg=0.0,
                       local_refinement: bool = True,
                       drum_refine_levels: int = 1,
                       band_margin: float = 1.5, materials=None,
                       absorber: str = "polar", samples: int = 0):
    """Build a globally coarse, drum-locally-refined 2D HP-MR problem.

    Coarse triangular lattice everywhere; each coarse cell whose centroid lies
    in the annular band the drum B4C arc occupies (radius in
    [DRUM_ABSORBER_INNER - band_margin, DRUM_RADIUS + band_margin] of any drum)
    is split one level into four sub-triangles, resolving the absorber directly
    at effective radial refinement ``2**drum_refine_levels * refine`` at a
    fraction of the cells a globally fine mesh would cost. The full radial band
    is refined at every azimuth, so one mesh serves any drum rotation. Polar
    volume mixing remains active on the refined cells: local refinement resolves
    flux variation while mixing conserves the curved absorber area and keeps
    rotation smooth.

    The coarse-to-fine interface is a conservative 2:1 hanging node handled by
    :func:`ndgpu.mesh.assemble_mesh`. Set ``local_refinement=False`` for an
    unstructured uniform-mesh reference equivalent to the structured lattice.
    """
    from ..mesh import assemble_mesh

    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float),
                             (len(_DRUM_SITES),))
    levels = int(drum_refine_levels) if local_refinement else 0
    if levels < 0:
        raise ValueError("drum_refine_levels must be >= 0")
    mats = _placeholder_materials() if materials is None else list(materials)
    if len(mats) != len(MATERIAL_NAMES):
        raise ValueError(f"expected {len(MATERIAL_NAMES)} materials")
    if absorber not in ("raster", "polar"):
        raise ValueError("absorber must be 'raster' or 'polar'")
    raster = hpmr_raster(refine, angles, paint_absorber=False)   # drums = DRUM_BE
    mm = raster.material_map
    drum_xy, arc_az = _drum_geometry(angles)
    drum_xy = np.asarray(drum_xy)
    arc_half = math.radians(DRUM_ARC_HALF_DEG)
    lo, hi = DRUM_ABSORBER_INNER - band_margin, DRUM_RADIUS + band_margin

    def absorber_material(cc):
        """DRUM_ABSORBER if the point is in a drum's arc, else DRUM_BE."""
        d2 = (drum_xy[:, 0] - cc[0]) ** 2 + (drum_xy[:, 1] - cc[1]) ** 2
        d = int(d2.argmin()); rr = math.sqrt(d2[d])
        if DRUM_ABSORBER_INNER < rr <= DRUM_RADIUS:
            dphi = (math.atan2(cc[1] - drum_xy[d, 1], cc[0] - drum_xy[d, 0])
                    - arc_az[d] + math.pi) % (2 * math.pi) - math.pi
            if abs(dphi) <= arc_half:
                return DRUM_ABSORBER
        return DRUM_BE

    node_at, coords = {}, []
    def node(p):
        k = (round(float(p[0]), 6), round(float(p[1]), 6))
        i = node_at.get(k)
        if i is None:
            i = len(coords); node_at[k] = i; coords.append([float(p[0]), float(p[1])])
        return i

    cells, cmat = [], []

    def append_triangle(vertices, material_at, levels_left):
        if levels_left == 0:
            cells.append(tuple(node(p) for p in vertices))
            cmat.append(material_at(np.mean(vertices, axis=0)))
            return
        mids = [(vertices[i] + vertices[(i + 1) % 3]) / 2.0 for i in range(3)]
        children = ((vertices[0], mids[0], mids[2]),
                    (mids[0], vertices[1], mids[1]),
                    (mids[2], mids[1], vertices[2]),
                    (mids[0], mids[1], mids[2]))
        for child in children:
            append_triangle(np.asarray(child), material_at, levels_left - 1)

    ni, nj, _ = mm.shape
    for a in range(ni):
        for b in range(nj):
            for t in (0, 1):
                if mm[a, b, t] == 0:
                    continue
                V = raster.cell_vertices(a, b, t)
                cc = V.mean(0)
                base = int(mm[a, b, t])
                # material by centroid: absorber arc overlaid on drum-body cells,
                # everything else its lattice material (raster). Refinement only
                # changes the *resolution* at which this is sampled.
                if absorber == "raster" and base == DRUM_BE:
                    mat_of = lambda p: absorber_material(p)
                else:
                    mat_of = lambda p: base
                rmin = math.sqrt(float(((drum_xy - cc) ** 2).sum(1).min()))
                append_triangle(V, mat_of, levels if lo <= rmin <= hi else 0)

    mesh = assemble_mesh(np.array(coords), cells, cmat)
    cmat = np.asarray(cmat)
    mix_material = mix_weight = None
    if absorber == "polar":
        mix_weight = absorber_fraction_mesh(mesh, cmat, angles, samples=samples)
        mix_material = np.where(
            mix_weight > 0.0, DRUM_ABSORBER, -1).astype(np.int64)
    return HpmrMeshProblem(
        mesh=mesh, materials=mats, cell_material=cmat, alpha_boundary=0.5,
        mix_material=mix_material, mix_weight=mix_weight,
        drum_angle_deg=np.array(angles), global_refine=int(refine),
        drum_refine=int(refine) * 2**levels)


def build_hpmr3d_local(refine: int = 8, nz: int = 10,
                       drum_angle_deg=0.0, *, drum_refine_levels: int = 3,
                       band_margin: float = 1.5, materials=None,
                       absorber: str = "polar", samples: int = 0):
    """Build the 3-D HP-MR on a locally refined extruded radial mesh.

    The validated 2-D mesh is retained as a tensor-product radial plane rather
    than converted into a general prism mesh. This preserves every recursive
    midpoint interface exactly and permits axial slab decomposition without a
    3-D graph partitioner. The radial topology is fixed while the polar volume
    fractions rotate.

    ``nz`` must align the 20 cm axial reflector interfaces with layer faces.
    The preferred production geometry is global ``refine=8`` with three local
    drum-band levels; smaller values are intended for verification only.
    """
    from ..extruded_mesh import ExtrudedMeshGrid

    if not isinstance(nz, (int, np.integer)) or nz < 1 or not np.isclose(
            AXIAL_REFLECTOR_HEIGHT / (TOTAL_HEIGHT / nz),
            round(AXIAL_REFLECTOR_HEIGHT / (TOTAL_HEIGHT / nz))):
        raise ValueError(
            "nz must place layer boundaries at the 20 cm axial reflectors")
    mats = _placeholder_materials(three_d=True) if materials is None \
        else list(materials)
    if len(mats) != len(MATERIAL_NAMES_3D):
        raise ValueError(f"expected {len(MATERIAL_NAMES_3D)} materials "
                         f"({', '.join(MATERIAL_NAMES_3D)}), got {len(mats)}")

    radial = build_hpmr2d_local(
        refine=refine, drum_angle_deg=drum_angle_deg,
        local_refinement=True, drum_refine_levels=drum_refine_levels,
        band_margin=band_margin, absorber=absorber, samples=samples)
    grid = ExtrudedMeshGrid(radial.mesh, height=TOTAL_HEIGHT, nz=int(nz))
    material_map = np.repeat(radial.cell_material[:, None], nz, axis=1)

    z_centres = (np.arange(nz) + 0.5) * grid.dz
    reflector = ((z_centres < AXIAL_REFLECTOR_HEIGHT)
                 | (z_centres > TOTAL_HEIGHT - AXIAL_REFLECTOR_HEIGHT))
    fuelish = ((material_map == FUEL) | (material_map == CENTRAL))
    material_map[fuelish & reflector[None, :]] = AXIAL_REFLECTOR

    def extrude(values):
        if values is None:
            return None
        return np.repeat(np.asarray(values)[:, None], nz, axis=1)

    return HpmrExtrudedMeshProblem(
        grid=grid, materials=mats, material_map=material_map,
        active=np.ones(grid.shape, dtype=bool), mask_bc="vacuum",
        bc=("reflective", "reflective", "vacuum"),
        kinetics=HPMR_KINETICS,
        mix_material=extrude(radial.mix_material),
        mix_weight=extrude(radial.mix_weight),
        drum_angle_deg=np.array(radial.drum_angle_deg),
        global_refine=radial.global_refine,
        drum_refine=radial.drum_refine)


def with_hpmr3d_local_drum_angle(problem: HpmrExtrudedMeshProblem,
                                 drum_angle_deg, *, samples: int = 0):
    """Reuse an extruded local mesh at a new polar-mixed drum angle.

    Local drum-band refinement covers every azimuth, so rotating the polar
    absorber changes only its volume fractions. Geometry, axial material map,
    materials, and boundary metadata are shared with ``problem``.
    """
    if not isinstance(problem, HpmrExtrudedMeshProblem):
        raise TypeError("problem must be an HpmrExtrudedMeshProblem")
    if problem.mix_material is None or problem.mix_weight is None:
        raise ValueError("drum-angle reuse requires polar volume mixing")

    grid = problem.grid
    radial_material = np.asarray(problem.material_map)[:, grid.nz // 2]
    angles = np.broadcast_to(np.asarray(drum_angle_deg, dtype=float),
                             (len(_DRUM_SITES),))
    radial_weight = absorber_fraction_mesh(
        grid.mesh, radial_material, angles, samples=samples)
    mix_weight = np.repeat(radial_weight[:, None], grid.nz, axis=1)
    mix_material = np.where(
        mix_weight > 0.0, DRUM_ABSORBER, -1).astype(np.int64)
    return replace(
        problem, mix_material=mix_material, mix_weight=mix_weight,
        drum_angle_deg=np.array(angles))


def hpmr_locally_refined_mesh(refine: int = 3, drum_angle_deg=0.0,
                              refine_drums: bool = True, band_margin: float = 1.5,
                              materials=None):
    """Legacy centroid-painted local mesh tuple.

    New work should use :func:`build_hpmr2d_local`, which combines the same
    conservative 2:1 refinement with polar volume mixing.
    """
    problem = build_hpmr2d_local(
        refine=refine, drum_angle_deg=drum_angle_deg,
        local_refinement=refine_drums, band_margin=band_margin,
        materials=materials, absorber="raster")
    return (problem.mesh, problem.cell_material, problem.materials,
            problem.alpha_boundary)


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
    # The rasterizer's physical frame (side + lattice origin). Needed to place
    # anything given in CORE coordinates onto the mesh -- e.g. pin positions for
    # pin-power reconstruction (ndgpu.pin_power), which must land in the same
    # frame the raster used or the sampled flux is taken from the wrong cells.
    raster: object = None


def build_hpmr2d(refine: int = 4, drum_angle_deg=0.0,
                 materials: list | None = None,
                 absorber: str = "raster", samples: int = 10) -> HpmrProblem:
    """Assemble the 2D HP-MR core on a body-fitted triangular mesh.

    refine         : triangles per hex = 6 refine^2; also sets the drum-arc
                     rasterization fidelity (>= 4 recommended for "raster").
    drum_angle_deg : absorber-arc rotation, scalar or one value per drum (12).
                     0 = arc toward the core centre (inserted), 180 = outward
                     (withdrawn).
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
                       mix_material=mix_material, mix_weight=mix_weight,
                       raster=raster)


def hpmr_transport_mask(problem: HpmrProblem, region: str = "drum") -> np.ndarray:
    """Cells that keep the full transport (SP3/SDPN) block in a hybrid solve;
    every other cell runs pure diffusion. Pass the result as ``hybrid_mask`` to
    a Tri SP3/SDPN eigen-solver.

    region
        "drum" (default): the whole rotating drum body -- the beryllium plus
            its B4C arc. Running transport over the entire drum gives a Be
            buffer around the strong absorber, so the reflective hybrid
            interface at the drum/reflector boundary sees an already-decayed
            second moment; this is the recommended, closure-insensitive choice.
        "absorber": only the B4C arc cells (raster: material == DRUM_ABSORBER;
            polar: the volume-mixed cells, mix_weight > 0). The tightest
            transport region -- fewest transport cells, but the reflective
            interface then sits right at the absorber edge.
    """
    mmap = np.asarray(problem.material_map)
    if region == "drum":
        mask = (mmap == DRUM_BE) | (mmap == DRUM_ABSORBER)
    elif region == "absorber":
        if problem.mix_weight is not None:         # polar: arc is volume-mixed
            mask = np.asarray(problem.mix_weight) > 0.0
        else:                                      # raster: arc is its own id
            mask = mmap == DRUM_ABSORBER
    else:
        raise ValueError(f"region must be 'drum' or 'absorber', got {region!r}")
    return mask & (mmap > 0)                        # never mark excised void cells


def build_hpmr3d(refine: int = 4, nz: int = 20, drum_angle_deg=0.0,
                 materials: list | None = None,
                 absorber: str = "raster", samples: int = 10) -> HpmrProblem:
    """Assemble the 3D HP-MR: the 2D radial core extruded to 200 cm.

    The fueled length is 160 cm with 20 cm beryllium axial reflectors above
    and below (fuel and central cells become AXIAL_REFLECTOR there); the
    radial Be ring and the drums -- including their B4C arcs -- run the full
    height, as in the VTB Serpent model. Vacuum on both z faces.

    refine, drum_angle_deg, absorber, samples : as in :func:`build_hpmr2d`.
    Because the drums (arc included) run the full height, the ``"polar"``
    absorber volume-mixing is the same in every axial layer, so the 2D
    ``mix_material``/``mix_weight`` are simply extruded over z.
    nz        : number of uniform axial layers over the 200 cm height; must be
                a multiple of 10 so layer boundaries fall on the 20/180 cm
                core-reflector interfaces (nz=20 -> dz=10 cm).
    materials : optional replacement list ordered as MATERIAL_NAMES_3D.

    Use with TriDiffusionEigenSolver, passing ``bc=p.bc`` for the vacuum z
    faces (and the mix arrays for absorber="polar")::

        p = build_hpmr3d(refine=4, nz=20, absorber="polar")
        res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                      active=p.active, mask_bc=p.mask_bc,
                                      bc=p.bc, mix_material=p.mix_material,
                                      mix_weight=p.mix_weight).solve()
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
    p2d = build_hpmr2d(refine=refine, drum_angle_deg=angles,
                       absorber=absorber, samples=samples)

    def extrude(a):
        return None if a is None else np.repeat(a[..., None], nz, axis=3)

    mmap = extrude(p2d.material_map)
    mix_material = extrude(p2d.mix_material)
    mix_weight = extrude(p2d.mix_weight)

    dz = TOTAL_HEIGHT / nz
    zc = (np.arange(nz) + 0.5) * dz
    refl = (zc < AXIAL_REFLECTOR_HEIGHT) | (zc > TOTAL_HEIGHT - AXIAL_REFLECTOR_HEIGHT)
    fuelish = (mmap == FUEL) | (mmap == CENTRAL)
    mmap[fuelish & refl[None, None, None, :]] = AXIAL_REFLECTOR
    grid = TriGrid(shape=mmap.shape, side=p2d.grid.side, height=TOTAL_HEIGHT)
    return HpmrProblem(grid=grid, materials=mats, material_map=mmap,
                       active=mmap > 0, mask_bc="vacuum",
                       bc=("reflective", "reflective", "vacuum"),
                       kinetics=HPMR_KINETICS, drum_angle_deg=np.array(angles),
                       mix_material=mix_material, mix_weight=mix_weight,
                       # The extrusion keeps the 2D raster's physical frame, so
                       # anything given in core coordinates (pin positions,
                       # coupling-mesh vertices) can be placed in 3D too.
                       raster=p2d.raster)


def hpmr_sn_homogenization(refine: int = 6, n_polar: int = 2, n_azi: int = 12,
                           scheme: str = "scb", device: str = "auto",
                           tol_k: float = 1e-7, tol_source: float = 1e-6,
                           materials=None):
    """Transport-weighted homogenization of the HP-MR assembly: one tri-S_N solve
    of the pin-resolved geometry, collapsed two ways.

    The assembly constants for :func:`build_hpmr2d` are normally taken from a
    *diffusion* solve of the pin lattice. The pin cell is precisely where
    diffusion is least trustworthy, and it shows: S_N gives k_inf = 1.177297
    against diffusion's 1.186813 at refine 6, a ~950 pcm difference that the flux
    shape carries into the homogenized cross sections too. This routes the
    homogenization through a transport reference instead, for ~91 s of CPU.

      ``assembly``       1 material, flux-weighted over the whole assembly. Feed
                         to :func:`hpmr_materials_builtin` for a core built on
                         transport-weighted constants.
      ``pin_materials``  the same flux collapsed per *pin type* instead -- 4
                         materials (graphite, fuel_compact, moderator_pin,
                         heatpipe_pin), each clad sharing its pin's region so the
                         homogenization performs the clad smearing. Diagnostic
                         here, and the natural starting point for a per-pin SPH
                         correction.

    Note this is plain flux weighting, NOT SPH: reaction rates are exact only at
    the reference flux, and a coarse solve has a different one. See
    :func:`~ndgpu.sph.sph_correct` for the correction that closes that gap
    (``method="jfnk"`` -- Picard runs away).

    Returns a dict with keys: k_inf, pin_materials, assembly, region_flux,
    region_volume, problem, flux.
    """
    from ..sph import flux_weighted_homogenize
    from ..tri_sn import TriSNTransportSolver
    from .hpmr_assembly import build_hpmr_assembly2d

    asm = build_hpmr_assembly2d(refine=refine, materials=materials)
    r = TriSNTransportSolver(
        asm.grid, asm.materials, asm.material_map, bc="periodic", scheme=scheme,
        device=device, n_polar=n_polar, n_azi=n_azi,
        mix_material=asm.mix_material, mix_weight=asm.mix_weight,
    ).solve(tol_k=tol_k, tol_source=tol_source)
    if not r.converged:
        raise RuntimeError(f"S_N assembly reference did not converge: {r}")
    flux = asnumpy(r.flux)

    # Pin types, not materials: each clad shares its pin's region, which is what
    # turns the homogenization into the smearing we want.
    #   0 graphite | 1 fuel | 2 moderator+clad | 3 heat pipe+wall
    region_of = np.array([0, 1, 2, 2, 3, 3])
    mm = np.asarray(asm.mix_material)
    kw = dict(cell_volume=asm.grid.cell_volume, mix_material=asm.mix_material,
              mix_weight=asm.mix_weight)
    pins, rflux, rvol = flux_weighted_homogenize(
        flux, asm.materials, asm.material_map, region_of[asm.material_map],
        mix_region_map=region_of[np.clip(mm, 0, None)], **kw)
    whole, _, _ = flux_weighted_homogenize(
        flux, asm.materials, asm.material_map,
        np.zeros(asm.material_map.shape, dtype=np.int64),
        mix_region_map=np.zeros(mm.shape, dtype=np.int64), **kw)

    return dict(k_inf=float(r.k_eff), pin_materials=pins, assembly=whole[0],
                region_flux=rflux, region_volume=rvol, problem=asm, flux=flux)
