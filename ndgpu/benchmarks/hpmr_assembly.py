"""Pin-resolved (heterogeneous) HP-MR fuel assembly on a periodic rhombic cell.

:mod:`ndgpu.benchmarks.hpmr` models the HP-MR at ASSEMBLY level: each fuel hex
is one flat-flux-homogenized material, mixed from its pin constituents by
volume fraction (``_ASSEMBLY_VOLUME_FRACTIONS``). That homogenization discards
the intra-assembly flux shape, and no amount of equivalence factoring recovers
it -- SPH and GET correct assembly-integrated rates, not the shape inside.
This module builds the heterogeneous problem that shape comes from: the 127-pin
lattice resolved explicitly, solved with S_N, giving the intra-assembly form
function used to reconstruct pin power from a homogenized full-core solution.

Unit cell
---------
An infinite lattice of identical assemblies. tri-S_N has no reflective law
(vacuum or periodic only), so the cell is NOT a reflected hexagon -- it is the
60 degree RHOMBUS spanned by the two assembly-lattice vectors, with periodic
wrap. That is the exact infinite-lattice condition rather than an approximation
to it: the tri grid's (nr, nc) indexing already IS the rhombic lattice, and the
solver's periodic BC is a plain index wrap, so the tiling is exact. The rhombus
carries the same area as the hexagonal assembly it replaces,
(sqrt3/2) PITCH^2 = 619.79 cm^2.

Pin lattice
-----------
Read from the VTB source deck, not inferred: ``virtual_test_bed``
``microreactors/mrad/Serpent_Model/serpent_input.i``, lattice card
``lat 20  2  0.0 0.0 19 19 2.3`` -- a 19x19 X-type hexagonal lattice at
**2.3 cm** pin pitch, embedded here as :data:`VTB_LATTICE_MAP`. The deck's
universes map to the same Griffin/YakXs IDs the assembly model already uses,
via its cell cards:

    universe 1 -> heat pipe    fill 815 inside r 0.97, wall 817 to r 1.07   x37
    universe 2 -> moderator    fill 802 inside r 0.825, clad 816 to r 0.92  x27
    universe 3 -> fuel compact fill 801 inside r 1.00 ("% fuel compact")    x63
    universe 9 -> graphite     fill 803                                     x234

Checks against independent quantities, all passing: the 63/27/37 counts and
those radii reproduce every one of the six documented assembly volume
fractions (0.3193 / 0.0931 / 0.0227 / 0.1765 / 0.0383 / 0.3501); the 127 pins
fit inside the 26.752 cm ``hexyprism`` assembly with 0.355 cm clearance; and
the minimum centre spacing is the full 2.3 cm pitch, comfortably above the
2.070 cm a heat pipe (1.07) needs beside a fuel compact (1.00).

That last check is why the pitch matters. An earlier reconstruction here put
the 127 pins on a compact 7-ring lattice at PITCH/13 = 2.0578 cm, which
reproduces the volume fractions equally well -- they depend only on counts and
radii -- but is geometrically impossible: it overlaps heat pipe and fuel by
0.0122 cm, and no pitch can fix it while 13 sites span the assembly. The real
lattice is looser (2.3 cm) and does not span the assembly, leaving a graphite
rim. Volume fractions alone do not determine a lattice; the packing check is
what catches it.

"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from ..backend import asnumpy
from ..power import power_density
from ..tri import TriGrid
from .hpmr import PITCH

# Pin lattice, read from the VTB Serpent deck (lat 20).
PIN_PITCH = 2.3                        # cm, from `lat 20 2 0.0 0.0 19 19 2.3`
LATTICE_NX = LATTICE_NY = 19
ROTATION_DEG = 30.0                    # from `trans 20 0.0 0.0 0.0 0.0 0.0 30.`

# The assembly pin map, transcribed from lattice 20. Rows are listed top-down
# exactly as the deck prints them; each row is offset a further half pitch in
# +x, which is what the deck's indentation depicts.
#   F = fuel compact, M = YH2 moderator, H = heat pipe, . = graphite monolith
VTB_LATTICE_MAP = (
    "...................",
    "...................",
    "...................",
    ".........HFHFHFH...",
    "........FFMFMFMF...",
    ".......HMHFHFHFH...",
    "......FFFFMFMFMF...",
    ".....HMHMHFHFHFH...",
    "....FFFFFFMFMFMF...",
    "...HMHMHMHFHFHFH...",
    "...FFFFFFMFMFMF....",
    "...HMHMHFHFHFH.....",
    "...FFFFMFMFMF......",
    "...HMHFHFHFH.......",
    "...FFMFMFMF........",
    "...HFHFHFH.........",
    "...................",
    "...................",
    "...................",
)
_MAP_KIND = {"F": "fuel", "M": "mod", "H": "hp"}

R_FUEL = 1.0                           # fuel compact radius, cm
R_MOD, R_MOD_SHELL = 0.825, 0.92       # YH2 moderator pin + clad
R_HP, R_HP_SHELL = 0.97, 1.07          # heat pipe + wall

N_FUEL, N_MOD, N_HP = 63, 27, 37

# material_map indices for this model (pin level, NOT hpmr's assembly level).
GRAPHITE, FUEL, MODERATOR, MOD_SHELL, HEATPIPE, HP_SHELL = range(6)
PIN_MATERIAL_NAMES = ("graphite", "fuel_compact", "moderator", "mod_shell",
                      "heatpipe", "hp_shell")

# Griffin/YakXs IDs, same library as the assembly model.
PIN_XS_IDS = (803, 801, 802, 816, 815, 817)   # ordered as PIN_MATERIAL_NAMES

_SQRT3 = math.sqrt(3.0)


def _barycentric_samples(k):
    """k-order barycentric sample points (u, v) inside the unit triangle."""
    pts = []
    for i in range(k):
        for j in range(k - i):
            pts.append(((i + 1.0 / 3.0) / k, (j + 1.0 / 3.0) / k))
    return pts


def vtb_pin_lattice(pitch=PIN_PITCH, lattice_map=VTB_LATTICE_MAP):
    """(centres, kinds) for the 127 pins, from the VTB lattice map.

    X-type hexagonal lattice: within a printed row the sites step by `pitch`
    in x, and each row down steps half a pitch in +x and sqrt(3)/2 pitch in -y.
    Centres are returned relative to the pin cluster's centroid, which is the
    assembly centre.

    The deck's ``trans 20 0.0 0.0 0.0 0.0 0.0 30.`` rotation IS APPLIED. It is
    not cosmetic: as listed the cluster fits a flat-y hexagon, but the assembly
    lattice vectors used here lie at 0 and 60 degrees, which tile a flat-x one.
    Skipping the rotation leaves the cluster 27.600 cm across against a 26.752
    cm lattice pitch, so pins in neighbouring assemblies overlap -- measured at
    0.848 cm between centres where 2.07 cm is needed. Voronoi assignment then
    clips them and the pin volume fractions come out ~2% low, with the error
    GROWING under refinement rather than converging.
    """
    xy, kinds = [], []
    for j, row in enumerate(lattice_map):
        for i, ch in enumerate(row):
            if ch == ".":
                continue
            xy.append(((i + 0.5 * j) * pitch, -j * (_SQRT3 / 2.0) * pitch))
            kinds.append(_MAP_KIND[ch])
    xy = np.asarray(xy, dtype=float)
    xy -= xy.mean(axis=0)
    a = math.radians(ROTATION_DEG)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return xy @ rot.T, kinds


@dataclass
class AssemblyProblem:
    grid: TriGrid
    material_map: np.ndarray        # (nr, nc, 2) index into PIN_MATERIAL_NAMES
    materials: list
    mix_material: np.ndarray        # (nr, nc, 2) volume-mix partner, -1 if pure
    mix_weight: np.ndarray          # (nr, nc, 2) partner area fraction
    pin_index: np.ndarray           # (nr, nc, 2) pin ordinal, -1 for monolith
    pin_centres: np.ndarray         # (n_pins, 2) xy of each pin
    pin_kind: list                  # "fuel" | "mod" | "hp" per pin
    volume_fractions: dict          # achieved, by material name


def build_hpmr_assembly2d(refine=8, materials=None, lattice_map=None,
                          samples=6):
    """Pin-resolved HP-MR assembly on the periodic rhombic unit cell.

    refine : mesh cells per pin pitch. The triangle side is
             PIN_PITCH / refine, so refine = 8 gives ~0.26 cm cells and ~4
             cells across a fuel-compact radius. The achieved volume fractions
             (returned) are the convergence measure -- compare them against
             hpmr._ASSEMBLY_VOLUME_FRACTIONS.

    Returns an :class:`AssemblyProblem`. Solve it with periodic BCs:

        TriSNTransportSolver(p.grid, p.materials, p.material_map,
                             bc="periodic", scheme="scb")

    There is no ``active`` mask -- the rhombus is full, which is what makes the
    periodic wrap an exact lattice.
    """
    n = 13 * refine                             # cells along each rhombus edge
    side = PITCH / n
    grid = TriGrid(shape=(n, n, 2), side=side)

    centres, kinds = vtb_pin_lattice(
        lattice_map=VTB_LATTICE_MAP if lattice_map is None else lattice_map)

    # Cell centroids of the rhombic tri lattice. Rows advance along a1, columns
    # along a2; a down triangle's centroid sits at 1/3 of the rhombus diagonal,
    # an up triangle's at 2/3 -- the same convention hexraster uses.
    a1 = np.array([side, 0.0])
    a2 = np.array([side * 0.5, side * (_SQRT3 / 2.0)])
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    base = ii[..., None] * a2 + jj[..., None] * a1      # rhombus corner
    cen = np.empty((n, n, 2, 2))
    cen[:, :, 0, :] = base + (a1 + a2) / 3.0            # down triangle
    cen[:, :, 1, :] = base + 2.0 * (a1 + a2) / 3.0      # up triangle

    # Lattice vectors of the ASSEMBLY (the rhombus edges).
    A1, A2 = n * a1, n * a2
    # Centre the pin cluster in the cell: the 7-ring hex is built about the
    # origin, so shift it to the rhombus centroid.
    shift = 0.5 * (A1 + A2)
    cpin = centres + shift

    # Sub-sample each triangle and assign every sample to its nearest pin over
    # the 9 periodic images (a pin near an edge serves cells on the far side,
    # which is what makes the lattice continuous). Nearest-pin is a Voronoi
    # assignment, which also resolves the slight pin overlap noted in the module
    # docstring without double-counting area.
    tri_a1, tri_a2 = a1, a2
    bary = _barycentric_samples(samples)
    frac_counts = np.zeros((n * n * 2, len(PIN_MATERIAL_NAMES)))
    pin_votes = np.full((n * n * 2, len(bary)), -1, dtype=np.int64)
    kind_arr = np.array([{"fuel": 0, "mod": 1, "hp": 2}[k] for k in kinds])
    images = np.concatenate([cpin + di * A1 + dj * A2
                             for di in (-1, 0, 1) for dj in (-1, 0, 1)])
    img_pin = np.tile(np.arange(len(cpin)), 9)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(images)
    except ImportError:                       # pragma: no cover
        tree = None

    for s, (u, v) in enumerate(bary):
        # sample point inside each triangle, in the same down/up convention
        pts = np.empty((n, n, 2, 2))
        off = u * tri_a1 + v * tri_a2
        pts[:, :, 0, :] = base + off
        pts[:, :, 1, :] = base + (tri_a1 + tri_a2) - off
        q = pts.reshape(-1, 2)
        if tree is not None:
            d, k = tree.query(q)
        else:
            dd = np.linalg.norm(q[:, None, :] - images[None, :, :], axis=2)
            k = dd.argmin(axis=1); d = dd[np.arange(dd.shape[0]), k]
        pin = img_pin[k]
        kk = kind_arr[pin]
        mat_s = np.full(q.shape[0], GRAPHITE, dtype=np.int64)
        inpin = np.zeros(q.shape[0], bool)
        for is_k, rin, rout, m_in, m_out in (
                (kk == 0, R_FUEL, None, FUEL, None),
                (kk == 1, R_MOD, R_MOD_SHELL, MODERATOR, MOD_SHELL),
                (kk == 2, R_HP, R_HP_SHELL, HEATPIPE, HP_SHELL)):
            sel = is_k & (d <= rin)
            mat_s[sel] = m_in; inpin |= sel
            if rout is not None:
                sel = is_k & (d > rin) & (d <= rout)
                mat_s[sel] = m_out; inpin |= sel
        frac_counts[np.arange(q.shape[0]), mat_s] += 1.0
        pin_votes[inpin, s] = pin[inpin]

    frac_cell = frac_counts / len(bary)
    order = np.argsort(-frac_cell, axis=1)
    base_mat = order[:, 0]
    second = order[:, 1]
    w2 = frac_cell[np.arange(frac_cell.shape[0]), second]
    # Volume mixing: the operator carries one base material per cell plus one
    # mix partner and weight (see hpmr's absorber="polar"). A boundary cell is
    # therefore represented by its two dominant constituents at their true area
    # fractions, not snapped to whichever one owns the centroid -- which is what
    # makes coarse meshes usable at all when a 1 cm pin sits in a 2 cm cell.
    mix_material = np.where(w2 > 0.0, second, -1).astype(np.int64).reshape(n, n, 2)
    mix_weight = np.where(w2 > 0.0, w2, 0.0).reshape(n, n, 2)

    # A cell's pin ownership is the pin most of its in-pin samples voted for.
    pin_of = np.full(n * n * 2, -1, dtype=np.int64)
    for c in range(pin_votes.shape[0]):
        v = pin_votes[c][pin_votes[c] >= 0]
        if v.size:
            pin_of[c] = np.bincount(v).argmax()

    material_map = base_mat.reshape(n, n, 2)
    pin_index = pin_of.reshape(n, n, 2)
    # Achieved fractions from the MIXED representation, which is what the solver
    # actually sees: (1-w) to the base material and w to the partner.
    frac = {}
    tot = frac_cell.shape[0]
    mw_f, mm_f = mix_weight.reshape(-1), mix_material.reshape(-1)
    for i, name in enumerate(PIN_MATERIAL_NAMES):
        v = float(((base_mat == i) * (1.0 - mw_f)).sum()
                  + ((mm_f == i) * mw_f).sum())
        frac[name] = v / tot

    if materials is None:
        materials = pin_materials_builtin()
    return AssemblyProblem(grid=grid, material_map=material_map,
                           materials=materials, mix_material=mix_material,
                           mix_weight=mix_weight, pin_index=pin_index,
                           pin_centres=cpin, pin_kind=kinds,
                           volume_fractions=frac)


def pin_materials_builtin():
    """The six pin materials, VENDORED in the repo -- no download, no XS file.

    Real 11-group ENDF/B-8 transport-corrected constants, extracted once from
    the VTB Griffin library (``fullcore_xml_G11_endfb8_ss_tr.xml``, grid index
    "3 3" ~ 800 K) and stored as ``data/hpmr_pin_xs_g11.npz`` -- 10 KB, versus
    a 1.5 MB library that lives outside the repo. This is what makes the
    heterogeneous assembly a self-contained example case: geometry from
    :data:`VTB_LATTICE_MAP`, materials from here, nothing external.

    Use :func:`assembly_pin_materials` instead to read a library directly (a
    different temperature node, or an updated evaluation).
    """
    from ..materials import Material
    path = os.path.join(os.path.dirname(__file__), "data",
                        "hpmr_pin_xs_g11.npz")
    d = np.load(path, allow_pickle=False)
    out = []
    for name in PIN_MATERIAL_NAMES:
        chi = d[f"{name}.chi"]
        out.append(Material(name=f"hpmr-{name}", diffusion=d[f"{name}.D"],
                            sigma_a=d[f"{name}.sa"], nu_sigma_f=d[f"{name}.nsf"],
                            sigma_s=d[f"{name}.ss"],
                            chi=chi if chi.sum() > 0 else None,
                            kappa_fission=(d[f"{name}.kf"]
                                           if d[f"{name}.kf"].sum() > 0 else None)))
    return out


def assembly_pin_materials(xs_path, grid_index="3 3"):
    """The six pin materials read straight from the Griffin/YakXs library.

    The whole point of the heterogeneous model: these are the region-level
    libraries the assembly model smears together with ``volume_homogenize``.
    """
    from ..griffin_xs import read_library
    lib = read_library(xs_path, PIN_XS_IDS, grid_index)
    return [lib[i] for i in PIN_XS_IDS]


def _placeholder_pin_materials():
    """Two-group stand-ins, same spirit as hpmr's placeholders.

    Magnitudes are plausible for a graphite/YH2 TRISO lattice -- good for
    exercising geometry and form-function machinery, NOT for predicting k.
    Use :func:`assembly_pin_materials` with the real library for physics.
    """
    from ..materials import Material

    def m(name, D, sa, nsf, s12, chi0=0.0):
        return Material(name=name, diffusion=D, sigma_a=sa, nu_sigma_f=nsf,
                        sigma_s=np.array([[0.0, s12], [0.0, 0.0]]),
                        chi=[chi0, 1.0 - chi0] if chi0 else [1.0, 0.0])

    return [
        m("graphite", [1.30, 0.90], [0.0004, 0.0035], [0.0, 0.0], 0.030),
        m("fuel_compact", [1.15, 0.75], [0.0090, 0.0900], [0.0070, 0.150], 0.020, 1.0),
        m("moderator", [0.90, 0.40], [0.0012, 0.0180], [0.0, 0.0], 0.070),
        m("mod_shell", [1.05, 0.55], [0.0035, 0.0250], [0.0, 0.0], 0.025),
        m("heatpipe", [1.60, 1.20], [0.0008, 0.0060], [0.0, 0.0], 0.012),
        m("hp_shell", [1.00, 0.50], [0.0040, 0.0300], [0.0, 0.0], 0.022),
    ]


def pin_powers(problem, flux, materials=None):
    """Pin-wise fission power from a heterogeneous flux -- the form function.

    Returns (power, normalized): the per-pin power sum over groups of
    nu_Sigma_f * phi * V, and the same normalized to mean 1 over FUEL pins,
    which is the intra-assembly form function pin-power reconstruction applies
    to a homogenized full-core solution.
    """
    mats = problem.materials if materials is None else materials
    # POWER is kappa*Sigma_f * phi, not nu*Sigma_f * phi -- see
    # ndgpu.power.fission_energy_xs. Summing over groups before the scatter is
    # identical to scattering group by group, and cheaper.
    dV = problem.grid.cell_volume
    dens = power_density(flux, mats, problem.material_map)
    pidx = problem.pin_index
    n_pins = len(problem.pin_kind)
    power = np.zeros(n_pins)
    contrib = (dens * dV).reshape(-1)
    flat = pidx.reshape(-1)
    np.add.at(power, flat[flat >= 0], contrib[flat >= 0])
    fuel = np.array([k == "fuel" for k in problem.pin_kind])
    scale = power[fuel].mean()
    return power, (power / scale if scale > 0 else power)


def pin_fluxes(problem, flux):
    """Per-pin, per-group average scalar flux, and its form function.

    Returns (phi, form) with shape (n_pins, G) each: the volume-average flux
    over each pin's own cells, and the same normalized so every group's mean
    over the FUEL pins is 1. The group-wise form function is what pin-FLUX
    reconstruction rides on -- pin power needs only the fission-weighted sum,
    but a flux map needs the shape group by group, because thermal and fast
    flux peak in different places (thermal in the moderator pins, fast in the
    compacts).
    """
    flux = asnumpy(flux)
    G = flux.shape[0]
    dV = problem.grid.cell_volume
    pidx = problem.pin_index.reshape(-1)
    n_pins = len(problem.pin_kind)
    sel = pidx >= 0
    vol = np.zeros(n_pins)
    np.add.at(vol, pidx[sel], np.full(int(sel.sum()), dV))
    phi = np.zeros((n_pins, G))
    for g in range(G):
        f = flux[g].reshape(-1)
        acc = np.zeros(n_pins)
        np.add.at(acc, pidx[sel], f[sel] * dV)
        phi[:, g] = acc / np.maximum(vol, 1e-300)
    fuel = np.array([k == "fuel" for k in problem.pin_kind])
    form = phi / np.maximum(phi[fuel].mean(axis=0), 1e-300)[None, :]
    return phi, form
