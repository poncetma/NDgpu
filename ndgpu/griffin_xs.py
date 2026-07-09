"""Reader for Griffin / YakXs (ISOXML) multigroup cross-section libraries.

Griffin (the NEAMS transport code) and its MGXS tooling store macroscopic
multigroup cross sections in the ``<YakXs>`` XML format -- the format the
INL/ANL Virtual Test Bed ships its reactor libraries in (e.g. the HP-MR
microreactor's ``fullcore_xml_G11_endfb8_ss_tr.xml``). This module reads such
a library into NDgpu ``Material`` objects so real, evaluated-nuclear-data
cross sections (here ENDF/B-8, transport-corrected) can drive the NDgpu
solvers in place of hand-made constants.

Format, as decoded from the VTB HP-MR library and verified by the balance
``Total = Absorption + sum_g' Scattering(g->g')`` closing to ~1e-5:

- one ``Multigroup_Cross_Section_Library`` per material, keyed by integer ID;
- each is tabulated on a temperature grid (``Tfuel``/``Tmod``), one ``Table``
  per grid node, selected here by ``grid_index`` (e.g. "3 3" ~ 800 K);
- inside a table, the ``pseudo`` isotope carries ``Total`` (transport XS, so
  D = 1/(3 Total)), ``Absorption`` (Sigma_a), ``nuFission``,
  ``FissionSpectrum`` (chi) and a Legendre-expanded ``Scattering`` matrix
  whose P0 moment (the first NGroup rows) is the diffusion scattering matrix;
- the ``Scattering`` ``Profile`` gives, per *sink* group, the ``first last``
  range of *source* groups scattering into it; the ``Value`` rows list those
  entries, so entry k of sink row g is Sigma_s(source=first+k -> g).

These libraries are region/pin level (fuel compact, moderator pin, heat pipe,
reflector, ...), not assembly-homogenized. :func:`volume_homogenize` mixes a
set of them by volume fraction -- the flat-flux homogenization that is the
*input* to a superhomogenization (SPH) correction. Rigorous SPH factors need a
region-wise transport flux reference and are applied on top (see the
``sph_factors`` argument), typically from a companion transport/FEMFFUSION run.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .materials import Material


def _isotope(path, mid, grid_index):
    root = ET.parse(path).getroot()
    libs = root.find("Multigroup_Cross_Section_Libraries")
    G = int(libs.get("NGroup"))
    lib = libs.find(f"Multigroup_Cross_Section_Library[@ID='{mid}']")
    if lib is None:
        raise KeyError(f"material {mid} not in {path}")
    table = lib.find(f"Table[@gridIndex='{grid_index}']")
    if table is None:
        avail = [t.get("gridIndex") for t in lib.findall("Table")]
        raise KeyError(f"grid index {grid_index!r} not in material {mid}; "
                       f"available: {avail}")
    return table.find("Isotope"), G


def read_material(path, mid, grid_index="3 3", name=None) -> Material:
    """Read one library material (by integer ID) into a ``Material``.

    grid_index : the temperature-grid node "i j" to read (1-based Tfuel/Tmod
                 indices as a string, e.g. "3 3" for the 3rd node of each).
    """
    iso, G = _isotope(path, mid, grid_index)

    def vec(tag):
        el = iso.find(tag)
        return np.array([float(x) for x in el.text.split()]) if el is not None \
            else np.zeros(G)

    total = vec("Total")
    sigma_a = vec("Absorption")
    nu_sigma_f = vec("nuFission")
    chi = vec("FissionSpectrum")

    # P0 scattering: first G rows of the profile/value blocks. Profile row g
    # (sink) gives the source-group range; NDgpu wants sigma_s[source, sink].
    sc = iso.find("Scattering")
    prof = [ln.split() for ln in sc.find("Profile").text.split("\n") if ln.strip()]
    val = [ln.split() for ln in sc.find("Value").text.split("\n") if ln.strip()]
    sigma_s = np.zeros((G, G))
    for sink in range(G):
        first, last = int(prof[sink][0]), int(prof[sink][1])
        for k, source in enumerate(range(first, last + 1)):
            sigma_s[source - 1, sink] = float(val[sink][k])

    fissile = bool(np.any(nu_sigma_f > 0))
    if fissile and chi.sum() > 0:
        chi = chi / chi.sum()
    elif not fissile:
        chi = None
    return Material(name=name or f"griffin-{mid}", diffusion=1.0 / (3.0 * total),
                    sigma_a=sigma_a, nu_sigma_f=nu_sigma_f, sigma_s=sigma_s,
                    chi=chi, total=total)


def read_library(path, ids, grid_index="3 3") -> dict:
    """Read several materials by ID into ``{id: Material}``."""
    return {mid: read_material(path, mid, grid_index) for mid in ids}


def volume_homogenize(materials, fractions, chi_from=None, name="homogenized",
                      sph_factors=None) -> Material:
    """Flat-flux (volume-weighted) homogenization of several ``Material`` s.

    materials   : dict {id: Material} (all same group count).
    fractions   : dict {id: volume fraction}; need not sum to 1 (normalized).
    chi_from    : id whose fission spectrum to use (default: the first fissile
                  material) -- only fissile regions emit, so chi is taken from
                  the fuel rather than volume-mixed.
    sph_factors : optional per-group SPH multipliers applied to the homogenized
                  total/absorption/production (a rigorous SPH correction from a
                  transport reference); default None = plain volume homogenization.

    Total, Sigma_a, nuSigma_f and the scattering matrix are volume-averaged
    (exact reaction-rate conservation under a flat flux); D = 1/(3 <Total>).
    """
    ids = list(fractions)
    G = materials[ids[0]].n_groups
    w = np.array([fractions[i] for i in ids], dtype=float)
    w = w / w.sum()

    total = sum(wi * materials[i].sigma_t for wi, i in zip(w, ids))
    sigma_a = sum(wi * materials[i].sigma_a for wi, i in zip(w, ids))
    nu_sigma_f = sum(wi * materials[i].nu_sigma_f for wi, i in zip(w, ids))
    sigma_s = sum(wi * materials[i].sigma_s for wi, i in zip(w, ids))

    if chi_from is None:
        chi_from = next((i for i in ids if materials[i].is_fissile), ids[0])
    chi = materials[chi_from].chi

    if sph_factors is not None:
        mu = np.asarray(sph_factors, dtype=float)
        total = total * mu
        sigma_a = sigma_a * mu
        nu_sigma_f = nu_sigma_f * mu
        sigma_s = sigma_s * mu[:, None]

    fissile = bool(np.any(nu_sigma_f > 0))
    return Material(name=name, diffusion=1.0 / (3.0 * total), sigma_a=sigma_a,
                    nu_sigma_f=nu_sigma_f, sigma_s=sigma_s,
                    chi=chi if fissile else None, total=total)
