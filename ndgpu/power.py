"""Fission power density: the source term the thermal solver consumes.

A k-eigenvalue flux carries an arbitrary normalization (the power iteration
renormalizes it every outer so the mean fission source is 1), so a *shape* is
all a criticality solve can give you. Coupling to heat transfer needs the
absolute thing -- watts per cubic centimetre -- which comes from the shape plus
one externally imposed number, the rated thermal power:

    q'''(r)  =  P_rated * kappa_Sigma_f . phi(r) / sum_cells kappa_Sigma_f . phi V

Normalizing this way means the absolute units of the library's ``kappaFission``
never enter: whatever they are (J or MeV per fission times Sigma_f), they cancel
between numerator and denominator, and only the *shape* of kappa_Sigma_f is
used. That is the honest reading of the data, because multigroup libraries do
not agree on those units and the reactor's power is a design specification
rather than something the eigenvalue solve computes.
"""

from __future__ import annotations

import numpy as np

from .backend import asnumpy


def fission_energy_xs(materials):
    """(M, G) energy-release cross section table, and the attribute it came from.

    POWER is proportional to kappa*Sigma_f, not nu*Sigma_f. Both vanish outside
    the fuel, so either puts power only where it belongs -- but nu varies by
    group (on the HP-MR's 11-group set kappaSigma_f/nuSigma_f spans 1.12x), so
    nu-weighting tilts the distribution by the local spectrum. Fall back to
    nu_sigma_f only when no material carries kappaFission, and report which was
    used so callers can say so.
    """
    mats = list(materials)
    key = ("kappa_fission" if any(getattr(m, "kappa_fission", None) is not None
                                  for m in mats) else "nu_sigma_f")

    def row(m):
        v = getattr(m, key, None)
        return np.atleast_1d(m.nu_sigma_f if v is None else v)

    return np.array([row(m) for m in mats], dtype=float), key


def power_density(flux, materials, material_map=None, *, total_power=None,
                  cell_volume=1.0, volume_weight=None, mix_material=None,
                  mix_weight=None, active=None):
    """Per-cell fission power density on the host, shape ``grid.shape``.

    flux         : (G, *grid.shape), on device or host.
    material_map : integer map into ``materials``; None for a single material.
    total_power  : rated thermal power in W. Given, the result is W/cm^3 and
                   integrates to exactly this over the cell volumes. Omitted,
                   the raw kappa_Sigma_f . phi edit is returned, which carries
                   the flux's own arbitrary normalization.
    cell_volume  : scalar cm^3 (uniform grids). Only the *product* with
                   volume_weight matters, and only up to a constant when
                   total_power is given.
    volume_weight: optional per-cell relative volume factor, shape grid.shape --
                   the radial metric ``grid.cylindrical_metrics()[0]`` on r-z
                   grids, where cells are annuli and cell_volume alone is wrong.
    mix_material / mix_weight : the solver's two-material volume blend. Cross
                   sections blend linearly (exact flat-flux reaction-rate
                   averaging), matching ``Fields``.
    active       : bool mask; excised cells produce no power.
    """
    flux = asnumpy(flux)
    G = flux.shape[0]
    table, _ = fission_energy_xs(materials)
    if table.shape[1] != G:
        raise ValueError(f"flux has {G} groups, materials have {table.shape[1]}")

    if material_map is None:
        if len(table) > 1:
            raise ValueError("material_map is required with multiple materials")
        xs = np.broadcast_to(table[0].reshape(G, *([1] * (flux.ndim - 1))),
                             flux.shape)
    else:
        mmap = np.asarray(material_map)
        if mmap.shape != flux.shape[1:]:
            raise ValueError(f"material_map shape {mmap.shape} != flux cell "
                             f"shape {flux.shape[1:]}")
        xs = np.stack([table[:, g][mmap] for g in range(G)])
        if mix_material is not None:
            mm2 = np.asarray(mix_material)
            w = np.asarray(mix_weight, dtype=float)
            has = mm2 >= 0
            base = np.maximum(mm2, 0)
            for g in range(G):
                other = table[:, g][base]
                xs[g] = np.where(has, (1.0 - w) * xs[g] + w * other, xs[g])

    dens = (xs * flux).sum(axis=0)
    if active is not None:
        dens = np.where(np.asarray(active).astype(bool), dens, 0.0)
    if total_power is None:
        return dens

    vol = float(cell_volume)
    if volume_weight is not None:
        vol = vol * asnumpy(volume_weight)
    integral = float(np.sum(dens * vol))
    if integral <= 0.0:
        raise ValueError("no fission power in the flux: cannot normalize to "
                         "a rated power (is the flux zero, or the material "
                         "map missing its fuel?)")
    return dens * (float(total_power) / integral)
