"""Pin-power reconstruction: homogeneous core flux x heterogeneous form function.

A full-core diffusion solve on assembly-homogenized constants knows the power
of each assembly but nothing about the 127 pins inside it -- homogenization
discarded that shape, and equivalence factoring (SPH, GET) does not bring it
back, because it corrects assembly-integrated reaction rates rather than the
distribution within a region. Reconstruction supplies the missing shape from a
separate heterogeneous calculation:

    P_pin  =  S(x_pin)  *  f(pin)

``f`` is the intra-assembly FORM FUNCTION, from a single-assembly heterogeneous
S_N solve in an infinite lattice (:mod:`ndgpu.benchmarks.hpmr_assembly`),
normalized to mean 1 over the fuel pins. ``S`` is the homogeneous power shape
from the core solve, evaluated AT the pin's position rather than averaged over
the assembly -- which is the point of "spanning the assembly": across a core
with drums and a reflector the coarse flux tilts substantially from one side of
an assembly to the other, and a single per-assembly scalar throws that tilt
away. The two factors are separable to the extent that the local spectrum
resembles the infinite-lattice one; that assumption weakens next to a
reflector, a drum, or a strong flux gradient, which is exactly where the
reconstruction error concentrates.

The form function is generated once per assembly TYPE and per state (drum angle
changes it), then applied everywhere that type appears.
"""

from __future__ import annotations

from .backend import asnumpy
from .power import power_density
import numpy as np



def tri_cell_centroids(raster):
    """(nr, nc, 2, 2) Cartesian centroids of a rasterized core.

    Vectorizes ``TriRaster.cell_centroid`` exactly -- same basis
    ``av = side (sqrt3/2, 1/2)``, ``bv = side (0, 1)`` and the same
    ``(i0, j0)`` lattice origin, with the down triangle's centroid at 1/3 of
    the rhombus diagonal and the up triangle's at 2/3. It must be the raster's
    own frame: a generic tri-lattice convention differs by a rotation and an
    offset, which would silently sample the flux from the wrong cells.
    """
    mmap = np.asarray(raster.material_map)
    nr, nc = mmap.shape[0], mmap.shape[1]
    h = raster.side
    av = np.array([h * np.sqrt(3.0) / 2.0, h * 0.5])
    bv = np.array([0.0, h])
    i0, j0 = raster.origin
    aa, bb = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    O = ((i0 + aa - 1)[..., None] * bv + (j0 + bb - 1)[..., None] * av)
    cen = np.empty((nr, nc, 2, 2))
    cen[:, :, 0, :] = O + (av + bv) / 3.0
    cen[:, :, 1, :] = O + 2.0 * (av + bv) / 3.0
    return cen


def sample_power_shape(raster, flux, materials, material_map, points,
                       mix_material=None, mix_weight=None, active=None):
    """Homogeneous fission-power density sampled at arbitrary points.

    Nearest-cell lookup on the triangular mesh. Returns one value per point;
    points falling on an inactive cell return 0. This is ``S`` above -- the
    smooth shape the reconstruction rides on, and it varies WITHIN an assembly
    because the core mesh resolves sub-assembly detail.
    """
    # kappa*Sigma_f, not nu*Sigma_f -- see ndgpu.power.fission_energy_xs. Power
    # lives only in the fuel (both vanish elsewhere); the difference is the
    # group weighting. Unnormalized: this is a shape, and it is the caller's
    # form-function normalization that fixes the scale.
    dens = power_density(flux, materials, material_map,
                         mix_material=mix_material, mix_weight=mix_weight,
                         active=active)

    cen = tri_cell_centroids(raster).reshape(-1, 2)
    val = dens.reshape(-1)
    try:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(cen).query(np.asarray(points))
    except ImportError:                                    # pragma: no cover
        d = np.linalg.norm(np.asarray(points)[:, None, :] - cen[None, :, :], axis=2)
        idx = d.argmin(axis=1)
    return val[idx]


def reconstruct_pin_powers(raster, flux, materials, material_map, sites,
                           pin_centres, form, pin_kind=None,
                           mix_material=None, mix_weight=None, active=None):
    """Reconstruct pin powers across a whole core.

    sites       : (n_sites, 2) xy of each assembly centre to reconstruct.
    pin_centres : (n_pins, 2) pin positions relative to an assembly centre.
    form        : (n_pins,) intra-assembly form function, mean 1 over fuel pins.

    Returns (power, xy) with shape (n_sites, n_pins) and
    (n_sites, n_pins, 2) -- absolute pin positions, so the result can be
    mapped or reduced without recomputing geometry.

    The core flux is sampled at every pin's own position, so an assembly
    sitting in a flux gradient gets a tilted reconstruction rather than a
    uniform scaling of the form function.
    """
    sites = np.asarray(sites, dtype=float)
    pin_centres = np.asarray(pin_centres, dtype=float)
    form = np.asarray(form, dtype=float)
    xy = sites[:, None, :] + pin_centres[None, :, :]
    flat = xy.reshape(-1, 2)
    S = sample_power_shape(raster, flux, materials, material_map, flat,
                           mix_material=mix_material, mix_weight=mix_weight,
                           active=active)
    power = S.reshape(len(sites), len(pin_centres)) * form[None, :]
    if pin_kind is not None:
        fuel = np.array([k == "fuel" for k in pin_kind])
        power = np.where(fuel[None, :], power, 0.0)
    return power, xy


def peaking(power, pin_kind=None):
    """Pin peaking factor: max pin power over the mean across fuel pins."""
    p = np.asarray(power, dtype=float)
    if pin_kind is not None:
        fuel = np.array([k == "fuel" for k in pin_kind])
        vals = p[:, fuel]
    else:
        vals = p[p > 0].reshape(1, -1) if p.ndim == 2 else p
    m = vals[vals > 0].mean()
    return float(vals.max() / m), float(vals.max()), float(m)
