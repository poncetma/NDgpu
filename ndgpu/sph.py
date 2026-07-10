"""Superhomogenization (SPH) against a transport reference.

Coarse diffusion with flux-weighted homogenized cross sections does not, on its
own, reproduce a transport reference: collapsing a heterogeneous region to one
set of constants preserves that region's *reaction rates at the reference flux*,
but the coarse diffusion flux differs from the reference, so the rates (and the
eigenvalue) drift. SPH restores the equivalence by multiplying each region-group
cross section by a factor mu chosen so the coarse flux matches the reference
region flux again.

This module builds the pipeline in pieces:

* :func:`flux_weighted_homogenize` -- collapse a fine reference solution into one
  Material per coarse region, flux-and-volume weighted (this step preserves the
  reference reaction rates by construction).
* the SPH factor solve (added incrementally) -- iterate mu until the coarse
  diffusion region fluxes match the reference.

The transport reference is any ndgpu eigensolver's scalar-flux Result -- e.g.
SP3EigenSolver or TriSP3EigenSolver, the "transport" NDgpu offers above
diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import Material


def _cell_tables(materials, material_map):
    """Per-cell cross-section tables from a material map, flattened to (N, ...).

    Returns a dict of arrays indexed by flat cell: sigma_a, nu_sigma_f, sigma_t,
    diffusion, chi (each (N, G)) and sigma_s ((N, G, G)).
    """
    mats = list(materials)
    G = mats[0].n_groups
    flat = np.asarray(material_map).reshape(-1)
    sa = np.array([m.sigma_a for m in mats])          # (M, G)
    nf = np.array([m.nu_sigma_f for m in mats])
    st = np.array([m.sigma_t for m in mats])
    df = np.array([m.diffusion for m in mats])
    ch = np.array([m.chi for m in mats])
    ss = np.array([m.sigma_s for m in mats])          # (M, G, G)
    return dict(sigma_a=sa[flat], nu_sigma_f=nf[flat], sigma_t=st[flat],
                diffusion=df[flat], chi=ch[flat], sigma_s=ss[flat], G=G)


def flux_weighted_homogenize(flux, materials, material_map, region_map,
                             cell_volume=1.0):
    """Collapse a fine reference solution into one Material per coarse region.

    flux         : (G, *shape) reference scalar flux (e.g. an SP3 Result.flux).
    materials    : fine material list. material_map : (*shape) index into it.
    region_map   : (*shape) coarse-region index in [0, R); the homogenization
                   regions (e.g. one per assembly).
    cell_volume  : scalar cell volume, or a (*shape) array (uniform grids: pass
                   the constant; it cancels in the ratios but sets the scale of
                   the returned region volumes/fluxes).

    Returns (homogenized_materials, region_flux, region_volume):
      homogenized_materials : list of R Materials, each cross section
        flux-and-volume weighted over its region and group so that
        Sigma_hom * <phi>_region * V_region == the region's fine reaction rate.
      region_flux           : (R, G) volume-average scalar flux per region.
      region_volume         : (R,) total volume per region.

    Reaction-rate preservation is exact by construction (see
    tests/verification/test_sph.py); it is what lets the SPH factor solve, added
    next, recover the reference eigenvalue rather than just its flux shape.
    """
    tab = _cell_tables(materials, material_map)
    G = tab["G"]
    n = np.asarray(region_map).reshape(-1)
    R = int(n.max()) + 1
    phi = np.asarray(flux).reshape(G, -1)             # (G, N)
    V = np.broadcast_to(np.asarray(cell_volume, dtype=float),
                        n.shape).reshape(-1)          # (N,)

    region_flux = np.zeros((R, G))
    region_volume = np.zeros(R)
    mats_out = []
    for i in range(R):
        cells = np.where(n == i)[0]
        Vi = V[cells]                                 # (ni,)
        w = phi[:, cells] * Vi                        # (G, ni) flux-volume weight
        wsum = w.sum(axis=1)                          # (G,)
        vol = Vi.sum()
        region_volume[i] = vol
        region_flux[i] = (phi[:, cells] * Vi).sum(axis=1) / vol   # volume-average flux

        def fw(table):                                # flux-volume weighted, (G,)
            return (table[cells].T * w).sum(axis=1) / wsum        # (G,)

        sigma_a = fw(tab["sigma_a"])
        nu_sigma_f = fw(tab["nu_sigma_f"])
        sigma_t = fw(tab["sigma_t"])
        # diffusion: flux-weight the transport cross section 1/(3D), then invert
        sigma_tr = (( (1.0 / (3.0 * tab["diffusion"]))[cells].T * w).sum(axis=1) / wsum)
        diffusion = 1.0 / (3.0 * sigma_tr)
        # scattering g->g' weighted by the source-group (g) flux-volume weight
        ss = tab["sigma_s"][cells]                    # (ni, G, G)
        sigma_s = np.zeros((G, G))
        for g in range(G):
            wg = w[g]
            denom = wg.sum()
            if denom > 0:
                sigma_s[g] = (ss[:, g, :].T * wg).sum(axis=1) / denom
        # chi: fission-production weighted over the region
        prod = (tab["nu_sigma_f"][cells] * phi[:, cells].T).sum(axis=1)   # (ni,) sum_g nuSf phi
        pw = prod * Vi
        chi = ((tab["chi"][cells].T * pw).sum(axis=1) / pw.sum()
               if pw.sum() > 0 else tab["chi"][cells][0])

        mats_out.append(Material(
            name=f"sph-region-{i}", diffusion=diffusion, sigma_a=sigma_a,
            nu_sigma_f=nu_sigma_f, sigma_s=sigma_s,
            chi=(chi if np.isclose(chi.sum(), 1.0) else np.array([1.0] + [0.0] * (G - 1))),
            total=sigma_t))
    return mats_out, region_flux, region_volume


def region_average(flux, region_map):
    """Volume-average scalar flux per coarse region, (R, G).

    flux : (G, *shape) scalar flux; region_map : (*shape) region index. Assumes
    equal cell volumes (uniform grid), matching :func:`flux_weighted_homogenize`.
    """
    n = np.asarray(region_map).reshape(-1)
    R = int(n.max()) + 1
    f = np.asarray(flux).reshape(np.asarray(flux).shape[0], -1)
    return np.array([[f[g][n == i].mean() for g in range(f.shape[0])] for i in range(R)])


def _scale_material(mat, mu):
    """Multiply every cross section of `mat` by mu[g], and D by 1/mu[g].

    Scaling the transport cross section 1/(3D) by mu is D -> D/mu. Multiplying
    *all* cross sections (removal, scatter, production, transport) by the SPH
    factor is what makes the corrected reaction rate mu * Sigma * Phi reproduce
    the reference rate once mu * Phi == phi_ref.
    """
    mu = np.asarray(mu, dtype=float)
    return Material(name=mat.name + "-sph", diffusion=mat.diffusion / mu,
                    sigma_a=mat.sigma_a * mu, nu_sigma_f=mat.nu_sigma_f * mu,
                    sigma_s=mat.sigma_s * mu[:, None], chi=mat.chi,
                    total=(mat.sigma_t * mu))


@dataclass
class SphResult:
    corrected_materials: list        # R Materials with SPH-corrected cross sections
    factors: np.ndarray              # (R, G) SPH factors mu
    k_eff: float                     # eigenvalue of the corrected coarse solve
    iterations: int
    converged: bool


def sph_correct(homogenized_materials, region_map, reference_region_flux, solve,
                max_iter=200, tol=1e-8, relax=1.0) -> SphResult:
    """Solve for the SPH factors that make a coarse solve reproduce the reference.

    The superhomogenization factor mu_{i,g} multiplies every cross section of
    region i, group g (and divides its diffusion coefficient) so that the
    corrected coarse reaction rate reproduces the reference: the defining
    condition is ``mu_{i,g} * Phi_{i,g} == phi_ref_{i,g}`` (not Phi == phi_ref),
    which preserves the region reaction rates and hence the eigenvalue on the
    generation geometry. The fixed point ``mu <- phi_ref / Phi(mu)`` is iterated
    to convergence; fluxes are normalized to a common total each step so only
    the shape matters.

    homogenized_materials : R Materials (e.g. from
        :func:`flux_weighted_homogenize`), indexed by region.
    region_map            : (*shape) region index; the coarse material map.
    reference_region_flux : (R, G) reference region fluxes (the second return of
        :func:`flux_weighted_homogenize`).
    solve                 : callable(materials) -> (region_flux (R, G), k_eff),
        running the coarse diffusion solve with the given per-region materials
        (the caller wires up grid, geometry and boundary conditions).
    relax                 : optional under-relaxation in (0, 1] on log(mu);
        1.0 is the plain fixed point (stable for well-homogenized generation
        problems).
    """
    R = len(homogenized_materials)
    ref = np.asarray(reference_region_flux, dtype=float)
    ref_n = ref / ref.sum()
    mu = np.ones_like(ref)
    k = float("nan")
    converged = False
    for it in range(1, max_iter + 1):
        region_flux, k = solve([_scale_material(homogenized_materials[i], mu[i])
                                for i in range(R)])
        phi = np.asarray(region_flux, dtype=float)
        mu_new = ref_n / (phi / phi.sum())
        if relax != 1.0:
            mu_new = mu ** (1.0 - relax) * mu_new ** relax
        err = np.abs(mu_new / mu - 1.0).max()
        mu = mu_new
        if err < tol:
            converged = True
            break
    corrected = [_scale_material(homogenized_materials[i], mu[i]) for i in range(R)]
    return SphResult(corrected_materials=corrected, factors=mu, k_eff=k,
                     iterations=it, converged=converged)

