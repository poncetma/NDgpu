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

The transport reference is any ndgpu eigensolver's scalar-flux Result -- any of
the simplified-transport families NDgpu offers above diffusion: SP3/TriSP3, the
double-PN SDP1 and SDP2/TriSDP1, TriSDP2, or the standard SPN orders. For every
one of these the Result carries the *physical* scalar flux phi0 (the
reconstructed 0th angular moment: SP3 forms phi0 = Phi1 - 2 phi2, SDPN dots the
even-moment vector with its phi0 closure weights), which is exactly the field
that drives the isotropic reaction rates SPH preserves -- so swapping the
reference between families feeds SPH genuinely different angular closures of the
same problem rather than a re-extraction of one field. See
examples/hpmr_sph_reference_families.py for an SP3-vs-SDP1-vs-SDP2 comparison on
the HP-MR drum arc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import asnumpy
from .linalg import anderson_step as _anderson_step
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


def _mixed_entries(material_map, region_map, cell_volume, shape_n,
                   mix_material=None, mix_weight=None, mix_region_map=None):
    """Expand per-cell volume mixing into fractional sub-cell entries.

    A cell blended between two materials (the solvers' ``mix_material`` /
    ``mix_weight``, e.g. the volume-mixed control-drum arc) is not one material,
    so homogenizing it by ``material_map`` alone attributes the whole cell to the
    base material and loses the absorber's reaction rate. Splitting it into two
    entries -- (1-w) of the volume with the base material, w with the mix partner,
    both carrying the same cell-average flux -- is what keeps
    ``Sigma_hom <phi> V == the fine reaction rate`` exact.

    Returns (cell_index, region, volume, material_index) arrays over entries.
    """
    idx = np.arange(shape_n)
    base_mat = np.asarray(material_map).reshape(-1)
    base_reg = np.asarray(region_map).reshape(-1)
    V = np.broadcast_to(np.asarray(cell_volume, dtype=float),
                        base_mat.shape).reshape(-1).astype(float)
    if mix_material is None or mix_weight is None:
        return idx, base_reg, V, base_mat
    mm = np.asarray(mix_material).reshape(-1)
    mw = np.asarray(mix_weight, dtype=float).reshape(-1)
    live = (mw > 0.0) & (mm >= 0)
    mreg = (mm if mix_region_map is None
            else np.asarray(mix_region_map).reshape(-1))
    return (np.concatenate([idx, idx[live]]),
            np.concatenate([base_reg, mreg[live]]),
            np.concatenate([V * (1.0 - np.where(live, mw, 0.0)), V[live] * mw[live]]),
            np.concatenate([base_mat, mm[live]]))


def flux_weighted_homogenize(flux, materials, material_map, region_map,
                             cell_volume=1.0, mix_material=None,
                             mix_weight=None, mix_region_map=None):
    """Collapse a fine reference solution into one Material per coarse region.

    flux         : (G, *shape) reference scalar flux (e.g. an SP3 Result.flux,
                   or a TriSNTransportSolver SNResult.flux -- any reference that
                   carries a physical scalar flux).
    materials    : fine material list. material_map : (*shape) index into it.
    region_map   : (*shape) coarse-region index in [0, R); the homogenization
                   regions (e.g. one per assembly).
    cell_volume  : scalar cell volume, or a (*shape) array (uniform grids: pass
                   the constant; it cancels in the ratios but sets the scale of
                   the returned region volumes/fluxes).
    mix_material, mix_weight : the solvers' per-cell two-material blend. Supply
                   them and each blended cell is split into two fractional
                   entries, so a volume-mixed absorber arc is homogenized at its
                   true volume fraction instead of being attributed wholly to
                   the base material. Without this, SPH is restricted to a
                   rasterized (discrete per-cell) geometry, which needs a very
                   fine mesh before a thin arc is resolved at all.
    mix_region_map : region index for the mixed component; defaults to
                   ``mix_material`` (correct for the common "one region per
                   material" mapping).

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
    flux = asnumpy(flux)          # device Result.flux -> host (see module note)
    G = int(flux.shape[0])
    ncell = int(np.asarray(material_map).size)
    cells, n, V, matidx = _mixed_entries(material_map, region_map, cell_volume,
                                         ncell, mix_material, mix_weight,
                                         mix_region_map)
    tab = _cell_tables(materials, matidx)
    R = int(max(n.max(), np.asarray(region_map).max())) + 1
    phi = flux.reshape(G, -1)[:, cells]               # (G, n_entries)

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

        # Fall back to volume weighting in any group with no flux (a void or a
        # decoupled region), so the region still yields a finite Material rather
        # than a division-by-zero; such regions carry no reaction rate and are
        # frozen (mu = 1) by the SPH solve anyway.
        wsafe = np.where(wsum > 0, wsum, 1.0)

        def fw(table):                                # flux-volume weighted, (G,)
            fluxw = (table[cells].T * w).sum(axis=1) / wsafe      # (G,)
            volw = (table[cells].T * Vi).sum(axis=1) / vol        # (G,)
            return np.where(wsum > 0, fluxw, volw)

        sigma_a = fw(tab["sigma_a"])
        nu_sigma_f = fw(tab["nu_sigma_f"])
        sigma_t = fw(tab["sigma_t"])
        # diffusion: flux-weight the transport cross section 1/(3D), then invert
        sigma_tr = fw(1.0 / (3.0 * tab["diffusion"]))
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


def production_weight(materials, region_volume):
    """Normalization weight for :func:`sph_correct`: nu*Sigma_f * V per region-group.

    Labour\u00e9 et al. (Ann. Nucl. Energy 2019) identify SPH's normalization factor
    as the defect that breaks leakage equivalence whenever a vacuum boundary is
    present: dividing each flux by its own *sum* matches shape while discarding
    the absolute level leakage depends on. Normalizing instead against total
    fission production is a physical integral, consistent with the reaction
    rates SPH preserves.

    Measured on HP-MR (drums withdrawn, S_N reference, JFNK):

        sum-normalized          dk = 746 pcm, absorption err 0.27%, leakage 37.1%
        production-normalized   dk = 185 pcm, absorption err 0.00%, leakage 14.8%

    Use it for any geometry with a vacuum boundary. For a fully reflected
    assembly the plain sum normalization is fine (and is still the default).
    """
    return np.array([np.asarray(m.nu_sigma_f, dtype=float)
                     * float(region_volume[i])
                     for i, m in enumerate(materials)])


def region_average(flux, region_map, mix_material=None, mix_weight=None,
                   mix_region_map=None):
    """Volume-average scalar flux per coarse region, (R, G).

    flux : (G, *shape) scalar flux; region_map : (*shape) region index. Assumes
    equal cell volumes (uniform grid), matching :func:`flux_weighted_homogenize`.
    Pass the same ``mix_material`` / ``mix_weight`` used there so the coarse
    solve's region fluxes are averaged over the *same* fractional volumes -- SPH
    compares these two directly, so an inconsistent split biases every factor.
    """
    f = asnumpy(flux)
    G = f.shape[0]
    cells, n, V, _ = _mixed_entries(region_map, region_map, 1.0, f[0].size,
                                    mix_material, mix_weight, mix_region_map)
    ff = f.reshape(G, -1)[:, cells]
    R = int(max(n.max(), np.asarray(region_map).max())) + 1
    out = np.zeros((R, G))
    for i in range(R):
        m = n == i
        vol = V[m].sum()
        if vol > 0:
            out[i] = (ff[:, m] * V[m]).sum(axis=1) / vol
    return out


def _scale_material_get(mat, mu, nu):
    """SPH factor mu on the reaction cross sections, GET factor nu on D alone.

    Plain SPH ties leakage to reaction rates: it scales every cross section by
    mu and D by 1/mu, so one factor per region-group must serve both. That is
    exactly why it preserves absorption but not leakage against an S_N reference
    (measured: absorption 0.27%, leakage 37% at HP-MR drums withdrawn). Letting D
    carry its own factor nu adds the missing degree of freedom in the SPH idiom
    -- no discontinuity factors in the stencil, no operator change -- and the two
    conditions (region reaction rate, region leakage) then determine both.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    nu = np.atleast_1d(np.asarray(nu, dtype=float))
    return Material(
        name=f"{mat.name}-get", diffusion=np.asarray(mat.diffusion) * nu,
        sigma_a=np.asarray(mat.sigma_a) * mu,
        nu_sigma_f=np.asarray(mat.nu_sigma_f) * mu,
        sigma_s=np.asarray(mat.sigma_s) * mu[:, None],
        chi=np.asarray(mat.chi), total=np.asarray(mat.sigma_t) * mu,
        # kappa*Sigma_f is a macroscopic reaction-rate coefficient too. Losing
        # it made an SPH-corrected model silently fall back to nu*Sigma_f for
        # thermal coupling, changing the power shape by energy group.
        kappa_fission=(None if mat.kappa_fission is None else
                       np.asarray(mat.kappa_fission) * mu))


def sph_get_correct(homogenized_materials, region_map, reference_region_flux,
                    reference_region_leakage, solve, tol=1e-8,
                    max_iter=200, method="hybr") -> SphResult:
    """SPH + a generalized-equivalence leakage factor, solved together by JFNK.

    Unknowns per live region-group: log(mu) and log(nu). Conditions:

        mu * Phi   = phi_ref        (region reaction rate, plain SPH)
        L(mu, nu)  = L_ref          (region net leakage, the GET addition)

    solve : callable(materials, region_df) -> (region_flux (R,G), k_eff,
        region_leakage (R,G)). region_df is the (R,G) discontinuity factor the
        caller expands onto cells and passes to the solver as ``df=``. Leakage
        is the region's net current out, i.e. production/k - absorption.

    The second factor is a *discontinuity factor on the region surface*, not a
    scaling of the region's D. That distinction is the whole point: a region-D
    scaling moves leakage only through the global balance (measured: drum-only
    nu x1.10 gave +12.9 pcm against -129 pcm for a global scaling, i.e. it has
    almost no local authority) and leaves the Newton system rank-deficient,
    because per-region leakage is not independent of per-region reaction rate
    once global balance holds. A DF acts per *face*, asymmetrically, so it can
    redistribute leakage between the faces of one region -- which is exactly
    what classical GET is for, and why it must live in the stencil.

    Newton is used rather than a fixed point for the reason recorded in
    sph_correct: the mu iteration is unstable, not merely slow.
    """
    from scipy.optimize import newton_krylov, root
    try:
        from scipy.optimize import NoConvergence
    except ImportError:                      # pragma: no cover
        from scipy.optimize.nonlin import NoConvergence

    R = len(homogenized_materials)
    ref = np.asarray(reference_region_flux, dtype=float)
    lref = np.asarray(reference_region_leakage, dtype=float)
    shape = ref.shape
    live = ((ref > ref.max() * 1e-12)
            & (np.abs(lref) > np.abs(lref).max() * 1e-12)).reshape(-1)
    n = int(live.sum())
    state = {"k": float("nan"), "n": 0}

    def build(z):
        # Clip the exponents: Newton probes wide steps, and exp() underflowing
        # to 0 makes a zero total cross section (a hard Material error) rather
        # than a large residual it could back off from.
        # Separate bounds: mu is a cross-section scaling and may legitimately
        # be far from 1, but a discontinuity factor is a ratio of surface to
        # region-average flux and is physically O(1). Left unbounded Newton
        # probes f ~ 1e5, which wrecks the conditioning of the (now
        # non-symmetric) operator and GMRES fails outright.
        z = np.asarray(z, dtype=float)
        x = np.zeros(ref.size); y = np.zeros(ref.size)
        x[live] = np.clip(z[:n], -12.0, 12.0)
        y[live] = np.clip(z[n:], -1.2, 1.2)          # f in [0.30, 3.3]
        mu, nu = np.exp(x).reshape(shape), np.exp(y).reshape(shape)
        return mu, nu, [_scale_material(homogenized_materials[i], mu[i])
                        for i in range(R)]

    def residual(z):
        state["n"] += 1
        mu, nu, mats = build(z)
        phi, k, leak = solve(mats, nu)
        state["k"] = k
        phi = np.asarray(phi, float).reshape(-1)
        leak = np.asarray(leak, float).reshape(-1)
        r1 = np.full(ref.size, 0.0); r2 = np.full(ref.size, 0.0)
        ok = live & (phi > 0)
        r1[ok] = np.log(mu.reshape(-1)[ok] * phi[ok] / ref.reshape(-1)[ok])
        r1[live & ~ok] = 1e3
        sgn = live & (leak * lref.reshape(-1) > 0)
        r2[sgn] = np.log(np.abs(leak[sgn]) / np.abs(lref.reshape(-1)[sgn]))
        r2[live & ~sgn] = 1e3            # leakage flipped sign: reject
        return np.concatenate([r1[live], r2[live]])

    z0 = np.zeros(2 * n)
    converged = True
    if method == "hybr":
        # Dense-Jacobian trust region, as for sph_correct. The joint mu+DF
        # system is 2n ~ 24 unknowns -- far too small for Jacobian-free Krylov,
        # which failed here two different ways (rank-deficient Jacobian on one
        # drum state, a near-miss GMRES tolerance on the other). Forming the
        # 2n x 2n Jacobian costs 2n+1 coarse solves and lets MINPACK take
        # properly bounded steps.
        sol = root(residual, z0, method="hybr",
                   options=dict(xtol=max(tol * 1e-4, 1e-14)))
        z, converged = sol.x, bool(sol.success)
    else:
        try:
            z = newton_krylov(residual, z0, f_tol=tol, maxiter=max_iter,
                              method="lgmres")
        except NoConvergence as e:
            z = np.asarray(e.args[0]); converged = False
    mu, nu, mats = build(z)
    _, k, _ = solve(mats, nu)
    return SphResult(corrected_materials=mats, factors=np.stack([mu, nu]),
                     k_eff=k, iterations=state["n"], converged=converged)


# How the diffusion coefficient is corrected alongside the cross sections.
#   "inverse"  D -> D / mu   -- transport-consistent (Sigma_t x mu => D = 1/3Sigma_tr)
#   "direct"   D -> D * mu   -- Laboure et al. 2019 Eq. 7 / Ortensi et al. 2018c,
#                               an empirical LEAKAGE correction "to better match
#                               the reference currents"
# These differ in SIGN of the leakage correction on every region with mu != 1.
# The paper is explicit about "direct", but measured on ndgpu's finite-volume
# leaky-colorset test "inverse" is markedly better (SPH closes 4x of the
# homogenization error vs 2x), so this is exposed as a choice and measured per
# problem rather than assumed. The paper's method is CFEM; ndgpu's is FV.
D_SCALING = "inverse"


def _scale_material(mat, mu, d_scaling=None):
    """SPH-corrected material: every cross section x mu, and D x mu.

    Sigma_m,g = mu_m,g Sigma_ref_m,g            (Laboure et al. 2019, Eq. 4)
    D_m,g     = mu_m,g D_ref_m,g                (Eq. 7)

    NOTE the diffusion coefficient is multiplied by mu, NOT divided. Dividing is
    what "consistency" suggests (if Sigma_t scales by mu then D = 1/(3 Sigma_tr)
    scales by 1/mu) and is what this function did originally -- but SPH's D
    correction is a deliberate empirical choice "to better match the reference
    currents", i.e. it is a LEAKAGE correction, not a transport-consistent one.
    Getting it backwards inverts the sign of the leakage correction on every
    region with mu != 1, which is why region leakage would not converge to the
    reference no matter how the factors were solved for.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    d_scaling = D_SCALING if d_scaling is None else d_scaling
    dfac = mu if d_scaling == "direct" else 1.0 / mu
    return Material(
        name=f"{mat.name}-sph", diffusion=np.asarray(mat.diffusion) * dfac,
        sigma_a=np.asarray(mat.sigma_a) * mu,
        nu_sigma_f=np.asarray(mat.nu_sigma_f) * mu,
        sigma_s=np.asarray(mat.sigma_s) * mu[:, None],
        chi=np.asarray(mat.chi), total=np.asarray(mat.sigma_t) * mu,
        # Preserve the energy-release reaction rate for coupled power edits.
        kappa_fission=(None if mat.kappa_fission is None else
                       np.asarray(mat.kappa_fission) * mu))

@dataclass
class SphResult:
    corrected_materials: list        # R Materials with SPH-corrected cross sections
    factors: np.ndarray              # (R, G) SPH factors mu
    k_eff: float                     # eigenvalue of the corrected coarse solve
    iterations: int
    converged: bool
    # Set only by sph_df_correct (GET): per-region discontinuity factors, per
    # vacuum-boundary-region boundary coefficients, and the least-squares cost
    # at the solution -- the diagnostic that says whether a nonzero residual is
    # a formulation failure or the per-cell (rather than per-face) DF limit.
    df: np.ndarray = None
    bcf: np.ndarray = None
    cost: float = float("nan")


# Strength of the linear pull back to mu = 1 from an infeasible trial point.
# Only its sign and scale matter: the Newton step it induces is -x regardless.
_INFEASIBLE_PULL = 10.0


def _sph_residual(x_live, live, ln_ref_n, shape, homogenized_materials, solve,
                  state):
    """SPH residual in log space, restricted to the live entries.

    F(x) = ln(phi_ref / sum) - ln(Phi(exp x) / sum) - x, i.e. the amount by
    which log(mu) still has to move. Its root is the SPH condition
    mu * Phi = phi_ref. Frozen entries are excluded rather than zeroed so the
    Jacobian has no null directions for the Newton solve.
    """
    R = len(homogenized_materials)
    x_live = np.clip(np.asarray(x_live, dtype=float), -8.0, 8.0)
    x = np.zeros(live.size)
    x[live] = x_live
    mu = np.exp(x).reshape(shape)
    region_flux, k = solve([_scale_material(homogenized_materials[i], mu[i])
                            for i in range(R)])
    state["k"] = k
    state["nfev"] += 1
    phi = np.asarray(region_flux, dtype=float).reshape(-1)
    live_phi = phi[live]
    if not np.all(live_phi > 0.0) or not np.isfinite(live_phi).all():
        # A trial mu drove the coarse solve to a non-positive (or non-finite)
        # region flux, where log() is undefined.
        #
        # Returning a CONSTANT here (the original guard) is a trap, not a
        # barrier: a constant residual has zero Jacobian, so the Krylov solve
        # finds no direction back and Newton simply stays put -- observed on
        # HP-MR as |F| pinned at 106.826 for every subsequent iteration, with
        # the step length varying and the residual not responding at all.
        #
        # A linear pull toward x = 0 (mu = 1, always feasible) instead has
        # Jacobian PENALTY*I, so the Newton step is exactly -x: one step back
        # into the feasible region, from which the real residual takes over.
        return _INFEASIBLE_PULL * np.asarray(x_live, dtype=float)
    if state.get("absolute"):
        denom = 1.0                       # no per-iteration normalization
    else:
        nw = state.get("nw")
        denom = live_phi.sum() if nw is None else (nw * live_phi).sum()
    ln_phi_n = np.log(live_phi / denom)
    r = (ln_ref_n[live] - ln_phi_n) - x_live
    sc = state.get("scale")
    return r if sc is None else r * sc


def sph_correct(homogenized_materials, region_map, reference_region_flux, solve,
                max_iter=200, tol=1e-8, relax=1.0, depth=5,
                method="anderson", norm_weight=None, mu0=None,
                residual_scale=None, region_volume=None,
                verbose=False) -> SphResult:
    """Solve for the SPH factors that make a coarse solve reproduce the reference.

    The superhomogenization factor mu_{i,g} multiplies every cross section of
    region i, group g (and divides its diffusion coefficient) so that the
    corrected coarse reaction rate reproduces the reference: the defining
    condition is ``mu_{i,g} * Phi_{i,g} == phi_ref_{i,g}`` (not Phi == phi_ref),
    which preserves the region reaction rates and hence the eigenvalue on the
    generation geometry. The fixed point ``mu <- phi_ref / Phi(mu)`` is iterated
    to convergence in log(mu) space; fluxes are normalized to a common total each
    step so only the shape matters.

    For a reflective single assembly the plain fixed point converges. On a leaky
    colorset or whole core the fixed point oscillates (a region whose flux dips
    gets its absorption scaled up, dipping it further), so the iteration is
    Anderson-accelerated over a short history of log-factor residuals; set
    ``depth=1`` to recover the plain (optionally relaxed) fixed point.

    homogenized_materials : R Materials (e.g. from
        :func:`flux_weighted_homogenize`), indexed by region.
    region_map            : (*shape) region index; the coarse material map.
    reference_region_flux : (R, G) reference region fluxes (the second return of
        :func:`flux_weighted_homogenize`).
    solve                 : callable(materials) -> (region_flux (R, G), k_eff),
        running the coarse diffusion solve with the given per-region materials
        (the caller wires up grid, geometry and boundary conditions).
    relax                 : damping beta in (0, 1] on the log(mu) update; 1.0 is
        undamped. depth : Anderson history length (1 disables acceleration).
    residual_scale        : row scaling for the Newton residual.

        *** "auto" IS BROKEN -- DO NOT USE. *** It weights each region-group by
        its share of total production, which scales DOWN precisely the
        low-production entries (reflector, drum absorber) that carry the leakage
        physics. Newton then meets the tolerance without solving those rows:
        measured on HP-MR it reports conv=True with mu == 1 in every region,
        i.e. no correction at all, and k falls back to uncorrected diffusion
        (inserted 770 pcm, withdrawn 745 pcm vs the S_N reference). That is
        FALSE convergence and is worse than the stall it was meant to cure,
        because it looks like success. Any row scaling here must not shrink the
        rows that need solving; scale by sensitivity, not by magnitude.
    mu0                   : optional initial factors (R, G), default all ones.
        Warm-starting a production-normalized solve from the converged
        sum-normalized factors is far more robust than starting from 1.
    method                : "anderson" (default) accelerates the fixed point;
        "jfnk" instead solves the residual F(x) = 0 directly with a
        Jacobian-free Newton-Krylov method (scipy's Newton-Krylov root find).

        The fixed point is not merely slow but *unstable* whenever the coarse
        flux in a strongly absorbing region falls below the reference: the SPH
        condition is mu*Phi = phi_ref, so a deficit demands mu > 1, which scales
        that region's absorption up, depresses Phi further, and demands a larger
        mu again. That is a monotone runaway, not an oscillation, so damping and
        Anderson cannot fix it -- on the HP-MR drum arc against an S_N reference
        mu climbed past 10^3. Newton sees the coupling through the Jacobian and
        converges these cases; this is the PJFNK-SPH approach of Ortensi et al.
        (Ann. Nucl. Energy, INL/Griffin), reported to converge configurations
        with reflectors and vacuum boundaries that the Picard iteration cannot.
    """
    if isinstance(norm_weight, str) and norm_weight == "auto":
        # Automatic selection. The two normalizations are complementary and
        # neither wins everywhere -- measured on HP-MR against an S_N reference:
        #
        #     drums inserted   sum 14 pcm (885 nfev)  | production 615 pcm (2017, failed)
        #     drums withdrawn  sum 746 pcm (762, failed) | production 185 pcm (53 nfev)
        #
        # so the choice changes the *conditioning* of the nonlinear problem, not
        # just the answer. Usefully, the case that converges is also far cheaper,
        # so trying production first and falling back to sum costs little when it
        # is the wrong choice and saves a great deal when it is right. Selecting
        # on the solve's own convergence avoids having to predict from drum angle
        # or leakage fraction, neither of which separates the two cleanly.
        if region_volume is None:
            raise ValueError(
                "norm_weight='auto' needs region_volume (the third return of "
                "flux_weighted_homogenize): the production weight is "
                "nu*Sigma_f * V, and HP-MR region volumes span ~170x, so unit "
                "volumes are a materially different -- and singular -- weighting")
        nwp = production_weight(homogenized_materials, region_volume)
        first = sph_correct(homogenized_materials, region_map,
                            reference_region_flux, solve, max_iter=max_iter,
                            tol=tol, relax=relax, depth=depth, method=method,
                            norm_weight=nwp, mu0=mu0,
                            region_volume=region_volume, verbose=verbose)
        if first.converged:
            return first
        return sph_correct(homogenized_materials, region_map,
                           reference_region_flux, solve, max_iter=max_iter,
                           tol=tol, relax=relax, depth=depth, method=method,
                           norm_weight=None, mu0=mu0,
                           region_volume=region_volume, verbose=verbose)

    R = len(homogenized_materials)
    ref = np.asarray(reference_region_flux, dtype=float)
    shape = ref.shape
    # Freeze regions/groups with no reference flux (void, decoupled reflector):
    # they carry no reaction rate, so their factor stays mu = 1 and is excluded
    # from the residual and the flux normalization.
    live = (ref > ref.max() * 1e-12).reshape(-1)
    ref_flat = ref.reshape(-1)
    # Normalization. Labouré et al. (Ann. Nucl. Energy 2019) identify the SPH
    # normalization factor as the defect that breaks leakage equivalence when a
    # vacuum boundary is present: scaling each flux by its own *sum* matches
    # shape while discarding the absolute level the leakage depends on. Passing
    # norm_weight (e.g. nu*Sigma_f * V per region-group) normalizes by a
    # physical integral -- total production -- instead, which is consistent with
    # the reaction rates SPH preserves.
    absolute = isinstance(norm_weight, str) and norm_weight == "none"
    nw = (np.ones(ref.size) if norm_weight is None or absolute
          else np.asarray(norm_weight, dtype=float).reshape(-1))
    ln_ref_n = np.zeros(ref.size)
    if absolute:
        # Laboure et al. remedy (i): REMOVE the normalization. Normalization
        # factors are needed for uniqueness only on purely reflecting problems
        # (Hebert 1981); with a vacuum boundary they are unnecessary AND
        # introduce homogenization inconsistency -- and, because they leave the
        # solution non-unique, two solvers can converge to points that differ
        # (observed: hybr and jfnk 27 pcm apart, both reporting success).
        # The reference is instead rescaled ONCE by a constant (their Eq. 8) so
        # the two fluxes are comparable; that is a conditioning device, not a
        # per-iteration normalization.
        phi0, _ = solve(list(homogenized_materials))
        phi0 = np.asarray(phi0, dtype=float).reshape(-1)
        const = phi0[live].sum() / ref_flat[live].sum()
        ln_ref_n[live] = np.log(ref_flat[live] * const)
    else:
        ln_ref_n[live] = np.log(ref_flat[live] / (nw[live] * ref_flat[live]).sum())
    x = np.zeros(ref.size)                            # x = log(mu), start mu = 1
    if mu0 is not None:
        # Warm start. Production normalization fixes the leakage inconsistency
        # but makes the Newton solve harder from mu = 1: the weight is dominated
        # by the fuel regions, so low-production entries are badly scaled and the
        # line search collapses to steps of ~1e-3 (observed: |F| creeping 1% per
        # iteration toward maxiter). Starting from the converged sum-normalized
        # factors skips that stalled region.
        m0 = np.asarray(mu0, dtype=float).reshape(-1)
        ok0 = m0 > 0
        x[ok0] = np.log(m0[ok0])

    if method in ("hybr", "lm"):
        # Dense-Jacobian Newton with a trust region. The SPH system is TINY --
        # one unknown per live region-group, ~12 on HP-MR -- so the Jacobian-free
        # Krylov machinery of "jfnk" is the wrong tool: it estimates directional
        # derivatives one at a time and, when the rows are poorly scaled, stalls
        # (measured: 2017 evaluations without converging on the inserted drum
        # state under production normalization). Forming the full Jacobian costs
        # n+1 residual evaluations -- 13 coarse solves -- and MINPACK's hybrid
        # trust region then takes proper steps instead of crawling.
        from scipy.optimize import root
        state = {"k": float("nan"), "nfev": 0, "scale": None,
                 "absolute": absolute,
                 "nw": None if (norm_weight is None or absolute) else nw[live]}
        sol = root(_sph_residual, x[live], method=method,
                   args=(live, ln_ref_n, shape, homogenized_materials, solve,
                         state),
                   # MINPACK's xtol is a tolerance on the SOLUTION step, not
                   # the residual, and its default (1.5e-8) is looser than the
                   # residual tolerance we actually want. Passing tol straight
                   # through stopped early: 41 pcm where JFNK reached 14 pcm on
                   # the same problem. Tighten it well below tol.
                   options=dict(xtol=max(tol * 1e-4, 1e-14)))
        x = np.zeros(ref.size)
        x[live] = sol.x
        mu = np.exp(x).reshape(shape)
        corrected = [_scale_material(homogenized_materials[i], mu[i])
                     for i in range(R)]
        _, k = solve(corrected)
        return SphResult(corrected_materials=corrected, factors=mu, k_eff=k,
                         iterations=state["nfev"], converged=bool(sol.success))

    if method == "jfnk":
        from scipy.optimize import root
        rs = residual_scale
        if isinstance(rs, str) and rs == "auto":
            share = (nw * ref_flat)[live]
            share = np.sqrt(np.maximum(share, 0.0) / max(share.sum(), 1e-300))
            rs = share / max(share.mean(), 1e-300)
        elif rs is not None:
            rs = np.asarray(rs, dtype=float).reshape(-1)[live]
        state = {"k": float("nan"), "nfev": 0, "scale": rs,
                 "absolute": absolute,
                 "nw": None if (norm_weight is None or absolute) else nw[live]}
        sol = root(_sph_residual, x[live], method="krylov",
                   args=(live, ln_ref_n, shape, homogenized_materials, solve,
                         state),
                   options=dict(fatol=tol, maxiter=max_iter, disp=bool(verbose)))
        x = np.zeros(ref.size)
        x[live] = sol.x
        mu = np.exp(x).reshape(shape)
        corrected = [_scale_material(homogenized_materials[i], mu[i])
                     for i in range(R)]
        # Re-solve so k_eff belongs to the returned materials.
        _, k = solve(corrected)
        return SphResult(corrected_materials=corrected, factors=mu, k_eff=k,
                         iterations=state["nfev"], converged=bool(sol.success))
    if method != "anderson":
        raise ValueError(f"method must be 'anderson', 'jfnk', 'hybr' or 'lm', "
                         f"got {method!r}")

    X, F = [], []
    k = float("nan")
    converged = False
    for it in range(1, max_iter + 1):
        mu = np.exp(x).reshape(shape)
        region_flux, k = solve([_scale_material(homogenized_materials[i], mu[i])
                                for i in range(R)])
        phi = np.asarray(region_flux, dtype=float).reshape(-1)
        ln_phi_n = np.zeros(ref.size)
        ln_phi_n[live] = np.log(phi[live] if absolute
                                else phi[live] / (nw[live] * phi[live]).sum())
        f = (ln_ref_n - ln_phi_n) - x                 # residual of x = g(x)
        f[~live] = 0.0                                # frozen regions never move
        err = np.abs(np.expm1(f)).max()               # |mu_new/mu - 1|
        if err < tol:
            converged = True
            break
        X.append(x); F.append(f)
        if len(F) > depth:
            X.pop(0); F.pop(0)
        x = _anderson_step(X, F, relax)
    mu = np.exp(x).reshape(shape)
    corrected = [_scale_material(homogenized_materials[i], mu[i]) for i in range(R)]
    return SphResult(corrected_materials=corrected, factors=mu, k_eff=k,
                     iterations=it, converged=converged)



def sph_correct_monolithic(homogenized_materials, region_map,
                           reference_region_flux, k_ref, balance_residual,
                           region_avg, flux0, tol=1e-8, max_iter=100,
                           precondition=None, verbose=False) -> SphResult:
    """Monolithic PJFNK-SPH: solve for flux and SPH factors in one Newton system.

    :func:`sph_correct` treats the coarse solve as a black box, so every residual
    evaluation costs a full eigenvalue solve. Ortensi et al. instead recast the
    problem as a *source-free steady state* in which flux and SPH factors are one
    nonlinear vector, updating the factors inside each linear iteration; that is
    what buys their reported 5x (diffusion) to 10-15x (transport) speedup over
    the Picard approach.

    The recast: fix the eigenvalue at the *reference* k and let the factors make
    the coarse operator critical there. With mu unknown, the unknowns and
    equations balance exactly:

        R1(phi, mu) = [A(mu) - F(mu)/k_ref] phi          (per cell, per group)
        R2(phi, mu) = log(mu_i,g * Phi_i,g / phi_ref_i,g) (per region, per group)

    R1 alone is homogeneous in phi and would admit phi = 0; R2 pins both the
    scale and the factors, which is exactly the non-linearity the paper notes
    "allows convergence to a non-zero solution". No inner eigensolve appears.

    balance_residual : callable(materials, flux) -> array, R1 for the given
        SPH-corrected materials and flux (the caller owns geometry/BCs).
    region_avg       : callable(flux) -> (R, G) region-average flux, consistent
        with the volume split used to build reference_region_flux.
    flux0            : initial flux guess, shape (G, *cells).
    precondition     : callable(flux_block) -> approximately J^-1 applied to the
        flux block (the natural choice is the diffusion inverse at the current
        factors). This is the "P" of PJFNK and it is not optional at scale: with
        ~10^4 flux unknowns the inner Krylov stalls without it and the factors
        never leave mu = 1, whereas the black-box solve (a dozen unknowns) needs
        no preconditioner at all.
    k_ref            : the reference eigenvalue the corrected coarse operator
        must reproduce.
    """
    from scipy.optimize import newton_krylov
    try:                      # moved to the public API in newer scipy
        from scipy.optimize import NoConvergence
    except ImportError:       # pragma: no cover
        from scipy.optimize.nonlin import NoConvergence

    R = len(homogenized_materials)
    ref = np.asarray(reference_region_flux, dtype=float)
    live = (ref > ref.max() * 1e-12).reshape(-1)
    flux0 = np.asarray(asnumpy(flux0), dtype=float)
    fshape = flux0.shape
    nflux = flux0.size
    scale = float(np.abs(flux0).max()) or 1.0
    nfev = [0]

    def unpack(v):
        phi = v[:nflux].reshape(fshape) * scale
        x = np.zeros(ref.size)
        x[live] = v[nflux:]
        return phi, np.exp(x).reshape(ref.shape)

    def residual(v):
        nfev[0] += 1
        phi, mu = unpack(v)
        mats = [_scale_material(homogenized_materials[i], mu[i])
                for i in range(R)]
        r1 = np.asarray(balance_residual(mats, phi), dtype=float).ravel() / scale
        Phi = np.asarray(region_avg(phi), dtype=float).reshape(-1)
        r2 = np.zeros(ref.size)
        good = live & (Phi > 0)
        r2[good] = np.log(mu.reshape(-1)[good] * Phi[good] / ref.reshape(-1)[good])
        r2[live & ~good] = 1e3          # reject factors that kill a region
        return np.concatenate([r1, r2[live]])

    v0 = np.concatenate([flux0.ravel() / scale, np.zeros(int(live.sum()))])
    inner_M = None
    if precondition is not None:
        from scipy.sparse.linalg import LinearOperator

        def _apply(v):
            out = np.empty_like(v)
            out[:nflux] = np.asarray(
                precondition(v[:nflux].reshape(fshape)), dtype=float).ravel()
            out[nflux:] = v[nflux:]          # factors: identity block
            return out

        inner_M = LinearOperator((v0.size, v0.size), matvec=_apply, dtype=float)
    # Capture the running iterate: on NoConvergence scipy may hand back the
    # starting point, which would silently report mu = 1 as "the answer".
    best = {"x": v0.copy(), "f": np.inf}

    def _cb(x, fx):
        n = float(np.linalg.norm(np.atleast_1d(fx)))
        if n < best["f"]:
            best["f"], best["x"] = n, np.array(x, copy=True)

    converged = True
    try:
        v = newton_krylov(residual, v0, f_tol=tol, maxiter=max_iter,
                          method="lgmres", inner_M=inner_M, callback=_cb,
                          verbose=verbose)
    except NoConvergence:
        v = best["x"]
        converged = False
    phi, mu = unpack(v)
    corrected = [_scale_material(homogenized_materials[i], mu[i])
                 for i in range(R)]
    return SphResult(corrected_materials=corrected, factors=mu, k_eff=k_ref,
                     iterations=nfev[0], converged=converged)


def sph_df_correct(homogenized_materials, region_map, reference_region_flux,
                   reference_interfaces, reference_boundaries, solve,
                   tol=1e-8, max_iter=200, method="lm", bounds=1.2,
                   df_pairs=None, verbose=False):
    """SPH + discontinuity factors + boundary coefficients, solved jointly.

    SHELVED (July 2026) -- correct and verified, but not the supported path.
    On HP-MR it does not earn its cost: plain diffusion already reproduces the
    control-drum WORTH to -20 pcm (its ~800 pcm absolute error at each drum state
    cancels in the difference), better than SPH (-52) and better than every DF
    variant here (bcf-only +63, near-boundary -261, full -259). SPH's value is
    absolute k (994 -> 41 pcm), so quote that, not worth. Use plain
    :func:`sph_correct`; this is kept for future work, not deleted.

    If picked up again, the state is:
      * per-SURFACE DF works (see ndgpu.tri.face_df_from_pairs); per-region does
        not -- it is over-determined and pegs at the bounds.
      * df_pairs=[] (bcf-only) is the best and cheapest variant AND keeps the
        operator symmetric, so CG stays valid. Start there.
      * the interface variants stall at cost ~5e-2 while bcf-only reaches ~5e-5.
        That is NOT a DoF limit. The coarse partial current is reconstructed from
        diffusion's P1 closure (J_out = phi_s/4 + J/2); asking it to match an S_N
        half-range integral beside a near-black absorber demands something the
        diffusion model structurally cannot produce. Progress needs a better
        coarse partial-current reconstruction (or an SP3/SN coarse operator),
        not more factors.
      * diagnostic that found every convention bug: at unit factors the
        coarse/reference ratio must be ONE constant across all rows. Distinct
        clusters = convention bug; spread within a cluster = the physics DF is
        meant to fix.

    Equivalence conditions, following Laboure et al. (PHYSOR 2018 / Ann. Nucl.
    Energy 2019) rather than the region net-leakage condition tried earlier:

        mu_m,g  : region reaction rate      mu Phi = phi_ref
        DF      : per region-pair interface, reference PARTIAL OUTGOING current
        BCf     : per vacuum-boundary region, reference partial outgoing current

    A region's NET leakage is a sum over its faces and so does not determine the
    per-face partials -- that is why the earlier net-leakage condition left the
    factors underdetermined and they pegged at their bounds.

    solve : callable(materials, df_by_pair, bcf_by_region)
            -> (region_flux (R,G), k_eff, interfaces, boundaries)
        df_by_pair maps an ORDERED region pair to that surface's per-group
        factors and bcf_by_region maps a boundary region to its per-group
        coefficients -- the same keys as the reference dicts (see
        TriSNTransportSolver.aggregate_partial_currents,
        ndgpu.tri.face_df_from_pairs and ndgpu.tri.tri_partial_currents).

    One unknown per SURFACE, not per region. Per-region factors are
    over-determined -- a region touching n neighbours gets n reference
    conditions but one unknown -- and least squares then drives them to their
    bounds (measured on HP-MR: cost stalled at 0.678 with every drum factor
    pegged at exp(+-1.2)). Per-surface factors match GET's actual prescription
    and the count of reference conditions.

    GAUGE. The coarse solve is an EIGENproblem, so its solution is fixed only up
    to a constant -- ndgpu normalizes it to sum(fission source) = n_cells, which
    has nothing to do with the reference's normalization. Comparing partial
    currents in absolute terms therefore compares two arbitrarily-scaled
    solutions, and no factor can absorb the difference: scaling mu, DF or BCf
    changes the PHYSICS, and the solver then re-normalizes to the same gauge
    anyway. (Observed: the factors ran to their bounds and k moved thousands of
    pcm.) Every residual row here is a log ratio and the gauge multiplies the
    coarse flux and every current by one constant, so it enters each row as the
    same additive shift -- removed exactly, and least-squares-optimally, by
    subtracting the mean of the rows. What is left is the physical content: the
    coarse and reference solutions must agree up to one overall constant, which
    is all an eigenproblem can require. This is a gauge fix applied identically
    to all rows, NOT a Laboure-style per-region normalization weight (which
    varies with the factors and does break equivalence).
    """
    from scipy.optimize import least_squares

    R = len(homogenized_materials)
    ref = np.asarray(reference_region_flux, dtype=float)
    G = ref.shape[1]
    live = (ref > ref.max() * 1e-12).reshape(-1)
    bkeys = sorted(reference_boundaries)
    # Which interior surfaces get a DF unknown. df_pairs=[] is the least
    # invasive variant -- boundary coefficients only, no interior DF at all, so
    # the operator stays SYMMETRIC and CG still applies. A surface with no
    # unknown also gets no condition: imposing its reference current with
    # nothing free to meet it is what over-determines the fit.
    ikeys = (sorted(reference_interfaces) if df_pairs is None
             else [tuple(k) for k in df_pairs])
    unknown = set(ikeys) - set(reference_interfaces)
    if unknown:
        raise ValueError(f"df_pairs not in the reference interfaces: {sorted(unknown)}")
    nmu, ndf, nbc = int(live.sum()), len(ikeys) * G, len(bkeys) * G
    state = {"k": float("nan"), "n": 0, "cost": float("nan")}

    def unpack(z):
        z = np.clip(np.asarray(z, dtype=float), -bounds, bounds)
        x = np.zeros(ref.size); x[live] = z[:nmu]
        mu = np.exp(x).reshape(ref.shape)
        df = np.exp(z[nmu:nmu + ndf]).reshape(len(ikeys), G)
        bc = np.exp(z[nmu + ndf:]).reshape(len(bkeys), G)
        return mu, df, bc

    def as_dicts(df, bc):
        return ({k: df[i] for i, k in enumerate(ikeys)},
                {k: bc[i] for i, k in enumerate(bkeys)})

    def residual(z):
        state["n"] += 1
        mu, df, bc = unpack(z)
        mats = [_scale_material(homogenized_materials[i], mu[i]) for i in range(R)]
        df_d, bc_d = as_dicts(df, bc)
        phi, k, itf, bnd = solve(mats, df_d, bc_d)
        state["k"] = k
        phi = np.asarray(phi, dtype=float).reshape(-1)
        rows, bad = [], []
        ok = live & (phi > 0)
        rr = np.zeros(ref.size)
        rr[ok] = np.log(mu.reshape(-1)[ok] * phi[ok] / ref.reshape(-1)[ok])
        rows.append(rr[live])
        bad.append(~ok[live])
        for g in range(G):
            for keys, dst, src in ((ikeys, itf, reference_interfaces),
                                   (bkeys, bnd, reference_boundaries)):
                for key in keys:
                    a = dst.get(key, [0.0, 0.0])[0]
                    b = src[key][0]
                    good = a > 0 and b > 0
                    rows.append([np.log(a / b) if good else 0.0])
                    bad.append([not good])
        q = np.concatenate([np.atleast_1d(v) for v in rows])
        bad = np.concatenate([np.atleast_1d(v) for v in bad]).astype(bool)
        # Gauge fix (see docstring). Taken over the FLUX rows only, not all rows.
        # Letting the current rows help set it lets them shift every mu by a
        # common factor, and mu is not a gauge: D_SCALING = "inverse" makes
        # D ~ 1/mu, so a global mu shift changes leakage NON-proportionally and
        # therefore changes k. Reading it off the flux rows pins mu at the level
        # SPH itself would choose and leaves DF/BCf to carry the currents.
        # (All-row gauge measured: every variant converged but k sat 2900-4100
        # pcm low, systematically, with cost as small as 2e-7.)
        gauge_rows = (~bad[:nmu])
        if gauge_rows.any():
            q = q - q[:nmu][gauge_rows].mean()
        elif (~bad).any():
            q = q - q[~bad].mean()
        q[bad] = 1e2
        # DF normalization, appended AFTER the gauge shift -- it is not a
        # log-ratio of solution quantities, so including it in the mean would
        # corrupt both it and the gauge. Scaling EVERY interface factor by a
        # common c leaves the operator bit-identical (the face pair is
        # (f_L C, f_R C) with C = kf / (f_L/2D_L + f_R/2D_R), so c cancels),
        # an exact null direction distinct from the eigenvector gauge. Pin it by
        # driving the geometric mean of the DF to 1; being null, this cannot
        # compete with the physical conditions, it only restores uniqueness.
        if ndf:
            q = np.append(q, np.log(df).mean())
        state["cost"] = 0.5 * float(q @ q)
        if verbose:
            print(f"    [sph_df] nfev={state['n']:4d} cost={state['cost']:.4e} "
                  f"k={state['k']:.6f}", flush=True)
        return q

    # Start from mu = DF = BCf = 1. No scale-ratio warm start is needed (or
    # meaningful) now that the gauge is removed inside the residual.
    z0 = np.zeros(nmu + ndf + nbc)

    sol = least_squares(residual, z0, method="lm" if method == "lm" else "trf",
                        xtol=tol, ftol=tol, max_nfev=max_iter * (len(z0) + 1))
    mu, df, bc = unpack(sol.x)
    mats = [_scale_material(homogenized_materials[i], mu[i]) for i in range(R)]
    df_d, bc_d = as_dicts(df, bc)          # solve() takes dicts, not raw arrays
    _, k, _, _ = solve(mats, df_d, bc_d)
    # factors is mu alone: df is now per SURFACE, so it no longer shares mu's
    # (R, G) shape and cannot be stacked with it -- it travels in its own field.
    return SphResult(corrected_materials=mats,
                     factors=mu, k_eff=k,
                     iterations=state["n"], df=df, bcf=bc, cost=float(sol.cost),
                     converged=bool(sol.success and sol.cost < 1e-6))
