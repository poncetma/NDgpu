"""Temperature feedback: how a hot core differs, neutronically, from a cold one.

This is what makes a neutronics/thermal coupling a coupling rather than a
one-way post-processing step. Without it, power heats the core and the story
ends; with it, the temperature comes back and changes the cross sections, and
the two physics have to agree with each other.

Two effects, both linearized about a reference temperature at which the cross
sections were evaluated:

**Doppler broadening.** As fuel heats, thermal motion of the nuclei broadens
the resonance absorption peaks. The resonances are self-shielded -- the flux is
depressed exactly where the cross section spikes -- so broadening exposes more
of each resonance to the flux and the *effective* absorption rises. Empirically
it goes as sqrt(T) over the range of interest, which is the classical result
for a Doppler-broadened resonance integral:

    dSigma_a,g(T) = c_D * Sigma_a,g(T_ref) * (sqrt(T) - sqrt(T_ref))

It is applied **additively to absorption**, never as a scale on the removal
cross section: removal is `sigma_a + out-scatter`, so scaling it would drag the
scattering along and quietly change the spectrum as well as the absorption.
This is why :class:`ndgpu.solver.Fields` keeps ``sigma_a`` separately.

**Density / expansion.** Heating lowers the density, so every macroscopic cross
section scales with the material's atom density while the diffusion
coefficient, being an inverse transport cross section, scales the other way:

    f(T) = 1 - beta_exp * (T - T_ref);   Sigma *= f,   D /= f

Unlike Doppler this one *is* legitimately multiplicative on removal, because it
scales absorption and the whole scattering matrix by the same factor. The
emission spectrum chi is untouched: it is a normalized probability, not a
cross section.

**An honest limitation.** The mesh does not move. So `beta_exp` here is a
density effect on a frozen geometry -- the migration area grows as 1/f^2 while
the core dimensions stay put. That is a standard and defensible reduction of a
fuel-density coefficient, but it is not thermal expansion of the core, and it
should not be reported as one.

Both coefficients are per material, so a reflector can be given a different (or
zero) response than the fuel, and they blend across a volume-mixed cell through
exactly the rules the cross sections themselves use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThermalFeedback:
    """Per-material temperature coefficients.

    t_ref      : (M,) reference temperature in K -- the temperature at which
                 each material's tabulated cross sections were evaluated. For
                 the HP-MR's Griffin library that is the ``grid_index`` node,
                 ~800 K.
    doppler    : (M,) or (M, G) coefficient c_D in K^-1/2 acting on absorption.
                 A per-group form lets the resonance range carry the effect
                 while thermal groups do not; a scalar per material applies it
                 to every group. Zero for non-fuel.
    expansion  : (M,) density coefficient beta_exp in 1/K, or None.
    doppler_updates_diffusion : also feed the added absorption into the
                 transport cross section, 1/D += 3 dSigma_a. Exact when
                 D = 1/(3 Sigma_tr), which is what ``volume_homogenize``
                 produces. Turn off for libraries carrying an independent
                 ``total``.
    """

    t_ref: np.ndarray
    doppler: np.ndarray
    expansion: np.ndarray | None = None
    doppler_updates_diffusion: bool = True

    def __post_init__(self):
        self.t_ref = np.atleast_1d(np.asarray(self.t_ref, dtype=float))
        self.doppler = np.asarray(self.doppler, dtype=float)
        if self.doppler.ndim == 1:
            self.doppler = self.doppler.reshape(-1, 1)      # broadcast over groups
        if len(self.doppler) != len(self.t_ref):
            raise ValueError("doppler must have one row per material")
        if np.any(self.t_ref <= 0):
            raise ValueError("reference temperatures must be positive (kelvin)")
        if self.expansion is not None:
            self.expansion = np.atleast_1d(np.asarray(self.expansion, dtype=float))
            if len(self.expansion) != len(self.t_ref):
                raise ValueError("expansion must have one value per material")

    @property
    def n_materials(self) -> int:
        return len(self.t_ref)

    def hook(self, temperature):
        """Build the ``xs_update`` callable for a given temperature field.

        Pass the result as ``xs_update=`` to any eigen solver. The closure
        captures the temperature, so a coupling driver makes a new one each
        iteration -- which is also when the solver (and hence its operators)
        must be rebuilt.
        """
        # Deliberately NOT np.asarray: on GPU the temperature is a CuPy array
        # and numpy refuses the implicit conversion, so forcing it here would
        # make the whole coupled path CPU-only. The array module is resolved
        # from the fields' own backend in _apply instead.
        T = temperature

        def update(fields):
            self._apply(fields, T)

        return update

    def _apply(self, fields, temperature):
        blend = fields.blend
        xp = blend.xp
        G = fields.n_groups
        # xp.asarray keeps a device array on the device and lifts a host array
        # onto it; either way the arithmetic below stays on one backend.
        T = xp.asarray(temperature, dtype=blend.dtype)
        if T.shape != blend.shape:
            raise ValueError(f"temperature shape {T.shape} != grid shape "
                             f"{blend.shape}")
        n_cols = self.doppler.shape[1]
        if n_cols not in (1, G):
            # Otherwise the group lookup below would clamp to the last column
            # and silently give every remaining group the wrong coefficient.
            raise ValueError(
                f"doppler must have 1 column (applied to every group) or one "
                f"per group ({G}), got {n_cols}")

        # Coefficients ride onto the mesh through the SAME blend the cross
        # sections used, so a drum cell that is 30% B4C is 30% B4C here too.
        t_ref = blend.linear(self.t_ref)
        # Clamp: sqrt of a negative temperature is not a physics question, and
        # a coupling iterate can transiently undershoot before it converges.
        dsqrt = xp.sqrt(xp.maximum(T, 0.0)) - xp.sqrt(t_ref)

        if self.expansion is not None:
            f = 1.0 - blend.linear(self.expansion) * (T - t_ref)
            if float(xp.min(f)) <= 0.0:
                raise ValueError(
                    "the expansion law drove a density factor to zero or "
                    "below; the linearization is only valid for "
                    "beta_exp * (T - T_ref) << 1")
        else:
            f = None

        for g in range(G):
            c_d = blend.linear(self.doppler[:, 0 if n_cols == 1 else g])
            d_sa = c_d * fields.sigma_a[g] * dsqrt

            fields.sigma_a[g] = fields.sigma_a[g] + d_sa
            fields.removal[g] = fields.removal[g] + d_sa
            fields.sigma_t[g] = fields.sigma_t[g] + d_sa
            if self.doppler_updates_diffusion:
                # 1/(1/D + 3 dSa) written as D/(1 + 3 dSa D): algebraically the
                # same, but exact when dSa = 0 (the reciprocal-of-a-reciprocal
                # form loses a bit or two there, which would make a switched-off
                # feedback measurably different from no feedback at all).
                fields.diffusion[g] = fields.diffusion[g] / (
                    1.0 + 3.0 * d_sa * fields.diffusion[g])

            if f is not None:
                fields.sigma_a[g] = fields.sigma_a[g] * f
                fields.removal[g] = fields.removal[g] * f
                fields.sigma_t[g] = fields.sigma_t[g] * f
                fields.nu_sigma_f[g] = fields.nu_sigma_f[g] * f
                fields.diffusion[g] = fields.diffusion[g] / f
                for gt in range(G):
                    if fields.sigma_s[g][gt] is not None:
                        fields.sigma_s[g][gt] = fields.sigma_s[g][gt] * f
        # chi is deliberately untouched: it is a normalized emission spectrum,
        # not a cross section, and scaling it would lose fission neutrons.


@dataclass
class TabulatedFeedback:
    """Temperature feedback by interpolating a library's own branches.

    The honest version of :class:`ThermalFeedback`. Rather than scaling one
    tabulation node by an analytic sqrt(T) law with a fitted coefficient, this
    reads the cross sections the evaluation actually produced at each
    temperature node and interpolates between them. Everything moves the way
    the data says: absorption, production, the full scattering matrix, the
    diffusion coefficient and the emission spectrum, in every group at once.

    material : index into the solver's materials list whose data varies.
    temperatures : (N,) the tabulation nodes, ascending, K.
    tables : mapping of Fields attribute -> array, ``(N, G)`` for the per-group
        quantities and ``(N, G, G)`` for ``sigma_s``.

    Linear in T between nodes and clamped outside them -- extrapolating a
    resonance-broadening curve past the evaluated range is not something the
    data supports, and a coupling iterate can transiently overshoot.
    """

    material: int
    temperatures: np.ndarray
    tables: dict
    #: Index of the node the solver's OWN material data was tabulated at.
    reference: int = 2

    def __post_init__(self):
        self.temperatures = np.asarray(self.temperatures, dtype=float)
        if self.temperatures.ndim != 1 or len(self.temperatures) < 2:
            raise ValueError("need at least two temperature nodes")
        if np.any(np.diff(self.temperatures) <= 0):
            raise ValueError("temperature nodes must be strictly ascending")
        self.tables = {k: np.asarray(v, dtype=float)
                       for k, v in self.tables.items()}
        n = len(self.temperatures)
        for k, v in self.tables.items():
            if v.shape[0] != n:
                raise ValueError(f"table {k!r} has {v.shape[0]} nodes, "
                                 f"expected {n}")

    @property
    def t_ref(self):
        """Present for interface parity with ThermalFeedback (drivers use it
        to name a reference state); the middle node."""
        return self.temperatures[len(self.temperatures) // 2:][:1]

    def hook(self, temperature):
        T = temperature

        def update(fields):
            self._apply(fields, T)

        return update

    def _apply(self, fields, temperature):
        blend = fields.blend
        xp = blend.xp
        G = fields.n_groups
        T = xp.asarray(temperature, dtype=blend.dtype)
        if T.shape != blend.shape:
            raise ValueError(f"temperature shape {T.shape} != grid shape "
                             f"{blend.shape}")

        nodes = xp.asarray(self.temperatures, dtype=blend.dtype)
        n = len(self.temperatures)
        # Bracketing interval and linear weight, per cell.
        idx = xp.clip(xp.searchsorted(nodes, T) - 1, 0, n - 2)
        lo = nodes[idx]
        hi = nodes[idx + 1]
        w = xp.clip((T - lo) / (hi - lo), 0.0, 1.0)

        # Only this material's cells take the branch data. In the HP-MR the
        # fuel is never volume-mixed (mixing is a drum-arc device), so a plain
        # index test is exact; a mixed fuel cell would need the blend applied
        # to the interpolated values instead.
        mmap = blend.mmap
        if mmap is None:
            here = xp.ones(blend.shape, dtype=bool)
        else:
            here = mmap == self.material
            if blend.mix and bool(xp.any(here & blend.active_mix)):
                raise ValueError(
                    "TabulatedFeedback material is volume-mixed in some cells; "
                    "interpolating there would ignore the blend partner")

        # Applied as a RATIO to the reference node, not as an absolute value.
        # Setting the cross sections outright would silently erase any other
        # perturbation to the same material -- a uniform absorption insertion
        # for a transient, an SPH factor, a density scaling -- because the fuel
        # is most of the core and the feedback would overwrite all of it. As a
        # ratio the feedback expresses what it actually knows ("this is how much
        # the data moves with temperature") and composes with everything else.
        ref = int(self.reference)

        def interp(table_2d):
            col = xp.asarray(table_2d, dtype=blend.dtype)
            v = (1.0 - w) * col[idx] + w * col[idx + 1]
            base = col[ref]
            return v / base if float(abs(base)) > 0.0 else v * 0.0 + 1.0

        for attr in ("sigma_a", "nu_sigma_f", "diffusion", "sigma_t"):
            tab = self.tables.get(attr)
            if tab is None:
                continue
            for g in range(G):
                fl = getattr(fields, attr)
                fl[g] = xp.where(here, fl[g] * interp(tab[:, g]), fl[g])
        # chi is a normalized spectrum, so a ratio would not preserve its sum;
        # its temperature dependence is negligible and it is left alone.

        s_tab = self.tables.get("sigma_s")
        if s_tab is not None:
            for gf in range(G):
                for gt in range(G):
                    if gf == gt:
                        continue
                    col = s_tab[:, gf, gt]
                    if not np.any(col):
                        continue
                    cur = fields.sigma_s[gf][gt]
                    if cur is None:
                        continue
                    fields.sigma_s[gf][gt] = xp.where(
                        here, cur * interp(col), cur)

        # removal is a derived quantity: sigma_a plus everything that scatters
        # OUT of the group. Recompute it wherever the data moved, or the
        # operator keeps the old group coupling.
        for g in range(G):
            out_scatter = None
            for gt in range(G):
                if gt == g:
                    continue
                s = fields.sigma_s[g][gt]
                if s is None:
                    continue
                out_scatter = s if out_scatter is None else out_scatter + s
            rem = fields.sigma_a[g] if out_scatter is None else fields.sigma_a[g] + out_scatter
            fields.removal[g] = xp.where(here, rem, fields.removal[g])


def uniform(feedback: ThermalFeedback, shape, temperature):
    """``xs_update`` for a uniform temperature -- the probe used to measure a
    coefficient without involving the thermal solver at all."""
    return feedback.hook(np.full(shape, float(temperature)))


def measure_pcm_per_K(solver_factory, shape, feedback, t_cal=800.0, dT=50.0,
                      **solve_kwargs):
    """Reactivity coefficient in pcm/K by central difference about ``t_cal``.

    solver_factory : callable ``xs_update -> solver`` building the eigen solver
                     for a given feedback hook (so this works for any geometry
                     and any angular approximation).

    Returns alpha = 1e5 * (1/k(T-dT) - 1/k(T+dT)) / (2 dT), i.e. drho/dT. A
    physical fuel coefficient is NEGATIVE: hotter fuel absorbs more, k falls.
    """
    ks = []
    for t in (t_cal - dT, t_cal + dT):
        res = solver_factory(uniform(feedback, shape, t)).solve(**solve_kwargs)
        if not res.converged:
            raise RuntimeError(f"eigen solve did not converge at T = {t} K")
        ks.append(res.k_eff)
    return 1e5 * (1.0 / ks[0] - 1.0 / ks[1]) / (2.0 * dT)


def scale_to(feedback: ThermalFeedback, factor: float) -> ThermalFeedback:
    """The same feedback with both coefficient vectors scaled.

    To first order reactivity is linear in the coefficient amplitude
    (perturbation theory), so one measurement fixes the scale needed to hit a
    target pcm/K -- which is how :func:`calibrate` works.
    """
    return ThermalFeedback(
        t_ref=feedback.t_ref.copy(),
        doppler=feedback.doppler * factor,
        expansion=None if feedback.expansion is None else feedback.expansion * factor,
        doppler_updates_diffusion=feedback.doppler_updates_diffusion)


def calibrate(solver_factory, shape, feedback, target_pcm_per_K,
              t_cal=800.0, dT=50.0, **solve_kwargs):
    """Scale a feedback's coefficients to reproduce a requested pcm/K.

    The shape of the response (which materials and groups respond, and in what
    ratio) is the caller's physics; only its amplitude is fitted. Returns
    (calibrated_feedback, measured_alpha).
    """
    if target_pcm_per_K >= 0.0:
        raise ValueError(
            "a fuel temperature coefficient must be negative; a positive one "
            "is a reactor that heats itself into a runaway, not a tuning choice")
    alpha = measure_pcm_per_K(solver_factory, shape, feedback, t_cal, dT,
                              **solve_kwargs)
    if alpha == 0.0:
        raise ValueError("the trial feedback produced no reactivity change; "
                         "check that the responding materials are the fuel")
    scaled = scale_to(feedback, target_pcm_per_K / alpha)
    return scaled, measure_pcm_per_K(solver_factory, shape, scaled, t_cal, dT,
                                     **solve_kwargs)
