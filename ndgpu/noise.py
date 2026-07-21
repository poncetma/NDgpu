"""Frequency-domain multigroup neutron noise (linearized diffusion).

Neutron noise analysis studies the small stationary fluctuations delta-phi(r, t)
that a fluctuating cross section delta-Sigma(r, t) induces on top of a critical
mean flux phi_0(r). Writing every quantity as mean + fluctuation, dropping
products of fluctuations, and Fourier transforming in time (d/dt -> i w) turns
the time-dependent diffusion + precursor equations into a *fixed-source*
problem at each angular frequency w, one complex linear solve per frequency:

    [ -div(D_g grad .) + Sigma_r,g + i w / v_g ] delta-phi_g
        - sum_{g'!=g} Sigma_s,g'->g delta-phi_g'
        - (chi_eff,g(w) / k) sum_g' nuSigma_f,g' delta-phi_g'  =  S_noise,g ,

with the frequency-dependent effective fission spectrum

    chi_eff,g(w) = chi_g - sum_i chi_d,i,g beta_i * i w / (i w + lambda_i)

(the exact analog of the transient backward-Euler weight, 1/(1+lambda dt) ->
i w/(i w+lambda): at w -> 0 the full spectrum chi_g returns -- the reactor is
neutrally stable at DC -- and at w -> inf only the prompt part (1-beta) chi_p,g
survives). The right-hand side is the *noise source*: every cross-section
fluctuation multiplying the static flux,

    S_noise,g = - delta-Sigma_r,g phi_0,g
                + sum_{g'!=g} delta-Sigma_s,g'->g phi_0,g'
                + (chi_eff,g(w) / k) sum_g' delta(nuSigma_f)_g' phi_0,g' .

This is exactly the transient within-step fixed point (see ndgpu.transient)
with the real shift 1/(v dt) replaced by the imaginary shift i w/v, the start-
of-step precursor source folded analytically into chi_eff, and phi^n/(v dt)
replaced by S_noise. The reuse is total: the within-group operator is
ndgpu.operator.GroupOperator built with a *complex* removal (its apply/inv_diag
are elementwise, so complex data flows through unchanged), which is complex
*symmetric* -- solved by COCG (ndgpu.linalg.cocg) -- and the group/fission
coupling is closed by the same Anderson-accelerated Gauss-Seidel sweep.

Validation: for a homogeneous, fully reflected (leakage-free) reactor a
spatially uniform delta-Sigma_a fluctuation drives a spatially flat response
whose amplitude follows the zero-power reactor transfer function exactly (see
:func:`zero_power_transfer_function` and tests/validation/test_noise_*), which
pins the i w/v term, the delayed-neutron feedback in chi_eff(w), and the source
construction against point-kinetics theory.

Angular approximations: diffusion (scalar flux) and the standard SPN family --
SP3/SP5/SP7 in the diffusive U-form -- share this machinery. For SPN the time
term i w/v enters the (complex-symmetric) even-moment block as theta and the
scalar flux drives fission/scatter/source through the block's src/phi0 weights;
the moment-coupled Marshak vacuum boundary is available (``marshak_vacuum``),
and reduces to diffusion's alpha=1/2 Robin term at order 0.

Scope: global kinetics data (velocities (G,), beta (I,), chi_delayed None / (G,)
/ (I, G)); the material_map and the optional mix arrays give full spatial
heterogeneity. Per-material velocities/beta (2D kinetics tables) are not yet
supported here -- they are a direct extension via the same Fields.map_table
helpers the transient solver uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .backend import asnumpy, device_name, get_backend, synchronize
from .grid import Grid
from .linalg import cocg, neumann_preconditioner
from .materials import Kinetics
from .spn import SDPNGroupOperator, _SPN_C, _SPN_G
from .stencil import BC_VACUUM, BC_ZERO_FLUX, GroupOperator
from .solver import DiffusionEigenSolver, SPNEigenSolver


def zero_power_transfer_function(omega, kinetics: Kinetics, generation_time: float):
    """Zero-power reactor transfer function G(w) = delta-P/P_0 per unit reactivity.

    The point-kinetics response of a critical reactor with no feedback,

        G(w) = 1 / [ i w ( Lambda + sum_i beta_i / (i w + lambda_i) ) ] ,

    Lambda the prompt neutron generation time. ``omega`` may be a scalar or an
    array (returns the matching complex array). This is the amplitude that a
    space-independent perturbation of reactivity delta-rho induces on the flux;
    the neutron-noise solver reproduces it exactly in the flat, leakage-free
    limit, and departs from it (space/energy dependence) as w grows.
    """
    w = np.asarray(omega, dtype=np.float64)
    beta, lam = kinetics.beta, kinetics.decay
    delayed = np.sum(beta[:, None] / (1j * w[None, :] + lam[:, None]), axis=0) \
        if w.ndim else np.sum(beta / (1j * float(w) + lam))
    return 1.0 / (1j * w * (generation_time + delayed))


@dataclass
class NoiseSource:
    """Fluctuating cross sections that drive the noise, as complex phasors at
    one frequency (amplitude and phase of each oscillating cross section).

    Every field is per energy group, each entry either None, a scalar, or a
    complex array of grid shape (real inputs are promoted). Omitted mechanisms
    contribute nothing.

    d_sigma_a    : absorption fluctuation delta-Sigma_a,g -- the classic
                   "absorber of variation" (a vibrating/oscillating absorber, a
                   fluctuating poison), length G.
    d_nu_sigma_f : fission-production fluctuation delta(nuSigma_f)_g, length G.
    d_sigma_s    : scattering-matrix fluctuation delta-Sigma_s[g_from][g_to], a
                   G x G nested list (None entries allowed). Contributes both an
                   out-scatter term to the removal fluctuation of g_from and an
                   in-scatter source into g_to.

    A moving material interface (vibrating fuel/reflector) is expressed as the
    delta-Sigma over the swept cells for each affected reaction.
    """

    d_sigma_a: list | None = None
    d_nu_sigma_f: list | None = None
    d_sigma_s: list | None = None


@dataclass
class NoiseResult:
    omega: float
    d_flux: object          # list of G complex arrays (nx, ny, nz), on device
    flux0: object           # list of G real static-flux arrays, on device
    k_eff: float
    converged: bool
    sweeps: int
    inner_iterations: int
    solve_seconds: float
    device: str
    change_history: list = field(default_factory=list)

    @property
    def frequency_hz(self) -> float:
        return self.omega / (2.0 * np.pi)

    def d_flux_numpy(self):
        return [asnumpy(d) for d in self.d_flux]

    def relative(self):
        """delta-phi_g / phi_0,g per group (complex), the fractional flux noise."""
        return [d / f0 for d, f0 in zip(self.d_flux, self.flux0)]

    def __repr__(self):
        status = "converged" if self.converged else "NOT CONVERGED"
        return (f"NoiseResult(f={self.frequency_hz:.4g} Hz, {status}, "
                f"{self.sweeps} sweeps / {self.inner_iterations} inners, "
                f"{self.solve_seconds:.2f} s on {self.device})")


def _anderson_complex(hist, raw, xp):
    """Complex Anderson (Type-II) update from (S_j, G(S_j)) pairs, latest last.

    Minimizes ||sum alpha_j f_j|| over the recent fixed-point residuals
    f_j = G(S_j) - S_j via the (Hermitian) normal equations -- the complex
    version of ndgpu.solver._anderson_source. Returns the plain iterate with
    fewer than two pairs or if the small solve is ill-posed / the coefficients
    blow up (the fixed point is unchanged either way)."""
    if len(hist) < 2:
        return raw
    res = [Gj - Sj for Sj, Gj in hist]
    dres = [res[i] - res[-1] for i in range(len(res) - 1)]
    m = len(dres)
    cd = lambda a, b: complex(xp.sum(a.conj() * b))
    A = np.array([[cd(dres[i], dres[j]) for j in range(m)] for i in range(m)],
                 dtype=np.complex128)
    b = np.array([-cd(dres[i], res[-1]) for i in range(m)], dtype=np.complex128)
    A[np.diag_indices(m)] += 1e-12 * (abs(np.trace(A)) + 1e-300)
    try:
        gamma = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return raw
    if not np.all(np.abs(gamma) < 1e4):
        return raw
    out = raw
    for j in range(m):
        out = out + complex(gamma[j]) * (hist[j][1] - hist[-1][1])
    return out


# Anderson restart safeguard, as in the transient solver: a sweep residual
# growing past this factor means the stored history no longer describes the
# fixed-point map (upscatter makes one Gauss-Seidel sweep a stateful function
# of the flux iterate), so the history is dropped.
_ANDERSON_RESTART_GROWTH = 1.5


class NoiseSolver:
    """Frequency-domain multigroup neutron-noise solver (linearized diffusion).

    Solves the static k-eigenvalue problem once to obtain the critical mean flux
    phi_0 and k_eff, then answers noise queries: for a given cross-section
    fluctuation (:class:`NoiseSource`) and angular frequency ``omega`` (rad/s),
    :meth:`solve` returns the complex flux fluctuation delta-phi_g(r).

    Parameters mirror :class:`~ndgpu.solver.DiffusionEigenSolver` /
    :class:`~ndgpu.transient.TransientSolver`:

    grid, materials, material_map : the static reactor (fissile), as for the
        eigensolver. materials may be a single Material or a list indexed by
        material_map; mix_material / mix_weight add the same optional per-cell
        two-material blend (static in frequency).
    kinetics : Kinetics -- velocities (one per group), delayed families
        (beta, decay), and optional chi_delayed. Global data only (see module
        docstring).
    angular : "diffusion" (default), "sp3", "sp5" or "sp7" -- the angular
        approximation. The SPN orders carry M = (N+1)/2 even moments in the
        diffusive U-form (the same block as :class:`~ndgpu.SPNEigenSolver`): the
        frequency-domain time term i w/v enters the block as theta (exactly as
        the transient's 1/(v dt) does) and the SPN block stays complex
        symmetric, so the same COCG within-group solve applies. The noise
        source, chi_eff(w) and scattering all act on the scalar flux phi0, so
        only the operator, the moment state and the RHS distribution change. SPN
        captures the transport effects (steep gradients, strong absorbers) that
        diffusion noise misses.
    marshak_vacuum : use the exact moment-coupled Marshak vacuum boundary on
        vacuum faces (SPN only; the coupled -n.D grad U = (g (x) I) U condition
        of the SPN family) instead of the per-moment Robin (alpha=1/2)
        approximation. Diffusion's vacuum boundary already *is* the Marshak
        condition (alpha=1/2), so the flag is inert there. Matches FEMFFUSION's
        SPN noise boundary treatment; leaves reflective/zero-flux faces
        unchanged.
    bc, device, dtype, active, mask_bc, precond_degree : as for the eigensolver.
    """

    _SPN_ORDER = {"sp3": 1, "sp5": 2, "sp7": 3}

    def __init__(self, grid: Grid, materials, material_map=None,
                 *, kinetics: Kinetics, angular: str = "diffusion",
                 marshak_vacuum: bool = False,
                 bc=BC_ZERO_FLUX, device: str = "auto",
                 dtype=np.float64, active=None, mask_bc=BC_VACUUM,
                 mix_material=None, mix_weight=None, precond_degree: int = 0):
        if kinetics.per_material:
            raise NotImplementedError(
                "NoiseSolver supports only global kinetics data; per-material "
                "velocities/beta are not yet implemented (see module docstring)")
        if angular != "diffusion" and angular not in self._SPN_ORDER:
            raise ValueError("angular must be 'diffusion' or one of "
                             f"{sorted(self._SPN_ORDER)}, got {angular!r}")
        self.angular = angular
        self._spn = angular != "diffusion"
        self._order = self._SPN_ORDER.get(angular, 0)
        self.marshak_vacuum = bool(marshak_vacuum)
        self.grid = grid
        self.kinetics = kinetics
        self.bc = bc
        self.active = active
        self.mask_bc = mask_bc
        self.precond_degree = int(precond_degree)
        self.xp = xp = get_backend(device)
        self.device = device_name(xp)
        self.dtype = np.dtype(dtype)
        self.cdtype = np.dtype(np.complex64 if self.dtype == np.float32
                               else np.complex128)

        # One static eigen-solve gives phi_0, k_eff and the cross-section
        # fields; everything downstream reuses them (the maps are static). The
        # SPN static solve runs at the same order and Marshak boundary as the
        # noise operator, so phi_0 is a consistent equilibrium.
        common = dict(
            bc=bc, device="cpu" if xp is np else "gpu", dtype=self.dtype,
            active=active, mask_bc=mask_bc,
            mix_material=mix_material, mix_weight=mix_weight)
        if self._spn:
            eig_cls = type(f"_SPN{self._order}", (SPNEigenSolver,),
                           {"_order": self._order})
            self.eig = eig_cls(grid, materials, material_map,
                               marshak_vacuum=self.marshak_vacuum, **common)
        else:
            self.eig = DiffusionEigenSolver(grid, materials, material_map, **common)
        res = self.eig.solve(tol_k=1e-9, tol_source=1e-8)
        if not res.converged:
            raise RuntimeError(f"static state did not converge: {res}")
        self.k_eff = res.k_eff
        self.fields = self.eig.fields
        self.n_groups = G = self.eig.n_groups
        if kinetics.velocities.shape[-1] != G:
            raise ValueError("kinetics.velocities must have one entry per group")
        self.flux0 = [res.flux[g].copy() for g in range(G)]
        # Per-group 1/v (scalar per group; global velocities).
        self._inv_v = [1.0 / float(kinetics.velocities[g]) for g in range(G)]
        # SPN U-form: how the scalar source spreads over moment rows and how the
        # scalar flux is recovered (constant in frequency; from the static op).
        if self._spn:
            self._src_weights = list(self.eig.ops[0].src_weights)
            self._phi0_weights = list(self.eig.ops[0].phi0_weights)

    # -- point-kinetics reference parameters --------------------------------
    def generation_time(self) -> float:
        """Prompt neutron generation time Lambda = <phi*, (1/v) phi> /
        <phi*, F phi> (fission scaled by 1/k, as in the noise operator).

        Uses the forward flux as its own weight (self-adjoint leakage+removal);
        exact for the flat, symmetric problems used to validate against point
        kinetics, and a consistent generation time for the transfer function
        otherwise."""
        xp, G = self.xp, self.n_groups
        num = sum(self._inv_v[g] * xp.sum(self.flux0[g]) for g in range(G))
        prod = self.fields.fission_source(self.flux0)
        den = xp.sum(prod) / self.k_eff
        return float(num / den)

    # -- source assembly -----------------------------------------------------
    def _as_field(self, val):
        if val is None:
            return None
        arr = self.xp.asarray(np.asarray(val), dtype=self.cdtype)
        if arr.ndim == 0:
            return arr  # scalar broadcasts against grid-shaped flux
        if arr.shape != self.grid.shape:
            raise ValueError(f"noise field shape {arr.shape} != grid "
                             f"shape {self.grid.shape}")
        return arr

    def _noise_source(self, source: NoiseSource, chi_eff):
        """Assemble S_noise,g from the cross-section fluctuations and phi_0."""
        xp, G = self.xp, self.n_groups
        phi0 = self.flux0
        da = ([self._as_field(source.d_sigma_a[g]) for g in range(G)]
              if source.d_sigma_a is not None else [None] * G)
        dnf = ([self._as_field(source.d_nu_sigma_f[g]) for g in range(G)]
               if source.d_nu_sigma_f is not None else [None] * G)
        ds = None
        if source.d_sigma_s is not None:
            ds = [[self._as_field(source.d_sigma_s[gf][gt]) for gt in range(G)]
                  for gf in range(G)]

        # delta-Sigma_r,g = delta-Sigma_a,g + sum_{g'!=g} delta-Sigma_s,g->g'.
        Sn = []
        for g in range(G):
            s = xp.zeros(self.grid.shape, dtype=self.cdtype)
            d_removal = da[g]
            if ds is not None:
                for gt in range(G):
                    if gt != g and ds[g][gt] is not None:
                        d_removal = (ds[g][gt] if d_removal is None
                                     else d_removal + ds[g][gt])
            if d_removal is not None:
                s = s - d_removal * phi0[g]
            if ds is not None:                       # in-scatter fluctuation
                for gf in range(G):
                    if gf != g and ds[gf][g] is not None:
                        s = s + ds[gf][g] * phi0[gf]
            Sn.append(s)

        # Fission-production fluctuation feeds chi_eff(w)/k, like the mean source.
        if any(d is not None for d in dnf):
            dprod = xp.zeros(self.grid.shape, dtype=self.cdtype)
            for g in range(G):
                if dnf[g] is not None:
                    dprod = dprod + dnf[g] * phi0[g]
            for g in range(G):
                Sn[g] = Sn[g] + (chi_eff[g] / self.k_eff) * dprod
        return Sn

    def _chi_eff(self, omega):
        """chi_eff,g(w) = chi_g - sum_i chi_d,i,g beta_i * i w/(i w + lambda_i),
        a per-group complex field (delayed feedback folded into the spectrum)."""
        kin = self.kinetics
        s = 1j * omega / (1j * omega + kin.decay)     # per family, complex
        chi = self.fields.chi
        cd = kin.chi_delayed
        if cd is None:                                # delayed = material chi
            fac = complex(1.0 - np.sum(kin.beta * s))
            return [chi[g] * fac for g in range(self.n_groups)]
        if cd.ndim == 1:                              # single global spectrum
            bs = complex(np.sum(kin.beta * s))
            return [chi[g] - complex(cd[g]) * bs for g in range(self.n_groups)]
        bc = kin.beta * s                             # (I,G) per-family spectra
        return [chi[g] - complex(np.dot(cd[:, g], bc))
                for g in range(self.n_groups)]

    # -- angular approximation hooks (diffusion scalar flux vs SPN moments) ---
    def _build_ops(self, xp, fields, omega, G):
        """Per-group frequency-domain within-group operators. The time term
        i w/v enters as the complex removal shift (diffusion) or the SPN block
        time parameter theta; both stay complex symmetric. The Marshak vacuum
        boundary matrix (SPN, when requested) couples the moments on vacuum
        faces."""
        if self._spn:
            bg = _SPN_G[self._order] if self.marshak_vacuum else None
            return [SDPNGroupOperator(
                xp, self.grid, fields.diffusion[g], fields.sigma_t[g],
                fields.removal[g], order=self._order, bc=self.bc,
                active=self.active, mask_bc=self.mask_bc, coeffs=_SPN_C,
                boundary_g=bg, theta=1j * omega * self._inv_v[g])
                for g in range(G)]
        return [GroupOperator(
            xp, self.grid, fields.diffusion[g],
            fields.removal[g] + 1j * omega * self._inv_v[g],
            bc=self.bc, active=self.active, mask_bc=self.mask_bc)
            for g in range(G)]

    def _zero_state(self, xp):
        """Complex zero solution state per group (scalar, or an (M, *grid)
        even-moment block for SPN)."""
        shape = ((self._order + 1,) + self.grid.shape) if self._spn else self.grid.shape
        return [xp.zeros(shape, dtype=self.cdtype) for _ in range(self.n_groups)]

    def _phi0(self, u):
        """Scalar flux of one group's state: phi0 = phi0_weights . U for SPN
        (the diffusive U-form recovers the scalar flux from the moments)."""
        if not self._spn:
            return u
        w = self._phi0_weights
        phi = w[0] * u[0]
        for j in range(1, len(w)):
            if w[j] != 0.0:
                phi = phi + w[j] * u[j]
        return phi

    def _rhs(self, xp, q0):
        """Distribute an isotropic source q0 over the group's moment rows by the
        SPN source weights (src_weights); diffusion takes q0 as is."""
        if not self._spn:
            return q0
        w = self._src_weights
        rhs = xp.empty((self._order + 1,) + self.grid.shape, dtype=self.cdtype)
        for i in range(len(w)):
            rhs[i] = w[i] * q0
        return rhs

    # -- the solve -----------------------------------------------------------
    def solve(self, source: NoiseSource, omega: float, *, tol: float = 1e-8,
              max_sweeps: int = 500, anderson_depth: int = 8,
              scatter_subsweeps: int | None = None,
              inner_rtol_floor: float = 1e-12, critical_adjust: bool = True,
              verbose: bool = False) -> NoiseResult:
        """Solve the noise problem at angular frequency ``omega`` (rad/s).

        The complex fission-source fixed point is closed by an
        Anderson-accelerated Gauss-Seidel sweep over the groups; each
        within-group system is the complex-symmetric operator solved by COCG.
        ``scatter_subsweeps`` (auto: 3 with upscatter, else 1) and the Anderson
        restart safeguard play the same roles as in the transient step.

        critical_adjust : divide the base fission production by k_eff so the
            unperturbed static state is an exact equilibrium of the noise
            operator (the physically correct critical reactor -- its noise
            operator is singular at w=0, the DC resonance of point kinetics).
            True by default, and the convention FEMFFUSION also uses (its
            make_critical() scales nu_sigma_f by 1/k_eff before assembling the
            noise operator). Set False to take the noise about the *un-adjusted*
            fundamental eigenmode (raw nu_sigma_f); the two differ by
            O(1 - 1/k_eff). The fission *perturbation* term in the source always
            carries 1/k_eff, so this flag is inert for perturbations that do not
            touch nu_sigma_f.
        """
        xp, G = self.xp, self.n_groups
        synchronize(xp)
        t0 = time.perf_counter()
        fields = self.fields
        k0 = self.k_eff
        fscale = 1.0 / k0 if critical_adjust else 1.0

        # Within-group operators carry the imaginary time shift i w/v_g (as the
        # complex removal, or the SP3 block theta) -> complex symmetric.
        ops = self._build_ops(xp, fields, omega, G)
        preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                            self.precond_degree) for op in ops]
        src_w = getattr(ops[0], "rhs_weight", None)   # cylindrical metric

        chi_eff = self._chi_eff(omega)
        Sn = self._noise_source(source, chi_eff)

        has_upscatter = any(fields.sigma_s[gf][gt] is not None
                            for gt in range(G) for gf in range(gt + 1, G))
        n_sub = (int(scatter_subsweeps) if scatter_subsweeps
                 else (3 if has_upscatter else 1))

        rnorm = lambda u: float(xp.sqrt(xp.sum((u.conj() * u).real)))
        # State per group (scalar flux, or the SP3 moment block); phi0 is the
        # scalar flux that drives fission, scatter and the source.
        u = self._zero_state(xp)
        phi = [self._phi0(u[g]) for g in range(G)]
        S = fields.fission_source(phi) * fscale       # complex fission source, =0
        change = 1.0
        change_prev = np.inf
        hist: list = []
        inner_total = 0
        converged = False
        hist_change = []

        for sweep in range(1, max_sweeps + 1):
            rtol = min(1e-6, max(1e-3 * change, 1e-3 * tol, inner_rtol_floor))
            for _ in range(n_sub):
                for g in range(G):
                    q0 = chi_eff[g] * S + Sn[g]
                    for gf in range(G):
                        s = fields.sigma_s[gf][g]
                        if gf != g and s is not None:
                            q0 = q0 + s * phi[gf]
                    if src_w is not None:
                        q0 = q0 * src_w
                    u[g], n_it = cocg(ops[g].apply, self._rhs(xp, q0), u[g],
                                      ops[g].inv_diag, xp, rtol=rtol,
                                      precond=preconds[g])
                    phi[g] = self._phi0(u[g])
                    inner_total += n_it
            G_S = fields.fission_source(phi) * fscale
            delta = G_S - S
            denom = rnorm(G_S)
            change = rnorm(delta) / denom if denom > 0 else rnorm(delta)
            hist_change.append(change)
            if verbose:
                print(f"  sweep {sweep:4d}  change = {change:.3e}")
            if change < tol:
                S = G_S
                converged = True
                break
            if change > _ANDERSON_RESTART_GROWTH * change_prev:
                hist = []
            change_prev = change
            hist.append((S, G_S))
            hist = hist[-anderson_depth:]
            S = _anderson_complex(hist, G_S, xp) if anderson_depth > 1 else G_S

        synchronize(xp)
        return NoiseResult(
            omega=float(omega), d_flux=phi, flux0=self.flux0, k_eff=k0,
            converged=converged, sweeps=sweep, inner_iterations=inner_total,
            solve_seconds=time.perf_counter() - t0, device=self.device,
            change_history=hist_change)
