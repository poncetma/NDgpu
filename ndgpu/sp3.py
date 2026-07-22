"""Symmetrized within-group SP3 / SDP1 two-moment block operator.

The dedicated two-moment block (Phi1, phi2) for the simplified-P3 and
simplified-double-P1 approximations, kept separate from the general order-N
SDPN/SPN machinery in ``spn.py`` because its 2x2 structure admits a hand-written
symmetric form that is the fast CG special case used by the SP3/SDP1 solvers and
their transient/triangular subclasses.
"""

from __future__ import annotations

from .stencil import BC_REFLECTIVE, BC_VACUUM, BC_ZERO_FLUX, GroupOperator


class SP3GroupOperator:
    """Symmetrized within-group SP3 / SDP1 block operator (``variant=``).

    The SP3 equations for one group, in the moments Phi1 = phi0 + 2*phi2 and
    phi2 (Brantley & Larsen form, isotropic sources):

        -div(D1 grad Phi1) + Sig0 * Phi1 - 2*Sig0 * phi2            = q0
        -div(D2 grad phi2) + (Sig2 + 4/5*Sig0) * phi2
                           - 2/5*Sig0 * Phi1                        = -2/5 q0

    with D1 = 1/(3*Sig1), D2 = 9/(35*Sig3), Sig0 the group removal cross
    section and Sig_l = Sigma_t - Sigma_s,l (here Sig1 uses the
    transport-corrected total via the material's D, and Sig2 = Sig3 = Sigma_t
    since l >= 2 scattering moments are not part of the data model).

    Multiplying the second equation by 5 makes the 2x2 block system symmetric
    (off-diagonal blocks both -2*Sig0) and positive definite
    (5*Sig0*Sig2 > 0), so the coupled system is solved directly by CG on
    states of shape (2, nx, ny, nz). The scalar flux is phi0 = Phi1 - 2*phi2.

    Boundary conditions: "reflective" is exact for SPN; "zero-flux" imposes
    phi = 0 on the surface for both moments (a good approximation whenever the
    physical vacuum boundary sits behind a reflector; Marshak vacuum
    conditions, which couple the moments at the boundary, are not implemented).

    The spatial discretization of both moment operators is supplied by
    ``op_cls`` (default :class:`~ndgpu.stencil.GroupOperator`, the structured
    Cartesian stencil). Passing a compatible operator -- e.g. the triangular
    ``TriGroupOperator`` -- yields SP3 on that geometry with no change to the
    angular block: the SP3 coupling only cares that ``op_cls`` exposes
    ``apply`` and ``inv_diag`` and accepts (xp, grid, D, removal, bc, active,
    mask_bc).
    """

    def __init__(self, xp, grid, D1, sigma_t, removal, bc=BC_ZERO_FLUX,
                 active=None, mask_bc=BC_VACUUM, op_cls=None, variant="sp3",
                 theta=None, hybrid_mask=None, hybrid_mask_bc=BC_REFLECTIVE,
                 hybrid_confine=False):
        self.xp = xp
        op_cls = op_cls or GroupOperator
        # Hybrid SP3/diffusion: run transport (the second moment phi2) only where
        # a per-cell mask marks it -- e.g. the control-drum absorber -- and plain
        # diffusion elsewhere. The mask zeroes phi2's *source* and the moment
        # coupling outside itself (below and in the driver's _rhs), so there the
        # block's first row is exactly -div(D1 grad Phi1) + Sig0 Phi1 = q0 with
        # phi0 = Phi1: identical to the diffusion solver. The transport
        # correction is therefore *generated* only in the masked region; Phi1
        # (the net-current-carrying moment) stays one global operator, so it is
        # continuous across the interface.
        #
        # hybrid_confine chooses what phi2 does outside the mask:
        #   False (default, "faithful"): phi2 keeps its full-domain operator, so
        #     with no source it relaxes to -div(D2 grad phi2) + Sig2 phi2 = 0 and
        #     decays smoothly out of the drum -- exactly the SP3 boundary layer,
        #     no interface closure to choose, and well-posed on the triangular
        #     mesh (no void border needed). This tracks full SP3 closely because
        #     the only physics dropped is phi2 sourcing in the near-diffusive
        #     bulk. This is the recommended method.
        #   True ("confined"): phi2's operator is excised outside the mask
        #     (pinned to exactly zero), so the scalar flux is bit-for-bit the
        #     diffusion solution there. Cheaper in principle (fewer live
        #     unknowns) but introduces an interface closure -- hybrid_mask_bc,
        #     reflective by default (zero phi2 current; the only law the tri
        #     stencil admits for an interior region without a void border) --
        #     which over/under-predicts the drum worth unless the region is
        #     extended until phi2 has decayed at its boundary.
        # hybrid_mask=None recovers the ordinary full-domain SP3 block.
        hm = None
        self.hybrid_confine = bool(hybrid_confine)
        if hybrid_mask is not None:
            hm = xp.asarray(hybrid_mask).astype(bool)
            if hm.shape != grid.shape:
                raise ValueError(f"hybrid_mask shape {hm.shape} != grid shape "
                                 f"{grid.shape}")
        if hm is not None and self.hybrid_confine:
            hi_active = hm if active is None else (
                xp.asarray(active).astype(bool) & hm)
            hi_mask_bc = hybrid_mask_bc
        else:
            hi_active, hi_mask_bc = active, mask_bc
        # Time-dependent term: a backward-Euler step adds theta = 1/(v*dt) times
        # the moments' time derivatives, with the odd-moment time derivatives
        # neglected (the standard quasi-static closure of time-SP3 kinetics).
        # Row 0 is the phi0 balance, so it carries theta*(Phi1 - 2 phi2): the
        # Phi1 diagonal *and* the phi2 coupling of row 0 both pick up theta.
        # Row 1 is NOT just theta*phi2: eliminating the odd moments substitutes
        # the balance equation into the second-moment equation, dragging in
        # -(2/5) d(phi0)/dt -- the exact row-1 time term (in this 5x-scaled row)
        # is theta*(9 phi2 - 2 Phi1). Equivalently, in the U-form the time
        # matrix is theta * sum_m c^(m) = theta*[[1,-2/3],[-2/3,1]] (identical
        # for SP3 and SDP1), which is non-diagonal but symmetric -- see the
        # sympy derivation in tests/verification/test_sdpn_derivation.py.
        # theta=None -> steady.
        m1_removal = removal if theta is None else removal + theta
        self.moment1 = op_cls(xp, grid, D1, m1_removal, bc=bc,
                              active=active, mask_bc=mask_bc)
        # The "sdp1" variant is the simplified double-P1 approximation of
        # Carreno, Vidal-Ferrandiz, Ginestar & Verdu, Ann. Nucl. Energy 207
        # (2024) 110675. Working their Eqs. (39)-(40) into these same moments
        # (Phi1 = phi0 + 2 phi2, phi2) gives a moment-1 equation *identical* to
        # SP3 and a moment-2 equation with the identical reaction
        # (Sig2 + 4/5 Sig0), coupling (2 Sig0) and RHS -- the sole difference
        # is the second-moment diffusion coefficient:
        #   "sp3"  -> D2 = 9/(35 Sigma_3)   (Brantley & Larsen simplified P3)
        #   "sdp1" -> D2 = 1/(5  Sigma_3)   (= 7/9 of the SP3 value)
        # Built from the half-range (double-PN) angular expansion, SDP1 tracks
        # discontinuous angular flux better than SP3 at the same cost, so it is
        # more accurate across strongly heterogeneous media (steep flux
        # gradients at water gaps / strong absorbers) -- see the paper's 1D BWR
        # results, where SDP1 beats SP3 for equal degrees of freedom.
        if variant == "sp3":
            D2 = 9.0 / (35.0 * sigma_t)
        elif variant == "sdp1":
            D2 = 1.0 / (5.0 * sigma_t)
        else:
            raise ValueError(f"variant must be 'sp3' or 'sdp1', got {variant!r}")
        m2_reaction = sigma_t + 0.8 * removal
        if theta is not None:
            # 5x-scaled row 1 carries 9*theta*phi2 -> 9/5 on the unscaled row.
            m2_reaction = m2_reaction + 1.8 * theta
        self.moment2 = op_cls(xp, grid, D2, m2_reaction, bc=bc,
                              active=hi_active, mask_bc=hi_mask_bc)
        # The moment-coupling term is a cell (volume) term, so it carries the
        # same metric weight as the moment operators' removal (1 on Cartesian).
        w = getattr(self.moment1, "rhs_weight", None)
        wt = 1.0 if w is None else w
        # Row-0 phi2 coupling picks up theta (the -2 theta phi2 half of
        # theta*phi0) and row-1's Phi1 coupling picks up the -2 theta Phi1 half
        # of its time term (see the theta note above): the exact time matrix is
        # symmetric, so both off-diagonal blocks are the *same* term, the
        # transient block stays symmetric (and SPD -- the scaled time matrix
        # [[1,-2],[-2,9]] is positive definite), and CG still applies.
        c = 2.0 * removal if theta is None else 2.0 * (removal + theta)
        self.coupling = c * wt
        # Hybrid: zero the moment coupling outside the phi2 subdomain, so the
        # first row degenerates to plain diffusion there (row 0 loses its phi2
        # term and row 1 becomes 5*phi2 = 0). Kept as a float field the driver
        # also uses to mask the phi2 source row of the RHS.
        self.hybrid_mask = None
        if hm is not None:
            self.hybrid_mask = hm.astype(self.coupling.dtype)
            self.coupling = self.coupling * self.hybrid_mask
        self.rhs_weight = w
        self.inv_diag = xp.stack([self.moment1.inv_diag, self.moment2.inv_diag / 5.0])

    def apply(self, u):
        """Return the block operator applied to u = (Phi1, phi2)."""
        out = self.xp.empty_like(u)
        out[0] = self.moment1.apply(u[0]) - self.coupling * u[1]
        out[1] = 5.0 * self.moment2.apply(u[1]) - self.coupling * u[0]
        return out
