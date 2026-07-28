"""Time-integration schemes for ndgpu's transient solvers.

Every scheme ndgpu supports can be written in one shape -- the *backward-Euler
shape* the engines already solve:

    (A + a0 theta) u^{n+1} = S^{n+1} + a0 theta Psi^n,     theta = 1/(v_g dt)

so an engine never learns which scheme is running: it is handed a collision
shift ``a0 theta`` and a carried field ``Psi``. A scheme supplies only

  * ``a0(step)``  -- the shift multiplier (may differ on the first step, where
    a multistep formula has no history yet), and
  * the rule advancing ``Psi``.

Concretely, with (1/v) du/dt = F(u) = -A u + S:

  backward Euler (BDF1)   a0 = 1      Psi = u^n
  theta-method            a0 = 1/w    Psi^{n+1} = u + ((1-w)/w)(u - Psi^n)
  BDF2                    a0 = 3/2    Psi = (4 u^n - u^{n-1})/3

The BDF2 carried field follows from theta(3/2 u^{n+1} - 2u^n + 1/2 u^{n-1}) =
F^{n+1}: the history term theta(2u^n - u^{n-1}/2) equals a0 theta Psi exactly
when Psi = (4u^n - u^{n-1})/3.

Precursors
----------
Writing any BDF as ``(a0 C^{n+1} - H)/dt = beta S^{n+1} - lambda C^{n+1}`` with
the history combination ``H = sum_{j>=1} (-a_j) C^{n+1-j}`` gives

    C^{n+1} = (dt beta S^{n+1} + H) / (a0 + lambda dt)

so the analytic substitution into the flux equation carries

    bcoef_i   = beta_i a0 / (a0 + lambda_i dt)      (end-of-step fission weight)
    decayed_i = lambda_i H_i / (a0 + lambda_i dt)   (delayed source from history)

which reduces to the familiar ``beta/(1 + lambda dt)`` at a0 = 1. Consistency
of a BDF forces ``sum_j a_j = 0``, hence ``H = a0 C`` at equilibrium, hence
``C^{n+1} = C`` exactly: the unperturbed t=0 state stays an exact equilibrium
for *any* a0, which is what keeps power evolution driven purely by the applied
perturbation. The theta-method is not of this form (it carries F^n, not past
states), so it has no matching precursor treatment and is prompt-only.
"""

from __future__ import annotations


def _lin(xs, coeffs):
    """Linear combination of per-group array lists."""
    out = []
    for parts in zip(*xs):
        acc = coeffs[0] * parts[0]
        for c, p in zip(coeffs[1:], parts[1:]):
            acc = acc + c * p
        out.append(acc)
    return out


class TimeScheme:
    """Base: backward Euler (BDF1). Stateless -- Psi is just the last solution."""

    name = "backward-euler"
    order = 1
    is_bdf = True                 # has a matching analytic precursor treatment

    def __init__(self):
        self._u = None

    def start(self, u0):
        """Seed from the (critically adjusted) steady state."""
        self._u = list(u0)
        return list(u0)

    def a0(self, step):
        return 1.0

    def carried(self, step):
        return list(self._u)

    def push(self, u_new):
        self._u = list(u_new)

    # ---- precursors: BDF form (see module docstring) ----------------------
    def _history(self, C_hist):
        """H = sum_{j>=1} (-a_j) C^{n+1-j}; C^n for BDF1."""
        return list(C_hist[0])

    def precursor_bcoef(self, step, beta, lam, dt):
        """beta_i a0/(a0 + lam_i dt) -- the end-of-step fission weight."""
        a0 = self.a0(step)
        return [b * a0 / (a0 + l * dt) for b, l in zip(beta, lam)]

    def precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt):
        """lam_i H_i/(a0 + lam_i dt) -- the delayed source from history."""
        a0 = self.a0(step)
        H = self._history(C_hist)
        return [(l / (a0 + l * dt)) * h for l, h in zip(lam, H)]

    def precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt):
        """C^{n+1} = (dt beta S^{n+1} + H)/(a0 + lam_i dt)."""
        a0 = self.a0(step)
        H = self._history(C_hist)
        return [(h + (dt * b) * S_new) / (a0 + l * dt)
                for h, b, l in zip(H, beta, lam)]


class ThetaMethod(TimeScheme):
    """theta-method with weight w on the end-of-step operator (w=1/2 is CN).

    A-stable for w <= 1/2 but *not* L-stable at w = 1/2, so a prompt jump rings
    rather than decays; opt-in, and prompt-only (no BDF-form precursors).
    """

    is_bdf = True          # has its own analytic precursor treatment (below)

    def __init__(self, w):
        super().__init__()
        if not 0.0 < w <= 1.0:
            raise ValueError(f"time_weight must be in (0, 1], got {w}")
        self.w = float(w)
        self.order = 2 if self.w == 0.5 else 1
        self.name = "crank-nicolson" if self.w == 0.5 else f"theta({self.w})"
        self._carry = (1.0 - self.w) / self.w
        self._psi = None

    def start(self, u0):
        self._psi = list(u0)
        return list(u0)

    def a0(self, step):
        return 1.0 / self.w

    def carried(self, step):
        return list(self._psi)

    def push(self, u_new):
        if self._carry == 0.0:
            self._psi = list(u_new)
        else:
            self._psi = [u + self._carry * (u - p)
                         for u, p in zip(u_new, self._psi)]

    # ---- precursors: theta-method form ------------------------------------
    # Applying the same weight to dC/dt = beta S - lambda C gives
    #     C^{n+1}(1 + w lam dt) = C^n(1 - (1-w) lam dt)
    #                             + dt beta [w S^{n+1} + (1-w) S^n]
    # so the end-of-step fission weight carries beta/(1 + w lam dt) -- the
    # backward-Euler coefficient with dt -> w dt -- and the history source
    # needs S^n as well as C^n (the BDF form needs only past C).
    # Equilibrium survives: with C = beta S0/lam the numerator collapses to
    # (beta S0/lam)(1 + w lam dt), so C^{n+1} = C exactly, and the flux-side
    # cancellation is identical to the BDF case.

    def precursor_bcoef(self, step, beta, lam, dt):
        return [b / (1.0 + self.w * l * dt) for b, l in zip(beta, lam)]

    def precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt):
        u = 1.0 - self.w
        return [(l / (1.0 + self.w * l * dt))
                * (c * (1.0 - u * l * dt) + (u * dt * b) * S_prev)
                for l, c, b in zip(lam, C_hist[0], beta)]

    def precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt):
        u = 1.0 - self.w
        return [(c * (1.0 - u * l * dt)
                 + (dt * b) * (self.w * S_new + u * S_prev))
                / (1.0 + self.w * l * dt)
                for c, l, b in zip(C_hist[0], lam, beta)]


class BDF2(TimeScheme):
    """Second-order backward differentiation. A-stable *and* L-stable.

    Unlike Crank-Nicolson it damps the prompt mode (amplification -> 0 rather
    than -> -1), which is why it is the right second-order default for stiff
    reactor kinetics. Bootstrapped with one backward-Euler step, so the first
    step is first-order accurate; take it small if that matters.
    """

    name = "bdf2"
    order = 2

    def __init__(self):
        super().__init__()
        self._prev = None          # u^{n-1}

    def start(self, u0):
        self._u = list(u0)
        self._prev = None
        return list(u0)

    def a0(self, step):
        # Keyed on the step index, NOT on whether history exists: the driver
        # advances the carried flux field and the precursors at different
        # points in the step, and a state-dependent a0 would report different
        # values to the two (solving the flux at a0=1 while updating
        # precursors at a0=3/2, which silently breaks the equilibrium).
        return 1.0 if step <= 1 else 1.5

    def carried(self, step):
        if self._prev is None:                     # bootstrap: backward Euler
            return list(self._u)
        # a0 Psi = 2 u^n - u^{n-1}/2  with a0 = 3/2
        return _lin([self._u, self._prev], [4.0 / 3.0, -1.0 / 3.0])

    def push(self, u_new):
        self._prev = self._u
        self._u = list(u_new)

    def _history(self, C_hist):
        """H = C^n (bootstrap) or 2 C^n - C^{n-1}/2 (BDF2)."""
        if len(C_hist) < 2 or C_hist[1] is None:
            return list(C_hist[0])
        return _lin([C_hist[0], C_hist[1]], [2.0, -0.5])


def make_time_scheme(time_scheme="backward-euler", time_weight=None):
    """Resolve the ``time_scheme`` / ``time_weight`` arguments to an object."""
    if time_weight is not None and time_weight != 1.0:
        if time_scheme not in ("backward-euler", "theta"):
            raise ValueError("time_weight applies to the theta-method only; "
                             f"got time_scheme={time_scheme!r}")
        return ThetaMethod(time_weight)
    name = str(time_scheme).lower()
    if name in ("backward-euler", "be", "bdf1"):
        return TimeScheme()
    if name == "bdf2":
        return BDF2()
    if name in ("crank-nicolson", "cn"):
        return ThetaMethod(0.5)
    raise ValueError(f"unknown time_scheme {time_scheme!r} (backward-euler, "
                     f"bdf2, crank-nicolson)")
