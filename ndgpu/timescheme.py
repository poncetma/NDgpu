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
  BDFq                    a0 = a[q,0] Psi = H/a0, q = min(step, max_order)

For example, BDF2 has ``a = (3/2, -2, 1/2)``.  Its history term is
``H = 2 u^n - u^{n-1}/2``, hence ``Psi = H/a0 =
(4 u^n - u^{n-1})/3``.  Higher orders use the same representation, so the
spatial engines remain unaware of the time order.

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

from fractions import Fraction

import numpy as np

from . import kernels
from .backend import asnumpy


def _lin(xs, coeffs):
    """Linear combination of per-group array lists."""
    out = []
    for parts in zip(*xs):
        xp = kernels.module_of(parts[0])
        acc = xp.empty_like(parts[0])
        xp.multiply(parts[0], coeffs[0], out=acc)
        for c, p in zip(coeffs[1:], parts[1:]):
            kernels.axpy_inplace(xp, acc, p, c)
        out.append(acc)
    return out


def _error_norm_sums(corrected, predicted, rtol, atol):
    """Return defect/scale squared norms with one device-to-host transfer.

    Reductions remain as zero-dimensional device arrays while all state fields
    are visited.  Besides requiring only one GPU synchronization for the whole
    multigroup/precursor state, streaming the fields avoids retaining two more
    complete copies of that state as residual and scale lists.
    """
    xp = kernels.module_of(corrected[0])
    numerator = denominator = None
    for c, p in zip(corrected, predicted):
        residual = c - p
        n = kernels.dot(xp, residual, residual)
        if atol == 0.0:
            d = (rtol * rtol) * kernels.dot(xp, c, c)
        else:
            scale = atol + rtol * xp.abs(c)
            d = kernels.dot(xp, scale, scale)
        numerator = n if numerator is None else numerator + n
        denominator = d if denominator is None else denominator + d
    # A single two-scalar copy is the only synchronization in this routine.
    pair = asnumpy(xp.stack((numerator, denominator)))
    return float(pair[0]), float(pair[1])


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

    def prepare_step(self, step, dt):
        """Prepare a possibly variable-width step (one-step schemes: no-op)."""

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

    A-stable for w >= 1/2 but *not* L-stable at w = 1/2, so a prompt jump rings
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


# Constant-step BDF coefficients, normalized so that
#     sum_j a_j u^{n+1-j} = dt f(u^{n+1}).
# Fractions keep the defining table exact; conversion to float happens only
# when a coefficient is used with a numerical field.
_BDF_COEFFICIENTS = {
    1: (Fraction(1), Fraction(-1)),
    2: (Fraction(3, 2), Fraction(-2), Fraction(1, 2)),
    3: (Fraction(11, 6), Fraction(-3), Fraction(3, 2), Fraction(-1, 3)),
    4: (Fraction(25, 12), Fraction(-4), Fraction(3),
        Fraction(-4, 3), Fraction(1, 4)),
    5: (Fraction(137, 60), Fraction(-5), Fraction(5),
        Fraction(-10, 3), Fraction(5, 4), Fraction(-1, 5)),
    6: (Fraction(49, 20), Fraction(-6), Fraction(15, 2),
        Fraction(-20, 3), Fraction(15, 4), Fraction(-6, 5),
        Fraction(1, 6)),
}


class BDF(TimeScheme):
    """Constant-step BDF with startup order ramping from one to ``max_order``.

    Orders one and two are A-stable.  Orders three through six are only
    A(alpha)-stable, with a shrinking stability wedge, so they remain opt-in
    for reactor kinetics.  All orders are strongly damping on the negative
    real axis, but rapid control motion can excite modes outside that wedge.

    The first accepted step uses BDF1, the next BDF2, and so on until the
    requested maximum order is reached.  This avoids inventing unavailable
    pre-history and preserves an initially critical equilibrium exactly.
    """

    is_bdf = True

    def __init__(self, max_order=2):
        super().__init__()
        max_order = int(max_order)
        if max_order not in _BDF_COEFFICIENTS:
            raise ValueError("BDF max_order must be between 1 and 6")
        self.max_order = max_order
        self.order = max_order
        self.name = ("backward-euler" if max_order == 1
                     else f"bdf{max_order}")
        self._history_u = []       # [u^n, u^{n-1}, ...]
        self._dt_history = []      # [dt_n, dt_{n-1}, ...], accepted widths
        self._prepared = None
        self._automatic_order = False
        self._selected_order = max_order

    def start(self, u0):
        self._history_u = [list(u0)]
        self._dt_history = []
        self._prepared = None
        if self._automatic_order:
            self._selected_order = 1
        self._u = self._history_u[0]
        return list(u0)

    def order_at(self, step):
        # History availability, rather than the absolute step number, also
        # makes an explicit discontinuity restart drop immediately to BDF1.
        available = (len(self._history_u) if self._history_u
                     else max(int(step), 1))
        cap = self._selected_order if self._automatic_order else self.max_order
        return min(max(int(step), 1), cap, available)

    def enable_order_selection(self):
        """Let a controller choose the active order up to ``max_order``."""
        self._automatic_order = True
        self._selected_order = 1
        self._prepared = None

    def select_order(self, order):
        """Select the order used by the next prepared step."""
        order = int(order)
        if not self._automatic_order:
            raise RuntimeError("BDF automatic order selection is not enabled")
        if order < 1 or order > self.max_order:
            raise ValueError("selected BDF order must be between one and max_order")
        self._selected_order = order
        self._prepared = None

    @property
    def history_length(self):
        """Number of accepted states available to recurrence/predictors."""
        return len(self._history_u)

    def coefficients(self, step):
        if self._prepared is not None and self._prepared[0] == int(step):
            return self._prepared[2]
        return tuple(float(a) for a in _BDF_COEFFICIENTS[self.order_at(step)])

    def prepare_step(self, step, dt):
        """Construct variable-step coefficients from accepted step widths.

        With nodes measured backwards from the new time and scaled by the
        proposed width, the coefficients satisfy
        ``sum_j a_j x_j**m = delta[m, 1]`` through degree q.  This is the
        non-uniform-grid BDF formula; equal widths reproduce the exact table.
        """
        step, dt = int(step), float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("BDF step width must be finite and positive")
        q = self.order_at(step)
        if len(self._dt_history) < q - 1:
            raise ValueError("insufficient accepted step-width history")
        nodes = [0.0]
        distance = dt
        for j in range(1, q + 1):
            nodes.append(-distance / dt)
            if j < q:
                distance += self._dt_history[j - 1]
        vandermonde = np.asarray(
            [[x**m for x in nodes] for m in range(q + 1)], dtype=float)
        rhs = np.zeros(q + 1)
        rhs[1] = 1.0
        coeffs = tuple(float(a) for a in np.linalg.solve(vandermonde, rhs))
        self._prepared = (step, dt, coeffs)

    def a0(self, step):
        return self.coefficients(step)[0]

    @staticmethod
    def _combine_history(history, coeffs):
        if len(history) < len(coeffs):
            raise ValueError("insufficient BDF history for requested order")
        return _lin(history[:len(coeffs)], coeffs)

    def carried(self, step):
        a = self.coefficients(step)
        # H = sum_{j>=1} -a_j u^{n+1-j}; the engine expects Psi = H/a0.
        return self._combine_history(
            self._history_u, [-a_j / a[0] for a_j in a[1:]])

    def predict_history(self, history, dt, *, max_degree=None):
        """Extrapolate an accepted state history one step forward.

        ``history`` is ordered newest first and may contain any state with the
        same list-of-arrays representation as the flux.  This lets the
        adaptive controller apply the same polynomial predictor to precursor
        and coupled state without maintaining a second, subtly different set
        of time nodes.

        ``max_degree`` is primarily useful for candidate-order diagnostics.
        The production predictor uses the scheme's configured maximum order.
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("BDF prediction width must be finite and positive")
        history = list(history)
        degree = self.max_order if max_degree is None else int(max_degree)
        if degree < 0 or degree > self.max_order:
            raise ValueError("BDF prediction degree must be between zero and "
                             "max_order")
        n_points = min(len(history), degree + 1)
        if n_points == 0:
            raise RuntimeError("BDF predictor needs non-empty accepted history")
        if len(self._dt_history) < n_points - 1:
            raise ValueError("insufficient accepted step-width history")
        nodes = [0.0]
        distance = 0.0
        for j in range(1, n_points):
            distance += self._dt_history[j - 1]
            nodes.append(-distance)
        target = dt
        weights = []
        for j, xj in enumerate(nodes):
            w = 1.0
            for m, xm in enumerate(nodes):
                if m != j:
                    w *= (target - xm) / (xj - xm)
            weights.append(w)
        return _lin(history[:n_points], weights)

    def predict(self, dt, *, max_degree=None):
        """Polynomially extrapolate accepted flux history one step forward.

        This is the multilevel BDF predictor used to initialize nonlinear
        feedback iterations and estimate local error. Up to ``max_order + 1``
        accepted states are retained, giving a degree-``max_order`` predictor
        once startup is complete; earlier steps use every available state.
        Actual nonuniform time nodes are respected. Startup with one state is
        a constant predictor. No state is mutated, so a rejected/probed step
        is free to call it repeatedly.
        """
        if not self._history_u:
            raise RuntimeError("BDF predictor must be seeded with start()")
        return self.predict_history(
            self._history_u, dt, max_degree=max_degree)

    @staticmethod
    def error_norm(corrected, predicted, *, rtol, atol=0.0):
        """Normalized predictor--corrector defect used for step acceptance.

        This follows Eq. (44) of Cherezov et al.: the Euclidean norm of the
        state correction divided by the norm of ``atol + rtol*corrected``.
        Both arguments are iterables so multigroup fluxes, precursor families,
        and thermal fields can be evaluated independently or concatenated.
        A value no greater than one passes the local error test.
        """
        rtol, atol = float(rtol), float(atol)
        if not np.isfinite(rtol) or rtol <= 0.0:
            raise ValueError("BDF error rtol must be finite and positive")
        if not np.isfinite(atol) or atol < 0.0:
            raise ValueError("BDF error atol must be finite and non-negative")
        corrected, predicted = list(corrected), list(predicted)
        if len(corrected) != len(predicted) or not corrected:
            raise ValueError("corrected and predicted states must have equal, "
                             "non-zero lengths")
        numerator, denominator = _error_norm_sums(
            corrected, predicted, rtol, atol)
        if denominator == 0.0:
            return 0.0 if numerator == 0.0 else np.inf
        return float(np.sqrt(numerator / denominator))

    def push(self, u_new):
        self._history_u.insert(0, list(u_new))
        # The BDFq recurrence needs q accepted states, while a degree-q LTE
        # predictor needs q+1. Retain that one extra state solely for error
        # estimation; coefficients/carried() still consume at most q.
        del self._history_u[self.max_order + 1:]
        if self._prepared is not None:
            self._dt_history.insert(0, self._prepared[1])
            del self._dt_history[self.max_order:]
        self._u = self._history_u[0]

    def _history(self, C_hist, step=None):
        # Precursor callers know the step; legacy callers do not.  The number
        # of populated entries identifies the same startup order in the latter
        # case because histories are advanced exactly once per accepted step.
        q = (min(len(C_hist), self.max_order) if step is None
             else self.order_at(step))
        a = (self.coefficients(step) if step is not None else
             tuple(float(x) for x in _BDF_COEFFICIENTS[q]))
        return self._combine_history(C_hist, [-a_j for a_j in a[1:]])

    def precursor_decayed(self, step, C_hist, S_prev, beta, lam, dt):
        a0 = self.a0(step)
        H = self._history(C_hist, step)
        return [(l / (a0 + l * dt)) * h for l, h in zip(lam, H)]

    def precursor_update(self, step, C_hist, S_new, S_prev, beta, lam, dt):
        a0 = self.a0(step)
        H = self._history(C_hist, step)
        return [(h + (dt * b) * S_new) / (a0 + l * dt)
                for h, b, l in zip(H, beta, lam)]


class BDF2(BDF):
    """Backward-compatible spelling for the second-order BDF scheme."""

    def __init__(self):
        super().__init__(2)


class BDFStepController:
    """Bounded predictor--corrector step-width controller.

    The accepted-step proposal is the order-``q`` form of Cherezov et al.
    Eq. (45), with conservative growth/shrink limits. A failed error test uses
    the paper's explicit factor-two reduction. Candidate order selection
    compares the ``q-1``, ``q``, and ``q+1`` width proposals without another
    spatial solve.
    """

    def __init__(self, *, safety=1.25, floor=1e-6,
                 min_factor=0.2, max_factor=5.0,
                 rejection_strategy="half", reject_max_factor=0.5):
        safety = float(safety)
        floor = float(floor)
        min_factor, max_factor = float(min_factor), float(max_factor)
        if not np.isfinite(safety) or safety <= 1.0:
            raise ValueError("BDF controller safety must be finite and > 1")
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("BDF controller floor must be finite and >= 0")
        if not (0.0 < min_factor <= 1.0 <= max_factor):
            raise ValueError("BDF controller factors must bracket one")
        if rejection_strategy not in ("half", "error"):
            raise ValueError("BDF rejection_strategy must be 'half' or 'error'")
        reject_max_factor = float(reject_max_factor)
        if (not np.isfinite(reject_max_factor)
                or not min_factor <= reject_max_factor < 1.0):
            raise ValueError(
                "BDF reject_max_factor must be finite, >= min_factor, and < 1")
        self.safety = safety
        self.floor = floor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.rejection_strategy = rejection_strategy
        self.reject_max_factor = reject_max_factor

    def factor(self, error, order, *, accepted=True):
        """Return the multiplicative proposal ``h_new / h``."""
        error, order = float(error), int(order)
        if order < 1 or order > 6:
            raise ValueError("BDF controller order must be between 1 and 6")
        if not np.isfinite(error) or error < 0.0:
            if np.isinf(error) and error > 0.0:
                return (0.5 if self.rejection_strategy == "half"
                        else self.min_factor)
            raise ValueError("BDF controller error must be non-negative")
        if not accepted or error > 1.0:
            if self.rejection_strategy == "half":
                return 0.5
            raw = 1.0 / (self.safety
                         * (error + self.floor) ** (1.0 / (order + 1)))
            return min(max(raw, self.min_factor), self.reject_max_factor)
        raw = 1.0 / (self.safety
                     * (error + self.floor) ** (1.0 / (order + 1)))
        return min(max(raw, self.min_factor), self.max_factor)

    def propose(self, dt, error, order, *, accepted=True):
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("BDF controller step width must be positive")
        return dt * self.factor(error, order, accepted=accepted)

    def choose_order(self, dt, errors, current_order, *, hysteresis=1.05):
        """Choose the candidate with the largest safe next-step proposal.

        ``errors`` maps candidate BDF orders to normalized defects. An order
        change must improve the proposed width by ``hysteresis`` over retaining
        the current order, which suppresses one-step order chatter.
        """
        dt, current_order = float(dt), int(current_order)
        hysteresis = float(hysteresis)
        if current_order not in errors:
            raise ValueError("candidate errors must include current_order")
        if not np.isfinite(hysteresis) or hysteresis < 1.0:
            raise ValueError("BDF order hysteresis must be finite and >= 1")
        proposals = {
            int(order): self.propose(
                dt, error, int(order), accepted=float(error) <= 1.0)
            for order, error in errors.items()
        }
        best = max(proposals, key=lambda order: (proposals[order], order))
        if (best != current_order
                and proposals[best] < hysteresis * proposals[current_order]):
            best = current_order
        return best, proposals[best]


def make_time_scheme(time_scheme="backward-euler", time_weight=None):
    """Resolve the ``time_scheme`` / ``time_weight`` arguments to an object."""
    if time_weight is not None and time_weight != 1.0:
        if time_scheme not in ("backward-euler", "theta"):
            raise ValueError("time_weight applies to the theta-method only; "
                             f"got time_scheme={time_scheme!r}")
        return ThetaMethod(time_weight)
    name = str(time_scheme).lower()
    if name in ("backward-euler", "be"):
        return TimeScheme()
    if name == "bdf1":
        return BDF(1)
    if name.startswith("bdf") and name[3:].isdigit():
        return BDF(int(name[3:]))
    if name in ("crank-nicolson", "cn"):
        return ThetaMethod(0.5)
    raise ValueError(f"unknown time_scheme {time_scheme!r} (backward-euler, "
                     f"bdf2..bdf6, crank-nicolson)")
