"""Matrix-free preconditioned Krylov solvers (CG and restarted GMRES).

Written against the NumPy/CuPy-common API: on GPU all vectors stay device-
resident and every operation is a CUDA kernel; the implicit device->host
syncs are the scalar reductions (one per CG iteration, one per orthogonalized
basis vector for GMRES).

CG is the default everywhere -- the discretized diffusion/SP3 operators are
kept symmetric positive definite by construction (volume-weighted stencils on
cylindrical grids exist for exactly this reason). GMRES and BiCGStab are the
escape hatches for operators that cannot be symmetrized: both share pcg's
signature and stopping rule, so any solver can swap them in via
``linear_solver="gmres"`` / ``"bicgstab"``. GMRES has a monotone residual but
stores `restart` basis vectors; BiCGStab runs in constant memory with two
operator applies per step but converges irregularly and can restart on
breakdown -- the usual Krylov trade-off (BiCGStab is what CFD codes like
OpenFOAM default to for asymmetric matrices).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect

import numpy as np

from . import kernels
from .backend import asnumpy


@dataclass
class PCGWorkspace:
    """Persistent vector storage for a sequence of same-shaped PCG solves.

    Transient group solves revisit the same operator shape thousands of times.
    Allocating ``x``, ``r``, ``z``, ``p``, and ``A p`` on every visit is cheap
    on a CPU but creates allocator traffic and prevents CUDA-graph capture on a
    GPU.  A workspace owns those five arrays and may be reused after the
    previous solution is no longer needed independently.  In ndgpu's
    transient drivers there is one workspace per energy group, and the
    returned solution remains the workspace's ``x`` array between calls.

    ``operator_out`` declares that ``apply_A(v, out=array)`` is supported.
    ndgpu's Cartesian and triangular finite-volume operators provide that
    interface.  Generic user callables can still use the workspace with
    ``operator_out=False``; only their operator result remains allocationful.
    """

    x: object
    r: object
    z: object
    p: object
    ap: object
    operator_out: bool = False
    fallback_start: object | None = None
    fallback_count: int = 0
    graph_scalars: dict = field(default_factory=dict)
    graph_key: tuple | None = None
    graph: object | None = None
    graph_error: str | None = None
    graph_captures: int = 0
    graph_replays: int = 0

    @classmethod
    def like(cls, template, *, operator_out=False, fallback=False):
        xp = kernels.module_of(template)
        values = [xp.empty_like(template) for _ in range(5)]
        start = xp.zeros_like(template) if fallback else None
        return cls(*values, operator_out=bool(operator_out),
                   fallback_start=start)

    def scalars(self, xp, dtype):
        """Persistent 0-D recurrence coefficients used by CUDA graphs."""
        dtype = np.dtype(dtype)
        if not self.graph_scalars:
            self.graph_scalars = {
                name: xp.empty((), dtype=dtype)
                for name in ("rz", "rz_new", "pap", "alpha", "beta")}
        return self.graph_scalars

    def clear_graph(self):
        self.graph_key = None
        self.graph = None
        self.graph_error = None

    def validate(self, template):
        """Reject accidental reuse across incompatible shape/dtype/backends."""
        xp = kernels.module_of(template)
        for name in ("x", "r", "z", "p", "ap"):
            value = getattr(self, name)
            if kernels.module_of(value) is not xp:
                raise ValueError("PCG workspace backend does not match the solve")
            if value.shape != template.shape or value.dtype != template.dtype:
                raise ValueError(
                    f"PCG workspace {name} has shape/dtype "
                    f"{value.shape}/{value.dtype}, expected "
                    f"{template.shape}/{template.dtype}")
        if self.fallback_start is not None:
            value = self.fallback_start
            if (kernels.module_of(value) is not xp
                    or value.shape != template.shape
                    or value.dtype != template.dtype):
                raise ValueError("PCG fallback workspace does not match the solve")


@dataclass
class FGMRESWorkspace:
    """Persistent storage for repeated same-shaped flexible GMRES solves.

    The contiguous ``V`` and ``Z`` arrays make GPU Arnoldi projections a
    matrix-vector reduction instead of one host-synchronizing dot product per
    stored vector.  Allocating the maximum restart basis once also avoids
    repeatedly growing and releasing two lists of large device arrays at every
    adaptive step attempt.
    """

    x: object
    r: object
    w: object
    V: object
    Z: object
    restart: int
    operator_out: bool = False

    @classmethod
    def like(cls, template, restart=30, *, operator_out=False):
        restart = int(restart)
        if restart < 1:
            raise ValueError("restart must be positive")
        xp = kernels.module_of(template)
        shape = tuple(template.shape)
        return cls(
            xp.empty_like(template), xp.empty_like(template),
            xp.empty_like(template),
            xp.empty((restart + 1,) + shape, dtype=template.dtype),
            xp.empty((restart,) + shape, dtype=template.dtype),
            restart, bool(operator_out))

    def validate(self, template, restart):
        xp = kernels.module_of(template)
        if int(restart) > self.restart:
            raise ValueError("FGMRES workspace restart capacity is too small")
        for name in ("x", "r", "w"):
            value = getattr(self, name)
            if (kernels.module_of(value) is not xp
                    or value.shape != template.shape
                    or value.dtype != template.dtype):
                raise ValueError(f"FGMRES workspace {name} is incompatible")
        expected = tuple(template.shape)
        if (kernels.module_of(self.V) is not xp
                or self.V.shape[1:] != expected
                or self.V.dtype != template.dtype
                or kernels.module_of(self.Z) is not xp
                or self.Z.shape[1:] != expected
                or self.Z.dtype != template.dtype):
            raise ValueError("FGMRES basis workspace is incompatible")


def pcg_workspaces(operators, templates, *, fallback=False):
    """Build one compatible :class:`PCGWorkspace` per energy group."""
    if len(operators) != len(templates):
        raise ValueError("operators and templates must have the same length")
    result = []
    for operator, value in zip(operators, templates):
        try:
            supports_out = "out" in inspect.signature(operator.apply).parameters
        except (TypeError, ValueError):
            supports_out = False
        result.append(PCGWorkspace.like(
            value, operator_out=supports_out, fallback=fallback))
    return result


class MixedPrecisionPreconditioner:
    """FP32 polynomial preconditioner for an FP64 outer Krylov solve.

    ``apply_A_low`` and ``inv_diag_low`` belong to a shadow low-precision
    operator. Input residuals are cast into persistent low-precision storage;
    only the approximate correction is cast back to the outer dtype. Physical
    operator applications, recurrence coefficients, true residuals, and
    stopping decisions remain in the outer solver's dtype.
    """

    ndgpu_out = True
    ndgpu_scratch = False
    mixed_precision = True

    def __init__(self, apply_A_low, inv_diag_low, degree, outer_dtype):
        self.apply_A = apply_A_low
        self.inv_diag = inv_diag_low
        self.degree = int(degree)
        self.outer_dtype = np.dtype(outer_dtype)
        self.xp = kernels.module_of(inv_diag_low)
        self.r = self.xp.empty_like(inv_diag_low)
        self.z = self.xp.empty_like(inv_diag_low)
        self.az = self.xp.empty_like(inv_diag_low)
        try:
            self.operator_out = (
                "out" in inspect.signature(apply_A_low).parameters)
        except (TypeError, ValueError):
            self.operator_out = False

    def __call__(self, residual, out=None):
        kernels.mixed_jacobi_start(
            self.xp, residual, self.inv_diag, self.r, self.z)
        for _ in range(self.degree):
            if self.operator_out:
                az = self.apply_A(self.z, out=self.az)
            else:
                self.xp.copyto(self.az, self.apply_A(self.z))
                az = self.az
            kernels.neumann_step(
                self.xp, self.z, self.r, az, self.inv_diag)
        if out is None:
            return self.z.astype(self.outer_dtype)
        out[...] = self.z
        return out


def mixed_precision_preconditioner(apply_A_low, inv_diag_low, degree,
                                   outer_dtype=np.float64):
    """Construct an allocation-stable low-precision polynomial preconditioner."""
    low_dtype = np.dtype(inv_diag_low.dtype)
    outer_dtype = np.dtype(outer_dtype)
    if low_dtype.itemsize >= outer_dtype.itemsize:
        raise ValueError("mixed preconditioner dtype must be narrower than outer dtype")
    return MixedPrecisionPreconditioner(
        apply_A_low, inv_diag_low, degree, outer_dtype)


def neumann_preconditioner(apply_A, inv_diag, degree):
    """Truncated Neumann-series (polynomial) preconditioner.

    Approximates A^-1 by the first `degree`+1 terms of the Neumann series in
    the Jacobi splitting,  sum_k (I - D^-1 A)^k D^-1,  evaluated as `degree`
    damped-Jacobi sweeps: z <- z + D^-1 (r - A z). Symmetric positive
    definite whenever Jacobi converges (diagonally dominant A), so CG theory
    still applies. Each application costs `degree` extra operator applies --
    pure stencil streaming, no dot products -- and cuts the CG iteration
    count (and with it the per-iteration global reductions, the only
    synchronization points on the GPU). Degree 0 is plain Jacobi.
    Cf. Zhang et al., NED 316 (2017): Neumann-PCG was the fastest of the
    preconditioners compared on GPU for fine-mesh FV diffusion.
    """
    xp = kernels.module_of(inv_diag)
    try:
        operator_out = "out" in inspect.signature(apply_A).parameters
    except (TypeError, ValueError):
        operator_out = False

    def apply(r, out=None, scratch=None):
        if out is None:
            z = inv_diag * r
        else:
            xp.multiply(inv_diag, r, out=out)
            z = out
        for _ in range(degree):
            if scratch is None:
                az = apply_A(z)
            elif operator_out:
                az = apply_A(z, out=scratch)
            else:
                xp.copyto(scratch, apply_A(z))
                az = scratch
            kernels.neumann_step(xp, z, r, az, inv_diag)
        return z

    # PCG uses this opt-in instead of guessing whether an arbitrary user
    # preconditioner accepts ``out=`` (and potentially hiding a real TypeError).
    apply.ndgpu_out = True
    apply.ndgpu_scratch = True
    return apply


def _callable_owner(value):
    """Stable identity for a bound method, whose wrapper object is ephemeral."""
    return getattr(value, "__self__", value)


def _pcg_graph_capable(xp, workspace, precond, vector_size):
    """Whether one PCG iteration is allocation-free and capturable."""
    return (kernels.is_cupy(xp) and workspace is not None
            and workspace.operator_out
            and kernels.use_fused(xp, "krylov")
            and vector_size <= kernels.DOT_FUSED_MAX
            and (precond is None or getattr(precond, "ndgpu_out", False)))


def pcg(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
        precond=None, check_every=1, raise_on_fail=True, workspace=None,
        residual_replace_every=0, fallback_precond=None, graph_block=0):
    """Solve A x = b with preconditioned CG.

    apply_A  : callable, x -> A x (A symmetric positive definite)
    b        : right-hand side (any shape; treated as a flat vector)
    x0       : initial guess (warm start), not modified
    inv_diag : elementwise inverse of diag(A) (the Jacobi preconditioner)
    precond  : optional callable r -> z applying a custom SPD preconditioner
               (e.g. neumann_preconditioner); default is Jacobi via inv_diag
    check_every : test convergence only every k iterations. Each test is a
               ``float(...)`` device->host reduction (a sync); on GPU, spacing
               them out cuts the per-iteration stall when many iterations are
               expected (e.g. Jacobi-CG on a diffusion operator).
    raise_on_fail : if False, return the current iterate after ``maxiter``
               instead of raising -- for use as an *inexact* preconditioner
               (e.g. a synthetic-acceleration solve) where a partial result is
               still useful.
    workspace : optional :class:`PCGWorkspace` reused across same-shaped solves.
               Its solution array is returned directly and remains valid until
               that workspace is used again.  With ``operator_out=True`` the
               operator must accept ``apply_A(v, out=array)``.
    residual_replace_every : recompute the FP64 true residual and restart the
               recurrence every N iterations. Zero disables replacement. This
               limits recursive-residual drift when a lower-precision
               preconditioner is used.
    fallback_precond : optional safe preconditioner used to restart from zero
               when the first solve develops a non-finite checked residual or
               reaches ``maxiter``. Intended for automatic FP64 fallback from
               a mixed-precision preconditioner.
    graph_block : on CuPy, capture this many allocation-free PCG iterations
               into one CUDA graph. It must equal ``check_every`` so graph and
               uncaptured paths make convergence decisions at identical
               iterations. Unsupported configurations fall back transparently
               and record ``workspace.graph_error``.

    Returns (x, n_iterations).
    """
    # The vector updates and reductions go through ndgpu.kernels, which fuses
    # each of them into one CUDA kernel on GPU (and is the plain expression on
    # CPU). That matters here more than anywhere else: a CG iteration is three
    # reductions and three vector updates over arrays that fit in cache, so
    # written as separate array expressions it is almost entirely kernel-launch
    # overhead. The coefficients deliberately stay 0-d *device* scalars -- only
    # the convergence test (every `check_every` iterations) touches the host.
    dot = lambda u, v: kernels.dot(xp, u, v)
    M = precond if precond is not None else (lambda r: inv_diag * r)
    residual_replace_every = int(residual_replace_every)
    graph_block = int(graph_block)
    if residual_replace_every < 0:
        raise ValueError("residual_replace_every must be non-negative")
    if graph_block < 0:
        raise ValueError("graph_block must be non-negative")
    if graph_block and graph_block != check_every:
        raise ValueError("graph_block must equal check_every")

    def apply_preconditioner(source, out=None):
        if out is None:
            return M(source)
        if precond is None:
            xp.multiply(inv_diag, source, out=out)
            return out
        if getattr(precond, "ndgpu_out", False):
            kwargs = ({"scratch": ap_buf}
                      if getattr(precond, "ndgpu_scratch", False) else {})
            return precond(source, out=out, **kwargs)
        xp.copyto(out, precond(source))
        return out

    if workspace is None:
        x = x0.copy()
        r = b - apply_A(x)
        z_buf = p_buf = ap_buf = None
    else:
        workspace.validate(x0)
        x, r = workspace.x, workspace.r
        z_buf, p_buf, ap_buf = workspace.z, workspace.p, workspace.ap
        if x0 is not x:
            xp.copyto(x, x0)
        if workspace.operator_out:
            apply_A(x, out=ap_buf)
        else:
            xp.copyto(ap_buf, apply_A(x))
        xp.subtract(b, ap_buf, out=r)

    def retry_with_fallback(iterations):
        if fallback_precond is None:
            return None
        if workspace is None:
            safe_start = xp.zeros_like(x0)
        else:
            if workspace.fallback_start is None:
                workspace.fallback_start = xp.zeros_like(x0)
            safe_start = workspace.fallback_start
            workspace.fallback_count += 1
        solved, fallback_iterations = pcg(
            apply_A, b, safe_start, inv_diag, xp, rtol=rtol, atol=atol,
            maxiter=maxiter, precond=fallback_precond,
            check_every=check_every, raise_on_fail=raise_on_fail,
            workspace=workspace,
            residual_replace_every=residual_replace_every)
        return solved, iterations + fallback_iterations

    stop2 = max(rtol * float(xp.sqrt(dot(b, b))), atol) ** 2
    if float(dot(r, r)) <= stop2:
        return x, 0

    z = apply_preconditioner(r, z_buf)
    if workspace is None:
        p = z.copy()
    else:
        xp.copyto(p_buf, z)
        p = p_buf
    rz = dot(r, z)

    graph_active = bool(graph_block and _pcg_graph_capable(
        xp, workspace, precond, b.size))
    if graph_block and not graph_active and workspace is not None:
        workspace.graph_error = (
            "capture needs CuPy fused reductions, an out= operator and an "
            f"allocation-free preconditioner; vector size must be <= "
            f"{kernels.DOT_FUSED_MAX}")

    if graph_active:
        scalars = workspace.scalars(xp, b.dtype)
        kernels.dot(xp, r, z, out=scalars["rz"])
        graph_key = (id(_callable_owner(apply_A)), id(precond), graph_block,
                     tuple(b.shape), np.dtype(b.dtype).str)
        if workspace.graph_key != graph_key:
            workspace.clear_graph()
            workspace.graph_key = graph_key

        def graph_iteration():
            apply_A(p, out=ap_buf)
            kernels.dot(xp, p, ap_buf, out=scalars["pap"])
            kernels.scalar_divide(
                xp, scalars["alpha"], scalars["rz"], scalars["pap"])
            kernels.cg_update(
                xp, x, r, p, ap_buf, scalars["alpha"])
            apply_preconditioner(r, z_buf)
            kernels.dot(xp, r, z_buf, out=scalars["rz_new"])
            kernels.scalar_divide(
                xp, scalars["beta"], scalars["rz_new"], scalars["rz"])
            kernels.cg_direction(xp, p, z_buf, scalars["beta"])
            kernels.scalar_copy(xp, scalars["rz"], scalars["rz_new"])

        def direct_block(width):
            for _ in range(width):
                graph_iteration()

        iterations = 0
        # The first direct block compiles every kernel and populates CuPy's
        # memory pool without adding artificial work: it is the solve's actual
        # first block. Capture starts only after that warm block unless this
        # workspace already owns a compatible graph from an earlier solve.
        warmed = workspace.graph is not None or workspace.graph_error is not None
        while iterations < maxiter:
            width = min(graph_block, maxiter - iterations)
            if width != graph_block or not warmed:
                direct_block(width)
                warmed = True
            elif workspace.graph is not None:
                workspace.graph.launch()
                workspace.graph_replays += 1
            elif workspace.graph_error is None:
                stream = xp.cuda.Stream(non_blocking=True)
                try:
                    with stream:
                        stream.begin_capture()
                        direct_block(graph_block)
                        workspace.graph = stream.end_capture()
                    workspace.graph_captures += 1
                    workspace.graph.launch()
                    workspace.graph_replays += 1
                except Exception as exc:
                    try:
                        stream.end_capture()
                    except Exception:
                        pass
                    workspace.graph_error = f"{type(exc).__name__}: {exc}"
                    direct_block(graph_block)
            else:
                direct_block(graph_block)
            iterations += width

            replaced = (residual_replace_every > 0
                        and iterations % residual_replace_every == 0)
            if replaced:
                apply_A(x, out=ap_buf)
                xp.subtract(b, ap_buf, out=r)
                apply_preconditioner(r, z_buf)
                xp.copyto(p_buf, z_buf)
                kernels.dot(xp, r, z_buf, out=scalars["rz"])

            if replaced or iterations % check_every == 0:
                residual2 = float(dot(r, r))
                if np.isfinite(residual2) and residual2 <= stop2:
                    return x, iterations
                if not np.isfinite(residual2):
                    retried = retry_with_fallback(iterations)
                    if retried is not None:
                        return retried
                    break
        retried = retry_with_fallback(iterations)
        if retried is not None:
            return retried
        if raise_on_fail:
            raise RuntimeError(
                f"PCG failed to converge in {maxiter} iterations "
                f"(residual {float(xp.sqrt(dot(r, r))):.3e}, "
                f"target {stop2**0.5:.3e})")
        return x, iterations

    for it in range(1, maxiter + 1):
        if workspace is None:
            Ap = apply_A(p)
        elif workspace.operator_out:
            Ap = apply_A(p, out=ap_buf)
        else:
            xp.copyto(ap_buf, apply_A(p))
            Ap = ap_buf
        alpha = rz / dot(p, Ap)
        kernels.cg_update(xp, x, r, p, Ap, alpha)
        replaced = (residual_replace_every > 0
                    and it % residual_replace_every == 0)
        if replaced:
            if workspace is None:
                r = b - apply_A(x)
            elif workspace.operator_out:
                apply_A(x, out=ap_buf)
                xp.subtract(b, ap_buf, out=r)
            else:
                xp.copyto(ap_buf, apply_A(x))
                xp.subtract(b, ap_buf, out=r)
        if replaced or it % check_every == 0:
            residual2 = float(dot(r, r))
            if np.isfinite(residual2) and residual2 <= stop2:
                return x, it
            if not np.isfinite(residual2):
                retried = retry_with_fallback(it)
                if retried is not None:
                    return retried
                break
        z = apply_preconditioner(r, z_buf)
        if replaced:
            if workspace is None:
                p = z.copy()
            else:
                xp.copyto(p_buf, z)
                p = p_buf
            rz = dot(r, z)
            continue
        rz_new = dot(r, z)
        kernels.cg_direction(xp, p, z, rz_new / rz)
        rz = rz_new
    retried = retry_with_fallback(it)
    if retried is not None:
        return retried
    if raise_on_fail:
        raise RuntimeError(
            f"PCG failed to converge in {maxiter} iterations "
            f"(residual {float(xp.sqrt(dot(r, r))):.3e}, target {stop2**0.5:.3e})"
        )
    return x, maxiter


def fixed_pcg(apply_A, b, x0, inv_diag, xp, iterations=1, precond=None,
              workspace=None):
    """Apply a fixed number of PCG steps without convergence synchronization.

    This is an inexact-preconditioner primitive, not a standalone converged
    solve.  Flexible GMRES supplies the true outer residual test.  Keeping the
    PCG work fixed means every recurrence coefficient remains a zero-dimensional
    device scalar and no per-group value is transferred to the host.  Tiny
    denominator regularization makes an already-exact warm start a no-op.

    Returns ``(x, iterations)`` with the requested deterministic work count.
    """
    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("fixed PCG iterations must be non-negative")
    M = precond if precond is not None else (lambda r: inv_diag * r)

    if workspace is None:
        x = x0.copy()
        r = b - apply_A(x)
        z_buf = p_buf = ap_buf = None
    else:
        workspace.validate(x0)
        x, r = workspace.x, workspace.r
        z_buf, p_buf, ap_buf = workspace.z, workspace.p, workspace.ap
        if x0 is not x:
            xp.copyto(x, x0)
        if workspace.operator_out:
            apply_A(x, out=ap_buf)
        else:
            xp.copyto(ap_buf, apply_A(x))
        xp.subtract(b, ap_buf, out=r)
    if iterations == 0:
        return x, 0

    def apply_preconditioner(source, out=None):
        if out is None:
            return M(source)
        if precond is None:
            xp.multiply(inv_diag, source, out=out)
            return out
        if getattr(precond, "ndgpu_out", False):
            kwargs = ({"scratch": ap_buf}
                      if getattr(precond, "ndgpu_scratch", False) else {})
            return precond(source, out=out, **kwargs)
        xp.copyto(out, precond(source))
        return out

    z = apply_preconditioner(r, z_buf)
    if workspace is None:
        p = z.copy()
    else:
        xp.copyto(p_buf, z)
        p = p_buf
    rz = kernels.dot(xp, r, z)
    tiny = np.finfo(b.dtype).tiny
    for step in range(iterations):
        if workspace is None:
            ap = apply_A(p)
        elif workspace.operator_out:
            ap = apply_A(p, out=ap_buf)
        else:
            xp.copyto(ap_buf, apply_A(p))
            ap = ap_buf
        alpha = rz / (kernels.dot(xp, p, ap) + tiny)
        kernels.cg_update(xp, x, r, p, ap, alpha)
        if step + 1 == iterations:
            break
        z = apply_preconditioner(r, z_buf)
        rz_new = kernels.dot(xp, r, z)
        kernels.cg_direction(xp, p, z, rz_new / (rz + tiny))
        rz = rz_new
    return x, iterations


def gmres(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
          precond=None, restart=30, raise_on_fail=True):
    """Solve A x = b with restarted, right-preconditioned GMRES(m).

    Same interface, warm start and stopping rule as :func:`pcg` (true
    residual norm <= max(rtol * ||b||, atol)), but A need *not* be symmetric
    positive definite -- this is the option for operators that cannot be
    symmetrized (upwinded drift terms, non-symmetric acceleration schemes,
    asymmetric couplings). Right preconditioning solves A M^-1 u = b with
    x = M^-1 u, so the Arnoldi residual *is* the true residual and the
    stopping test matches pcg's exactly; the preconditioner must be linear
    (Jacobi and the Neumann polynomial both are), which lets the update be
    reassembled as x += M(V y) without storing the preconditioned basis.

    Costs relative to CG on an SPD system: `restart` basis vectors of extra
    memory and O(j) dot products at Arnoldi step j -- use CG whenever SPD
    holds. `restart` trades memory/orthogonalization work against the
    convergence penalty of discarding the Krylov space each cycle.

    raise_on_fail : if False, return the current iterate after ``maxiter``
               instead of raising -- the same escape hatch :func:`pcg` has, for
               call sites where a partial result is still useful (an inexact
               within-group solve inside an outer iteration that revisits it).

    Returns (x, n_iterations) where an iteration is one operator apply.
    """
    dot = lambda u, v: float(xp.sum(u * v))
    norm = lambda u: float(xp.sqrt(xp.sum(u * u)))
    M = precond if precond is not None else (lambda r: inv_diag * r)

    x = x0.copy()
    stop = max(rtol * norm(b), atol)
    r = b - apply_A(x)
    res = norm(r)
    if res <= stop:
        return x, 0

    it = 0
    while it < maxiter:
        m = min(restart, maxiter - it)
        V = [r / res]                       # Arnoldi basis (device arrays)
        H = np.zeros((m + 1, m))            # Hessenberg, host
        cs, sn = np.zeros(m), np.zeros(m)   # Givens rotations
        g = np.zeros(m + 1)                 # rotated residual vector
        g[0] = res
        k = 0
        for j in range(m):
            it += 1
            w = apply_A(M(V[j]))
            h = np.zeros(j + 2)
            for i in range(j + 1):          # modified Gram-Schmidt
                h[i] = dot(w, V[i])
                w = w - h[i] * V[i]
            h[j + 1] = norm(w)
            breakdown = h[j + 1] <= 1e-300  # exact solution in the space
            if not breakdown:
                V.append(w / h[j + 1])
            # Rotate the new column into upper-triangular form.
            for i in range(j):
                h[i], h[i + 1] = (cs[i] * h[i] + sn[i] * h[i + 1],
                                  -sn[i] * h[i] + cs[i] * h[i + 1])
            denom = float(np.hypot(h[j], h[j + 1]))
            cs[j], sn[j] = (1.0, 0.0) if denom == 0.0 else (h[j] / denom, h[j + 1] / denom)
            h[j], h[j + 1] = denom, 0.0
            H[:j + 2, j] = h
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            k = j + 1
            if abs(g[k]) <= stop or breakdown:
                break
        y = np.linalg.solve(H[:k, :k], g[:k])   # small upper-triangular system
        dv = float(y[0]) * V[0]
        for j in range(1, k):
            dv += float(y[j]) * V[j]
        x += M(dv)                              # M linear: M(V y) = (M V) y
        r = b - apply_A(x)
        res = norm(r)
        if res <= stop:
            return x, it
    if raise_on_fail:
        raise RuntimeError(
            f"GMRES failed to converge in {maxiter} iterations "
            f"(residual {res:.3e}, target {stop:.3e})"
        )
    return x, it


def _fgmres_workspace_solve(apply_A, b, x0, xp, M, *, rtol, atol,
                            maxiter, restart, raise_on_fail, workspace):
    """Allocation-stable FGMRES implementation used when storage is supplied."""
    workspace.validate(x0, restart)
    x, r, w = workspace.x, workspace.r, workspace.w
    V, Z = workspace.V, workspace.Z

    def apply_operator(value, out):
        if workspace.operator_out:
            return apply_A(value, out=out)
        xp.copyto(out, apply_A(value))
        return out

    def apply_preconditioner(value, out):
        if getattr(M, "ndgpu_out", False):
            return M(value, out=out)
        xp.copyto(out, M(value))
        return out

    def norm(value):
        return float(xp.sqrt(kernels.dot(xp, value, value)))

    xp.copyto(x, x0)
    stop = max(float(rtol) * norm(b), float(atol))
    apply_operator(x, w)
    xp.subtract(b, w, out=r)
    res = norm(r)
    if res <= stop:
        return x, 0

    it = 0
    while it < maxiter:
        m = min(restart, maxiter - it)
        xp.multiply(r, 1.0 / res, out=V[0])
        H = np.zeros((m + 1, m))
        cs, sn = np.zeros(m), np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = res
        k = 0
        for j in range(m):
            it += 1
            apply_preconditioner(V[j], Z[j])
            apply_operator(Z[j], w)
            if kernels.is_cupy(xp):
                # Two-pass classical Gram--Schmidt (CGS2) retains MGS-level
                # orthogonality, but both projection passes remain on device.
                basis = V[:j + 1].reshape(j + 1, -1)
                flat_w = w.reshape(-1)
                h1 = basis @ flat_w
                kernels.basis_accumulate(
                    xp, w, V, h1, j + 1, alpha=-1.0)
                h2 = basis @ flat_w
                kernels.basis_accumulate(
                    xp, w, V, h2, j + 1, alpha=-1.0)
                h_device = h1 + h2
                wnorm = xp.sqrt(kernels.dot(xp, w, w))
                packed = xp.concatenate((h_device, wnorm.reshape(1)))
                host = asnumpy(packed)  # one host synchronization per Arnoldi step
                H[:j + 1, j] = host[:-1]
                H[j + 1, j] = float(host[-1])
            else:
                # Preserve modified Gram--Schmidt on CPU, where scalar access
                # does not synchronize a device and small BLAS calls cost more.
                for i in range(j + 1):
                    H[i, j] = float(kernels.dot(xp, V[i], w))
                    kernels.axpy_inplace(xp, w, V[i], -H[i, j])
                H[j + 1, j] = norm(w)
            breakdown = H[j + 1, j] <= 1e-300
            if not breakdown:
                xp.multiply(w, 1.0 / H[j + 1, j], out=V[j + 1])
            for i in range(j):
                h0, h1_host = H[i, j], H[i + 1, j]
                H[i, j] = cs[i] * h0 + sn[i] * h1_host
                H[i + 1, j] = -sn[i] * h0 + cs[i] * h1_host
            denom = float(np.hypot(H[j, j], H[j + 1, j]))
            if denom == 0.0:
                cs[j], sn[j] = 1.0, 0.0
            else:
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom
            H[j, j], H[j + 1, j] = denom, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            k = j + 1
            if abs(g[k]) <= stop or breakdown:
                break

        y = np.linalg.solve(H[:k, :k], g[:k])
        y_device = xp.asarray(y, dtype=x.dtype)
        kernels.basis_accumulate(xp, x, Z, y_device, k)
        apply_operator(x, w)
        xp.subtract(b, w, out=r)
        res = norm(r)
        if res <= stop:
            return x, it
    if raise_on_fail:
        raise RuntimeError(
            f"FGMRES failed to converge in {it} applies "
            f"(residual {res:.3e}, target {stop:.3e})")
    return x, it


def fgmres(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0,
           maxiter=5000, precond=None, restart=30, raise_on_fail=True,
           workspace=None):
    """Solve a real non-symmetric system with flexible restarted GMRES.

    This has the same public signature and true-residual stopping rule as
    :func:`gmres`, but stores every preconditioned Arnoldi vector ``Z[j]``.
    Consequently ``precond`` may be variable, nonlinear, or an inexact
    iterative solve.  That extra basis is the essential distinction for the
    monolithic transient prototype: one application of its right
    preconditioner is an energy-group Gauss--Seidel sweep whose within-group
    PCG iteration count can change from one Arnoldi vector to the next.

    ``inv_diag`` supplies the default Jacobi preconditioner for interface
    compatibility. ``workspace`` may be an :class:`FGMRESWorkspace`; on GPU
    this also batches all Arnoldi projections into one host synchronization per
    iteration. Returns ``(x, n_iterations)``, counting operator applies.
    """
    dot = lambda u, v: float(xp.sum(u * v))
    norm = lambda u: float(xp.sqrt(xp.sum(u * u)))
    M = precond if precond is not None else (lambda r: inv_diag * r)

    restart = int(restart)
    if restart < 1:
        raise ValueError("restart must be positive")
    if workspace is not None:
        return _fgmres_workspace_solve(
            apply_A, b, x0, xp, M, rtol=rtol, atol=atol,
            maxiter=maxiter, restart=restart, raise_on_fail=raise_on_fail,
            workspace=workspace)
    x = x0.copy()
    stop = max(rtol * norm(b), atol)
    r = b - apply_A(x)
    res = norm(r)
    if res <= stop:
        return x, 0

    it = 0
    while it < maxiter:
        m = min(restart, maxiter - it)
        V = [r / res]
        Z = []
        H = np.zeros((m + 1, m))
        cs, sn = np.zeros(m), np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = res
        k = 0
        for j in range(m):
            it += 1
            zj = M(V[j])
            Z.append(zj)
            w = apply_A(zj)
            for i in range(j + 1):
                H[i, j] = dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = norm(w)
            breakdown = H[j + 1, j] <= 1e-300
            if not breakdown:
                V.append(w / H[j + 1, j])
            for i in range(j):
                h0, h1 = H[i, j], H[i + 1, j]
                H[i, j] = cs[i] * h0 + sn[i] * h1
                H[i + 1, j] = -sn[i] * h0 + cs[i] * h1
            denom = float(np.hypot(H[j, j], H[j + 1, j]))
            if denom == 0.0:
                cs[j], sn[j] = 1.0, 0.0
            else:
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom
            H[j, j], H[j + 1, j] = denom, 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            k = j + 1
            if abs(g[k]) <= stop or breakdown:
                break

        y = np.linalg.solve(H[:k, :k], g[:k])
        for j in range(k):
            x = x + float(y[j]) * Z[j]
        # A true residual at every restart protects the stopping decision from
        # loss of orthogonality and from an inexact/variable preconditioner.
        r = b - apply_A(x)
        res = norm(r)
        if res <= stop:
            return x, it
    if raise_on_fail:
        raise RuntimeError(
            f"FGMRES failed to converge in {it} applies "
            f"(residual {res:.3e}, target {stop:.3e})")
    return x, it


def cocg(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
         precond=None):
    """Solve A x = b with preconditioned COCG for *complex-symmetric* A.

    Conjugate Orthogonal CG (van der Vorst & Melissen, IEEE Trans. Magn. 26
    (1990) 706): the CG recurrence with the *bilinear* (unconjugated) inner
    product ``(u, v) = sum_j u_j v_j`` in place of the Hermitian one. It solves
    A x = b whenever A = A^T (complex symmetric) at CG's cost -- one apply and
    two short vector updates per iteration, constant memory -- which is exactly
    the frequency-domain within-group operator  -div(D grad .) + Sigma_r + i w/v
    (a real SPD stencil plus a purely imaginary diagonal shift, so A = A^T but
    not A = A^H). The convergence test still uses the *true* Euclidean norm
    ||r|| = sqrt(sum |r_j|^2); only the recurrence coefficients use the
    bilinear form. Same interface, warm start and stopping rule as :func:`pcg`
    (to which it reduces for real A), so it slots into the same call sites.

    COCG can break down if a bilinear product vanishes on a non-null vector;
    that is rare for these near-SPD operators and is reported as a failure.

    Returns (x, n_iterations).
    """
    bdot = lambda u, v: kernels.dot(xp, u, v)         # bilinear (unconjugated)
    rnorm = lambda u: float(xp.sqrt(xp.sum((u.conj() * u).real)))
    M = precond if precond is not None else (lambda r: inv_diag * r)

    x = x0.copy()
    r = b - apply_A(x)
    stop = max(rtol * rnorm(b), atol)
    if rnorm(r) <= stop:
        return x, 0

    z = M(r)
    p = z.copy()
    rz = bdot(r, z)
    for it in range(1, maxiter + 1):
        Ap = apply_A(p)
        pAp = bdot(p, Ap)
        if pAp == 0:
            raise RuntimeError("COCG breakdown (p^T A p = 0)")
        alpha = rz / pAp
        kernels.cg_update(xp, x, r, p, Ap, alpha)
        if rnorm(r) <= stop:
            return x, it
        z = M(r)
        rz_new = bdot(r, z)
        kernels.cg_direction(xp, p, z, rz_new / rz)
        rz = rz_new
    raise RuntimeError(
        f"COCG failed to converge in {maxiter} iterations "
        f"(residual {rnorm(r):.3e}, target {stop:.3e})"
    )


def bicgstab(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
             precond=None):
    """Solve A x = b with preconditioned BiCGStab (van der Vorst).

    Same interface, warm start and stopping rule as :func:`pcg`. Like
    :func:`gmres` it does not require symmetry, but it uses short recurrences:
    constant memory (7 work vectors, vs. `restart` basis vectors for GMRES)
    and two operator+preconditioner applies per iteration -- the same
    trade-off that makes it the standard asymmetric solver in CFD codes
    (e.g. OpenFOAM's PBiCGStab). The price is a non-monotone residual and
    the possibility of breakdown (rho or omega ~ 0); breakdowns are handled
    by restarting the recurrence from the current residual, and reported as
    a failure only if no progress is possible.

    Returns (x, n_iterations) where an iteration is one operator apply
    (each BiCGStab step counts 2, keeping counts comparable with pcg/gmres).
    """
    dot = lambda u, v: float(xp.sum(u * v))
    norm2 = lambda u: float(xp.sum(u * u))
    M = precond if precond is not None else (lambda r: inv_diag * r)

    x = x0.copy()
    r = b - apply_A(x)
    stop2 = max(rtol * norm2(b) ** 0.5, atol) ** 2
    if norm2(r) <= stop2:
        return x, 0

    r_hat = r.copy()                      # fixed shadow residual
    rho = alpha = omega = 1.0
    v = p = None
    restarted = False
    it = 0
    while it < 2 * maxiter:
        rho_new = dot(r_hat, r)
        if abs(rho_new) < 1e-300 or (v is not None and omega == 0.0):
            # Lanczos breakdown: restart the recurrence from where we are.
            if restarted and norm2(r) > stop2:
                break                     # twice in a row -> give up
            r_hat = r.copy()
            rho = alpha = omega = 1.0
            v = p = None
            restarted = True
            continue
        if v is None:
            p = r.copy()
        else:
            beta = (rho_new / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
        rho = rho_new
        p_hat = M(p)
        v = apply_A(p_hat)
        it += 1
        denom = dot(r_hat, v)
        if abs(denom) < 1e-300:
            r_hat = r.copy(); rho = alpha = omega = 1.0; v = p = None
            if restarted:
                break
            restarted = True
            continue
        alpha = rho / denom
        s = r - alpha * v
        if norm2(s) <= stop2:             # converged at the half step
            x += alpha * p_hat
            return x, it
        s_hat = M(s)
        t = apply_A(s_hat)
        it += 1
        tt = norm2(t)
        omega = dot(t, s) / tt if tt > 0.0 else 0.0
        x += alpha * p_hat + omega * s_hat
        r = s - omega * t
        restarted = False
        if norm2(r) <= stop2:
            return x, it
    raise RuntimeError(
        f"BiCGStab failed to converge in {it} operator applies "
        f"(residual {norm2(r)**0.5:.3e}, target {stop2**0.5:.3e})"
    )


def fgmres_c(apply_A, b, x0, xp, precond, rtol=1e-8, atol=0.0, maxiter=1000,
             restart=50):
    """Flexible, complex restarted GMRES for a general (non-symmetric) complex
    operator A with a *variable / nonlinear* right preconditioner.

    Unlike :func:`gmres` (real, linear preconditioner) this uses Hermitian inner
    products and complex Givens rotations, and -- being *flexible* -- stores the
    preconditioned basis Z_j = precond(V_j) so ``precond`` may itself be an
    iterative solve (e.g. a block Gauss-Seidel sweep whose within-group systems
    are solved by COCG). That is exactly the preconditioner the frequency-domain
    neutron-noise operator wants: the block-diagonal within-group solves are
    strong and cheap, and FGMRES/Krylov-accelerates the group (scatter/fission)
    coupling, which the unaccelerated fixed point resolves only slowly near
    criticality. ``precond`` need not be linear; the price over stored-basis
    GMRES is a second set of ``restart`` vectors (the Z_j).

    Returns (x, n_outer) with n_outer the number of A applies (Arnoldi steps).
    """
    hdot = lambda u, v: complex(xp.sum(u.conj() * v))
    nrm = lambda u: float(xp.sqrt(xp.sum((u.conj() * u).real)))

    x = x0.copy()
    stop = max(rtol * nrm(b), atol)
    r = b - apply_A(x)
    res = nrm(r)
    if res <= stop:
        return x, 0
    it = 0
    while it < maxiter:
        m = min(restart, maxiter - it)
        V = [r / res]
        Z = []
        H = np.zeros((m + 1, m), dtype=np.complex128)
        cs = np.zeros(m, dtype=np.complex128)
        sn = np.zeros(m, dtype=np.complex128)
        g = np.zeros(m + 1, dtype=np.complex128)
        g[0] = res
        k = 0
        for j in range(m):
            it += 1
            zj = precond(V[j])                  # flexible: may be iterative
            Z.append(zj)
            w = apply_A(zj)
            for i in range(j + 1):              # modified Gram-Schmidt (Hermitian)
                H[i, j] = hdot(V[i], w)
                w = w - H[i, j] * V[i]
            hjj = nrm(w)
            H[j + 1, j] = hjj
            breakdown = hjj <= 1e-300
            if not breakdown:
                V.append(w / hjj)
            for i in range(j):                  # apply stored Givens to column j
                t1, t2 = H[i, j], H[i + 1, j]
                H[i, j] = cs[i] * t1 + sn[i] * t2
                H[i + 1, j] = -np.conj(sn[i]) * t1 + cs[i] * t2
            a, bb = H[j, j], H[j + 1, j]         # complex Givens to zero H[j+1,j]
            if bb == 0:
                c, s, rr = 1.0, 0.0 + 0j, a
            elif a == 0:
                c, s, rr = 0.0, 1.0 + 0j, bb
            else:
                aa, ab = abs(a), abs(bb)
                nu = np.hypot(aa, ab)
                c, s, rr = aa / nu, (a / aa) * np.conj(bb) / nu, (a / aa) * nu
            cs[j], sn[j] = c, s
            H[j, j], H[j + 1, j] = rr, 0.0
            g[j + 1] = -np.conj(s) * g[j]
            g[j] = c * g[j]
            k = j + 1
            if abs(g[k]) <= stop or breakdown:
                break
        y = np.linalg.solve(H[:k, :k], g[:k])
        for j in range(k):
            x = x + complex(y[j]) * Z[j]
        r = b - apply_A(x)
        res = nrm(r)
        if res <= stop:
            return x, it
    raise RuntimeError(
        f"FGMRES failed to converge in {it} applies "
        f"(residual {res:.3e}, target {stop:.3e})")


#: Drop the Anderson history when the residual grows by more than this factor.
#: A fixed-point map whose evaluation is itself an iterative solve is not a
#: stationary map -- warm starts and inner tolerances make G slightly different
#: each time it is called -- so a history built across such a change can point
#: the extrapolation the wrong way. Restarting costs a few plain iterations;
#: not restarting can diverge. (Same constant, same reason, as the noise
#: solver's Gauss-Seidel sweep.)
ANDERSON_RESTART_GROWTH = 1.5


def anderson_step(X, F, beta=1.0):
    """One Anderson update from histories of iterates and residuals.

    X : list of past iterates x_k (flat arrays). F : list of residuals
    f_k = g(x_k) - x_k. Returns the next iterate: the least-squares mixture of
    the history minimizing the combined residual, damped by ``beta``.

    With a single point this is plain relaxed fixed-point ``x + beta*f`` --
    which is what makes ``depth <= 1`` directly comparable to an external
    coupling tool's constant-relaxation scheme.

    This is the unconstrained Walker-Ni form. It is deliberately NOT the same
    algorithm as ``solver._anderson_source`` / ``noise._anderson_complex``,
    which use the Type-II normal equations with a Tikhonov floor and are tuned
    for the fission-source fixed point; keeping both is intentional.
    """
    m = len(F)
    fk = F[-1]
    if m == 1:
        return X[-1] + beta * fk
    dF = np.column_stack([F[i + 1] - F[i] for i in range(m - 1)])   # (n, m-1)
    dX = np.column_stack([X[i + 1] - X[i] for i in range(m - 1)])
    gamma, *_ = np.linalg.lstsq(dF, fk, rcond=None)
    return X[-1] + beta * fk - (dX + beta * dF) @ gamma


class AndersonAccelerator:
    """Stateful Anderson acceleration of a fixed-point map, with restart.

    Feed it the current iterate and the map's output; it returns the next
    iterate. ``depth <= 1`` degenerates to under-relaxed Picard
    ``beta*g(x) + (1-beta)*x``, written in exactly that form so it matches an
    external coupling tool's constant-relaxation arithmetic bit for bit.
    """

    def __init__(self, depth=5, beta=1.0):
        self.depth = int(depth)
        self.beta = float(beta)
        self.reset()

    def reset(self):
        self._X, self._F, self._last = [], [], None

    def step(self, x, gx):
        x = np.asarray(x, dtype=float)
        gx = np.asarray(gx, dtype=float)
        shape = x.shape
        xf, gf = x.reshape(-1), gx.reshape(-1)
        if self.depth <= 1:
            # The literal relaxation form, not the algebraically identical
            # x + beta*(g - x): the two differ in the last bits, and those bits
            # propagate through an inner Krylov solve's stopping decision.
            return (self.beta * gf + (1.0 - self.beta) * xf).reshape(shape)

        f = gf - xf
        norm = float(np.linalg.norm(f))
        if self._last is not None and norm > ANDERSON_RESTART_GROWTH * self._last:
            self.reset()
        self._last = norm

        self._X.append(xf)
        self._F.append(f)
        if len(self._F) > self.depth:
            self._X.pop(0)
            self._F.pop(0)
        try:
            return anderson_step(self._X, self._F, self.beta).reshape(shape)
        except np.linalg.LinAlgError:                       # pragma: no cover
            self.reset()
            return (xf + self.beta * f).reshape(shape)


LINEAR_SOLVERS = {"cg": pcg, "cocg": cocg, "gmres": gmres,
                  "fgmres": fgmres, "bicgstab": bicgstab}


def get_linear_solver(name):
    """Resolve a named solver or a callable with :func:`pcg`'s signature."""
    if callable(name):
        return name
    try:
        return LINEAR_SOLVERS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown linear solver {name!r}; use "
            f"{sorted(LINEAR_SOLVERS)} or a callable with pcg's signature")
