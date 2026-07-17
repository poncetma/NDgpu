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

import numpy as np


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
    if degree == 0:
        return lambda r: inv_diag * r

    def apply(r):
        z = inv_diag * r
        for _ in range(degree):
            z += inv_diag * (r - apply_A(z))
        return z

    return apply


def pcg(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
        precond=None):
    """Solve A x = b with preconditioned CG.

    apply_A  : callable, x -> A x (A symmetric positive definite)
    b        : right-hand side (any shape; treated as a flat vector)
    x0       : initial guess (warm start), not modified
    inv_diag : elementwise inverse of diag(A) (the Jacobi preconditioner)
    precond  : optional callable r -> z applying a custom SPD preconditioner
               (e.g. neumann_preconditioner); default is Jacobi via inv_diag

    Returns (x, n_iterations).
    """
    dot = lambda u, v: xp.sum(u * v)
    M = precond if precond is not None else (lambda r: inv_diag * r)

    x = x0.copy()
    r = b - apply_A(x)
    stop2 = max(rtol * float(xp.sqrt(dot(b, b))), atol) ** 2
    if float(dot(r, r)) <= stop2:
        return x, 0

    z = M(r)
    p = z.copy()
    rz = dot(r, z)
    for it in range(1, maxiter + 1):
        Ap = apply_A(p)
        alpha = rz / dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        if float(dot(r, r)) <= stop2:
            return x, it
        z = M(r)
        rz_new = dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError(
        f"PCG failed to converge in {maxiter} iterations "
        f"(residual {float(xp.sqrt(dot(r, r))):.3e}, target {stop2**0.5:.3e})"
    )


def gmres(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000,
          precond=None, restart=30):
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
    raise RuntimeError(
        f"GMRES failed to converge in {maxiter} iterations "
        f"(residual {res:.3e}, target {stop:.3e})"
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


LINEAR_SOLVERS = {"cg": pcg, "gmres": gmres, "bicgstab": bicgstab}


def get_linear_solver(name):
    """Resolve a linear-solver spec: "cg", "gmres", or a pcg-signature callable."""
    if callable(name):
        return name
    try:
        return LINEAR_SOLVERS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown linear solver {name!r}; use "
            f"{sorted(LINEAR_SOLVERS)} or a callable with pcg's signature")
