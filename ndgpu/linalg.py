"""Matrix-free preconditioned conjugate gradient.

Written against the NumPy/CuPy-common API: on GPU all vectors stay device-
resident and every operation is a CUDA kernel; the single implicit
device->host sync per iteration is the scalar convergence test.
"""

from __future__ import annotations


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
