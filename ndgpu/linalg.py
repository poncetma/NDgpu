"""Matrix-free preconditioned conjugate gradient.

Written against the NumPy/CuPy-common API: on GPU all vectors stay device-
resident and every operation is a CUDA kernel; the single implicit
device->host sync per iteration is the scalar convergence test.
"""

from __future__ import annotations


def pcg(apply_A, b, x0, inv_diag, xp, rtol=1e-6, atol=0.0, maxiter=5000):
    """Solve A x = b with Jacobi-preconditioned CG.

    apply_A  : callable, x -> A x (A symmetric positive definite)
    b        : right-hand side (any shape; treated as a flat vector)
    x0       : initial guess (warm start), not modified
    inv_diag : elementwise inverse of diag(A)

    Returns (x, n_iterations).
    """
    dot = lambda u, v: xp.sum(u * v)

    x = x0.copy()
    r = b - apply_A(x)
    stop2 = max(rtol * float(xp.sqrt(dot(b, b))), atol) ** 2
    if float(dot(r, r)) <= stop2:
        return x, 0

    z = inv_diag * r
    p = z.copy()
    rz = dot(r, z)
    for it in range(1, maxiter + 1):
        Ap = apply_A(p)
        alpha = rz / dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        if float(dot(r, r)) <= stop2:
            return x, it
        z = inv_diag * r
        rz_new = dot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    raise RuntimeError(
        f"PCG failed to converge in {maxiter} iterations "
        f"(residual {float(xp.sqrt(dot(r, r))):.3e}, target {stop2**0.5:.3e})"
    )
