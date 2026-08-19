"""Hand-written CUDA kernels for the hot inner loops (GPU only).

Everything else in ndgpu is written against the NumPy API that CuPy mirrors, so
one code path runs on both devices (see :mod:`ndgpu.backend`). That uniformity
is the right default, but it has a cost the profiler makes obvious: CuPy
dispatches *one kernel per array operation*, and the operators here are built
from many small operations on small grids. At the sizes ndgpu runs (1e4-1e6
cells) the arithmetic of a stencil apply takes microseconds while each kernel
launch costs a few microseconds regardless of grid size -- the solvers are
**launch-bound, not bandwidth-bound**.

This module is the deliberate exception: a handful of fused kernels that collapse
those operation chains into one launch each. Every entry point takes ``xp`` and
falls back to the ordinary NumPy/CuPy expression when the backend is NumPy, when
dtypes do not match the fused kernel's assumptions, or when fusion is switched
off -- so the CPU path and the validation suite are untouched, and any fused
kernel can be A/B'd against the generic path at runtime.

Switching off:  ``NDGPU_FUSED=0`` in the environment, or :func:`set_fused` for
all of them / :func:`set_fused_group` for one family ("stencil", "krylov",
"block", "groups", "adaptive"). Those switches are what the benchmark notebooks toggle, and the
per-group one exists because "is fusion faster" turned out to be the wrong
question -- see below.

Measured on a Tesla T4 (Colab, 2026-07-29). The fused stencil wins 6.4-7.3x at
every size from 8^3 to 128^3 and 1.26-1.90x end to end -- it is a win on every
case measured. The Krylov kernels were *not*: they cost 15-18% on large 3D
solves until :data:`DOT_FUSED_MAX` was introduced, because one of the five
kernels here (the `dot` reduction) is slower than the CuPy expression it
replaces above a certain length. Both facts came out of toggling the groups
independently, which is why :func:`set_fused_group` exists. Whole-solve
speedups after thresholding: 1.3x (IAEA-3D) to 2.4x (C5G7-2D SP3).

The lesson worth keeping: "fused is faster" is not true kernel by kernel. CuPy's
generic paths are hand-tuned in places (reductions dispatch to CUB), and a
custom kernel only wins where it is saving launches or traffic that CuPy cannot.

That run predates the order-N block kernels and the triangular stencil, which
are written and CPU-verified but not yet timed.
"""

from __future__ import annotations

import os

import numpy as np

_ENABLED = os.environ.get("NDGPU_FUSED", "1").lower() not in ("0", "false", "no")

#: Per-group switches, so a benchmark can attribute a change to one family of
#: kernels rather than to "fusion" as a whole. That distinction mattered: the
#: first T4 measurements showed the stencil winning 6-7x while large solves got
#: *slower*, which is only diagnosable by toggling the groups independently.
_GROUPS = {"stencil": True, "krylov": True, "block": True,
           "groups": True, "adaptive": True}


def set_fused(flag: bool) -> bool:
    """Enable/disable all fused kernels; returns the previous master state.

    Operators cache their fused buffers lazily, so toggling this mid-run is
    safe: the next apply simply takes the other branch.
    """
    global _ENABLED
    prev = _ENABLED
    _ENABLED = bool(flag)
    return prev


def set_fused_group(name: str, flag: bool) -> bool:
    """Enable/disable one named fused-kernel family."""
    if name not in _GROUPS:
        raise ValueError(f"unknown kernel group {name!r}; use {sorted(_GROUPS)}")
    prev = _GROUPS[name]
    _GROUPS[name] = bool(flag)
    return prev


def fused_enabled() -> bool:
    return _ENABLED


def is_cupy(xp) -> bool:
    """True if xp is the CuPy module (cheap; no import of cupy if absent)."""
    return xp is not np and getattr(xp, "__name__", "") == "cupy"


def use_fused(xp, group: str | None = None) -> bool:
    return _ENABLED and (group is None or _GROUPS[group]) and is_cupy(xp)


def module_of(a):
    """The array module (numpy or cupy) that owns `a`."""
    if hasattr(a, "__cuda_array_interface__"):
        import cupy

        return cupy
    # NumPy arithmetic on a 0-d array may return a NumPy scalar.  Treat it as
    # host data too instead of importing CuPy merely to classify it.
    if isinstance(a, (np.ndarray, np.generic)) or np.isscalar(a):
        return np
    import cupy

    return cupy


def _scalar_arg(a):
    """A 1-element view of a 0-d device scalar (raw kernel params want ndim>=1)."""
    return a.reshape(1) if getattr(a, "ndim", 1) == 0 else a


# --------------------------------------------------------------------------
# 7-point stencil
# --------------------------------------------------------------------------
# The generic apply is  out = diag*phi  followed by six shifted multiply-adds,
# i.e. 13 kernels and 7 full-size temporaries. Fused it is one kernel, one
# allocation, and one pass over phi with the seven stencil points served from
# cache instead of seven separate streaming reads.
#
# Index arithmetic: phi/diag/out are C-contiguous (nx, ny, nz), so the flat
# index is i = ii*ny*nz + j*nz + k. The face-coupling arrays are one cell short
# on their own axis -- wx is (nx-1, ny, nz), wy is (nx, ny-1, nz), wz is
# (nx, ny, nz-1) -- so each needs its own stride expression. w?[..] is the
# coupling of the face on the *low* side of the named cell index.
_STENCIL7_SRC = """
    const int k  = i % nz;
    const int j  = (i / nz) % ny;
    const int ii = i / (ny * nz);
    const int yz = ny * nz;
    T v = diag[i] * phi[i];
    if (ii > 0)      v -= wx[i - yz] * phi[i - yz];
    if (ii < nx - 1) v -= wx[i]      * phi[i + yz];
    const int wyb = ii * (ny - 1) * nz + k;
    if (j > 0)       v -= wy[wyb + (j - 1) * nz] * phi[i - nz];
    if (j < ny - 1)  v -= wy[wyb + j * nz]       * phi[i + nz];
    const int wzb = ii * ny * (nz - 1) + j * (nz - 1);
    if (k > 0)       v -= wz[wzb + k - 1] * phi[i - 1];
    if (k < nz - 1)  v -= wz[wzb + k]     * phi[i + 1];
    out = v;
"""

_kernels: dict = {}


def _stencil7(scaled: bool):
    """The fused stencil kernel; `scaled` folds in the cylindrical row scaling."""
    key = ("stencil7", scaled)
    if key not in _kernels:
        import cupy

        params = ("raw T phi, raw T diag, raw T wx, raw T wy, raw T wz, "
                  "int32 nx, int32 ny, int32 nz")
        src = _STENCIL7_SRC
        if scaled:
            params += ", raw T rs"
            src = src.replace("out = v;", "out = v * rs[i];")
        _kernels[key] = cupy.ElementwiseKernel(
            params, "T out", src,
            f"ndgpu_stencil7{'_scaled' if scaled else ''}")
    return _kernels[key]


def stencil7_apply(xp, phi, diag, wx, wy, wz, row_scale, out=None):
    """One fused 7-point apply. Returns None if the fused path does not apply.

    All arrays must be C-contiguous and share one dtype (the caller prepares
    them once; see GroupOperator._fused_arrays).
    """
    nx, ny, nz = phi.shape
    kern = _stencil7(row_scale is not None)
    if out is None:
        out = xp.empty_like(phi)
    args = [phi, diag, wx, wy, wz, np.int32(nx), np.int32(ny), np.int32(nz)]
    if row_scale is not None:
        args.append(row_scale)
    kern(*args, out)
    return out


# --------------------------------------------------------------------------
# Triangular stencil
# --------------------------------------------------------------------------
# Same treatment as the Cartesian stencil, for the (nx, ny, 2[, nz]) up/down
# triangle layout that everything HP-MR runs on. A down triangle couples to the
# up triangle of its own cell (shared hypotenuse), to up(i-1, j) and to
# up(i, j-1); an up triangle couples the other way. Each face family carries an
# ordered PAIR of weights (a, b) rather than one symmetric weight, because
# discontinuity factors make the operator non-symmetric -- which is why the
# branch on the sublattice `t` cannot be folded away.
#
# 2D grids are handled as nz = 1: a C-contiguous (nx, ny, 2) array has exactly
# the flat layout of (nx, ny, 2, 1), and the coupling arrays follow.
_TRI_SRC = """
    const int z  = i % nz;
    const int t  = (i / nz) % 2;
    const int j  = (i / (2 * nz)) % ny;
    const int ii = i / (ny * 2 * nz);
    const int cidx = (ii * ny + j) * nz + z;   // (nx, ny, nz) coupling index
    const int si = ny * 2 * nz;                // stride: i -> i+1
    const int sj = 2 * nz;                     // stride: j -> j+1
    const int st = nz;                         // stride: down -> up
    T v = diag[i] * phi[i];
    if (t == 0) {
        v -= b_hyp[cidx] * phi[i + st];
        if (ii > 0) v -= b_v[cidx - ny * nz] * phi[i - si + st];
        if (j  > 0) v -= b_h[(ii * (ny - 1) + j - 1) * nz + z] * phi[i - sj + st];
    } else {
        v -= a_hyp[cidx] * phi[i - st];
        if (ii < nx - 1) v -= a_v[cidx] * phi[i + si - st];
        if (j  < ny - 1) v -= a_h[(ii * (ny - 1) + j) * nz + z] * phi[i + sj - st];
    }
"""

_TRI_Z_SRC = """
    const int wzb = ((ii * ny + j) * 2 + t) * (nz - 1);
    if (z < nz - 1) v -= wz[wzb + z] * phi[i + 1];
    if (z > 0)      v -= wz[wzb + z - 1] * phi[i - 1];
"""


def _tri_stencil(axial: bool):
    key = ("tri", axial)
    if key not in _kernels:
        import cupy

        params = ("raw T phi, raw T diag, raw T a_hyp, raw T b_hyp, "
                  "raw T a_v, raw T b_v, raw T a_h, raw T b_h, "
                  "int32 nx, int32 ny, int32 nz")
        src = _TRI_SRC
        if axial:
            params += ", raw T wz"
            src += _TRI_Z_SRC
        _kernels[key] = cupy.ElementwiseKernel(
            params, "T out", src + "    out = v;\n",
            f"ndgpu_tri_stencil{'_z' if axial else ''}")
    return _kernels[key]


def tri_stencil_apply(xp, phi, diag, a_hyp, b_hyp, a_v, b_v, a_h, b_h, wz,
                      nx, ny, nz, out=None):
    """One fused triangular-FV apply. Arrays are prepared by the operator."""
    if out is None:
        out = xp.empty_like(phi)
    args = [phi, diag, a_hyp, b_hyp, a_v, b_v, a_h, b_h,
            np.int32(nx), np.int32(ny), np.int32(nz)]
    if wz is not None:
        args.append(wz)
    _tri_stencil(wz is not None)(*args, out)
    return out


# --------------------------------------------------------------------------
# Krylov vector kernels
# --------------------------------------------------------------------------
# Three reductions and three vector updates per CG iteration; written the
# obvious way that is 3 temporaries + ~9 kernels, fused it is 3 + 2.


def _dot_kernel():
    if "dot" not in _kernels:
        import cupy

        # Deliberately not cuBLAS dot: cuBLAS calls cannot be recorded into a
        # CUDA graph, and capturing the CG inner loop is the next step up from
        # here (cf. the same note in tri_sn._levels_exec).
        _kernels["dot"] = cupy.ReductionKernel(
            "T x, T y", "T out", "x * y", "a + b", "out = a", "0", "ndgpu_dot")
    return _kernels["dot"]


#: Element count above which `dot` hands back to ``xp.sum(u * v)``.
#:
#: The fused reduction saves a launch and a full-size temporary, which is what
#: matters while the reduction is short enough to be launch-bound. Once it is
#: long enough to be bandwidth-bound the comparison inverts hard: CuPy's ``sum``
#: dispatches to CUB, whose tuned device reduction beats a generic two-input
#: ReductionKernel by far more than the extra elementwise pass costs.
#:
#: Measured on a Tesla T4 (fused/generic; >1 means the fused kernel wins):
#:
#:     n        1024  4096  16384  65536  262144  1048576  4194304
#:     dot      2.16  1.98   2.07   1.56    0.36     0.19     0.18
#:
#: i.e. above the crossover the fused reduction runs at roughly a *fifth* of
#: CUB's throughput. The threshold is set to the largest measured size where it
#: still wins; the true crossover is somewhere in 65536-262144. Element count
#: rather than bytes is a simplification -- the underlying effect is bandwidth,
#: so a complex128 vector crosses over at half this length.
#:
#: The other two Krylov kernels win at every size measured (cg_update 1.4-6.6x,
#: cg_direction 1.7-2.0x) and are not thresholded.
#:
#: Note this leaves the allocation-free path in place exactly where CUDA graph
#: capture (Phase 4) applies -- small grids -- which is convenient, since
#: ``xp.sum`` allocates and cannot be captured.
DOT_FUSED_MAX = 1 << 16


def dot(xp, u, v, out=None):
    """Bilinear (unconjugated) inner product as a 0-d device scalar.

    Unconjugated on purpose: real CG and complex-symmetric COCG both want this
    form, and no call site here needs the Hermitian one.
    """
    if use_fused(xp, "krylov") and u.size <= DOT_FUSED_MAX:
        return _dot_kernel()(u, v, out=out)
    value = xp.sum(u * v)
    if out is None:
        return value
    out[...] = value
    return out


def scalar_divide(xp, out, numerator, denominator):
    """``out = numerator / denominator`` for persistent device scalars."""
    if use_fused(xp, "krylov"):
        _block_kernel(
            "ndgpu_scalar_divide", "raw T num, raw T den", "T out",
            "out = num[0] / den[0];")(
                _scalar_arg(numerator), _scalar_arg(denominator), out)
    else:
        xp.divide(numerator, denominator, out=out)
    return out


def scalar_copy(xp, out, value):
    """Copy one persistent device scalar without creating a temporary."""
    if use_fused(xp, "krylov"):
        _block_kernel(
            "ndgpu_scalar_copy", "raw T value", "T out",
            "out = value[0];")(_scalar_arg(value), out)
    else:
        xp.copyto(out, value)
    return out


# The updated vectors are declared as *output* parameters and modified in
# place: CuPy binds non-raw outputs by reference, so `x += ...` both reads and
# writes the caller's array and no aliasing between an input and an output is
# needed.
def _cg_update_kernel():
    if "cg_update" not in _kernels:
        import cupy

        _kernels["cg_update"] = cupy.ElementwiseKernel(
            "T p, T ap, raw T alpha", "T x, T r",
            "x += alpha[0] * p; r -= alpha[0] * ap;",
            "ndgpu_cg_update")
    return _kernels["cg_update"]


def cg_update(xp, x, r, p, ap, alpha):
    """x += alpha*p and r -= alpha*Ap, in place, in one launch.

    ``alpha`` stays a 0-d *device* array: reading it on the host would sync.
    """
    if use_fused(xp, "krylov"):
        _cg_update_kernel()(p, ap, _scalar_arg(alpha), x, r)
    else:
        x += alpha * p
        r -= alpha * ap


def _cg_direction_kernel():
    if "cg_direction" not in _kernels:
        import cupy

        _kernels["cg_direction"] = cupy.ElementwiseKernel(
            "T z, raw T beta", "T p", "p = z + beta[0] * p;",
            "ndgpu_cg_direction")
    return _kernels["cg_direction"]


def cg_direction(xp, p, z, beta):
    """p <- z + beta*p, in place, in one launch."""
    if use_fused(xp, "krylov"):
        _cg_direction_kernel()(z, _scalar_arg(beta), p)
    else:
        p *= beta
        p += z


def axpy_inplace(xp, out, x, alpha):
    """``out += alpha*x`` without the full-size product temporary on GPU.

    BDF extrapolation applies this operation repeatedly to every flux and
    precursor field.  Keeping it here makes that otherwise backend-neutral
    history algebra one launch and one memory pass per history coefficient.
    """
    if use_fused(xp, "adaptive"):
        _block_kernel(
            "ndgpu_adaptive_axpy", "T x, T alpha", "T out",
            "out += alpha * x;")(x, alpha, out)
    else:
        out += alpha * x


def basis_accumulate(xp, out, basis, coefficients, n_vectors, alpha=1.0):
    """Add a short linear combination of contiguous basis vectors to ``out``.

    This is the bandwidth-bound vector half of Arnoldi orthogonalization and
    solution reconstruction.  A single kernel replaces one launch and one
    full-size temporary per basis vector.
    """
    n_vectors = int(n_vectors)
    if n_vectors < 1:
        return out
    if use_fused(xp, "krylov"):
        _block_kernel(
            "ndgpu_basis_accumulate",
            "raw T basis, raw T coefficients, T alpha, int32 K, int64 N",
            "T out",
            "T v = out; for (int k = 0; k < K; ++k)"
            " v += alpha * coefficients[k] * basis[k * N + i]; out = v;")(
                basis, coefficients, alpha, np.int32(n_vectors),
                np.int64(out.size), out)
    else:
        for j in range(n_vectors):
            out += alpha * coefficients[j] * basis[j]
    return out


def neumann_step(xp, z, r, az, inv_diag):
    """One damped-Jacobi sweep  z <- z + inv_diag*(r - A z), in place.

    The Neumann-polynomial preconditioner's inner loop (linalg.neumann_
    preconditioner): generic it is 4 kernels and 3 temporaries per degree.
    """
    if use_fused(xp, "krylov"):
        if "neumann" not in _kernels:
            import cupy

            _kernels["neumann"] = cupy.ElementwiseKernel(
                "T r, T az, T inv_diag", "T z",
                "z += inv_diag * (r - az);", "ndgpu_neumann_step")
        _kernels["neumann"](r, az, inv_diag, z)
    else:
        z += inv_diag * (r - az)
    return z


def mixed_jacobi_start(xp, residual, inv_diag, residual_low, z_low):
    """Cast an outer residual and form the low-precision Jacobi correction.

    A mixed preconditioner needs both ``r_low = (L) residual`` and
    ``z_low = inv_diag_low * r_low``. Expressing those assignments separately
    costs two GPU launches; this performs both in one pass and one launch.
    """
    if use_fused(xp, "krylov"):
        _block_kernel(
            "ndgpu_mixed_jacobi_start",
            "T residual, L inv_diag", "L residual_low, L z_low",
            "residual_low = (L)residual; z_low = inv_diag * residual_low;")(
                residual, inv_diag, residual_low, z_low)
    else:
        residual_low[...] = residual
        xp.multiply(inv_diag, residual_low, out=z_low)
    return z_low


# --------------------------------------------------------------------------
# SPN / SDPN order-N block
# --------------------------------------------------------------------------
# The order-N block is M coupled fields on one grid. Written as array
# expressions its apply is by far the most launch-heavy operator in the repo
# (SDP3 measured at ~195 probe-kernels even with the stencil already fused),
# because every projection assembles a weighted combination of the M moments,
# runs a stencil on it, scatters it back weighted, and then a dense M x M
# reaction runs over every (i, j) pair. Each of those three steps collapses to
# one kernel that holds a cell's M moments in registers.
#
# Indexing: the block state is (M, *grid), C-contiguous, so the flat index is
# i = m*N + idx with N the number of grid cells.


def _block_kernel(key, in_params, out_params, body):
    if key not in _kernels:
        import cupy

        _kernels[key] = cupy.ElementwiseKernel(in_params, out_params, body, key)
    return _kernels[key]


def moment_gather(xp, u, w, out=None):
    """s = sum_m w[m] * u[m]  -- project the M moments onto one scalar field."""
    M = u.shape[0]
    N = u[0].size
    if use_fused(xp, "block"):
        if out is None:
            out = xp.empty(u.shape[1:], dtype=u.dtype)
        _block_kernel(
            "ndgpu_moment_gather",
            "raw T u, raw T w, int32 M, int32 N", "T out",
            "T v = 0; for (int m = 0; m < M; ++m) v += w[m] * u[m * N + i];"
            " out = v;")(u, w, np.int32(M), np.int32(N), out)
        return out
    s = None
    for m in range(M):
        wm = float(w[m])
        if wm != 0.0:
            s = wm * u[m] if s is None else s + wm * u[m]
    if s is None:
        s = xp.zeros(u.shape[1:], dtype=u.dtype)
    if out is not None:
        out[...] = s
        return out
    return s


def moment_scatter_add(xp, out, s, w):
    """out[m] += w[m] * s  -- distribute one scalar field back over M moments."""
    M = out.shape[0]
    if use_fused(xp, "block"):
        _block_kernel(
            "ndgpu_moment_scatter_add",
            "raw T s, raw T w, int32 N", "T out",
            "out += w[i / N] * s[i % N];")(s, w, np.int32(s.size), out)
        return out
    for m in range(M):
        wm = float(w[m])
        if wm != 0.0:
            out[m] += wm * s
    return out


def dense_react_add(xp, out, u, C, pairs):
    """out[i] += sum_j C[i, j] * u[j] for the dense M x M reaction block.

    ``C`` is the stacked (M, M, *grid) coupling, built only for the fused path
    (entries the operator does not have are zero there). ``pairs`` is the
    operator's own sparse ``{(i, j): field}`` mapping, which the generic path
    keeps using so the CPU does not pay for the zeros. The fused accumulator
    starts from ``out``, matching the summation order of the per-pair
    expression it replaces.
    """
    if use_fused(xp, "block"):
        _block_kernel(
            "ndgpu_dense_react_add",
            "raw T u, raw T c, int32 M, int32 N", "T out",
            "const int idx = i % N; const int m = i / N; T v = out;"
            " for (int j = 0; j < M; ++j) v += c[(m * M + j) * N + idx]"
            " * u[j * N + idx]; out = v;")(
                u, C, np.int32(out.shape[0]), np.int32(u[0].size), out)
        return out
    for (i, j), f in pairs.items():
        out[i] += f * u[j]
    return out


def stack_pairs(xp, pairs, M, shape, dtype):
    """Stack a sparse {(i, j): field} coupling into a dense (M, M, *shape).

    Built lazily by the block operators on first fused apply -- the dense form
    is what lets one kernel walk the whole M x M reaction, and the operators
    are dense in practice anyway (the congruence fills them in).
    """
    C = xp.zeros((M, M) + tuple(shape), dtype=dtype)
    for (i, j), f in pairs.items():
        C[i, j] = f
    return C


def batched_matvec(xp, out, A, b):
    """out[k, r] = sum_j A[k, r, j] * b[k, j] -- a batch of tiny dense matvecs.

    Written for the S_N sweep's per-cell 3x3 corner blocks. The obvious CuPy
    spelling is ``A @ b[..., None]``, but cuBLAS calls cannot be recorded into a
    CUDA graph, so the captured sweep had been using a broadcast-multiply into
    an (K, n, n) temporary followed by a reduction -- three times the memory
    traffic of the matvec it computes, paid on every level of every sweep. This
    is one kernel, one pass, no temporary, and *is* capturable: it removes the
    penalty rather than trading it against the graph.
    """
    if not is_cupy(xp):
        # Only reachable from a test forcing this path on CPU: the callers pick
        # the fused form solely on GPU. Allocates, which would be illegal inside
        # a CUDA graph capture -- and is fine here, since capture is GPU-only.
        out[...] = (A * b[:, None, :]).sum(axis=2)
        return out
    n = A.shape[-1]
    _block_kernel(
        "ndgpu_batched_matvec",
        "raw T A, raw T b, int32 n", "T out",
        "const int r = i % n; const int k = i / n; T v = 0;"
        " for (int j = 0; j < n; ++j) v += A[(k * n + r) * n + j] * b[k * n + j];"
        " out = v;")(A, b, np.int32(n), out)
    return out


# --------------------------------------------------------------------------
# Multigroup source assembly
# --------------------------------------------------------------------------


def group_accumulate(xp, out, W, P, alpha=1.0):
    """out += alpha*sum_g W[g] * P[g], contracting the group axis.

    ``W`` and ``P`` are (G, \\*grid); ``out`` is (\\*grid). This is the whole
    in-scatter row for one group, or the fission source over all groups, in one
    kernel -- the per-group Python loop it replaces costs two kernels and a
    full-size temporary per (g, g') pair, i.e. O(G^2) launches per outer.

    The NumPy path exists for *testing* only -- it is what lets the batched
    assembly's arithmetic (above all the scattering transpose, which a batched
    rewrite can silently get backwards) be checked without a GPU. Production
    callers gate on ``use_fused(xp, "groups")``, which is False on NumPy, so on
    CPU they keep their sparse loop, which skips the absent couplings instead
    of multiplying by materialized zeros.
    """
    if not use_fused(xp, "groups"):
        for g in range(W.shape[0]):
            # Match the fused kernel: accumulation and multiplication use the
            # seeded output's precision even when inputs have narrower dtypes.
            out += (alpha * W[g].astype(out.dtype, copy=False)
                    * P[g].astype(out.dtype, copy=False))
        return out
    _block_kernel(
        "ndgpu_group_accumulate",
        "raw W weights, raw P phi, T alpha, int32 G, int32 N", "T out",
        "T v = out; for (int g = 0; g < G; ++g)"
        " v += alpha * (T)weights[g * N + i]"
        " * (T)phi[g * N + i]; out = v;")(
            W, P, alpha, np.int32(W.shape[0]), np.int32(out.size), out)
    return out


def product_accumulate(xp, out, x, y, alpha=1.0):
    """``out += alpha*x*y`` in one launch, without a product temporary."""
    if use_fused(xp, "groups"):
        _block_kernel(
            "ndgpu_product_accumulate", "T x, T y, T alpha", "T out",
            "out += alpha * x * y;")(x, y, alpha, out)
    else:
        out += alpha * x * y
    return out


# --------------------------------------------------------------------------
# SP3 / SDP1 two-moment coupling
# --------------------------------------------------------------------------


def sp3_couple(xp, out, u, coupling):
    """Finish the SP3 block apply in place, given the two moment leakages.

    On entry ``out`` holds (A1 Phi1, A2 phi2); on exit it holds the coupled
    block  (A1 Phi1 - c phi2,  5 A2 phi2 - c Phi1). Written as one kernel
    because the two rows share both inputs, so the coupling costs one pass
    rather than four.
    """
    if use_fused(xp, "block"):
        if "sp3_couple" not in _kernels:
            import cupy

            _kernels["sp3_couple"] = cupy.ElementwiseKernel(
                "T u0, T u1, T c", "T o0, T o1",
                "o0 -= c * u1; o1 = 5 * o1 - c * u0;", "ndgpu_sp3_couple")
        _kernels["sp3_couple"](u[0], u[1], coupling, out[0], out[1])
    else:
        out[0] -= coupling * u[1]
        out[1] *= 5.0
        out[1] -= coupling * u[0]
    return out


def clear_cache() -> None:
    """Drop the compiled-kernel cache (tests / repeated dtype sweeps)."""
    _kernels.clear()
