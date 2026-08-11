"""Measurement tools for the GPU optimization work (Phase 0 of the plan).

The premise behind :mod:`ndgpu.kernels` is that the solvers are *launch-bound*:
each CuPy array operation is one CUDA kernel, each launch costs a few
microseconds of mostly grid-size-independent overhead, and the operators here
are built from a dozen or more such operations on grids small enough that the
arithmetic is faster than the dispatch. That is a claim about a particular GPU,
so it has to be measured rather than assumed -- these helpers do the measuring.

The key trick is :func:`effective_launches`: time an operation on a grid so
small that *only* overhead is left, divide by the measured cost of one trivial
kernel, and you get the number of kernels the operation actually launched. It
needs no profiler, no privileges, and works inside Colab, where `nsys` does not.

Everything degrades gracefully on CPU (NumPy), where it just reports wall time.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from .backend import synchronize
from .kernels import is_cupy


@contextmanager
def nvtx_range(name: str, color_id: int = 0):
    """Annotate a region for `nsys`/`ncu` traces; a no-op without CuPy."""
    try:
        import cupy

        cupy.cuda.nvtx.RangePush(name)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        cupy.cuda.nvtx.RangePop()


def timeit(fn, xp, n_repeat: int = 100, n_warmup: int = 10) -> float:
    """Mean wall-clock seconds per call of `fn`, device-synchronized.

    Wall clock rather than CUDA events on purpose: the overhead under
    investigation is partly *host*-side (CuPy's per-operation Python dispatch),
    and CUDA-event timing would hide it.
    """
    for _ in range(n_warmup):
        fn()
    synchronize(xp)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    synchronize(xp)
    return (time.perf_counter() - t0) / n_repeat


def launch_cost(xp, n_repeat: int = 2000) -> float:
    """Seconds of overhead for one trivial elementwise kernel on this backend.

    Measured on an array small enough that the memory traffic is irrelevant, so
    what is left is launch + CuPy dispatch. On NumPy this is the cost of one
    tiny ufunc call, which makes :func:`effective_launches` meaningful (if less
    interesting) on CPU too.
    """
    a = xp.ones(8, dtype=np.float64)
    return timeit(lambda: xp.multiply(a, 1.0, out=a), xp,
                  n_repeat=n_repeat, n_warmup=100)


def effective_launches(fn, xp, tiny_cost: float | None = None,
                       n_repeat: int = 500) -> float:
    """Estimated number of kernels `fn` launches, from its cost at tiny size.

    `fn` must operate on a grid small enough to be pure overhead (a few hundred
    cells); the caller is responsible for building it that way. The result is an
    estimate -- kernels differ in dispatch cost and CuPy fuses nothing on its
    own -- but it is accurate enough to tell 13 launches from 1, which is the
    only question being asked.
    """
    if tiny_cost is None:
        tiny_cost = launch_cost(xp)
    return timeit(fn, xp, n_repeat=n_repeat) / tiny_cost


def operator_profile(op, xp, shape, dtype=np.float64, n_repeat: int = 200):
    """Time one `op.apply` and report seconds and cell throughput.

    Returns a dict with keys: cells, seconds, gcells_per_s.
    """
    u = xp.ones(shape, dtype=dtype)
    sec = timeit(lambda: op.apply(u), xp, n_repeat=n_repeat)
    n = int(np.prod(shape))
    return dict(cells=n, seconds=sec, gcells_per_s=n / sec / 1e9)


def ab_compare(fn, xp, n_repeat: int = 200, label: str = ""):
    """Run `fn` with fusion off then on; return (off_s, on_s, speedup).

    The A/B the notebooks report. Both legs run the identical code path apart
    from the ndgpu.kernels dispatch, so the ratio isolates fusion.
    """
    from . import kernels

    prev = kernels.set_fused(False)
    try:
        off = timeit(fn, xp, n_repeat=n_repeat)
        kernels.set_fused(True)
        on = timeit(fn, xp, n_repeat=n_repeat)
    finally:
        kernels.set_fused(prev)
    return dict(label=label, off=off, on=on,
                speedup=(off / on if on > 0 else float("nan")))


#: Character budget for one printed table. Notebooks get printed to PDF, which
#: does not wrap or scroll a <pre> block -- it clips it, silently, from the right.
#: The speedup column is invariably the one that falls off the page, so tables
#: wider than this are split into stacked blocks instead of being truncated.
MAX_TABLE_WIDTH = 78


def report(rows, columns, title="", max_width=None):
    """Print a list of dicts as a fixed-width table (benchmark-skill style).

    Tables wider than ``max_width`` are split column-wise into stacked blocks,
    each repeating the first column as the row key, so nothing is lost when the
    output is printed to PDF. Pass ``max_width=0`` to disable.
    """
    if title:
        print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return
    limit = MAX_TABLE_WIDTH if max_width is None else max_width
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in columns}

    def emit(cols):
        print("  ".join(c.rjust(widths[c]) for c in cols))
        for r in rows:
            print("  ".join(_fmt(r.get(c)).rjust(widths[c]) for c in cols))

    def width_of(cols):
        return sum(widths[c] for c in cols) + 2 * (len(cols) - 1)

    if not limit or width_of(columns) <= limit:
        emit(columns)
        return
    key, rest = columns[0], list(columns[1:])
    block = [key]
    while rest:
        if len(block) > 1 and width_of(block + [rest[0]]) > limit:
            emit(block)
            print()
            block = [key]
        block.append(rest.pop(0))
    if len(block) > 1:
        emit(block)


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def device_supports_fusion(xp) -> bool:
    """True when the fused kernels are actually in play (CuPy + enabled)."""
    from . import kernels

    return kernels.use_fused(xp) and is_cupy(xp)
