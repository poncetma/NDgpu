"""Array-backend selection: CuPy on GPU, NumPy on CPU.

The whole solver is written against the NumPy API surface that CuPy mirrors
exactly, so a single code path runs on both devices. On GPU, every array
operation in the hot loops (stencil applies, axpy, dot products) executes as
a CUDA kernel on device-resident data; the only host<->device traffic is the
scalar convergence checks.
"""

from __future__ import annotations

import numpy as np


def get_backend(device: str = "auto"):
    """Return the array module (numpy or cupy) for the requested device.

    device: "auto" (GPU if available), "gpu"/"cuda" (require GPU), "cpu".
    """
    device = device.lower()
    if device == "cpu":
        return np
    if device in ("gpu", "cuda"):
        import cupy  # raises ImportError with a clear message if absent

        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("device='gpu' requested but no CUDA device found")
        return cupy
    if device == "auto":
        try:
            import cupy

            if cupy.cuda.runtime.getDeviceCount() >= 1:
                return cupy
        except Exception:
            pass
        return np
    raise ValueError(f"unknown device {device!r}; use 'auto', 'gpu', or 'cpu'")


def device_name(xp) -> str:
    if xp is np:
        return "cpu (numpy)"
    props = xp.cuda.runtime.getDeviceProperties(xp.cuda.runtime.getDevice())
    return f"cuda (cupy): {props['name'].decode()}"


def asnumpy(a):
    """Copy an array to host memory as a numpy array (no-op for numpy)."""
    return a.get() if hasattr(a, "get") else np.asarray(a)


def synchronize(xp) -> None:
    """Block until all queued device work is done (no-op on CPU). Needed for timing."""
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()
