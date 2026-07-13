"""One-group reflected slab reactor -- an analytic diffusion benchmark.

The reflected reactor is the classic textbook problem for reflector savings
(Lamarsh, *Introduction to Nuclear Reactor Theory*, Ch. 7; Glasstone & Sesonske,
*Nuclear Reactor Engineering*, Ch. 6). A fissile core is faced by a non-fissile
reflector, so the flux does not vanish at the core edge and the critical size
shrinks relative to a bare core.

For a symmetric one-group slab, using the mid-plane symmetry (reflective at
x = 0): the core spans [0, a] and the reflector [a, a + b], with zero flux at the
outer face x = a + b. The one-group diffusion equation gives

    core:       phi_c(x) = A cos(B x),          B^2 = (nuSf/k - Sa_c) / D_c
    reflector:  phi_r(x) = C sinh((a+b-x)/L),   L = sqrt(D_r / Sa_r)

(the cosine is flat at the symmetry plane; the sinh vanishes at the outer face).
Matching the flux and the net current D phi' at the interface x = a eliminates the
amplitudes and leaves the exact eigenvalue condition

    D_c B tan(B a) = (D_r / L) coth(b / L),

a transcendental equation with a single root B in (0, pi/(2a)); the eigenvalue is
then k = nuSf / (Sa_c + D_c B^2). This closed-form value is the reference the
finite-volume solve is checked against (it converges to it at second order), and
it exceeds the bare-core value k = nuSf / (Sa_c + D_c (pi/(2a))^2) -- the reflector
savings.

This module provides the analytic reference and a builder that runs on either the
low-level ``DiffusionEigenSolver`` or the high-level ``ndgpu.Model`` (the geometry
is a filled 1-D slab, so both express it directly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Material

# One-group constants: a fissile core (k_inf = nuSf/Sa = 1.15) faced by a good,
# lightly-absorbing reflector. Lengths in cm, cross sections in 1/cm.
CORE = Material(name="core", diffusion=[1.2], sigma_a=[0.10], nu_sigma_f=[0.115])
REFLECTOR = Material(name="reflector", diffusion=[1.1], sigma_a=[0.008],
                     nu_sigma_f=[0.0])

CORE_HALF_WIDTH = 25.0    # cm, a (half the physical core, via mid-plane symmetry)
REFLECTOR_WIDTH = 20.0    # cm, b


def bare_k(core: Material = CORE, a: float = CORE_HALF_WIDTH) -> float:
    """Analytic k of the same half-core with a bare (zero-flux) edge at x = a."""
    D, Sa, nuSf = core.diffusion[0], core.sigma_a[0], core.nu_sigma_f[0]
    B = math.pi / (2.0 * a)
    return nuSf / (Sa + D * B * B)


def reflected_k(core: Material = CORE, reflector: Material = REFLECTOR,
                a: float = CORE_HALF_WIDTH, b: float = REFLECTOR_WIDTH) -> float:
    """Exact analytic eigenvalue of the one-group reflected slab (see module doc).

    Solves ``D_c B tan(B a) = (D_r / L) coth(b / L)`` for the fundamental root B
    by bisection on ``(0, pi/(2a))`` -- the left side rises monotonically from 0
    to +inf there while the right side is a positive constant -- then returns
    ``k = nuSf / (Sa_c + D_c B^2)``.
    """
    Dc, Sac, nuSf = core.diffusion[0], core.sigma_a[0], core.nu_sigma_f[0]
    Dr, Sar = reflector.diffusion[0], reflector.sigma_a[0]
    if nuSf <= 0:
        raise ValueError("core must be fissile")
    L = math.sqrt(Dr / Sar)
    rhs = (Dr / L) / math.tanh(b / L)                      # (D_r/L) coth(b/L), constant

    def f(B):
        return Dc * B * math.tan(B * a) - rhs

    lo, hi = 1e-9, math.pi / (2.0 * a) - 1e-9              # single root in this bracket
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    B = 0.5 * (lo + hi)
    return nuSf / (Sac + Dc * B * B)


@dataclass
class ReflectedSlabProblem:
    grid: Grid
    materials: list          # [core, reflector]
    material_map: np.ndarray  # (nx, 1, 1): 0 = core, 1 = reflector
    bc: object
    k_reference: float       # exact analytic eigenvalue
    core_cells: int
    core_half_width: float
    reflector_width: float


def build_reflected_slab(cells: int = 90, core: Material = CORE,
                         reflector: Material = REFLECTOR,
                         a: float = CORE_HALF_WIDTH, b: float = REFLECTOR_WIDTH
                         ) -> ReflectedSlabProblem:
    """Assemble the reflected slab on a 1-D Cartesian grid.

    cells : total cells across the domain [0, a + b]; they are split between core
    and reflector in proportion to the widths (the interface is placed on a cell
    boundary). Boundary: reflective at x = 0 (mid-plane), zero flux at x = a + b.
    """
    total = a + b
    core_cells = max(1, round(cells * a / total))
    dx = total / cells
    # Snap the interface to a cell edge so both regions are whole cells.
    a_eff = core_cells * dx
    b_eff = total - a_eff

    mmap = np.ones((cells, 1, 1), dtype=np.int64)          # 1 = reflector
    mmap[:core_cells, 0, 0] = 0                            # 0 = core
    grid = Grid(shape=(cells, 1, 1), size=(total, dx, dx))
    bc = (("reflective", "zero-flux"), "reflective", "reflective")
    k_ref = reflected_k(core, reflector, a_eff, b_eff)     # reference for THIS interface
    return ReflectedSlabProblem(grid=grid, materials=[core, reflector],
                                material_map=mmap, bc=bc, k_reference=k_ref,
                                core_cells=core_cells, core_half_width=a_eff,
                                reflector_width=b_eff)
