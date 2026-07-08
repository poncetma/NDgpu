"""ndgpu — GPU-native multigroup neutron diffusion solver.

Steady-state k-eigenvalue reactor physics on 3D structured grids, matrix-free,
running natively on CUDA GPUs via CuPy with a NumPy CPU fallback.
"""

from .analytic import (geometric_buckling_box, k_bare_box, k_bare_box_sp3,
                       k_from_buckling, k_from_buckling_sp3, k_infinite)
from .backend import asnumpy, device_name, get_backend
from .grid import Grid
from .hex import HexDiffusionEigenSolver, HexGrid, offset_to_axial
from .materials import Kinetics, Material, ONE_GROUP_DEMO, PWR_TWO_GROUP
from .solver import DiffusionEigenSolver, Result, SP3EigenSolver
from .transient import TransientResult, TransientSolver

__version__ = "0.1.0"

__all__ = [
    "DiffusionEigenSolver",
    "SP3EigenSolver",
    "HexDiffusionEigenSolver",
    "TransientSolver",
    "TransientResult",
    "Grid",
    "HexGrid",
    "offset_to_axial",
    "Material",
    "Kinetics",
    "Result",
    "k_bare_box_sp3",
    "k_from_buckling_sp3",
    "ONE_GROUP_DEMO",
    "PWR_TWO_GROUP",
    "asnumpy",
    "device_name",
    "get_backend",
    "geometric_buckling_box",
    "k_bare_box",
    "k_from_buckling",
    "k_infinite",
]
