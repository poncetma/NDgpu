"""NDgpu — GPU-native multigroup neutron diffusion solver.

Steady-state k-eigenvalue reactor physics on 3D structured grids, matrix-free,
running natively on CUDA GPUs via CuPy with a NumPy CPU fallback.
"""

from .analytic import (geometric_buckling_box, k_bare_box, k_bare_box_sp3,
                       k_from_buckling, k_from_buckling_sp3, k_infinite)
from .backend import asnumpy, device_name, get_backend
from .femffusion import read_material_xml, read_xsec
from .griffin_xs import (read_library as read_griffin_library,
                        read_material as read_griffin_material,
                        volume_homogenize)
from .grid import Grid
from .hex import HexDiffusionEigenSolver, HexGrid, offset_to_axial
from .materials import Kinetics, Material, ONE_GROUP_DEMO, PWR_TWO_GROUP
from .mesh import (Mesh, MeshResult, UnstructuredDiffusionSolver, assemble_mesh,
                  assemble_mesh_3d, read_gmsh)
from .model import (HexLattice, MeshModel, Model, ModelResult, ReactorResult)
from .perturbation import first_order_reactivity
from .solver import DiffusionEigenSolver, Result, SP3EigenSolver
from .sph import (SphResult, flux_weighted_homogenize, region_average,
                  sph_correct)
from .tri import TriDiffusionEigenSolver, TriGrid, TriSP3EigenSolver
from .transient import TransientResult, TransientSolver

__version__ = "0.1.0"

__all__ = [
    "Model",
    "MeshModel",
    "HexLattice",
    "ModelResult",
    "ReactorResult",
    "DiffusionEigenSolver",
    "SP3EigenSolver",
    "TriDiffusionEigenSolver",
    "TriSP3EigenSolver",
    "TriGrid",
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
    "first_order_reactivity",
    "UnstructuredDiffusionSolver",
    "Mesh",
    "MeshResult",
    "read_gmsh",
    "assemble_mesh",
    "assemble_mesh_3d",
    "flux_weighted_homogenize",
    "region_average",
    "sph_correct",
    "SphResult",
    "read_material_xml",
    "read_xsec",
    "read_griffin_library",
    "read_griffin_material",
    "volume_homogenize",
]
