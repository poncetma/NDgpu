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
from .model import (HexLattice, MeshModel, Model, ModelResult, ReactorResult,
                   TransientModelResult)
from .noise import (NoiseResult, NoiseSolver, NoiseSource,
                    zero_power_transfer_function)
from .perturbation import first_order_reactivity
from .solver import (DiffusionEigenSolver, Result, SDP1EigenSolver,
                     SDP2EigenSolver, SDP3EigenSolver, SP1EigenSolver,
                     SP3EigenSolver, SP5EigenSolver, SP7EigenSolver)
from .hybrid_sn import HybridSNDiffusionSolver, HybridSNResult
from .sn import SNResult, SNTransportSolver, quadrature_2d
from .sph import (SphResult, flux_weighted_homogenize, region_average,
                  sph_correct)
from .tri import (TriDiffusionEigenSolver, TriGrid, TriSDP1EigenSolver,
                  TriSDP2EigenSolver, TriSDP3EigenSolver, TriSP1EigenSolver,
                  TriSP3EigenSolver, TriSP5EigenSolver, TriSP7EigenSolver)
from .transient import (TransientResult, TransientSDP1Solver,
                        TransientSDP3Solver, TransientSDPNSolver,
                        TransientSPNSolver, TransientSolver)

__version__ = "0.1.0"

__all__ = [
    "Model",
    "MeshModel",
    "HexLattice",
    "ModelResult",
    "ReactorResult",
    "TransientModelResult",
    "DiffusionEigenSolver",
    "SP1EigenSolver",
    "SP3EigenSolver",
    "SP5EigenSolver",
    "SP7EigenSolver",
    "SDP1EigenSolver",
    "SDP2EigenSolver",
    "SDP3EigenSolver",
    "TriDiffusionEigenSolver",
    "TriSP1EigenSolver",
    "TriSP3EigenSolver",
    "TriSP5EigenSolver",
    "TriSP7EigenSolver",
    "TriSDP1EigenSolver",
    "TriSDP2EigenSolver",
    "TriSDP3EigenSolver",
    "TriGrid",
    "HexDiffusionEigenSolver",
    "TransientSolver",
    "TransientSDP1Solver",
    "TransientSDPNSolver",
    "TransientSDP3Solver",
    "TransientSPNSolver",
    "TransientResult",
    "NoiseSolver",
    "NoiseSource",
    "NoiseResult",
    "zero_power_transfer_function",
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
    "SNTransportSolver",
    "SNResult",
    "quadrature_2d",
    "HybridSNDiffusionSolver",
    "HybridSNResult",
    "read_material_xml",
    "read_xsec",
    "read_griffin_library",
    "read_griffin_material",
    "volume_homogenize",
]
