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
from .model import (FeedbackSpec, HexLattice, MeshModel, Model, ModelResult,
                   ReactorResult, TransientModelResult, TriReactor, hex_disk,
                   hex_ring)
from .noise import (NoiseResult, NoiseSolver, NoiseSource,
                    zero_power_transfer_function)
from .perturbation import first_order_reactivity
from .power import fission_energy_xs, power_density
from .quasistatic import (EffectiveKinetics, FixedShapeQuasiStaticResult,
                          PointKineticsResult, QuasiStaticResult,
                          advance_point_kinetics,
                          equilibrium_precursors,
                          fixed_shape_coupled_transient,
                          integrate_point_kinetics,
                          project_effective_kinetics,
                          quasistatic_coupled_transient)
from .feedback import ThermalFeedback
from .thermal import ConductionSolver, ThermalMaterial, ThermalResult
from .solver import (DiffusionEigenSolver, Result, SDP1EigenSolver,
                     SDP2EigenSolver, SDP3EigenSolver, SP1EigenSolver,
                     SP3EigenSolver, SP5EigenSolver, SP7EigenSolver)
from .hybrid_sn import HybridSNDiffusionSolver, HybridSNResult
from .hybrid_tri_sn import HybridTriSNDiffusionSolver
from .ldfe import ldfe_quadrature_2d, ldfe_sphere, mirror_maps
from .sn import SNResult, SNTransportSolver, quadrature_2d
from .tri_sn import TriSNTransportSolver
from .sph import (SphResult, flux_weighted_homogenize, production_weight,
                  region_average,
                  sph_correct, sph_correct_monolithic,
                  sph_get_correct)
from .tri import (TriDiffusionEigenSolver, TriGrid, TriSDP1EigenSolver,
                  TriSDP2EigenSolver, TriSDP3EigenSolver, TriSP1EigenSolver,
                  TriSP3EigenSolver, TriSP5EigenSolver, TriSP7EigenSolver)
from .transient import (TransientResult, TransientSDP1Solver,
                        TransientSDP3Solver, TransientSDPNSolver,
                        TransientSNSolver, TransientSPNSolver, TransientSolver)

__version__ = "0.1.0"

__all__ = [
    "Model",
    "MeshModel",
    "HexLattice",
    "TriReactor",
    "FeedbackSpec",
    "hex_ring",
    "hex_disk",
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
    "TransientSNSolver",
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
    "power_density",
    "fission_energy_xs",
    "EffectiveKinetics",
    "FixedShapeQuasiStaticResult",
    "QuasiStaticResult",
    "PointKineticsResult",
    "project_effective_kinetics",
    "equilibrium_precursors",
    "advance_point_kinetics",
    "integrate_point_kinetics",
    "fixed_shape_coupled_transient",
    "quasistatic_coupled_transient",
    "ConductionSolver",
    "ThermalMaterial",
    "ThermalResult",
    "ThermalFeedback",
    "UnstructuredDiffusionSolver",
    "Mesh",
    "MeshResult",
    "read_gmsh",
    "assemble_mesh",
    "assemble_mesh_3d",
    "flux_weighted_homogenize",
    "region_average",
    "production_weight",
    "sph_correct",
    "sph_correct_monolithic",
    "sph_get_correct",
    "SphResult",
    "SNTransportSolver",
    "TriSNTransportSolver",
    "SNResult",
    "quadrature_2d",
    "ldfe_quadrature_2d",
    "ldfe_sphere",
    "mirror_maps",
    "HybridSNDiffusionSolver",
    "HybridSNResult",
    "HybridTriSNDiffusionSolver",
    "read_material_xml",
    "read_xsec",
    "read_griffin_library",
    "read_griffin_material",
    "volume_homogenize",
]
