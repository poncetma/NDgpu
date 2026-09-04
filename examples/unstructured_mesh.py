"""Steady TPFA reactor on a controlled unstructured mesh via ndgpu.MeshModel.

MeshModel runs the matrix-free two-point-flux solver on a compatible Gmsh 2.2
ASCII mesh, or a controlled Mesh assembled in code. It is not a general
CAD-mesh solver; see ``docs/unstructured_mesh_scope.md``. Here we build an
orthogonal 3-D hex mesh and paint a spherical fuel region by cell centroid. The
curved interface is therefore stair-stepped rather than geometrically clipped.

    m = ndgpu.MeshModel("core.msh")     # or MeshModel(mesh_object)
    m.fill(reflector).assign(fuel, tag=1).set_boundary("vacuum")
    print(m.run())

Usage: python examples/unstructured_mesh.py [cells_per_side] [radius_cm] [cpu|gpu|auto]
"""

import sys

import numpy as np

import ndgpu
from ndgpu import Material
from ndgpu.mesh import assemble_mesh_3d

n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
radius = float(sys.argv[2]) if len(sys.argv) > 2 else 35.0
device = sys.argv[3] if len(sys.argv) > 3 else "auto"

fuel = Material(name="fuel", diffusion=[1.26, 0.35], sigma_a=[0.012, 0.121],
                nu_sigma_f=[0.0085, 0.185], sigma_s=[[0.0, 0.026], [0.0, 0.0]], chi=[1, 0])
reflector = Material(name="reflector", diffusion=[1.15, 0.90], sigma_a=[0.0002, 0.005],
                     nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.045], [0.0, 0.0]])


def hex_mesh(n, L):
    """An n^3 grid of hexahedra over an L cm cube, as an unstructured Mesh."""
    dx = L / n
    nid, coords = {}, []

    def gid(i, j, k):
        if (i, j, k) not in nid:
            nid[(i, j, k)] = len(coords); coords.append((i * dx, j * dx, k * dx))
        return nid[(i, j, k)]

    cells = [(gid(i, j, k), gid(i + 1, j, k), gid(i + 1, j + 1, k), gid(i, j + 1, k),
              gid(i, j, k + 1), gid(i + 1, j, k + 1), gid(i + 1, j + 1, k + 1), gid(i, j + 1, k + 1))
             for i in range(n) for j in range(n) for k in range(n)]
    return assemble_mesh_3d(coords, cells, [0] * len(cells))


L = 120.0
mesh = hex_mesh(n, L)
centre = np.array([L / 2, L / 2, L / 2])

model = (
    ndgpu.MeshModel(mesh)
    .fill(reflector)
    .assign(fuel, where=lambda c: np.linalg.norm(c - centre) <= radius)   # a sphere of fuel
    .set_boundary("vacuum")
)

print(f"Spherical fuel region r={radius:.0f} cm in a {L:.0f} cm reflector cube, on {device}\n")
print(model.run(device=device))
