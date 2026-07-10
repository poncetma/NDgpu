"""Unstructured finite-volume diffusion on a Gmsh mesh.

Reads a Gmsh 2.2 ``.msh`` file (quad or triangle cells) and solves the
multigroup k-eigenvalue diffusion problem on the arbitrary geometry it
describes -- the same job FEMFFUSION or GeN-Foam do from a mesh, rather than
from a structured lattice. This is the general-geometry escape hatch: where the
structured Cartesian/hex operators need a lattice, this one needs only cells,
their vertices, and per-cell materials.

The scheme is a cell-centred two-point-flux finite volume: for a face shared by
cells i and j, the coupling is D_face * L_face / d(centroid_i, centroid_j) with
D_face the harmonic mean of the cell diffusion coefficients; boundary faces get
the Robin/albedo term (vacuum alpha = 1/2). The connectivity is irregular, so
the within-group operator is a sparse matrix (SciPy) rather than a shifted
stencil -- correct and general, at the cost of the matrix-free locality the
structured solvers enjoy. Power iteration drives the outer fission source.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


_VACUUM_ALPHA = 0.5


@dataclass
class Mesh:
    """Cells and faces read from a Gmsh file (2D, single z-plane)."""

    coords: np.ndarray              # (n_nodes, 2)
    cells: list                     # list of node-index tuples (per cell)
    cell_tag: np.ndarray            # (n_cells,) physical/elementary tag per cell
    centroid: np.ndarray            # (n_cells, 2)
    area: np.ndarray                # (n_cells,)
    faces: list                     # (i, j, length, centroid_distance) interior
    bfaces: list                    # (i, length, centroid_to_edge_distance) boundary

    @property
    def n_cells(self) -> int:
        return len(self.cells)


def read_gmsh(path: str) -> Mesh:
    """Parse a Gmsh 2.2 ASCII mesh of 2D quad (type 3) / triangle (type 2) cells."""
    lines = open(path).read().splitlines()
    ni = lines.index("$Nodes")
    nn = int(lines[ni + 1])
    coords = np.zeros((nn + 1, 2))
    for k in range(nn):
        t = lines[ni + 2 + k].split()
        coords[int(t[0])] = (float(t[1]), float(t[2]))
    ei = lines.index("$Elements")
    ne = int(lines[ei + 1])
    cells, tags = [], []
    for k in range(ne):
        t = [int(x) for x in lines[ei + 2 + k].split()]
        etype = t[1]
        if etype in (2, 3):                      # 3-node tri or 4-node quad
            nnode = 3 if etype == 2 else 4
            tags.append(t[3])                    # first tag (physical / assembly id)
            cells.append(tuple(t[5:5 + nnode]))
    return assemble_mesh(coords, cells, tags)


def assemble_mesh(coords, cells, tags) -> Mesh:
    """Build a Mesh (centroids, areas, interior/boundary faces) from polygons.

    coords : (n_nodes, 2) node coordinates. cells : list of node-index tuples.
    tags   : one integer per cell (material/assembly id). A face is a boundary
    face when only one cell borders it.

    Nonconforming (locally-refined) interfaces are supported: a one-cell edge
    (i, j) whose midpoint is itself a mesh node m, with the half-edges (i, m)
    and (m, j) each owned by another cell, is a coarse edge meeting two fine
    cells across a 2:1 hanging node. It is split into two interior faces
    coupling the coarse cell to each fine cell (each carrying its half-edge
    length) -- the standard conservative two-point-flux treatment. Purely
    conforming meshes are unaffected (their one-cell edges have no midpoint
    node, so they stay boundary faces).
    """
    coords = np.asarray(coords, dtype=float)
    cells = [tuple(c) for c in cells]
    nc = len(cells)
    centroid = np.zeros((nc, 2))
    area = np.zeros(nc)
    edge_cells = defaultdict(list)
    for c, ns in enumerate(cells):
        P = coords[list(ns)]
        x, y = P[:, 0], P[:, 1]
        area[c] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        centroid[c] = P.mean(0)
        m = len(ns)
        for a in range(m):
            edge_cells[tuple(sorted((ns[a], ns[(a + 1) % m])))].append(c)

    # Node lookup by (rounded) coordinate, over nodes actually used by cells,
    # for detecting a hanging-node midpoint on a one-cell edge.
    def key(p):
        return (round(float(p[0]), 6), round(float(p[1]), 6))
    used = {n for ns in cells for n in ns}
    node_at = {key(coords[n]): n for n in used}
    one_cell = {e: lst[0] for e, lst in edge_cells.items() if len(lst) == 1}

    def dist(a, b):
        return float(np.hypot(*(centroid[a] - centroid[b])))

    faces, bfaces, consumed = [], [], set()
    for e, ci in one_cell.items():
        if e in consumed:
            continue
        a, b = e
        mnode = node_at.get(key(0.5 * (coords[a] + coords[b])))
        if mnode is None or mnode in (a, b):
            continue
        e1, e2 = tuple(sorted((a, mnode))), tuple(sorted((mnode, b)))
        if e1 in one_cell and e2 in one_cell:      # coarse edge over two fine cells
            for half in (e1, e2):
                cj = one_cell[half]
                L = float(np.hypot(*(coords[half[0]] - coords[half[1]])))
                faces.append((ci, cj, L, dist(ci, cj)))
            consumed.update((e, e1, e2))

    for e, lst in edge_cells.items():
        if e in consumed:
            continue
        p0, p1 = coords[e[0]], coords[e[1]]
        L = float(np.hypot(*(p1 - p0)))
        if len(lst) == 2:
            i, j = lst
            faces.append((i, j, L, dist(i, j)))
        else:
            i = lst[0]
            db = float(np.hypot(*(centroid[i] - 0.5 * (p0 + p1))))
            bfaces.append((i, L, db))
    return Mesh(coords=coords, cells=cells, cell_tag=np.asarray(tags), centroid=centroid,
               area=area, faces=faces, bfaces=bfaces)


@dataclass
class MeshResult:
    k_eff: float
    flux: np.ndarray                # (G, n_cells)
    converged: bool
    outer_iterations: int
    solve_seconds: float


class UnstructuredDiffusionSolver:
    """Multigroup k-eigenvalue diffusion FV solver on an arbitrary Gmsh mesh.

    materials      : list of Material (all same group count, no upscatter).
    cell_material  : (n_cells,) index into `materials` for each cell.
    alpha_boundary : Robin coefficient on every boundary face (0.5 = vacuum,
                     0 = reflective, or a custom albedo).
    """

    def __init__(self, mesh: Mesh, materials, cell_material, alpha_boundary=_VACUUM_ALPHA):
        self.mesh = mesh
        self.mats = list(materials)
        self.cm = np.asarray(cell_material)
        self.alpha = float(alpha_boundary)
        self.G = self.mats[0].n_groups
        self.D = [np.array([self.mats[m].diffusion[g] for m in self.cm]) for g in range(self.G)]
        self.removal = [np.array([self.mats[m].removal[g] for m in self.cm]) for g in range(self.G)]
        self.nsf = [np.array([self.mats[m].nu_sigma_f[g] for m in self.cm]) for g in range(self.G)]
        self.chi = [np.array([self.mats[m].chi[g] for m in self.cm]) for g in range(self.G)]
        # downscatter g'->g (g' < g), no upscatter
        self.scat = {}
        for gf in range(self.G):
            for gt in range(self.G):
                if gt != gf:
                    col = np.array([self.mats[m].sigma_s[gf, gt] for m in self.cm])
                    if np.any(col):
                        self.scat[(gf, gt)] = col

    def _operator(self, g, extra_diag=None):
        """Sparse within-group operator A_g (integrated over cell volumes)."""
        m = self.mesh
        D = self.D[g]
        diag = (self.removal[g] * m.area).astype(float).copy()
        if extra_diag is not None:
            diag = diag + extra_diag
        I, J, V = [], [], []
        for i, j, L, d in m.faces:
            w = 2.0 * D[i] * D[j] / (D[i] + D[j]) * L / d
            diag[i] += w
            diag[j] += w
            I += [i, j]
            J += [j, i]
            V += [-w, -w]
        if self.alpha != 0.0:
            for i, L, db in m.bfaces:
                diag[i] += self.alpha * D[i] * L / (db * self.alpha + D[i])
        I += list(range(m.n_cells))
        J += list(range(m.n_cells))
        V += list(diag)
        return sp.csc_matrix((V, (I, J)), shape=(m.n_cells, m.n_cells))

    def solve(self, tol_k=1e-7, max_outer=1000) -> MeshResult:
        import time
        t0 = time.perf_counter()
        m = self.mesh
        A = m.area
        lu = [splu(self._operator(g)) for g in range(self.G)]
        phi = [np.ones(m.n_cells) for _ in range(self.G)]
        k = 1.0
        conv = False
        for outer in range(1, max_outer + 1):
            F = sum(self.nsf[g] * phi[g] for g in range(self.G))
            Ftot = F.sum()
            for g in range(self.G):
                q = self.chi[g] / k * F * A
                for (gf, gt), col in self.scat.items():
                    if gt == g:
                        q = q + col * phi[gf] * A
                phi[g] = lu[g].solve(q)
            Fn = sum(self.nsf[g] * phi[g] for g in range(self.G))
            k_new = k * Fn.sum() / Ftot
            if abs(k_new - k) < tol_k and outer > 5:
                k = k_new
                conv = True
                break
            k = k_new
            s = m.n_cells / (Fn * A).sum()
            for g in range(self.G):
                phi[g] *= s
        return MeshResult(k_eff=k, flux=np.array(phi), converged=conv,
                          outer_iterations=outer, solve_seconds=time.perf_counter() - t0)
