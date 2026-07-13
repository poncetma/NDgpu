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
the Robin/albedo term (vacuum alpha = 1/2). The connectivity is irregular, but
the within-group operator is still applied matrix-free as a pure *gather*: the
faces are stored as a row-wise ELLPACK adjacency (each cell's neighbours padded
to the maximum degree), and the apply reads each cell's neighbour fluxes and
combines them with per-slot weights -- no scatter, hence no GPU atomics, and a
coalesced write, the same access pattern that makes the structured stencils fast
on CUDA. The single NumPy/CuPy code path runs on CPU or GPU and reuses the
structured solvers' Jacobi/Neumann-PCG, putting the general-geometry track on the
GPU alongside the structured lattices rather than the earlier SciPy sparse direct
factorisation. Power iteration drives the outer fission source.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .backend import asnumpy, device_name, get_backend, synchronize
from .linalg import neumann_preconditioner, pcg


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
    device: str = "cpu (numpy)"
    inner_iterations: int = 0


def _build_ell(face_i, face_j, n):
    """Row-wise (ELLPACK) adjacency from a symmetric face list.

    Each interior face (i, j) couples both endpoints, so it appears once in row
    i (neighbour j) and once in row j (neighbour i). The rows are padded to the
    maximum cell degree K and stored column-major, (K, n): ``nbr[s, c]`` is the
    s-th neighbour cell of c and ``fidx[s, c]`` the face feeding that slot (-1 on
    padding). Padding neighbours point at the cell itself with a zero weight, so
    they are inert. This lets the operator apply be a *gather* (read each
    neighbour's flux) with a coalesced write, instead of a *scatter* (atomic adds
    on the GPU) -- the same access pattern that makes the structured stencils
    fast on CUDA.
    """
    nf = face_i.size
    owner = np.concatenate([face_i, face_j])
    neigh = np.concatenate([face_j, face_i])
    fidx = np.concatenate([np.arange(nf), np.arange(nf)]).astype(np.int64)
    deg = np.bincount(owner, minlength=n)
    K = max(int(deg.max()) if nf else 0, 1)
    offset = np.zeros(n, dtype=np.int64)
    if n:
        offset[1:] = np.cumsum(deg)[:-1]
    order = np.argsort(owner, kind="stable")
    owner_s = owner[order]
    slot = np.arange(owner.size) - offset[owner_s]           # 0..deg-1 within each row
    nbr = np.tile(np.arange(n, dtype=np.int64), (K, 1))      # (K, n), self-padded
    fmap = np.full((K, n), -1, dtype=np.int64)
    if nf:
        nbr[slot, owner_s] = neigh[order]
        fmap[slot, owner_s] = fidx[order]
    return nbr, fmap


class _MeshGroupOperator:
    """Matrix-free within-group FV operator on an unstructured mesh.

    Exposes the same ``apply`` / ``inv_diag`` interface as the structured
    :class:`~ndgpu.operator.GroupOperator`, so it drops straight into the shared
    Jacobi/Neumann-PCG. The connectivity is an explicit ELLPACK adjacency
    (:func:`_build_ell`) rather than a shifted stencil, but the apply is still a
    pure *gather*: for every cell read its neighbours' flux and combine with the
    per-slot weights, ``A phi = diag*phi - sum_s w[s]*phi[nbr[s]]``. No scatter,
    so no GPU atomics and the write is coalesced. The assembled operator --
    volume-integrated harmonic-mean two-point-flux leakage, plus removal and the
    Robin boundary term on the diagonal -- is symmetric and diagonally dominant,
    hence SPD, so CG converges.
    """

    def __init__(self, xp, nbr, w_ell, diag):
        self.xp = xp
        self.nbr = nbr                # (K, n) int32 neighbour cell per slot
        self.w_ell = w_ell            # (K, n) coupling weight per slot (0 on padding)
        self.diag = diag              # (n,) assembled diagonal
        self.inv_diag = 1.0 / diag    # Jacobi preconditioner

    def apply(self, phi):
        gathered = phi[self.nbr]                      # (K, n) neighbour flux
        return self.diag * phi - (self.w_ell * gathered).sum(axis=0)


class UnstructuredDiffusionSolver:
    """Multigroup k-eigenvalue diffusion FV solver on an arbitrary Gmsh mesh.

    materials      : list of Material (all same group count, up- or down-scatter).
    cell_material  : (n_cells,) index into `materials` for each cell.
    alpha_boundary : Robin coefficient on every boundary face (0.5 = vacuum,
                     0 = reflective, or a custom albedo).
    device         : "auto" (GPU if present) | "gpu" | "cpu".
    precond_degree : Neumann-polynomial preconditioner degree for the inner CG
                     (0 = plain Jacobi, the default), as on the structured
                     solvers.
    """

    def __init__(self, mesh: Mesh, materials, cell_material,
                 alpha_boundary=_VACUUM_ALPHA, device="auto",
                 precond_degree=0, dtype=np.float64):
        self.xp = xp = get_backend(device)
        self.device = device_name(xp)
        self.dtype = dtype
        self.mesh = mesh
        self.mats = list(materials)
        self.cm = cm = np.asarray(cell_material)
        self.alpha = float(alpha_boundary)
        self.G = G = self.mats[0].n_groups
        n = mesh.n_cells

        area = np.asarray(mesh.area, dtype=dtype)
        D = [np.array([self.mats[m].diffusion[g] for m in cm], dtype=dtype) for g in range(G)]
        removal = [np.array([self.mats[m].removal[g] for m in cm], dtype=dtype) for g in range(G)]

        # Face and boundary connectivity as flat arrays (host; moved to device).
        fi = np.array([f[0] for f in mesh.faces], dtype=np.int64)
        fj = np.array([f[1] for f in mesh.faces], dtype=np.int64)
        fL = np.array([f[2] for f in mesh.faces], dtype=dtype)
        fd = np.array([f[3] for f in mesh.faces], dtype=dtype)
        bi = np.array([b[0] for b in mesh.bfaces], dtype=np.int64)
        bL = np.array([b[1] for b in mesh.bfaces], dtype=dtype)
        bd = np.array([b[2] for b in mesh.bfaces], dtype=dtype)

        # ELLPACK adjacency (geometry, shared across groups); moved to device once.
        nbr, fmap = _build_ell(fi, fj, n)
        nbr_dev = xp.asarray(nbr.astype(np.int32))
        fpad = np.where(fmap >= 0, fmap, 0)                    # safe index for padding

        self.ops = []
        for g in range(G):
            Dg = D[g]
            Di, Dj = Dg[fi], Dg[fj]
            face_w = 2.0 * Di * Dj / (Di + Dj) * fL / fd       # harmonic-mean coupling
            # per-slot weights (0 where padded); diag from the same face weights.
            w_ell = np.where(fmap >= 0, face_w[fpad], 0.0).astype(dtype)
            diag = removal[g] * area
            diag = diag + np.bincount(fi, weights=face_w, minlength=n)
            diag = diag + np.bincount(fj, weights=face_w, minlength=n)
            if self.alpha != 0.0:
                bw = self.alpha * Dg[bi] * bL / (bd * self.alpha + Dg[bi])
                diag = diag + np.bincount(bi, weights=bw, minlength=n)
            self.ops.append(_MeshGroupOperator(
                xp, nbr_dev, xp.asarray(w_ell), xp.asarray(diag)))
        self.preconds = [neumann_preconditioner(op.apply, op.inv_diag,
                                                 int(precond_degree))
                         for op in self.ops]

        # Source data, device-resident.
        self.area = xp.asarray(area)
        self.nsf = [xp.asarray(np.array([self.mats[m].nu_sigma_f[g] for m in cm], dtype=dtype))
                    for g in range(G)]
        self.chi = [xp.asarray(np.array([self.mats[m].chi[g] for m in cm], dtype=dtype))
                    for g in range(G)]
        # scattering g'->g (both down- and up-scatter); lagged through the source.
        self.scat = {}
        for gf in range(G):
            for gt in range(G):
                if gt != gf:
                    col = np.array([self.mats[m].sigma_s[gf, gt] for m in cm], dtype=dtype)
                    if np.any(col):
                        self.scat[(gf, gt)] = xp.asarray(col)

    def solve(self, tol_k=1e-7, tol_source=1e-8, max_outer=1000,
              inner_rtol_floor=1e-11) -> MeshResult:
        xp, G = self.xp, self.G
        synchronize(xp)
        t0 = time.perf_counter()
        n = self.mesh.n_cells
        area = self.area
        phi = [xp.ones(n, dtype=self.dtype) for _ in range(G)]
        k = 1.0
        F = sum(self.nsf[g] * phi[g] for g in range(G))
        total = xp.sum(F)
        conv = False
        inner_total = 0
        src_err = 1.0
        for outer in range(1, max_outer + 1):
            # Inner CG tolerance tracks the outer residual: loose early, tight late.
            rtol = min(1e-3, max(0.1 * src_err, inner_rtol_floor, 0.01 * tol_source))
            for g in range(G):
                q = self.chi[g] / k * F * area
                for (gf, gt), col in self.scat.items():
                    if gt == g:
                        q = q + col * phi[gf] * area
                phi[g], n_it = pcg(self.ops[g].apply, q, phi[g], self.ops[g].inv_diag,
                                   xp, rtol=rtol, precond=self.preconds[g])
                inner_total += n_it

            Fn = sum(self.nsf[g] * phi[g] for g in range(G))
            total_new = xp.sum(Fn)
            k_new = k * float(total_new / total)

            diff = Fn / total_new - F / total
            src_err = float(xp.sqrt(xp.sum(diff * diff) / xp.sum((Fn / total_new) ** 2)))
            dk = abs(k_new - k)

            scale = n / float(total_new)
            for g in range(G):
                phi[g] = phi[g] * scale
            F = Fn * scale
            total = xp.sum(F)
            k = k_new

            if dk < tol_k and src_err < tol_source and outer > 5:
                conv = True
                break

        synchronize(xp)
        flux = np.array([asnumpy(phi[g]) for g in range(G)])
        return MeshResult(k_eff=k, flux=flux, converged=conv, outer_iterations=outer,
                          solve_seconds=time.perf_counter() - t0, device=self.device,
                          inner_iterations=inner_total)
