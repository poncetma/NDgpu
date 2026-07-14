"""Unstructured finite-volume diffusion on a Gmsh mesh.

Reads a Gmsh 2.2 ``.msh`` file -- 2D triangle/quad cells, or 3D tetrahedra,
hexahedra and prisms -- and solves the multigroup k-eigenvalue diffusion problem
on the arbitrary geometry it describes, the same job FEMFFUSION or GeN-Foam do
from a mesh rather than from a structured lattice. This is the general-geometry
escape hatch: where the structured Cartesian/hex operators need a lattice, this
one needs only cells, their vertices, and per-cell materials.

The scheme is a cell-centred two-point-flux finite volume: for a face shared by
cells i and j, the coupling is D_face * A_face / d(centroid_i, centroid_j) with
A_face the shared face measure (edge length in 2D, polygon area in 3D) and
D_face the harmonic mean of the cell diffusion coefficients; boundary faces get
the Robin/albedo term (vacuum alpha = 1/2). The 2D and 3D assemblers produce the
same face quantities, so the operator and solver below are dimension-agnostic.
The connectivity is irregular, but
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
    """Cells and faces of a 2D or 3D finite-volume mesh.

    ``area`` holds the cell measure (area in 2D, volume in 3D) and each face its
    measure (edge length in 2D, face area in 3D); the two-point-flux operator
    consumes these identically in either dimension.
    """

    coords: np.ndarray              # (n_nodes, 2) or (n_nodes, 3)
    cells: list                     # list of node-index tuples (per cell)
    cell_tag: np.ndarray            # (n_cells,) physical/elementary tag per cell
    centroid: np.ndarray            # (n_cells, 2) or (n_cells, 3)
    area: np.ndarray                # (n_cells,) cell measure (2D area / 3D volume)
    faces: list                     # (i, j, face_measure, centroid_distance) interior
    bfaces: list                    # (i, face_measure, centroid_to_face_distance) boundary

    @property
    def n_cells(self) -> int:
        return len(self.cells)


# Gmsh element type -> node count, split by dimension.
_GMSH_2D = {2: 3, 3: 4}                          # triangle, quad
_GMSH_3D = {4: 4, 5: 8, 6: 6}                     # tetrahedron, hexahedron, prism


def read_gmsh(path: str) -> Mesh:
    """Parse a Gmsh 2.2 ASCII mesh and assemble a Mesh.

    Handles 2D meshes of triangles (type 2) / quads (type 3) and 3D meshes of
    tetrahedra (4) / hexahedra (5) / prisms (6). If any 3D volume element is
    present the mesh is built in 3D (surface elements are ignored); otherwise the
    2D path is used.
    """
    lines = open(path).read().splitlines()
    ni = lines.index("$Nodes")
    nn = int(lines[ni + 1])
    coords = np.zeros((nn + 1, 3))
    for k in range(nn):
        t = lines[ni + 2 + k].split()
        coords[int(t[0])] = (float(t[1]), float(t[2]), float(t[3]))
    ei = lines.index("$Elements")
    ne = int(lines[ei + 1])
    elems = [[int(x) for x in lines[ei + 2 + k].split()] for k in range(ne)]
    is_3d = any(t[1] in _GMSH_3D for t in elems)
    table = _GMSH_3D if is_3d else _GMSH_2D
    cells, tags = [], []
    for t in elems:
        nnode = table.get(t[1])
        if nnode is not None:
            tags.append(t[3])                    # first tag (physical / assembly id)
            cells.append(tuple(t[5:5 + nnode]))
    if is_3d:
        return assemble_mesh_3d(coords, cells, tags)
    return assemble_mesh(coords[:, :2], cells, tags)


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


# Local face node-orderings per 3D element type, keyed by node count. Orientation
# is irrelevant here: face areas and cell volumes are taken in magnitude and a
# face is identified by the *set* of its nodes, so any consistent enumeration of
# each element's bounding faces works. (Gmsh 4=tet, 5=hex, 6=prism/wedge.)
_TET_FACES = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
_HEX_FACES = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
              (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
_PRISM_FACES = [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
_FACE_TEMPLATES = {4: _TET_FACES, 8: _HEX_FACES, 6: _PRISM_FACES}


def _poly_area_centroid_3d(pts):
    """Area and area-weighted centroid of a planar polygon in 3D (fan triangulation)."""
    p0 = pts[0]
    atot = 0.0
    csum = np.zeros(3)
    for i in range(1, len(pts) - 1):
        cr = np.cross(pts[i] - p0, pts[i + 1] - p0)
        a = 0.5 * float(np.linalg.norm(cr))
        atot += a
        csum += a * (p0 + pts[i] + pts[i + 1]) / 3.0
    return atot, (csum / atot if atot > 0 else pts.mean(0))


def _cell_volume_centroid_3d(cell_pts, faces_local):
    """Volume and volume-weighted centroid of a star-convex polyhedron.

    Decomposes into tetrahedra from a seed point (the vertex mean) to a fan
    triangulation of every bounding face. Exact for convex tets/hexes/prisms.
    """
    g = cell_pts.mean(0)
    vtot = 0.0
    csum = np.zeros(3)
    for f in faces_local:
        fp = cell_pts[list(f)]
        p0 = fp[0]
        for i in range(1, len(fp) - 1):
            v = abs(float(np.dot(np.cross(fp[i] - g, fp[i + 1] - g), p0 - g))) / 6.0
            vtot += v
            csum += v * (g + p0 + fp[i] + fp[i + 1]) / 4.0
    return vtot, (csum / vtot if vtot > 0 else g)


def assemble_mesh_3d(coords, cells, tags) -> Mesh:
    """Build a 3D Mesh (volumes, centroids, interior/boundary faces) from cells.

    coords : (n_nodes, 3) node coordinates. cells : list of node-index tuples,
    each a tetrahedron (4 nodes), prism/wedge (6) or hexahedron (8) in Gmsh
    ordering. tags : one integer per cell (material id). Two cells share a face
    when they share the same set of face nodes (conforming meshes only -- 3D
    hanging nodes are not split); a face touched by one cell is a boundary face.

    The returned Mesh carries the same two-point-flux quantities the 2D path
    does -- so `Mesh.area` holds the cell *volume* and each face its *area* and
    centroid-to-centroid (or centroid-to-face) distance -- and drives the exact
    same operator and solver.
    """
    coords = np.asarray(coords, dtype=float)
    cells = [tuple(c) for c in cells]
    nc = len(cells)
    centroid = np.zeros((nc, 3))
    volume = np.zeros(nc)
    face_cells = defaultdict(list)      # sorted global node key -> [(cell, global face nodes)]
    for c, ns in enumerate(cells):
        tmpl = _FACE_TEMPLATES.get(len(ns))
        if tmpl is None:
            raise ValueError(f"cell {c} has {len(ns)} nodes; expected a tet (4), "
                             "prism (6) or hex (8)")
        cell_pts = coords[list(ns)]
        volume[c], centroid[c] = _cell_volume_centroid_3d(cell_pts, tmpl)
        for fl in tmpl:
            gface = tuple(ns[k] for k in fl)
            face_cells[tuple(sorted(gface))].append((c, gface))

    def dist(a, b):
        return float(np.linalg.norm(centroid[a] - centroid[b]))

    faces, bfaces = [], []
    for lst in face_cells.values():
        gface = lst[0][1]
        A, fcent = _poly_area_centroid_3d(coords[list(gface)])
        if len(lst) == 2:
            i, j = lst[0][0], lst[1][0]
            faces.append((i, j, A, dist(i, j)))
        elif len(lst) == 1:
            i = lst[0][0]
            db = float(np.linalg.norm(centroid[i] - fcent))
            bfaces.append((i, A, db))
        else:
            raise ValueError(f"face {lst[0][1]} is shared by {len(lst)} cells "
                             "(non-manifold or non-conforming mesh)")
    return Mesh(coords=coords, cells=cells, cell_tag=np.asarray(tags),
                centroid=centroid, area=volume, faces=faces, bfaces=bfaces)


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
              inner_rtol_floor=1e-11, anderson_depth=8) -> MeshResult:
        """Power iteration on the fission source, Anderson-accelerated.

        The fission-source map ``F -> (1/k) sum_g nuSigma_f,g A_g^-1 chi_g F`` is a
        fixed point whose plain power iteration converges at the dominance ratio;
        for a loosely-coupled core (ratio near 1) that is hundreds of outers.
        Anderson acceleration mixes a short history of source residuals (the same
        scheme used in the transient step and the SPH solve), collapsing the slow
        modes so the eigenvector converges in far fewer outers. ``anderson_depth``
        <= 1 recovers the plain power iteration.
        """
        xp, G = self.xp, self.G
        synchronize(xp)
        t0 = time.perf_counter()
        n = self.mesh.n_cells
        area = self.area
        phi = [xp.ones(n, dtype=self.dtype) for _ in range(G)]
        k = 1.0
        F = sum(self.nsf[g] * phi[g] for g in range(G))
        F = F * (n / float(xp.sum(F)))                    # normalize source, sum = n
        conv = False
        inner_total = 0
        src_err = 1.0
        hist = []                                         # (F_in, raw_iterate) for Anderson
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
            total_new = float(xp.sum(Fn))
            k_new = k * total_new / n                     # F has sum n
            g_F = Fn * (n / total_new)                    # raw power iterate, sum = n
            r = g_F - F
            src_err = float(xp.sqrt(xp.sum(r * r) / xp.sum(g_F ** 2)))
            dk = abs(k_new - k)
            k = k_new
            if dk < tol_k and src_err < tol_source and outer > 5:
                F = g_F
                conv = True
                break

            # Anderson update: F <- residual-minimizing mix of the recent iterates.
            F_new = g_F
            if anderson_depth > 1:
                hist.append((F, g_F))
                if len(hist) > anderson_depth:
                    hist.pop(0)
                if len(hist) >= 2:
                    res = [Gj - Sj for Sj, Gj in hist]
                    dres = [res[i] - res[-1] for i in range(len(res) - 1)]
                    m = len(dres)
                    A = np.array([[float(xp.sum(dres[i] * dres[j])) for j in range(m)]
                                  for i in range(m)])
                    b = np.array([-float(xp.sum(dres[i] * res[-1])) for i in range(m)])
                    A[np.diag_indices(m)] += 1e-12 * (np.trace(A) + 1e-300)
                    try:
                        gamma = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        gamma = None
                    if gamma is not None and np.all(np.abs(gamma) < 1e4):
                        for j in range(m):
                            F_new = F_new + float(gamma[j]) * (hist[j][1] - hist[-1][1])
            F = F_new * (n / float(xp.sum(F_new)))        # renormalize, sum = n

        synchronize(xp)
        flux = np.array([asnumpy(phi[g]) for g in range(G)])
        return MeshResult(k_eff=k, flux=flux, converged=conv, outer_iterations=outer,
                          solve_seconds=time.perf_counter() - t0, device=self.device,
                          inner_iterations=inner_total)
