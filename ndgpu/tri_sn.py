"""Discrete-ordinates (S_N) transport on the body-fitted triangular mesh.

The triangular counterpart of ``ndgpu.sn`` (2D Cartesian S_N), so discrete-
ordinates transport -- and the hybrid S_N/diffusion drum treatment -- run on the
actual HP-MR hex/triangular core, not a Cartesian stand-in.

The mesh is the same structured equilateral-triangle lattice the diffusion/SP3
solvers use (:class:`ndgpu.tri.TriGrid`): cells stored on (nrows, ncols, 2) with
the last index the down/up triangle of each rhombus, every interior cell coupled
to three neighbours at fixed offsets. That structure is what makes S_N tractable
here. Two sweep engines share identical algebra (``engine=``):

* ``"lu"`` (CPU default): per ordinate, the sparse streaming+collision
  operator L_Omega = Omega.grad + Sigma_t is assembled once and LU-factorized;
  a "sweep" is a triangular solve. scipy-bound, CPU only; the only engine that
  supports the periodic torus and the hybrid iface coupling.
* ``"levels"`` (default on GPU via ``device=``; also slightly faster on CPU
  for large meshes): a level-scheduled wavefront. For vacuum bc the upwind
  dependency graph is a DAG (Omega . centroid strictly increases along every
  dependency edge), so cells sort into topological levels; a sweep is
  max-level sequential steps, each one batched gather/scatter update over
  *all* ordinates at once (plus batched per-cell 3x3 corner-block inverses
  for SCB) on the numpy/cupy array backend -- the GPU-friendly shape, and
  machine-precision identical to the LU engine. (cupy path untested: no GPU
  on the dev machine.)

Either way the within-group scattering fixed
point is collapsed by the same acceleration menu as the Cartesian solver
(``acceleration=``: "dsa" default, "dsa-gmres", "gmres", "si" -- see
``ndgpu.sn``), with the DSA error solve a triangular finite-volume diffusion
operator matching ``TriGroupOperator`` (harmonic face D = 1/(3 Sigma_t),
4D/h^2 coupling, Marshak-vacuum Robin on faces to excised/off-mesh cells --
the *error* equation has zero incoming there -- and periodic wrap on the
torus), one lazy sparse LU per group. All of it sits inside a fission power
iteration.

Upwind (step) differencing keeps every angular flux non-negative -- robust
through the near-black B4C drums where diamond differencing would ring -- at the
cost of being first-order in space (more numerically diffusive than the diamond
scheme of ``ndgpu.sn``). The equilateral triangle has edge length h, area
sqrt(3)/4 h^2, and the six (type, edge) outward normals below.

Boundaries: ``vacuum`` (the HP-MR core surface; incoming flux zero on edges
facing an excised/void cell or off-mesh) and ``periodic`` (wrap the lattice to a
torus -- an infinite medium whose flat-flux k is exactly k_inf, used to validate
the operator). CPU/numpy reference, not the GPU path.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, factorized, gmres

from .backend import asnumpy, get_backend
from .materials import Material
from .sn import SNResult, _anderson, _cmfd_power, quadrature_2d, quadrature_3d
from .solver import Fields
from .stencil import face_alpha, harmonic_mean
from .tri import TriGrid

_SQRT3_2 = np.sqrt(3.0) / 2.0

# (source type, neighbour rhombus offset di, dj, neighbour type, outward normal).
# Down triangle (t=0): hypotenuse->up(i,j), bottom->up(i-1,j), left->up(i,j-1).
# Up   triangle (t=1): hypotenuse->down(i,j), top->down(i+1,j), right->down(i,j+1).
_EDGES = [
    (0, 0, 0, 1, (_SQRT3_2, 0.5)),
    (0, -1, 0, 1, (0.0, -1.0)),
    (0, 0, -1, 1, (-_SQRT3_2, 0.5)),
    (1, 0, 0, 0, (-_SQRT3_2, -0.5)),
    (1, 1, 0, 0, (0.0, 1.0)),
    (1, 0, 1, 0, (_SQRT3_2, -0.5)),
]


class TriSNTransportSolver:
    """S_N transport k-eigenvalue solver on a triangular mesh.

    Parameters
    ----------
    grid          : TriGrid (2D; shape (nrows, ncols, 2)).
    materials     : a Material or list indexed by material_map.
    material_map  : int array of shape grid.shape; omit for a homogeneous medium.
    active        : optional bool mask of in-core cells (shape grid.shape). Cells
                    outside it are excised; active cells facing them see the
                    ``bc`` boundary law. Defaults to all-active.
    n_polar, n_azi: product-quadrature sizes (n_azi a multiple of 4).
    bc            : "vacuum" (default) or "periodic" (torus / infinite medium).
    scheme        : spatial differencing. "step" (default) -- upwind, robustly
                    non-negative, first-order. "scb" -- simple corner balance, a
                    second-order finite-volume scheme: each triangle is split into
                    three corner sub-volumes (3 unknowns per cell), with the cell
                    boundary half-edges upwinded to the neighbour's corner at the
                    shared vertex and the interior corner faces carrying the
                    average of the two corner fluxes. It is a genuine finite-volume
                    balance (not a difference stencil), stays linear so it
                    factorizes once, is exact for a flat flux (k_inf exact), and --
                    unlike the earlier edge-average scheme -- reaches second-order
                    convergence, resolving the HP-MR drum worth at far coarser mesh
                    than step. Costs ~3x the unknowns of step.
    """

    ACCELERATIONS = ("dsa", "dsa-gmres", "gmres", "si")

    def __init__(self, grid: TriGrid, materials, material_map=None, active=None,
                 n_polar: int = 3, n_azi: int = 12, bc: str = "vacuum",
                 scheme: str = "step", require_fissile: bool = True,
                 mix_material=None, mix_weight=None, acceleration: str = "dsa",
                 outer_acceleration: str = "cmfd", max_inner: int = 800,
                 engine: str | None = None, device: str = "cpu",
                 dsa_rtol: float = 1e-4, dsa_maxiter: int = 100,
                 cmfd_solver: str = "lu", sigma_t_shift=None):
        if len(grid.shape) not in (3, 4) or grid.shape[2] != 2:
            raise ValueError("tri-S_N grid shape must be (nr, nc, 2) or "
                             "(nr, nc, 2, nz) for extruded prisms")
        # bc: a single spec for all faces, or (radial, axial) for prisms. Each
        # spec is "vacuum" or "periodic" (tri-S_N has no reflective law).
        if isinstance(bc, (tuple, list)):
            if len(grid.shape) != 4:
                raise ValueError("per-(radial, axial) bc needs an extruded grid")
            bc_radial, bc_axial = bc
        else:
            bc_radial = bc_axial = bc
        for b in (bc_radial, bc_axial):
            if b not in ("vacuum", "periodic"):
                raise ValueError("bc specs must be 'vacuum' or 'periodic'")
        self.bc_radial, self.bc_axial = bc_radial, bc_axial
        if scheme not in ("step", "scb"):
            raise ValueError("scheme must be 'step' or 'scb'")
        if acceleration not in self.ACCELERATIONS:
            raise ValueError(f"acceleration must be one of {self.ACCELERATIONS}")
        if outer_acceleration not in ("cmfd", "power"):
            raise ValueError("outer_acceleration must be 'cmfd' or 'power'")
        self.xp = get_backend(device)
        if engine is None:                       # LU is scipy-bound: CPU only
            engine = "lu" if self.xp is np else "levels"
        if engine not in ("lu", "levels"):
            raise ValueError("engine must be 'lu' or 'levels'")
        if engine == "levels" and (self.bc_radial != "vacuum"
                                   or self.bc_axial != "vacuum"):
            raise ValueError("engine='levels' needs vacuum bc (the periodic "
                             "torus wraps the sweep dependency graph into cycles)")
        if engine == "lu" and self.xp is not np:
            raise ValueError("engine='lu' is CPU-only; use engine='levels'")
        self.engine = engine
        self.acceleration = acceleration
        self.outer_acceleration = outer_acceleration
        self.max_inner = max_inner
        self.dsa_rtol = dsa_rtol            # device DSA solve: loose tol + capped
        self.dsa_maxiter = dsa_maxiter      # iters (a preconditioner, not exact)
        self.cmfd_solver = cmfd_solver      # CMFD solve: "lu" or "mg" (O(N), 3D)
        self.scheme = scheme
        self.grid = grid
        self.nr, self.nc = grid.shape[0], grid.shape[1]
        self.bc = self.bc_radial                             # in-plane law (2D paths)
        self.h = grid.side
        self.area = (np.sqrt(3.0) / 4.0) * self.h ** 2
        self.is3d = len(grid.shape) == 4                     # extruded prisms
        if self.is3d:
            self.nz = grid.shape[3]
            self.dz = grid.dz
            self.vol = self.area * self.dz                   # prism cell measure
            if acceleration in ("gmres", "dsa-gmres"):       # Phase 3: dsa / si
                self.acceleration = "dsa"
        else:
            self.nz = 1
        self.N = self.nr * self.nc * 2 * self.nz
        self._measure = self.vol if self.is3d else self.area  # step cell measure

        mats = [materials] if isinstance(materials, Material) else list(materials)
        self.G = mats[0].n_groups
        mmap = (np.zeros(grid.shape, int) if material_map is None
                else np.asarray(material_map).reshape(grid.shape))
        self.active = (np.ones(grid.shape, bool) if active is None
                       else np.asarray(active).reshape(grid.shape))

        # Per-cell fields via the validated Fields blend, so the optional polar
        # volume-mixing (mix_material/mix_weight) that dilutes the thin B4C arc
        # into the drum cells applies to S_N exactly as it does to diffusion:
        # cross sections mix linearly, D harmonically, chi by fission share. With
        # Sigma_t = 1/(3D) (no explicit total) the linear Sigma_t mix equals the
        # harmonic-D mix, so S_N stays P1-consistent with the diffusion reference.
        f = Fields(np, grid, mats, mmap, np.float64,
                   mix_material=mix_material, mix_weight=mix_weight)
        self.st = np.stack(f.sigma_t)                        # (G, nr, nc, 2)
        removal = np.stack(f.removal)
        self.ss_self = np.maximum(self.st - removal, 0.0)
        # Transient collision shift: backward Euler adds theta_g = 1/(v_g dt) to
        # the total cross section. Applied *after* ss_self (within-group
        # scattering is untouched) and *before* _prefactor, so the prefactored
        # streaming+collision operator, the DSA matrix and the CMFD drift
        # operator all pick it up with no further plumbing -- the tri engine
        # factorizes its operator once, so the shift belongs at construction.
        self.sigma_t_shift = None
        if sigma_t_shift is not None:
            shift = np.asarray(sigma_t_shift, float).reshape(-1)
            if shift.size != self.G:
                raise ValueError("sigma_t_shift needs one value per group")
            if np.any(shift < 0.0):
                raise ValueError("sigma_t_shift must be non-negative")
            self.sigma_t_shift = shift
            self.st = self.st + shift.reshape((self.G,) + (1,) * (self.st.ndim - 1))
        self.nsf = np.stack(f.nu_sigma_f)
        self.chi = np.stack(f.chi)
        self.scatter = [[None] * self.G for _ in range(self.G)]
        for gf in range(self.G):
            for gt in range(self.G):
                if gf != gt and f.sigma_s[gf][gt] is not None:
                    self.scatter[gf][gt] = np.asarray(f.sigma_s[gf][gt])
        if require_fissile and not np.any(self.nsf):
            raise ValueError("no fissile material: k-eigenvalue is undefined")

        if self.is3d:
            self.mu, self.eta, self.xi, self.w = quadrature_3d(n_polar, n_azi)
        else:
            self.mu, self.eta, self.w = quadrature_2d(n_polar, n_azi)
            self.xi = None
        self.M = self.mu.size
        self._cell = np.arange(self.N).reshape(grid.shape)
        self._act_flat = self.active.reshape(-1)
        self._dsa_fac = [None] * self.G                      # lazy DSA LU per group
        self._sweep_count = 0
        if not self.is3d:
            self._build_nbr_maps()                           # CMFD/currents: 2D only
        self._prefactor()

    def _build_nbr_maps(self):
        """Per _EDGES entry: the neighbour lookup (clipped indices + activity
        masks) shared by the face-current accumulation and the CMFD assembly."""
        nr, nc = self.nr, self.nc
        ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        self._nbr_maps = []
        for (t, di, dj, tn, nrm) in _EDGES:
            ni, nj = ii + di, jj + dj
            if self.bc == "periodic":
                ni, nj = ni % nr, nj % nc
                inb = np.ones((nr, nc), bool)
            else:
                inb = (ni >= 0) & (ni < nr) & (nj >= 0) & (nj < nc)
            nic, njc = np.clip(ni, 0, nr - 1), np.clip(nj, 0, nc - 1)
            nbr_act = np.zeros((nr, nc), bool)
            nbr_act[inb] = self.active[ni[inb], nj[inb], tn]
            self._nbr_maps.append(
                (t, tn, nic, njc, nbr_act, self.active[:, :, t], nrm))

    def _prefactor(self):
        if self.is3d:
            if self.engine == "levels":
                if self.scheme == "scb":
                    self._build_corners_3d()
                self._setup_levels_3d()
            elif self.scheme == "scb":
                self._build_corners_3d()
                self._prefactor_scb_3d()
            else:
                self._prefactor_step_3d()
            return
        if self.engine == "levels":
            if self.scheme == "scb":
                self._build_corners()
            self._setup_levels()
            return
        if self.scheme == "scb":
            self._build_corners()
            self._prefactor_scb()
        else:
            self._prefactor_step()

    def _prefactor_step(self):
        """Assemble and LU-factorize L_Omega = Omega.grad + Sigma_t (upwind) for
        every ordinate and group, once."""
        h, area = self.h, self.area
        nr, nc = self.nr, self.nc
        cell = self._cell
        act = self.active
        # base diagonal per group = Sigma_t * area (collision); streaming adds to it.
        self._solvers = [[None] * self.M for _ in range(self.G)]
        for g in range(self.G):
            st_area = (self.st[g] * area).reshape(-1)
            for m in range(self.M):
                mu, eta = self.mu[m], self.eta[m]
                diag = st_area.copy()
                rows, cols, vals = [], [], []
                for (t, di, dj, tn, (nx, ny)) in _EDGES:
                    On = (mu * nx + eta * ny) * h            # (Omega.n) * edge length
                    src = cell[:, :, t]                      # (nr, nc) source cell ids
                    src_act = act[:, :, t]
                    ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
                    ni, nj = ii + di, jj + dj
                    if self.bc == "periodic":
                        ni %= nr; nj %= nc
                        inb = np.ones_like(ni, bool)
                    else:
                        inb = (ni >= 0) & (ni < nr) & (nj >= 0) & (nj < nc)
                    nbr = np.full((nr, nc), -1)
                    nbr[inb] = cell[ni[inb], nj[inb], tn]
                    nbr_act = np.zeros((nr, nc), bool)
                    nbr_act[inb] = act[ni[inb], nj[inb], tn]
                    valid = src_act & nbr_act                # interior coupled edge
                    if On > 0:                               # outflow: psi_face = psi_c
                        # leakage out across every edge of an active cell (to an
                        # active neighbour or across the vacuum/void boundary).
                        d = np.where(src_act, On, 0.0)
                        np.add.at(diag, src.reshape(-1), d.reshape(-1))
                    else:                                    # inflow: psi_face = psi_nbr
                        s = src[valid]; n = nbr[valid]
                        rows.append(s); cols.append(n)
                        vals.append(np.full(s.size, On))
                        # inflow across a vacuum/void boundary contributes 0.
                # inactive cells: identity rows (psi = 0)
                inact = ~self._act_flat
                diag = np.where(inact, 1.0, diag)
                rows.append(np.arange(self.N)); cols.append(np.arange(self.N))
                vals.append(diag)
                A = sp.csr_matrix((np.concatenate(vals),
                                   (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(self.N, self.N))
                # zero any stray couplings out of inactive rows
                self._solvers[g][m] = factorized(A.tocsc())

    @staticmethod
    def _add_face(On, src, src_act, nbr, nbr_act, diag, rows, cols, vals):
        """One face family's step-upwind contribution: outflow (On>0) adds |On|
        to the source diagonal (leakage out, to a neighbour or the boundary);
        inflow (On<0) couples On to the upwind neighbour, only where both cells
        are active (inflow across a vacuum/void boundary contributes 0)."""
        if On > 0.0:
            np.add.at(diag, src.reshape(-1),
                      np.where(src_act, On, 0.0).reshape(-1))
        elif On < 0.0:
            valid = src_act & nbr_act
            s = src[valid]
            rows.append(s); cols.append(nbr[valid])
            vals.append(np.full(s.size, On))

    def _prefactor_step_3d(self):
        """Assemble + LU-factorize L_Omega = Omega.grad + Sigma_t (step upwind)
        per ordinate and group on the extruded triangular-prism mesh. Each prism
        has 3 lateral faces (tri edges extruded: area h*dz, in-plane normal, the
        2D _EDGES normals) coupling within a z layer, plus 2 axial caps (area =
        tri area, normal +/-z) coupling adjacent layers via the z-cosine xi."""
        nr, nc, nz, N = self.nr, self.nc, self.nz, self.N
        h, area, dz, vol = self.h, self.area, self.dz, self.vol
        cell, act = self._cell, self.active                  # (nr, nc, 2, nz)
        ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        rad_per = self.bc_radial == "periodic"               # in-plane wrap
        ax_per = self.bc_axial == "periodic"                 # axial wrap
        self._solvers = [[None] * self.M for _ in range(self.G)]
        for g in range(self.G):
            st_vol = (self.st[g] * vol).reshape(-1)
            for m in range(self.M):
                mu, eta, xi = self.mu[m], self.eta[m], self.xi[m]
                diag = st_vol.copy()
                rows, cols, vals = [], [], []
                for (t, di, dj, tn, (nx, ny)) in _EDGES:      # lateral faces
                    On = (mu * nx + eta * ny) * (h * dz)
                    ni, nj = ii + di, jj + dj
                    if rad_per:
                        inb = np.ones((nr, nc), bool); ni2, nj2 = ni % nr, nj % nc
                    else:
                        inb = (ni >= 0) & (ni < nr) & (nj >= 0) & (nj < nc)
                        ni2, nj2 = np.clip(ni, 0, nr - 1), np.clip(nj, 0, nc - 1)
                    nbr = np.full((nr, nc, nz), -1)
                    nbr_act = np.zeros((nr, nc, nz), bool)
                    nbr[inb] = cell[ni2[inb], nj2[inb], tn, :]
                    nbr_act[inb] = act[ni2[inb], nj2[inb], tn, :]
                    self._add_face(On, cell[:, :, t, :], act[:, :, t, :],
                                   nbr, nbr_act, diag, rows, cols, vals)
                kk = np.arange(nz)
                for dk in (+1, -1):                           # axial caps
                    On = (xi if dk > 0 else -xi) * area
                    nk = kk + dk
                    if ax_per:
                        inbz = np.ones(nz, bool); nk2 = nk % nz
                    else:
                        inbz = (nk >= 0) & (nk < nz); nk2 = np.clip(nk, 0, nz - 1)
                    nbr = np.full((nr, nc, 2, nz), -1)
                    nbr_act = np.zeros((nr, nc, 2, nz), bool)
                    nbr[:, :, :, inbz] = cell[:, :, :, nk2[inbz]]
                    nbr_act[:, :, :, inbz] = act[:, :, :, nk2[inbz]]
                    self._add_face(On, cell, act, nbr, nbr_act,
                                   diag, rows, cols, vals)
                diag = np.where(~self._act_flat, 1.0, diag)   # excised: identity
                rows.append(np.arange(N)); cols.append(np.arange(N))
                vals.append(diag)
                A = sp.csr_matrix(
                    (np.concatenate([np.atleast_1d(v) for v in vals]),
                     (np.concatenate([np.atleast_1d(r) for r in rows]),
                      np.concatenate([np.atleast_1d(c) for c in cols]))),
                    shape=(N, N))
                self._solvers[g][m] = factorized(A.tocsc())

    def _setup_levels_3d(self):
        """Level schedule for the prism step sweep (engine='levels', GPU path).

        Same topological scheme as the 2D step engine but the upwind dependency
        graph adds axial edges, so a cell has up to FIVE inflow faces (3 lateral
        + 2 axial) instead of 3. The per-level gather tables are width-5; the
        width-agnostic _run_levels/_sweep_dev/CUDA-graph machinery is otherwise
        unchanged. Vacuum bc only (a periodic wrap would cycle the graph)."""
        xp, N, M = self.xp, self.N, self.M
        nr, nc, nz = self.nr, self.nc, self.nz
        h, area, dz = self.h, self.area, self.dz
        cell, act = self._cell, self.active
        act_flat = self._act_flat
        ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        NIN = 5
        outsum = np.zeros((M, N))
        in_nbr = np.zeros((M, N, NIN), np.int64)
        in_coef = np.zeros((M, N, NIN))
        in_cnt = np.zeros((M, N), np.int64)
        es = [[] for _ in range(M)]
        ed = [[] for _ in range(M)]
        # (s_all outflow ids, sv/nv inflow src/nbr ids, kind, g1, g2) per face
        faces = []
        for (t, di, dj, tn, (nx, ny)) in _EDGES:              # lateral
            src, src_act = cell[:, :, t, :], act[:, :, t, :]
            ni, nj = ii + di, jj + dj
            inb = (ni >= 0) & (ni < nr) & (nj >= 0) & (nj < nc)
            ni2, nj2 = np.clip(ni, 0, nr - 1), np.clip(nj, 0, nc - 1)
            nbr = np.full((nr, nc, nz), -1)
            nbr_act = np.zeros((nr, nc, nz), bool)
            nbr[inb] = cell[ni2[inb], nj2[inb], tn, :]
            nbr_act[inb] = act[ni2[inb], nj2[inb], tn, :]
            valid = src_act & nbr_act
            faces.append((src[src_act], src[valid], nbr[valid], "lat", nx, ny))
        for dk in (+1, -1):                                    # axial caps
            nk = np.arange(nz) + dk
            inbz = (nk >= 0) & (nk < nz)
            nk2 = np.clip(nk, 0, nz - 1)
            nbr = np.full((nr, nc, 2, nz), -1)
            nbr_act = np.zeros((nr, nc, 2, nz), bool)
            nbr[:, :, :, inbz] = cell[:, :, :, nk2[inbz]]
            nbr_act[:, :, :, inbz] = act[:, :, :, nk2[inbz]]
            valid = act & nbr_act
            faces.append((cell[act], cell[valid], nbr[valid], "ax", dk, None))
        for (s_all, sv, nv, kind, g1, g2) in faces:
            for m in range(M):
                if kind == "lat":
                    On = (self.mu[m] * g1 + self.eta[m] * g2) * (h * dz)
                else:
                    On = (self.xi[m] if g1 > 0 else -self.xi[m]) * area
                if On > 0.0:
                    outsum[m][s_all] += On
                elif On < 0.0:
                    es[m].append(nv); ed[m].append(sv)
                    slot = in_cnt[m][sv]
                    in_nbr[m, sv, slot] = nv
                    in_coef[m, sv, slot] = -On
                    in_cnt[m][sv] += 1
        # Kahn's algorithm per ordinate (identical to the 2D engine).
        lvl_of = np.full((M, N), -1, np.int64)
        n_act = int(act_flat.sum())
        for m in range(M):
            se = np.concatenate(es[m]) if es[m] else np.zeros(0, np.int64)
            de = np.concatenate(ed[m]) if ed[m] else np.zeros(0, np.int64)
            indeg = np.zeros(N, np.int64)
            np.add.at(indeg, de, 1)
            perm = np.argsort(se, kind="stable")
            se_s, de_s = se[perm], de[perm]
            indptr = np.searchsorted(se_s, np.arange(N + 1))
            ready = act_flat & (indeg == 0)
            done = l = 0
            while done < n_act:
                ids = np.where(ready)[0]
                if ids.size == 0:
                    raise RuntimeError("sweep dependency cycle (unexpected for "
                                       "vacuum bc)")
                lvl_of[m, ids] = l
                done += ids.size
                ready[ids] = False
                counts = indptr[ids + 1] - indptr[ids]
                tot = int(counts.sum())
                if tot:
                    starts = np.repeat(indptr[ids], counts)
                    offs = (np.arange(tot) - np.repeat(
                        np.concatenate(([0], np.cumsum(counts)[:-1])), counts))
                    deps = de_s[starts + offs]
                    np.subtract.at(indeg, deps, 1)
                    hit = deps[indeg[deps] == 0]
                    ready[hit[lvl_of[m, hit] < 0]] = True
                l += 1
        # per-level width-5 tables (same layout as the 2D step branch)
        mm, cc = np.where(lvl_of >= 0)
        lv = lvl_of[mm, cc]
        order = np.argsort(lv, kind="stable")
        mm, cc, lv = mm[order], cc[order], lv[order]
        bounds = np.searchsorted(lv, np.arange(int(lv.max()) + 2))
        spans = list(zip(bounds[:-1], bounds[1:]))
        i32 = np.int32
        self._w_xp = xp.asarray(self.w)
        self._lv_group = {}
        self._graphs = {}
        self.graphs_active = None
        self._graph_error = None
        self._act_flat_dev = xp.asarray(act_flat)
        if self.scheme == "scb":
            self._setup_levels_scb_tables_3d(mm, cc, spans)
            return
        pidx = mm * N + cc
        off = (np.arange(M) * N)[:, None, None]
        nbr_f = np.where(in_coef > 0.0, off + in_nbr, 0).reshape(M * N, NIN)
        coef = in_coef.reshape(M * N, NIN)
        self._lv_out = outsum
        self._levels = []
        for a, b in spans:
            p = pidx[a:b]
            self._levels.append((xp.asarray(cc[a:b].astype(i32)),
                                 xp.asarray(p.astype(i32)),
                                 xp.asarray(nbr_f[p].astype(i32)),
                                 xp.asarray(coef[p])))
        self._bufs = dict(
            psi=xp.zeros(M * N), rhs=xp.zeros(N),
            phiM=xp.zeros(N), psiw=xp.zeros((M, N)),
            wcol=xp.asarray(self.w[:, None]),
            work=[(xp.zeros((n, NIN)), xp.zeros(n), xp.zeros(n))
                  for n in (b - a for a, b in spans)])

    def _setup_levels_scb_tables_3d(self, mm, cc, spans):
        """SCB per-level tables for the prism levels engine: the corner external
        inflow gather (width 4 = 2 lateral half-edges + 2 axial caps) + buffers
        for the batched 3x3 corner-block sweep. The within-cell block (internal
        in-plane faces) is _lv_tables' Binv; only the EXTERNAL faces (upwind to
        a neighbour prism's corner) are gathered here."""
        xp, N, M = self.xp, self.N, self.M
        h, dz, area = self.h, self.dz, self.area
        d = self._scb
        ac, K = d["ac"], d["K"]
        aidx = np.full(N, -1, np.int64)
        aidx[ac] = np.arange(K)
        rr = aidx[cc]
        pidx = mm * K + rr
        h2d, Acap = (h / 2.0) * dz, area / 3.0
        off = (np.arange(M) * (3 * K))[:, None, None, None]
        extn, enbr = d["ext_n"], d["ext_nbr"]
        oe = (self.mu[:, None, None, None] * extn[None, ..., 0]
              + self.eta[:, None, None, None] * extn[None, ..., 1])      # (M,K,3,2)
        c_lat = np.where((oe < 0.0) & (enbr >= 0)[None], -oe * h2d, 0.0)
        n_lat = np.where(c_lat > 0.0, off + np.maximum(enbr, 0)[None], 0)
        oz = np.stack([self.xi, -self.xi], -1)[:, None, None, :]         # (M,1,1,2) up/down
        anbr = d["ax_nbr"]
        c_ax = np.where((oz < 0.0) & (anbr >= 0)[None], -oz * Acap, 0.0)
        n_ax = np.where(c_ax > 0.0, off + np.maximum(anbr, 0)[None], 0)
        coef = np.concatenate([c_lat, c_ax], 3).reshape(M * K, 3, 4)     # width 4
        nbr_f = np.concatenate([n_lat, n_ax], 3).reshape(M * K, 3, 4)
        self._lv_oe = oe                                                 # host, Binv
        self._levels = []
        for a, b in spans:
            p = pidx[a:b]
            sidx = ((p[:, None] * 3) + np.arange(3)).ravel()
            self._levels.append((xp.asarray(rr[a:b].astype(np.int32)),
                                 xp.asarray(p.astype(np.int32)),
                                 xp.asarray(sidx.astype(np.int32)),
                                 xp.asarray(nbr_f[p].astype(np.int32)),
                                 xp.asarray(coef[p])))
        self._bufs = dict(
            psi=xp.zeros(M * K * 3), rhs=xp.zeros(K),
            IF=xp.zeros((M * K, 3)),
            mk=xp.zeros((M, K)), mk_w=xp.zeros((M, K)), phk=xp.zeros(K),
            w3col=xp.asarray(self.w[:, None] / 3.0),
            work=[(xp.zeros((n, 3, 4)), xp.zeros((n, 3)), xp.zeros(n),
                   xp.zeros((n, 3)), xp.zeros((n, 3, 3)), xp.zeros((n, 3)))
                  for n in (b - a for a, b in spans)])
        self._ac_dev = xp.asarray(ac)
        self._bufs["phiN"] = xp.zeros(N)

    def _sweep_3d(self, g, src_flat):
        """One isotropic-source sweep on the prism mesh: phi = sum_m w_m psi_m,
        psi_m = L_Omega^-1 (src * cell-volume) for active cells."""
        self._sweep_count += 1
        rhs = np.where(self._act_flat, src_flat * self.vol, 0.0)
        phi = np.zeros(self.N)
        for m in range(self.M):
            phi += self.w[m] * self._solvers[g][m](rhs)
        return phi

    # ---- level-scheduled sweep (engine="levels", the GPU path) -------------
    def _setup_levels(self):
        """Topological level schedule for the batched sweep. Per ordinate the
        upwind dependency graph on the tri lattice is a DAG for vacuum bc (the
        potential Omega.x_centroid strictly increases along every dependency
        edge, faces with Omega.n = 0 carrying none), so cells sort into levels
        and each level is one batched update -- concatenated across all
        ordinates, so a sweep is max-level-count sequential steps of large
        gather/scatter (and, for SCB, batched 3x3 corner-block) kernels on the
        array backend. The unknowns and update algebra are identical to the LU
        engine's assembled systems, solved in topological order."""
        xp, N, M, h = self.xp, self.N, self.M, self.h
        cell, act = self._cell, self._act_flat
        scb = self.scheme == "scb"
        if not scb:
            outsum = np.zeros((M, N))
            in_nbr = np.zeros((M, N, 3), np.int64)
            in_coef = np.zeros((M, N, 3))
            in_cnt = np.zeros((M, N), np.int64)
        es = [[] for _ in range(M)]
        ed = [[] for _ in range(M)]
        for (t, tn, nic, njc, nbr_act, src_act, (nxv, nyv)) in self._nbr_maps:
            src = cell[:, :, t]
            nbr = cell[nic, njc, tn]
            valid = src_act & nbr_act
            s_all = src[src_act]
            sv, nv = src[valid], nbr[valid]
            for m in range(M):
                On = (self.mu[m] * nxv + self.eta[m] * nyv) * h
                if On > 0.0:
                    if not scb:
                        outsum[m][s_all] += On
                elif On < 0.0:
                    es[m].append(nv)
                    ed[m].append(sv)
                    if not scb:
                        slot = in_cnt[m][sv]
                        in_nbr[m, sv, slot] = nv
                        in_coef[m, sv, slot] = -On
                        in_cnt[m][sv] += 1
        # Kahn's algorithm per ordinate, vectorized over each level.
        lvl_of = np.full((M, N), -1, np.int64)
        n_act = int(act.sum())
        for m in range(M):
            se = (np.concatenate(es[m]) if es[m] else np.zeros(0, np.int64))
            de = (np.concatenate(ed[m]) if ed[m] else np.zeros(0, np.int64))
            indeg = np.zeros(N, np.int64)
            np.add.at(indeg, de, 1)
            perm = np.argsort(se, kind="stable")
            se_s, de_s = se[perm], de[perm]
            indptr = np.searchsorted(se_s, np.arange(N + 1))
            ready = act & (indeg == 0)
            done = 0
            l = 0
            while done < n_act:
                ids = np.where(ready)[0]
                if ids.size == 0:
                    raise RuntimeError("sweep dependency cycle (unexpected "
                                       "for vacuum bc)")
                lvl_of[m, ids] = l
                done += ids.size
                ready[ids] = False
                counts = indptr[ids + 1] - indptr[ids]
                tot = int(counts.sum())
                if tot:
                    starts = np.repeat(indptr[ids], counts)
                    offs = (np.arange(tot)
                            - np.repeat(np.concatenate(
                                ([0], np.cumsum(counts)[:-1])), counts))
                    deps = de_s[starts + offs]
                    np.subtract.at(indeg, deps, 1)
                    hit = deps[indeg[deps] == 0]
                    ready[hit[lvl_of[m, hit] < 0]] = True
                l += 1
        # (m, cell) pairs grouped by level, concatenated across ordinates.
        # Everything below is stored PER LEVEL, pre-gathered, so the sweep loop
        # is a short fixed sequence of allocation-free out= kernels per level
        # (few launches, and safe to record into a CUDA graph on cupy).
        mm, cc = np.where(lvl_of >= 0)
        lv = lvl_of[mm, cc]
        order = np.argsort(lv, kind="stable")
        mm, cc, lv = mm[order], cc[order], lv[order]
        bounds = np.searchsorted(lv, np.arange(int(lv.max()) + 2))
        spans = list(zip(bounds[:-1], bounds[1:]))
        i32 = np.int32
        self._w_xp = xp.asarray(self.w)
        self._lv_group = {}
        self._graphs = {}
        self.graphs_active = None                # None until a GPU sweep runs
        self._graph_error = None                 # reason capture fell back, if any
        self._act_flat_dev = xp.asarray(self._act_flat)   # device-resident masks
        if scb:
            d = self._scb
            ac, K = d["ac"], d["K"]
            aidx = np.full(N, -1, np.int64)
            aidx[ac] = np.arange(K)
            rr = aidx[cc]
            pidx = mm * K + rr
            # inflow gather tables per (m, corner-cell, corner, face)
            extn, nbr = d["ext_n"], d["ext_nbr"]
            oe = (self.mu[:, None, None, None] * extn[None, ..., 0]
                  + self.eta[:, None, None, None] * extn[None, ..., 1])
            coef = np.where((oe < 0.0) & (nbr >= 0)[None],
                            -oe * (h / 2.0), 0.0)              # (M, K, 3, 2)
            off = (np.arange(M) * (3 * K))[:, None, None, None]
            nbr_f = np.where(coef > 0.0, off + np.maximum(nbr, 0)[None],
                             0).reshape(M * K, 3, 2)
            coef = coef.reshape(M * K, 3, 2)
            self._lv_oe = oe                                   # host, Binv + iface
            self._levels = []
            for a, b in spans:
                p = pidx[a:b]
                sidx = ((p[:, None] * 3) + np.arange(3)).ravel()
                self._levels.append((xp.asarray(rr[a:b].astype(i32)),
                                     xp.asarray(p.astype(i32)),
                                     xp.asarray(sidx.astype(i32)),
                                     xp.asarray(nbr_f[p].astype(i32)),
                                     xp.asarray(coef[p])))
            n_max = max(b - a for a, b in spans)
            self._bufs = dict(
                psi=xp.zeros(M * K * 3), rhs=xp.zeros(K),
                IF=xp.zeros((M * K, 3)),
                mk=xp.zeros((M, K)), mk_w=xp.zeros((M, K)), phk=xp.zeros(K),
                w3col=xp.asarray(self.w[:, None] / 3.0),
                work=[(xp.zeros((n, 3, 2)), xp.zeros((n, 3)), xp.zeros(n),
                       xp.zeros((n, 3)), xp.zeros((n, 3, 3)), xp.zeros((n, 3)))
                      for n in (b - a for a, b in spans)])
            self._ac_dev = xp.asarray(ac)
            self._bufs["phiN"] = xp.zeros(N)          # full-N device phi scatter
        else:
            pidx = mm * N + cc
            off = (np.arange(M) * N)[:, None, None]
            nbr_f = np.where(in_coef > 0.0, off + in_nbr, 0).reshape(M * N, 3)
            coef = in_coef.reshape(M * N, 3)
            self._lv_out = outsum                              # host, per-group denom
            self._levels = []
            for a, b in spans:
                p = pidx[a:b]
                self._levels.append((xp.asarray(cc[a:b].astype(i32)),
                                     xp.asarray(p.astype(i32)),
                                     xp.asarray(nbr_f[p].astype(i32)),
                                     xp.asarray(coef[p])))
            self._bufs = dict(
                psi=xp.zeros(M * N), rhs=xp.zeros(N),
                phiM=xp.zeros(N), psiw=xp.zeros((M, N)),
                wcol=xp.asarray(self.w[:, None]),
                work=[(xp.zeros((n, 3)), xp.zeros(n), xp.zeros(n))
                      for n in (b - a for a, b in spans)])

    def _lv_tables(self, g):
        """Group-dependent sweep tables, pre-gathered per level: the step
        denominator, or the SCB per-cell 3x3 corner-block inverses (batched),
        per ordinate."""
        if g in self._lv_group:
            return self._lv_group[g]
        xp, M = self.xp, self.M
        if self.scheme == "step":
            denom = (self.st[g].reshape(-1)[None, :] * self._measure
                     + self._lv_out).reshape(-1)
            out = [xp.asarray(denom[asnumpy(lev[1])]) for lev in self._levels]
        else:
            d = self._scb
            K = d["K"]
            dzf = self.dz if self.is3d else 1.0              # lateral face area x dz
            h2 = (self.h / 2.0) * dzf
            hi = (self.h / (2.0 * np.sqrt(3.0))) * dzf
            A3 = (self.area / 3.0) * dzf                     # corner sub-volume
            ax_out = (np.abs(self.xi) * (self.area / 3.0)    # axial cap outflow
                      if self.is3d else np.zeros(M))
            st_c = self.st[g].reshape(-1)[d["ac"]]
            oi = (self.mu[:, None, None, None] * d["int_n"][None, ..., 0]
                  + self.eta[:, None, None, None] * d["int_n"][None, ..., 1])
            wloc = d["int_w"] - np.arange(K)[:, None, None] * 3   # (K, 3, 2)
            rK = np.arange(K)
            Binv = np.empty((M, K, 3, 3))
            for m in range(M):
                B = np.zeros((K, 3, 3))
                diag = (st_c[:, None] * A3
                        + (np.maximum(self._lv_oe[m], 0.0) * h2).sum(2)
                        + (oi[m] * hi * 0.5).sum(2) + ax_out[m])   # (K, 3)
                B[:, np.arange(3), np.arange(3)] = diag
                for lc in range(3):
                    for f in range(2):
                        B[rK, lc, wloc[:, lc, f]] += oi[m, :, lc, f] * hi * 0.5
                Binv[m] = np.linalg.inv(B)
            Binv = Binv.reshape(M * K, 3, 3)
            out = [xp.asarray(Binv[asnumpy(lev[1])]) for lev in self._levels]
        self._lv_group[g] = out
        return out

    def _run_levels(self, g, iface):
        """The level loop proper: a fixed sequence of allocation-free out=
        kernels on the persistent buffers (recordable into a CUDA graph).
        Inputs are read from bufs['rhs'] (+ bufs['IF']); outputs land in
        bufs['psi'] and the phi reduction buffer."""
        xp = self.xp
        bufs = self._bufs
        tab = self._lv_tables(g)
        psi = bufs["psi"]
        psi.fill(0.0)
        if self.scheme == "step":
            rhs = bufs["rhs"]
            for (cc, pidx, nbr, coef), den, (gl, red, rg) in zip(
                    self._levels, tab, bufs["work"]):
                xp.take(psi, nbr, out=gl)
                xp.multiply(gl, coef, out=gl)
                xp.sum(gl, axis=1, out=red)
                xp.take(rhs, cc, out=rg)
                xp.add(red, rg, out=red)
                xp.divide(red, den, out=red)
                xp.put(psi, pidx, red)
            # phi_n = sum_m w_m psi[m, n]: an elementwise weight + a reduction
            # (NOT matmul -- cuBLAS calls cannot be captured into a CUDA graph).
            xp.multiply(psi.reshape(self.M, self.N), bufs["wcol"],
                        out=bufs["psiw"])
            xp.sum(bufs["psiw"], axis=0, out=bufs["phiM"])
            return
        base = bufs["rhs"]
        IF = bufs["IF"]
        for (rr, pidx, sidx, nbr, coef), Binv, (gl, b2, bb, ifl, bm, p3) in zip(
                self._levels, tab, bufs["work"]):
            xp.take(psi, nbr, out=gl)
            xp.multiply(gl, coef, out=gl)
            xp.sum(gl, axis=2, out=b2)
            xp.take(base, rr, out=bb)
            xp.add(b2, bb[:, None], out=b2)
            if iface:
                xp.take(IF, pidx, axis=0, out=ifl)
                xp.add(b2, ifl, out=b2)
            # p3 = Binv @ b2, batched 3x3 matvec written as broadcast-multiply +
            # reduction so the captured loop calls no cuBLAS.
            xp.multiply(Binv, b2[:, None, :], out=bm)
            xp.sum(bm, axis=2, out=p3)
            xp.put(psi, sidx, p3)
        K = self._scb["K"]
        xp.sum(psi.reshape(self.M, K, 3), axis=2, out=bufs["mk"])
        # phi_k = sum_m (w_m/3) mk[m, k]: weight + reduction, no cuBLAS.
        xp.multiply(bufs["mk"], bufs["w3col"], out=bufs["mk_w"])
        xp.sum(bufs["mk_w"], axis=0, out=bufs["phk"])

    def _levels_exec(self, g, iface):
        """Execute the level loop -- directly on numpy; on cupy, captured once
        per (group, iface) into a CUDA graph and replayed as a single launch
        per sweep (the loop's kernels and buffer addresses are identical every
        sweep; only the input buffers' contents change). Any capture failure
        falls back permanently to the plain loop."""
        xp = self.xp
        if xp is np:
            self._run_levels(g, iface)
            return
        key = (g, iface)
        entry = self._graphs.get(key)
        if entry == "fallback":
            self._run_levels(g, iface)
            return
        if entry is None:
            self._lv_tables(g)                   # build outside the capture
            try:
                stream = xp.cuda.Stream(non_blocking=True)
                with stream:
                    # Warm-up: run once so the memory pool caches every scratch
                    # block and the kernels are compiled. Allocating from the
                    # pool (cudaMalloc) is illegal *during* capture, so the pool
                    # must already hold every block the captured run will reuse.
                    self._run_levels(g, iface)
                    stream.synchronize()
                    stream.begin_capture()
                    self._run_levels(g, iface)
                    graph = stream.end_capture()
                self._graphs[key] = graph
                self.graphs_active = True
            except Exception as e:               # capture unsupported/failed
                try:
                    stream.end_capture()
                except Exception:
                    pass
                self._graph_error = f"{type(e).__name__}: {e}"
                self._graphs[key] = "fallback"
                if self.graphs_active is None:
                    self.graphs_active = False
                self._run_levels(g, iface)
                return
            entry = self._graphs[key]
        entry.launch()                           # replays on the current stream

    def _buf_set(self, buf, values):
        """Copy host values into a (possibly device) persistent buffer."""
        if self.xp is np:
            np.copyto(buf, values)
        else:
            buf.set(np.ascontiguousarray(values))

    def _make_diff_solver(self, A_host, symmetric):
        """Return a callable ``solve(b) -> x`` for the assembled diffusion
        operator, on the solver's backend. NumPy: an exact sparse LU (scipy
        ``factorized``, cheap reused back-solves). CuPy: the operator is moved
        to the device once and solved iteratively -- Jacobi-preconditioned CG
        when ``symmetric`` (the DSA operator), else BiCGStab (the non-symmetric
        CMFD drift operator). On CuPy ``b`` and the returned ``x`` are device
        arrays, so the caller's iteration never leaves the GPU."""
        if self.xp is np:
            return factorized(A_host.tocsc())
        xp = self.xp
        import cupyx.scipy.sparse as csp
        from .linalg import pcg, bicgstab
        Ad = csp.csr_matrix(A_host.tocsr())
        inv_diag = 1.0 / Ad.diagonal()
        # DSA/CMFD is an ACCELERATOR, not part of the fixed point -- an inexact
        # solve only changes the outer rate, so cap the iterations and check
        # convergence rarely (each check is a GPU sync). The source-iteration
        # watchdog drops the acceleration if a too-weak solve stops contracting.
        rtol, maxit = self.dsa_rtol, self.dsa_maxiter
        chk = max(1, min(maxit, 25))

        def _solve(b):
            if symmetric:
                x, _ = pcg(lambda x: Ad @ x, b, xp.zeros_like(b), inv_diag, xp,
                           rtol=rtol, maxiter=maxit, check_every=chk,
                           raise_on_fail=False)
            else:
                x, _ = bicgstab(lambda x: Ad @ x, b, xp.zeros_like(b), inv_diag,
                                xp, rtol=rtol, maxiter=maxit)
            return x
        return _solve

    def _sweep_dev(self, g, src_dev):
        """Device-resident isotropic sweep for the levels engine: ``src_dev``
        is a backend array and the returned ``(phi_N, psi)`` are backend arrays
        too (no host round-trip), so a within-group iteration can stay on the
        GPU. ``phi_N`` is a reused buffer -- copy it if it must outlive the next
        sweep. Mirrors ``_sweep_levels`` exactly, only the source assembly and
        flux read-out move onto the device."""
        self._sweep_count += 1
        xp, bufs = self.xp, self._bufs
        if self.scheme == "step":
            rhs = bufs["rhs"]
            xp.multiply(src_dev, self._measure, out=rhs)
            rhs *= self._act_flat_dev                # zero the excised cells
            self._levels_exec(g, False)
            return bufs["phiM"].ravel(), bufs["psi"]
        ac = self._ac_dev
        xp.take(src_dev, ac, out=bufs["rhs"])
        bufs["rhs"] *= (self._measure / 3.0)
        self._levels_exec(g, False)
        pf = bufs["phiN"]
        pf.fill(0.0)
        pf[ac] = bufs["phk"].ravel()
        return pf, bufs["psi"]

    def _solve_group_dev(self, g, qext_flat, phi0, tol):
        """The (DSA-accelerated) within-group source iteration, run entirely on
        the backend for the levels engine: the sweep, the scattering source and
        the DSA diffusion correction all operate on device arrays, so the only
        host<->device traffic is ``qext_flat``/``phi0`` in and the flux out (per
        group per outer -- not per sweep). Numerically identical to the host
        loop in ``_solve_group``; on NumPy it simply runs on NumPy."""
        xp = self.xp
        ss = xp.asarray(self.ss_self[g].reshape(-1))
        b = self._sweep_dev(g, xp.asarray(qext_flat))[0].copy()
        tol = min(tol, 1e-4)
        accelerate = self.acceleration == "dsa"
        fac = self._dsa_factor(g) if accelerate else None
        phi = xp.asarray(np.asarray(phi0, float))
        prev = None
        bad = 0
        for _ in range(self.max_inner):
            half = b + self._sweep_dev(g, ss * phi)[0]
            new = half + fac(ss * (half - phi)) if accelerate else half
            d = float(xp.max(xp.abs(new - phi)))
            scale = max(float(xp.max(xp.abs(new))), 1e-300)
            phi = new
            if d <= tol * scale:
                break
            if accelerate:
                if prev is not None and d > prev:
                    bad += 1
                    if bad >= 3:
                        accelerate = False
                else:
                    bad = 0
                prev = d
        return asnumpy(phi)

    def _sweep_levels(self, g, src_flat, iface_in=None):
        """One batched level-scheduled sweep; returns (phi, psi) with psi the
        full per-ordinate angular flux (backend array, a reused buffer -- read
        it before the next sweep) for current folds. iface_in (SCB only)
        enters as a fixed source on the inflow interface half-edges, exactly
        as in the LU engine's ``_iface_rhs`` -- interface faces are boundary
        faces (no dependency edges), so the level schedule is unchanged."""
        bufs = self._bufs
        if self.scheme == "step":
            if iface_in is not None:
                raise ValueError("iface_in is SCB-only")
            self._buf_set(bufs["rhs"],
                          np.where(self._act_flat, src_flat * self._measure, 0.0))
            self._levels_exec(g, False)
            return asnumpy(bufs["phiM"]).ravel().copy(), bufs["psi"]
        d = self._scb
        ac, K = d["ac"], d["K"]
        self._buf_set(bufs["rhs"], src_flat[ac] * (self._measure / 3.0))
        iface = iface_in is not None
        if iface:
            psi_in, is_iface = iface_in
            cif = np.where(is_iface[None] & (self._lv_oe < 0.0),
                           -self._lv_oe * (self.h / 2.0), 0.0)  # (M, K, 3, 2)
            self._buf_set(bufs["IF"],
                          (cif * psi_in[None]).sum(3).reshape(self.M * K, 3))
        self._levels_exec(g, iface)
        phi = np.zeros(self.N)
        phi[ac] = asnumpy(bufs["phk"]).ravel()
        return phi, bufs["psi"]

    def _currents_from_psi(self, g, psi):
        """Fold the level-swept angular fluxes into the per-cell-edge net
        currents J6 (same convention as _sweep_currents), host-side."""
        psi = asnumpy(psi)
        M = self.M
        if self.scheme == "step":
            psi3 = psi.reshape(M, self.nr, self.nc, 2)
            J6 = np.zeros((6, self.nr, self.nc))
            for k, (t, tn, nic, njc, nbr_act, src_act, (nxv, nyv)) in \
                    enumerate(self._nbr_maps):
                for m in range(M):
                    On = self.mu[m] * nxv + self.eta[m] * nyv
                    if On > 0:
                        face = psi3[m, :, :, t]
                    else:
                        face = np.where(nbr_act, psi3[m, nic, njc, tn], 0.0)
                    J6[k] += (self.w[m] * On) * np.where(src_act, face, 0.0)
            return J6
        d = self._scb
        K = d["K"]
        psi3 = psi.reshape(M, K, 3)
        nbr = d["ext_nbr"]
        Jh = np.zeros((K, 3, 2))
        for m in range(M):
            oe = self._lv_oe[m]
            pm = psi3[m]
            nbr_psi = np.where(nbr >= 0, psi[m * 3 * K:][np.maximum(nbr, 0)], 0.0)
            face = np.where(oe > 0, pm[:, :, None], nbr_psi)
            Jh += self.w[m] * oe * (self.h / 2.0) * face
        return self._fold_half_currents(Jh)

    def _sweep(self, g, src_flat, iface_in=None):
        """phi = Sum_m w_m L_Omega^-1 (src * area) for an isotropic source.
        iface_in (SCB only) injects a hybrid incoming flux on interface edges."""
        if self.is3d and self.engine == "lu":
            if self.scheme == "scb":
                return self._sweep_scb_3d(g, src_flat)
            return self._sweep_3d(g, src_flat)
        self._sweep_count += 1
        if self.engine == "levels":
            if iface_in is not None:
                raise ValueError("engine='levels' does not support the hybrid "
                                 "iface_in coupling; use engine='lu'")
            return self._sweep_levels(g, src_flat)[0]
        if self.scheme == "scb":
            return self._sweep_scb(g, src_flat, iface_in)
        rhs = src_flat * self.area
        rhs = np.where(self._act_flat, rhs, 0.0)
        phi = np.zeros(self.N)
        for m in range(self.M):
            phi += self.w[m] * self._solvers[g][m](rhs)
        return phi

    # ---- transient engine adapter -----------------------------------------
    # The transport time term (1/v) dpsi/dt acts on the *angular* flux, so
    # backward Euler adds theta = 1/(v dt) to Sigma_t (folded in at construction
    # via sigma_t_shift, see __init__) and a per-ordinate source theta*psi_old.
    # TransientSNSolver drives this through the t_* methods only and never
    # interprets psi -- which matters here because the step scheme's psi lives on
    # cells (M, N) while SCB's lives on corner sub-volumes (M, K, 3), and the
    # time source must enter each corner balance with its own value.

    T_SHIFT_AT_CONSTRUCTION = True    # the operator is prefactored: shift at build
    T_CMFD_STEP = False               # transient CMFD on tri is Phase 3b

    def _t_require_lu(self):
        if self.engine != "lu":
            raise NotImplementedError(
                "transient tri-S_N needs engine='lu' (the level-scheduled sweep "
                "carries one shared cell source per ordinate; a per-ordinate time "
                "source needs wider level tables -- Phase 3b)")

    def _sweep_ang(self, g, src_flat, q_ang=None):
        """LU-engine sweep with an optional per-ordinate additive source and the
        full angular flux returned. ``q_ang`` is in the scheme's own layout --
        (M, N) cell values for step, (M, K, 3) corner values for SCB -- so the
        backward-Euler time source enters exactly the unknowns it belongs to.
        Returns (phi (N,), psi) with psi in that same layout.

        Both schemes solve the *same* prefactored per-ordinate operators as the
        steady sweeps; only the right-hand side gains the time source. Covers 2D
        triangles and 3D prisms (``_measure`` is the cell area or prism volume).
        """
        self._t_require_lu()
        self._sweep_count += 1
        meas = self._measure
        if self.scheme == "scb":
            d = self._scb
            ac, K = d["ac"], d["K"]
            base = np.repeat(src_flat[ac] * (meas / 3.0), 3).reshape(K, 3)
            phi = np.zeros(self.N)
            psi = np.empty((self.M, K, 3))
            for m in range(self.M):
                rhs = base if q_ang is None else base + q_ang[m] * (meas / 3.0)
                p = self._solvers[g][m](rhs.ravel()).reshape(K, 3)
                psi[m] = p
                phi[ac] += self.w[m] * p.mean(1)   # cell flux = mean of corners
            return phi, psi
        phi = np.zeros(self.N)
        psi = np.empty((self.M, self.N))
        for m in range(self.M):
            s = src_flat if q_ang is None else src_flat + q_ang[m]
            p = self._solvers[g][m](np.where(self._act_flat, s * meas, 0.0))
            psi[m] = p
            phi += self.w[m] * p
        return phi, psi

    def t_setup(self, theta):
        """theta is already folded into Sigma_t (sigma_t_shift at construction),
        so only the DSA factors -- which read the shifted Sigma_t -- are built."""
        self._t_require_lu()
        if self.sigma_t_shift is None or not np.allclose(
                self.sigma_t_shift, np.asarray(theta, float).reshape(-1)):
            raise ValueError(
                "tri-S_N transient engine must be constructed with "
                "sigma_t_shift = theta (T_SHIFT_AT_CONSTRUCTION)")
        self._t_dsa = [self._dsa_factor(g) if self.acceleration == "dsa" else None
                       for g in range(self.G)]

    def t_state0(self):
        return [None] * self.G          # vacuum/periodic fold into L_Omega

    def t_seed_psi(self, g, src, state):
        """Angular flux of the converged steady source -- the first step's
        psi_old. Called on the *unshifted* engine (the driver builds the shifted
        marching instance afterwards), so these sweeps use the true steady
        Sigma_t; only the psi layout is shared between the two instances."""
        _, psi = self._sweep_ang(g, np.asarray(src).reshape(-1))
        return psi, state

    def t_solve_group(self, g, qext, psi_old, tol, phi0, state):
        """DSA-accelerated within-group solve for one backward-Euler step, with
        the per-ordinate time source theta*psi_old held fixed. Mirrors the
        Cartesian engine; the boundary is folded into the operator, so there is
        no boundary fixed point."""
        theta = float(self.sigma_t_shift[g])
        q_ang = theta * psi_old
        qf = np.asarray(qext).reshape(-1)
        ss = self.ss_self[g].reshape(-1)
        fac = self._t_dsa[g]
        phi = (np.zeros(self.N) if phi0 is None
               else np.asarray(phi0).reshape(-1).copy())
        accelerate = fac is not None
        prev, bad = None, 0
        for _ in range(self.max_inner):
            half, _ = self._sweep_ang(g, ss * phi + qf, q_ang)
            new = half + fac(ss * (half - phi)) if accelerate else half
            d = float(np.max(np.abs(new - phi)))
            scale = max(float(np.max(np.abs(new))), 1e-300)
            phi = new
            if d <= tol * scale:
                break
            if accelerate:                       # same watchdog as the steady solve
                if prev is not None and d > prev:
                    bad += 1
                    if bad >= 3:
                        accelerate = False
                else:
                    bad = 0
                prev = d
        # DSA moves phi off the last sweep, so re-sweep at the converged flux to
        # return an angular flux consistent with it (this psi is psi_old next step).
        _, psi = self._sweep_ang(g, ss * phi + qf, q_ang)
        return phi.reshape(self.grid.shape), psi, state

    # ---- simple corner balance (SCB), second-order ------------------------
    def _build_corners(self):
        """Corner connectivity for SCB (direction-independent). Each active cell
        is split into three corner sub-volumes (one per vertex); each corner has
        two external half-edges (upwind-coupled to the neighbour cell's corner at
        the shared vertex) and two internal faces (to the cell's other corners).
        Builds, per corner, the external half-edge normals + neighbour corner ids
        and the internal face normals + same-cell corner ids."""
        s = _SQRT3_2
        # per cell type: for each edge -> (corner set, outward normal, neighbour
        # rhombus offset (di,dj,type), and this->neighbour local-corner map).
        edge_spec = {
            0: [({1, 2}, (s, 0.5), (0, 0, 1), {1: 0, 2: 1}),      # down hyp -> up(i,j)
                ({0, 1}, (0.0, -1.0), (-1, 0, 1), {0: 1, 1: 2}),  # down bot -> up(i-1,j)
                ({0, 2}, (-s, 0.5), (0, -1, 1), {0: 0, 2: 2})],   # down left -> up(i,j-1)
            1: [({0, 1}, (-s, -0.5), (0, 0, 0), {0: 1, 1: 2}),    # up hyp -> down(i,j)
                ({1, 2}, (0.0, 1.0), (1, 0, 0), {1: 0, 2: 1}),    # up top -> down(i+1,j)
                ({0, 2}, (s, -0.5), (0, 1, 0), {0: 0, 2: 2})],    # up right -> down(i,j+1)
        }
        # internal face normal from corner v toward corner w = unit(w - v).
        int_dir = {
            0: {(0, 1): (1.0, 0.0), (0, 2): (0.5, s), (1, 0): (-1.0, 0.0),
                (1, 2): (-0.5, s), (2, 0): (-0.5, -s), (2, 1): (0.5, -s)},
            1: {(0, 1): (-0.5, s), (0, 2): (0.5, s), (1, 0): (0.5, -s),
                (1, 2): (1.0, 0.0), (2, 0): (-0.5, -s), (2, 1): (-1.0, 0.0)},
        }
        ext_tab, int_tab = {}, {}
        edge_of = np.zeros((2, 3, 2), int)       # (type, corner, face) -> cell edge
        for t in (0, 1):
            for lc in (0, 1, 2):
                ext_tab[(t, lc)] = [(nrm, off, cmap[lc]) for (cs, nrm, off, cmap)
                                    in edge_spec[t] if lc in cs]
                edge_of[t, lc] = [e for e, (cs, *_ ) in enumerate(edge_spec[t])
                                  if lc in cs]
                others = [w for w in (0, 1, 2) if w != lc]
                int_tab[(t, lc)] = [(w, int_dir[t][(lc, w)]) for w in others]

        nr, nc = self.nr, self.nc
        cell = self._cell
        act = self.active
        ac = np.where(self._act_flat)[0]
        K = ac.size
        aidx = np.full(self.N, -1)
        aidx[ac] = np.arange(K)
        ext_n = np.zeros((K, 3, 2, 2))                       # [corner-cell, lc, face, xy]
        ext_nbr = np.full((K, 3, 2), -1)                    # neighbour corner global id
        ext_cell = np.full((K, 3, 2), -1)                   # full-mesh neighbour cell id
        int_n = np.zeros((K, 3, 2, 2))
        int_w = np.zeros((K, 3, 2), int)                    # same-cell corner global id
        ci, cj, ct = np.zeros(K, int), np.zeros(K, int), np.zeros(K, int)
        for r in range(K):
            c = ac[r]
            i, j, t = c // (nc * 2), (c // 2) % nc, c % 2
            ci[r], cj[r], ct[r] = i, j, t
            for lc in range(3):
                for f, (nrm, (di, dj, tn), nbr_lc) in enumerate(ext_tab[(t, lc)]):
                    ext_n[r, lc, f] = nrm
                    ni, nj = i + di, j + dj
                    if self.bc == "periodic":
                        ni, nj = ni % nr, nj % nc
                        ok = True
                    else:
                        ok = 0 <= ni < nr and 0 <= nj < nc
                    if ok:
                        nc_cell = cell[ni, nj, tn]
                        ext_cell[r, lc, f] = nc_cell
                        if act.reshape(-1)[nc_cell]:
                            ext_nbr[r, lc, f] = aidx[nc_cell] * 3 + nbr_lc
                for f, (w, nrm) in enumerate(int_tab[(t, lc)]):
                    int_n[r, lc, f] = nrm
                    int_w[r, lc, f] = r * 3 + w
        self._scb = {"ac": ac, "K": K, "ext_n": ext_n, "ext_nbr": ext_nbr,
                     "ext_cell": ext_cell, "int_n": int_n, "int_w": int_w,
                     "edge_of": edge_of, "ci": ci, "cj": cj, "ct": ct}

    def _prefactor_scb(self):
        """Assemble and factorize the 3*K corner system per ordinate and group."""
        d = self._scb
        K = d["K"]
        h2 = self.h / 2.0                                    # external half-edge length
        hi = self.h / (2.0 * np.sqrt(3.0))                  # internal face length
        A3 = self.area / 3.0                                 # corner volume
        row = (np.arange(K)[:, None, None] * 3 + np.arange(3)[None, :, None])
        row = np.broadcast_to(row, (K, 3, 2))
        self._solvers = [[None] * self.M for _ in range(self.G)]
        for g in range(self.G):
            st_c = self.st[g].reshape(-1)[d["ac"]]           # (K,)
            for m in range(self.M):
                oe = self.mu[m] * d["ext_n"][..., 0] + self.eta[m] * d["ext_n"][..., 1]
                oi = self.mu[m] * d["int_n"][..., 0] + self.eta[m] * d["int_n"][..., 1]
                # diagonal: collision + outflow external + internal self-share
                diag = st_c[:, None] * A3                    # (K, 3)
                diag = diag + (np.where(oe > 0, oe, 0.0) * h2).sum(2)
                diag = diag + (oi * hi * 0.5).sum(2)
                rid = (np.arange(K)[:, None] * 3 + np.arange(3)[None, :]).ravel()
                rows = [rid]; cols = [rid]; vals = [diag.ravel()]
                # external inflow -> neighbour corner (skip vacuum/void boundary)
                inflow = (oe < 0) & (d["ext_nbr"] >= 0)
                rows.append(row[inflow]); cols.append(d["ext_nbr"][inflow])
                vals.append((oe * h2)[inflow])
                # internal faces -> other-corner share
                rows.append(row.ravel()); cols.append(d["int_w"].ravel())
                vals.append((oi * hi * 0.5).ravel())
                Amat = sp.csr_matrix((np.concatenate(vals),
                                      (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(3 * K, 3 * K))
                self._solvers[g][m] = factorized(Amat.tocsc())

    def _iface_rhs(self, m, iface_in):
        """Per-ordinate corner RHS contribution from a prescribed incoming flux on
        interface half-edges (hybrid coupling): an inflow interface face moves its
        known incoming to the RHS, -(Omega.n)(h/2) psi_in. Returns (K,3)."""
        d = self._scb
        psi_in, is_iface = iface_in
        oe = self.mu[m] * d["ext_n"][..., 0] + self.eta[m] * d["ext_n"][..., 1]
        contrib = np.where(is_iface & (oe < 0), -oe * (self.h / 2.0) * psi_in, 0.0)
        return contrib.sum(2), oe

    def _sweep_scb(self, g, src_flat, iface_in=None):
        d = self._scb
        ac, K = d["ac"], d["K"]
        base = np.repeat(src_flat[ac] * (self.area / 3.0), 3)  # source into each corner
        phi = np.zeros(self.N)
        for m in range(self.M):
            rhs = base if iface_in is None else base + self._iface_rhs(m, iface_in)[0].ravel()
            psi = self._solvers[g][m](rhs).reshape(K, 3)
            phi[ac] += self.w[m] * psi.mean(1)                # cell flux = mean of corners
        return phi

    def _build_corners_3d(self):
        """Corner connectivity for SCB on the prism mesh: each active prism is
        split into three corner sub-prisms (the 2D corner quad extruded in z).
        A corner keeps its two lateral external half-edges + two lateral
        internal faces (the 2D structure) and gains two axial cap faces coupling
        the SAME corner in the prism directly above/below. 3 DoF per prism."""
        s = _SQRT3_2
        edge_spec = {
            0: [({1, 2}, (s, 0.5), (0, 0, 1), {1: 0, 2: 1}),
                ({0, 1}, (0.0, -1.0), (-1, 0, 1), {0: 1, 1: 2}),
                ({0, 2}, (-s, 0.5), (0, -1, 1), {0: 0, 2: 2})],
            1: [({0, 1}, (-s, -0.5), (0, 0, 0), {0: 1, 1: 2}),
                ({1, 2}, (0.0, 1.0), (1, 0, 0), {1: 0, 2: 1}),
                ({0, 2}, (s, -0.5), (0, 1, 0), {0: 0, 2: 2})],
        }
        int_dir = {
            0: {(0, 1): (1.0, 0.0), (0, 2): (0.5, s), (1, 0): (-1.0, 0.0),
                (1, 2): (-0.5, s), (2, 0): (-0.5, -s), (2, 1): (0.5, -s)},
            1: {(0, 1): (-0.5, s), (0, 2): (0.5, s), (1, 0): (0.5, -s),
                (1, 2): (1.0, 0.0), (2, 0): (-0.5, -s), (2, 1): (-1.0, 0.0)},
        }
        ext_tab, int_tab = {}, {}
        for t in (0, 1):
            for lc in (0, 1, 2):
                ext_tab[(t, lc)] = [(nrm, off, cmap[lc]) for (cs, nrm, off, cmap)
                                    in edge_spec[t] if lc in cs]
                int_tab[(t, lc)] = [(w, int_dir[t][(lc, w)])
                                    for w in (0, 1, 2) if w != lc]
        nr, nc, nz, N = self.nr, self.nc, self.nz, self.N
        cell, act_flat = self._cell, self._act_flat
        rad_per, ax_per = self.bc_radial == "periodic", self.bc_axial == "periodic"
        ac = np.where(act_flat)[0]
        K = ac.size
        aidx = np.full(N, -1)
        aidx[ac] = np.arange(K)
        ext_n = np.zeros((K, 3, 2, 2))
        ext_nbr = np.full((K, 3, 2), -1)
        int_n = np.zeros((K, 3, 2, 2))
        int_w = np.zeros((K, 3, 2), int)
        ax_nbr = np.full((K, 3, 2), -1)                      # [corner, lc, up/down]
        for r in range(K):
            c = ac[r]
            k = c % nz; t = (c // nz) % 2
            j = (c // (nz * 2)) % nc; i = c // (nz * 2 * nc)
            for lc in range(3):
                for f, (nrm, (di, dj, tn), nbr_lc) in enumerate(ext_tab[(t, lc)]):
                    ext_n[r, lc, f] = nrm
                    ni, nj = i + di, j + dj
                    ok = (rad_per or (0 <= ni < nr and 0 <= nj < nc))
                    if ok:
                        nb = cell[ni % nr, nj % nc, tn, k]
                        if act_flat[nb]:
                            ext_nbr[r, lc, f] = aidx[nb] * 3 + nbr_lc
                for f, (w, nrm) in enumerate(int_tab[(t, lc)]):
                    int_n[r, lc, f] = nrm
                    int_w[r, lc, f] = r * 3 + w
                for uf, dk in enumerate((+1, -1)):
                    nk = k + dk
                    if ax_per or (0 <= nk < nz):
                        nb = cell[i, j, t, nk % nz]
                        if act_flat[nb]:
                            ax_nbr[r, lc, uf] = aidx[nb] * 3 + lc
        # per cell edge (t, e) -> the (corner, ext-face) pairs on it, for folding
        # the corner half-edge currents into the _faces_3d per-cell-face current.
        edge_corners = {}
        for t in (0, 1):
            lc_edges = {lc: [e for e, (cs, *_) in enumerate(edge_spec[t])
                             if lc in cs] for lc in range(3)}
            for e, (cs, *_) in enumerate(edge_spec[t]):
                edge_corners[(t, e)] = [(lc, lc_edges[lc].index(e))
                                        for lc in sorted(cs)]
        self._scb = {"ac": ac, "K": K, "ext_n": ext_n, "ext_nbr": ext_nbr,
                     "int_n": int_n, "int_w": int_w, "ax_nbr": ax_nbr,
                     "edge_corners": edge_corners}

    def _prefactor_scb_3d(self):
        """Factorize the 3K corner system per ordinate/group on prisms: the 2D
        corner operator with lateral face areas scaled by dz, plus two axial cap
        faces (area = corner cap A3 = area/3, cosine +/-xi) upwind-coupled to the
        same corner above/below."""
        d = self._scb
        K = d["K"]
        dz = self.dz
        h2 = (self.h / 2.0) * dz                             # lateral ext area
        hi = (self.h / (2.0 * np.sqrt(3.0))) * dz            # lateral int area
        A3 = (self.area / 3.0) * dz                          # corner sub-prism volume
        Acap = self.area / 3.0                               # axial cap area
        row = np.broadcast_to(
            np.arange(K)[:, None, None] * 3 + np.arange(3)[None, :, None], (K, 3, 2))
        rid = (np.arange(K)[:, None] * 3 + np.arange(3)[None, :]).ravel()
        arow = np.broadcast_to(
            np.arange(K)[:, None, None] * 3 + np.arange(3)[None, :, None], (K, 3, 2))
        self._solvers = [[None] * self.M for _ in range(self.G)]
        for g in range(self.G):
            st_c = self.st[g].reshape(-1)[d["ac"]]
            for m in range(self.M):
                oe = self.mu[m] * d["ext_n"][..., 0] + self.eta[m] * d["ext_n"][..., 1]
                oi = self.mu[m] * d["int_n"][..., 0] + self.eta[m] * d["int_n"][..., 1]
                oz = np.array([self.xi[m], -self.xi[m]])      # up, down cap cosines
                diag = st_c[:, None] * A3                     # (K, 3) collision
                diag = diag + (np.where(oe > 0, oe, 0.0) * h2).sum(2)
                diag = diag + (oi * hi * 0.5).sum(2)
                diag = diag + (np.where(oz > 0, oz, 0.0) * Acap).sum()  # axial outflow
                rows = [rid]; cols = [rid]; vals = [diag.ravel()]
                inflow = (oe < 0) & (d["ext_nbr"] >= 0)       # lateral inflow
                rows.append(row[inflow]); cols.append(d["ext_nbr"][inflow])
                vals.append((oe * h2)[inflow])
                rows.append(row.ravel()); cols.append(d["int_w"].ravel())  # internal
                vals.append((oi * hi * 0.5).ravel())
                azin = (oz < 0)[None, None, :] & (d["ax_nbr"] >= 0)  # axial inflow
                coef = np.broadcast_to(oz * Acap, (K, 3, 2))
                rows.append(arow[azin]); cols.append(d["ax_nbr"][azin])
                vals.append(coef[azin])
                Amat = sp.csr_matrix((np.concatenate(vals),
                                      (np.concatenate(rows), np.concatenate(cols))),
                                     shape=(3 * K, 3 * K))
                self._solvers[g][m] = factorized(Amat.tocsc())

    def _sweep_scb_3d(self, g, src_flat):
        """One SCB prism sweep: phi = sum_m w_m mean_corner(psi_m), the corner
        source being src * corner-sub-prism-volume (vol/3)."""
        self._sweep_count += 1
        d = self._scb
        ac, K = d["ac"], d["K"]
        base = np.repeat(src_flat[ac] * (self.vol / 3.0), 3)
        phi = np.zeros(self.N)
        for m in range(self.M):
            psi = self._solvers[g][m](base).reshape(K, 3)
            phi[ac] += self.w[m] * psi.mean(1)
        return phi

    def _sweep_iface(self, g, src_flat, iface_in):
        """One SCB sweep that also accumulates the interface half-edge net
        currents from the same per-ordinate solves (the fused form of
        ``_sweep`` + ``interface_currents`` for the monolithic hybrid
        coupling). Returns (phi, J) with J shaped like interface_currents'."""
        if self.scheme != "scb":
            raise ValueError("_sweep_iface is SCB-only (hybrid drum boxes)")
        self._sweep_count += 1
        if self.engine == "levels":
            phi, psi = self._sweep_levels(g, src_flat, iface_in)
            psi3 = asnumpy(psi).reshape(self.M, self._scb["K"], 3)
            psi_in, is_iface = iface_in
            J = np.zeros((self._scb["K"], 3, 2))
            for m in range(self.M):
                oe = self._lv_oe[m]
                face = np.where(oe > 0, psi3[m][:, :, None], psi_in)
                J += self.w[m] * np.where(is_iface,
                                          oe * (self.h / 2.0) * face, 0.0)
            return phi, J
        d = self._scb
        ac, K = d["ac"], d["K"]
        base = np.repeat(src_flat[ac] * (self.area / 3.0), 3)
        psi_in, is_iface = iface_in
        phi = np.zeros(self.N)
        J = np.zeros((K, 3, 2))
        for m in range(self.M):
            add, oe = self._iface_rhs(m, iface_in)
            psi = self._solvers[g][m](base + add.ravel()).reshape(K, 3)
            phi[ac] += self.w[m] * psi.mean(1)
            face_flux = np.where(oe > 0, psi[:, :, None], psi_in)
            J += self.w[m] * np.where(is_iface, oe * (self.h / 2.0) * face_flux, 0.0)
        return phi, J

    def interface_currents(self, g, cell_source, iface_in):
        """Net current (drum -> bulk, outward-normal positive) on each interface
        half-edge, given the converged within-group source per cell (scatter +
        external) and the incoming from the bulk. Returns (K, 3, 2)."""
        self._sweep_count += 1                               # same cost as a sweep
        d = self._scb
        ac, K = d["ac"], d["K"]
        base = np.repeat(cell_source[ac] * (self.area / 3.0), 3)
        is_iface = iface_in[1]
        J = np.zeros((K, 3, 2))
        for m in range(self.M):
            add, oe = self._iface_rhs(m, iface_in)
            psi = self._solvers[g][m](base + add.ravel()).reshape(K, 3)
            # outflow face uses this corner's flux; inflow uses the incoming.
            face_flux = np.where(oe > 0, psi[:, :, None], iface_in[0])
            J += self.w[m] * np.where(is_iface, oe * (self.h / 2.0) * face_flux, 0.0)
        return J

    # ---- diffusion synthetic acceleration ---------------------------------
    def _dsa_matrix(self, g):
        """Assemble the group-g DSA diffusion operator (scipy CSR, host).

        Triangular finite-volume diffusion matching TriGroupOperator: harmonic
        face D = 1/(3 Sigma_t) with 4D/h^2 coupling, removal = Sigma_t -
        Sigma_s,gg. Faces from an active cell onto an excised/off-mesh cell get
        the Marshak-vacuum Robin term -- the within-group *error* equation has
        zero incoming flux there (any prescribed incoming, boundary or hybrid
        iface_in, is a fixed source) -- and the periodic torus wraps. Split out
        of _dsa_factor so the assembled operator can be reused (e.g. the Phase-0
        device-solver bake-off benchmarks solves on this real matrix)."""
        if self.is3d:
            return self._dsa_matrix_3d(g)
        nr, nc, N, h = self.nr, self.nc, self.N, self.h
        st = np.maximum(self.st[g].reshape(-1), 1e-12)
        ss = self.ss_self[g].reshape(-1)
        Dv = 1.0 / (3.0 * st)
        kf = 4.0 / (h * h)
        alpha = face_alpha("vacuum")
        cell, act = self._cell, self.active
        diag = np.maximum(st - ss, 1e-12).astype(float)
        rows, cols, vals = [], [], []
        ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        for (t, di, dj, tn, _) in _EDGES:
            src = cell[:, :, t]
            src_act = act[:, :, t]
            ni, nj = ii + di, jj + dj
            if self.bc == "periodic":
                ni, nj = ni % nr, nj % nc
                inb = np.ones_like(ni, bool)
            else:
                inb = (ni >= 0) & (ni < nr) & (nj >= 0) & (nj < nc)
            nbr = np.where(inb, cell[np.clip(ni, 0, nr - 1),
                                     np.clip(nj, 0, nc - 1), tn], -1)
            nbr_act = np.zeros((nr, nc), bool)
            nbr_act[inb] = act[ni[inb], nj[inb], tn]
            both = src_act & nbr_act                     # interior coupled face
            if both.any():
                w = harmonic_mean(Dv[src[both]], Dv[nbr[both]]) * kf
                rows.append(src[both]); cols.append(nbr[both]); vals.append(-w)
                np.add.at(diag, src[both], w)
            vac = src_act & ~both                        # zero-incoming error face
            if vac.any():
                Dc = Dv[src[vac]]
                term = (8.0 * Dc * alpha
                        / (h * (h * alpha + 2.0 * np.sqrt(3.0) * Dc)))
                np.add.at(diag, src[vac], term)
        diag = np.where(self._act_flat, diag, 1.0)       # excised: unit diagonal
        rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
        return sp.csr_matrix((np.concatenate([np.atleast_1d(v) for v in vals]),
                             (np.concatenate([np.atleast_1d(r) for r in rows]),
                              np.concatenate([np.atleast_1d(c) for c in cols]))),
                             shape=(N, N))

    def _dsa_matrix_3d(self, g):
        """The DSA diffusion operator on the extruded prism mesh: the 2D
        in-plane tri-FV coupling per z layer (harmonic 4D/h^2, tri-edge vacuum
        Robin) plus axial cap coupling harmonic(D)/dz^2 and the perpendicular-
        face vacuum Robin 2 D alpha/(dz (dz alpha + 2 D)) on z boundaries and
        axial excised faces -- matching TriGroupOperator's 3D stencil."""
        nr, nc, nz, N = self.nr, self.nc, self.nz, self.N
        h, dz = self.h, self.dz
        st = np.maximum(self.st[g].reshape(-1), 1e-12)
        ss = self.ss_self[g].reshape(-1)
        Dv = 1.0 / (3.0 * st)                            # length N, by flat id
        kf = 4.0 / (h * h)
        alpha = face_alpha("vacuum")
        cell, act = self._cell, self.active             # (nr, nc, 2, nz)
        diag = np.maximum(st - ss, 1e-12).astype(float)
        rows, cols, vals = [], [], []
        ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        rad_per = self.bc_radial == "periodic"
        ax_per = self.bc_axial == "periodic"
        for (t, di, dj, tn, _) in _EDGES:               # lateral, per z layer
            src = cell[:, :, t, :]
            src_act = act[:, :, t, :]
            ni, nj = ii + di, jj + dj
            if rad_per:
                inb = np.ones((nr, nc), bool); ni2, nj2 = ni % nr, nj % nc
            else:
                inb = (ni >= 0) & (ni < nr) & (nj >= 0) & (nj < nc)
                ni2, nj2 = np.clip(ni, 0, nr - 1), np.clip(nj, 0, nc - 1)
            nbr = np.full((nr, nc, nz), -1)
            nbr_act = np.zeros((nr, nc, nz), bool)
            nbr[inb] = cell[ni2[inb], nj2[inb], tn, :]
            nbr_act[inb] = act[ni2[inb], nj2[inb], tn, :]
            both = src_act & nbr_act
            if both.any():
                w = harmonic_mean(Dv[src[both]], Dv[nbr[both]]) * kf
                rows.append(src[both]); cols.append(nbr[both]); vals.append(-w)
                np.add.at(diag, src[both], w)
            vac = src_act & ~both
            if vac.any():
                Dc = Dv[src[vac]]
                np.add.at(diag, src[vac], 8.0 * Dc * alpha
                          / (h * (h * alpha + 2.0 * np.sqrt(3.0) * Dc)))
        kk = np.arange(nz)
        for dk in (+1, -1):                             # axial caps
            nk = kk + dk
            if ax_per:
                inbz = np.ones(nz, bool); nk2 = nk % nz
            else:
                inbz = (nk >= 0) & (nk < nz); nk2 = np.clip(nk, 0, nz - 1)
            nbr = np.full((nr, nc, 2, nz), -1)
            nbr_act = np.zeros((nr, nc, 2, nz), bool)
            nbr[:, :, :, inbz] = cell[:, :, :, nk2[inbz]]
            nbr_act[:, :, :, inbz] = act[:, :, :, nk2[inbz]]
            both = act & nbr_act
            if both.any():
                wz = harmonic_mean(Dv[cell[both]], Dv[nbr[both]]) / (dz * dz)
                rows.append(cell[both]); cols.append(nbr[both]); vals.append(-wz)
                np.add.at(diag, cell[both], wz)
            vac = act & ~both
            if vac.any():
                Dc = Dv[cell[vac]]
                np.add.at(diag, cell[vac],
                          2.0 * Dc * alpha / (dz * (dz * alpha + 2.0 * Dc)))
        diag = np.where(self._act_flat, diag, 1.0)       # excised: identity
        rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
        return sp.csr_matrix((np.concatenate([np.atleast_1d(v) for v in vals]),
                             (np.concatenate([np.atleast_1d(r) for r in rows]),
                              np.concatenate([np.atleast_1d(c) for c in cols]))),
                             shape=(N, N))

    def _dsa_factor(self, g):
        """Backend solver for the group-g DSA diffusion operator, built on
        first use (host: exact LU; device: capped inexact iterative)."""
        if self._dsa_fac[g] is None:
            self._dsa_fac[g] = self._make_diff_solver(self._dsa_matrix(g),
                                                      symmetric=True)
        return self._dsa_fac[g]

    # ---- CMFD outer acceleration ------------------------------------------
    def _sweep_currents(self, g, src_flat):
        """One sweep that also accumulates the per-cell-edge net currents.

        Returns (phi, J6): J6[k] (nr, nc) is the net current per unit edge
        length, outward from the type-t source cell, across edge type k of
        ``_EDGES`` (zero where the source cell is inactive). Face fluxes are
        the schemes' own: upwind cell flux for step, upwind corner flux per
        half-edge for SCB -- both make the transport cell balance exact, so
        div J + Sigma_t phi = src holds identically and CMFD's fixed point is
        the transport solution."""
        self._sweep_count += 1
        if self.engine == "levels":
            phi, psi = self._sweep_levels(g, src_flat)
            return phi, self._currents_from_psi(g, psi)
        if self.scheme == "scb":
            return self._sweep_currents_scb(g, src_flat)
        nr, nc = self.nr, self.nc
        rhs = np.where(self._act_flat, src_flat * self.area, 0.0)
        phi = np.zeros(self.N)
        J6 = np.zeros((6, nr, nc))
        for m in range(self.M):
            psi = self._solvers[g][m](rhs)
            phi += self.w[m] * psi
            psi3 = psi.reshape(nr, nc, 2)
            for k, (t, tn, nic, njc, nbr_act, src_act, (nxv, nyv)) in \
                    enumerate(self._nbr_maps):
                On = self.mu[m] * nxv + self.eta[m] * nyv
                if On > 0:                       # outflow: upwind = this cell
                    face = psi3[:, :, t]
                else:                            # inflow: upwind = neighbour (0 at bc)
                    face = np.where(nbr_act, psi3[nic, njc, tn], 0.0)
                J6[k] += (self.w[m] * On) * np.where(src_act, face, 0.0)
        return phi, J6

    def _sweep_currents_scb(self, g, src_flat):
        d = self._scb
        ac, K = d["ac"], d["K"]
        base = np.repeat(src_flat[ac] * (self.area / 3.0), 3)
        phi = np.zeros(self.N)
        Jh = np.zeros((K, 3, 2))                 # per half-edge (length h/2 folded in)
        for m in range(self.M):
            psi = self._solvers[g][m](base).reshape(K, 3)
            phi[ac] += self.w[m] * psi.mean(1)
            oe = self.mu[m] * d["ext_n"][..., 0] + self.eta[m] * d["ext_n"][..., 1]
            nbr = d["ext_nbr"]
            nbr_psi = np.where(nbr >= 0, psi.ravel()[np.clip(nbr, 0, None)], 0.0)
            face = np.where(oe > 0, psi[:, :, None], nbr_psi)
            Jh += self.w[m] * oe * (self.h / 2.0) * face
        return phi, self._fold_half_currents(Jh)

    def _fold_half_currents(self, Jh):
        """Fold per-half-edge currents (K, 3, 2; length h/2 folded in) into the
        per-cell-edge J6 (6, nr, nc), reported per unit length."""
        d = self._scb
        J6 = np.zeros((6, self.nr, self.nc))
        eo, ci, cj, ct = d["edge_of"], d["ci"], d["cj"], d["ct"]
        for t in (0, 1):
            sel = ct == t
            Je = np.zeros((int(sel.sum()), 3))
            for lc in range(3):
                for f in range(2):
                    Je[:, eo[t, lc, f]] += Jh[sel, lc, f]
            for e in range(3):
                J6[t * 3 + e][ci[sel], cj[sel]] = Je[:, e] / self.h
        return J6

    def _cmfd_factor(self, g, p, J6):
        """LU of the group-g drift-corrected triangular diffusion operator (the
        tri counterpart of the Cartesian CMFD matrix): interior face model
        J = -beta (phi_n - phi_c) + gamma (phi_n + phi_c) per unit length, with
        beta = sqrt(3) harm(D)/h (the TriGroupOperator coupling) plus the
        odCMFD-style thick-cell damping, gamma fitted to the transport current;
        faces onto inactive/off-mesh cells carry the transport leakage ratio.
        Assembled from both sides of every face (the transport currents are
        exactly antisymmetric), each side contributing its own row."""
        h, N = self.h, self.N
        s3 = np.sqrt(3.0)
        Lv = 4.0 / (s3 * h)                      # edge length / cell area
        st = self.st[g].reshape(-1)
        D = 1.0 / (3.0 * np.maximum(st, 1e-12))
        rem = st - self.ss_self[g].reshape(-1)
        cell = self._cell
        tiny = 1e-30
        diag = np.where(self._act_flat, rem, 1.0)
        rows, cols, vals = [np.arange(N)], [np.arange(N)], [diag]
        for k, (t, tn, nic, njc, nbr_act, src_act, _) in enumerate(self._nbr_maps):
            src = cell[:, :, t]
            nbr = cell[nic, njc, tn]
            both = src_act & nbr_act
            if both.any():
                s_, n_ = src[both], nbr[both]
                beta = s3 * harmonic_mean(D[s_], D[n_]) / h
                tau = np.maximum(st[s_], st[n_]) * h
                beta = beta + 0.25 * np.maximum(0.0, 1.0 - 1.0 / np.maximum(tau, tiny))
                gh = (J6[k][both] + beta * (p[n_] - p[s_])) \
                    / np.maximum(p[s_] + p[n_], tiny)
                rows.append(s_); cols.append(s_); vals.append(Lv * (beta + gh))
                rows.append(s_); cols.append(n_); vals.append(Lv * (-beta + gh))
            bnd = src_act & ~both                # vacuum / excised face
            if bnd.any():
                s_ = src[bnd]
                gb = J6[k][bnd] / np.maximum(p[s_], tiny)
                rows.append(s_); cols.append(s_); vals.append(Lv * gb)
        A = sp.csr_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(N, N))
        return self._make_cmfd_solver(A)

    def _faces_3d(self):
        """Directed faces of the prism mesh for CMFD, each yielded once per
        source side (currents are antisymmetric, so both sides give a row).
        Returns per face: source/neighbour flat-id + activity grids, the
        ordinate cosine Omega.n_hat(m), the face length/volume ratio Lv, the
        per-length diffusion coupling beta(Ds,Dn), and the normal-direction cell
        size (for odCMFD thick-cell damping)."""
        nr, nc, nz = self.nr, self.nc, self.nz
        h, dz, area = self.h, self.dz, self.area
        cell, act = self._cell, self.active
        ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        s3 = np.sqrt(3.0)
        Lv_lat, Lv_ax = 4.0 / (s3 * h), 1.0 / dz
        rad_per, ax_per = self.bc_radial == "periodic", self.bc_axial == "periodic"
        for (t, di, dj, tn, (nx, ny)) in _EDGES:              # lateral
            if rad_per:
                inb = np.ones((nr, nc), bool)
                ni, nj = (ii + di) % nr, (jj + dj) % nc
            else:
                inb = (ii + di >= 0) & (ii + di < nr) & (jj + dj >= 0) & (jj + dj < nc)
                ni, nj = np.clip(ii + di, 0, nr - 1), np.clip(jj + dj, 0, nc - 1)
            nbr = np.full((nr, nc, nz), -1)
            nbr_act = np.zeros((nr, nc, nz), bool)
            nbr[inb] = cell[ni[inb], nj[inb], tn, :]
            nbr_act[inb] = act[ni[inb], nj[inb], tn, :]
            yield (cell[:, :, t, :], act[:, :, t, :], nbr, nbr_act,
                   (nx, ny, 0.0), Lv_lat,
                   lambda Ds, Dn: s3 * harmonic_mean(Ds, Dn) / h, h)
        kk = np.arange(nz)
        for dk in (+1, -1):                                   # axial caps
            nk = kk + dk
            if ax_per:
                inbz = np.ones(nz, bool); nk2 = nk % nz
            else:
                inbz = (nk >= 0) & (nk < nz); nk2 = np.clip(nk, 0, nz - 1)
            nbr = np.full((nr, nc, 2, nz), -1)
            nbr_act = np.zeros((nr, nc, 2, nz), bool)
            nbr[:, :, :, inbz] = cell[:, :, :, nk2[inbz]]
            nbr_act[:, :, :, inbz] = act[:, :, :, nk2[inbz]]
            yield (cell, act, nbr, nbr_act, (0.0, 0.0, float(dk)), Lv_ax,
                   lambda Ds, Dn: harmonic_mean(Ds, Dn) / dz, dz)

    def _sweep_currents_3d(self, g, src_flat):
        """Prism sweep that also folds the per-face net current density
        J = sum_m w_m (Omega.n_hat) psi_upwind (upwind = source cell on outflow,
        neighbour on inflow), one array per directed face of _faces_3d, so the
        transport cell balance div J + Sigma_t phi = src holds and CMFD's fixed
        point is the transport solution."""
        M, N = self.M, self.N
        if self.engine == "levels":
            phi, psi = self._sweep_levels(g, src_flat)
            psi_mn = asnumpy(psi).reshape(M, N)
        else:
            self._sweep_count += 1
            rhs = np.where(self._act_flat, src_flat * self.vol, 0.0)
            psi_mn = np.stack([self._solvers[g][m](rhs) for m in range(M)])
            phi = (self.w[:, None] * psi_mn).sum(0)
        currents = []
        for (src, src_act, nbr, nbr_act, (nx, ny, nz_), Lv, beta, hd) \
                in self._faces_3d():
            J = np.zeros(src.shape)
            for m in range(M):
                c = self.mu[m] * nx + self.eta[m] * ny + self.xi[m] * nz_
                if c > 0.0:
                    face = psi_mn[m][src]
                elif c < 0.0:
                    face = np.where(nbr_act, psi_mn[m][nbr], 0.0)
                else:
                    continue
                J += (self.w[m] * c) * np.where(src_act, face, 0.0)
            currents.append(J)
        return phi, currents

    def _sweep_currents_scb_3d(self, g, src_flat):
        """CMFD currents for the SCB prism scheme, in the same _faces_3d
        per-cell-face convention as _sweep_currents_3d. The transport current
        density across a lateral cell edge is the mean of its two corner
        half-edge densities (each J = sum_m w_m (Omega.n_hat) psi_upwind, upwind
        = own corner on outflow, the neighbour prism's shared-vertex corner on
        inflow); across an axial cap it is the mean over the cell's three
        corners. Works for the LU and levels engines (corner angular flux)."""
        M, N = self.M, self.N
        d = self._scb
        ac, K = d["ac"], d["K"]
        if self.engine == "levels":
            phi, psi = self._sweep_levels(g, src_flat)
            psi3 = asnumpy(psi).reshape(M, K, 3)
        else:
            self._sweep_count += 1
            base = np.repeat(src_flat[ac] * (self.vol / 3.0), 3)
            psi3 = np.stack([self._solvers[g][m](base).reshape(K, 3)
                             for m in range(M)])
            phi = np.zeros(N)
            phi[ac] = (self.w[:, None] * psi3.mean(2)).sum(0)
        psi_flat = psi3.reshape(M, 3 * K)                    # (M, 3K)

        def upwind_density(cos, nbr_id):                     # (M,K,3,F) cosine
            valid = (nbr_id >= 0)[None]
            npsi = np.where(valid, psi_flat[:, np.maximum(nbr_id, 0)], 0.0)
            face = np.where(cos > 0.0, psi3[..., None], npsi)  # own vs neighbour
            return (self.w[:, None, None, None] * cos * face).sum(0)  # (K,3,F)

        extn = d["ext_n"]
        oe = (self.mu[:, None, None, None] * extn[None, ..., 0]
              + self.eta[:, None, None, None] * extn[None, ..., 1])   # (M,K,3,2)
        Jh = upwind_density(oe, d["ext_nbr"])                # lateral half-edges
        oz = np.stack([self.xi, -self.xi], -1)[:, None, None, :]      # (M,1,1,2)
        Ja = upwind_density(oz, d["ax_nbr"])                 # axial caps (K,3,2)
        aidx = np.full(N, -1)
        aidx[ac] = np.arange(K)
        cell = self._cell
        currents = []
        for i in range(6):                                   # lateral, _EDGES order
            t, e = (0, i) if i < 3 else (1, i - 3)
            r = aidx[cell[:, :, t, :]]                       # (nr,nc,nz), -1 excised
            mrk = r >= 0
            J = np.zeros(r.shape)
            for lc, f in d["edge_corners"][(t, e)]:          # the 2 corners on edge e
                J[mrk] += 0.5 * Jh[r[mrk], lc, f]
            currents.append(J)
        for uf in (0, 1):                                    # axial dk=+1, -1
            r = aidx[cell]                                   # (nr,nc,2,nz)
            mrk = r >= 0
            J = np.zeros(r.shape)
            J[mrk] = Ja[r[mrk], :, uf].mean(1)               # mean over 3 corners
            currents.append(J)
        return phi, currents

    def _cmfd_factor_3d(self, g, p, currents):
        """Solver for the group-g drift-corrected prism diffusion operator (LU,
        or multigrid when cmfd_solver='mg' -- 3D sparse LU is O(N^2)/O(N^4/3)
        fill and blows up, while AMG stays O(N))."""
        return self._make_cmfd_solver(self._cmfd_matrix_3d(g, p, currents))

    def _cmfd_matrix_3d(self, g, p, currents):
        """Assemble the drift-corrected FV diffusion operator on the prism mesh
        (scipy CSR, host): per directed face J = -beta (phi_n - phi_c) +
        gamma (phi_n + phi_c), beta the TriGroupOperator lateral/axial coupling
        with odCMFD thick-cell damping, gamma fitted to the transport current;
        excised/boundary faces carry the transport leakage ratio. Per volume
        (Lv = face length / cell volume), from both sides of every face."""
        N = self.N
        st = self.st[g].reshape(-1)
        D = 1.0 / (3.0 * np.maximum(st, 1e-12))
        rem = st - self.ss_self[g].reshape(-1)
        tiny = 1e-30
        diag = np.where(self._act_flat, rem, 1.0)
        rows, cols, vals = [np.arange(N)], [np.arange(N)], [diag]
        for (src, src_act, nbr, nbr_act, _n, Lv, beta_fn, hd), J in zip(
                self._faces_3d(), currents):
            both = src_act & nbr_act
            if both.any():
                s_, n_ = src[both], nbr[both]
                beta = beta_fn(D[s_], D[n_])
                tau = np.maximum(st[s_], st[n_]) * hd        # optical thickness
                beta = beta + 0.25 * np.maximum(0.0, 1.0 - 1.0
                                                / np.maximum(tau, tiny))
                gh = (J[both] + beta * (p[n_] - p[s_])) \
                    / np.maximum(p[s_] + p[n_], tiny)
                rows.append(s_); cols.append(s_); vals.append(Lv * (beta + gh))
                rows.append(s_); cols.append(n_); vals.append(Lv * (-beta + gh))
            bnd = src_act & ~both
            if bnd.any():
                s_ = src[bnd]
                gb = J[bnd] / np.maximum(p[s_], tiny)
                rows.append(s_); cols.append(s_); vals.append(Lv * gb)
        return sp.csr_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N, N))

    def _make_cmfd_solver(self, A):
        """Solver for a CMFD drift matrix. Default: scipy sparse LU. With
        cmfd_solver='mg': smoothed-aggregation multigrid preconditioning a
        BiCGStab solve (the drift operator is non-symmetric), which stays O(N)
        where the 3D LU factorisation blows up. On the GPU the multigrid
        hierarchy (built once on the host by pyamg) is moved to the device and
        the V-cycle + BiCGStab run there, so -- with the device-resident
        _cmfd_power -- the CMFD solves never leave the GPU."""
        if getattr(self, "cmfd_solver", "lu") != "mg":
            return factorized(A.tocsc())
        try:
            import pyamg
        except ImportError as e:                             # optional dependency
            raise ImportError("cmfd_solver='mg' needs pyamg (pip install pyamg) "
                              "-- the O(N) CMFD solve for large 3D meshes") from e
        xp = self.xp
        ml = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=400)
        if xp is np:
            return lambda b: ml.solve(b, tol=1e-9, accel="bicgstab", maxiter=200)
        import cupyx.scipy.sparse as csp
        from .linalg import bicgstab
        levels = []
        for L in ml.levels[:-1]:
            La = L.A.tocsr()
            levels.append((csp.csr_matrix(La), xp.asarray(1.0 / La.diagonal()),
                           csp.csr_matrix(L.P.tocsr()), csp.csr_matrix(L.R.tocsr())))
        coarse = factorized(ml.levels[-1].A.tocsc())          # tiny, host
        Ad = csp.csr_matrix(A.tocsr())
        w = 0.7                                               # damped Jacobi

        def vcycle(b, i=0):
            if i == len(levels):
                return xp.asarray(coarse(asnumpy(b)))         # coarse solve on host
            La, Dinv, P, R = levels[i]
            x = xp.zeros_like(b)
            for _ in range(2):
                x = x + w * Dinv * (b - La @ x)
            x = x + P @ vcycle(R @ (b - La @ x), i + 1)
            for _ in range(2):
                x = x + w * Dinv * (b - La @ x)
            return x

        def solve(b):
            x, _ = bicgstab(lambda y: Ad @ y, b, xp.zeros_like(b), None, xp,
                            rtol=1e-9, maxiter=200, precond=vcycle)
            return x
        return solve

    def _solve_group(self, g, qext_flat, phi0, tol, iface_in=None):
        """Within-group scattering solve with the configured acceleration; the
        boundary is vacuum/periodic (folded into L_Omega), so no boundary fixed
        point is needed. iface_in injects a hybrid incoming flux on interface
        half-edges (a fixed source, so only b carries it)."""
        ss = self.ss_self[g].reshape(-1)
        b = self._sweep(g, qext_flat, iface_in)              # source-only response
        tol = min(tol, 1e-4)

        if self.acceleration in ("gmres", "dsa-gmres"):
            def op(x):                                       # (I - T) x, T = scatter sweep
                return x - self._sweep(g, ss * x)

            A = LinearOperator((self.N, self.N), matvec=op, dtype=float)
            M = None
            if self.acceleration == "dsa-gmres":
                fac = self._dsa_factor(g)

                def prec(x):                                 # M = I + F^-1 Sigma_s
                    return x + fac(ss * x)
                M = LinearOperator((self.N, self.N), matvec=prec, dtype=float)
            phi, _ = gmres(A, b, x0=phi0, M=M, rtol=tol, atol=0.0, maxiter=400)
            return phi

        # levels engine: run the whole source iteration device-resident.
        if self.engine == "levels" and iface_in is None:
            return self._solve_group_dev(g, qext_flat, phi0, tol)

        # (DSA-accelerated) source iteration, mirroring the Cartesian solver:
        # each sweep is followed by the diffusion error correction; if the
        # update norm stops contracting the acceleration is dropped.
        accelerate = self.acceleration == "dsa"
        fac = self._dsa_factor(g) if accelerate else None
        phi = np.asarray(phi0, float).copy()
        prev = None
        bad = 0
        for _ in range(self.max_inner):
            half = b + self._sweep(g, ss * phi)
            new = half + fac(ss * (half - phi)) if accelerate else half
            d = np.max(np.abs(new - phi))
            scale = max(np.max(np.abs(new)), 1e-300)
            phi = new
            if d <= tol * scale:
                break
            if accelerate:
                if prev is not None and d > prev:
                    bad += 1
                    if bad >= 3:
                        accelerate = False
                else:
                    bad = 0
                prev = d
        return phi

    def _sync(self):
        """Block until queued device work is done -- makes wall-clock timers
        honest on GPU (a no-op on NumPy)."""
        if self.xp is not np:
            self.xp.cuda.Stream.null.synchronize()

    def solve(self, tol_k: float = 1e-7, tol_source: float = 1e-6,
              max_outer: int = 500, verbose: bool = False) -> SNResult:
        t0 = time.perf_counter()
        G, N = self.G, self.N
        sweeps0 = self._sweep_count
        self.t_groups = self.t_cmfd = self.t_power = 0.0   # component wall times
        phi = [np.where(self._act_flat, 1.0, 0.0) for _ in range(G)]
        nsf = [self.nsf[g].reshape(-1) for g in range(G)]
        chi = [self.chi[g].reshape(-1) for g in range(G)]
        ss = [self.ss_self[g].reshape(-1) for g in range(G)]
        scat = [[None if self.scatter[gf][g] is None
                 else self.scatter[gf][g].reshape(-1) for g in range(G)]
                for gf in range(G)]
        fiss = sum(nsf[g] * phi[g] for g in range(G))
        n_act = float(self._act_flat.sum())
        total = fiss.sum()
        k = 1.0
        prev_rel = prev_err = 1.0
        k_hist = []
        converged = False
        outer = 0
        hist = []                                            # Anderson (fsrc_in, raw)
        for outer in range(1, max_outer + 1):
            fs = fiss / k
            tol = min(1e-3, max(0.05 * prev_rel, 0.01 * tol_k, 1e-10))
            fiss_in = fiss
            phi_new = [None] * G
            tg = time.perf_counter()
            for g in range(G):
                q = chi[g] * fs
                for gf in range(G):
                    if gf != g and scat[gf][g] is not None:
                        src = phi_new[gf] if gf < g else phi[gf]
                        q = q + scat[gf][g] * src
                phi_new[g] = self._solve_group(g, q, phi[g], tol)
            self._sync()
            self.t_groups += time.perf_counter() - tg
            cmfd_ok = False
            if self.outer_acceleration == "cmfd":
                # one current-accumulating sweep per group with the converged
                # within-group source -> consistent (flux, current) pairs; the
                # drift-corrected diffusion eigensolve replaces the iterate.
                tc = time.perf_counter()
                phi_h, facs = [None] * G, [None] * G
                for g in range(G):
                    q = chi[g] * fs
                    for gf in range(G):
                        if gf != g and scat[gf][g] is not None:
                            q = q + scat[gf][g] * phi_new[gf]
                    if self.is3d:
                        cur_fn = (self._sweep_currents_scb_3d if self.scheme == "scb"
                                  else self._sweep_currents_3d)
                        ps, cur = cur_fn(g, ss[g] * phi_new[g] + q)
                        facs[g] = self._cmfd_factor_3d(g, ps, cur)
                    else:
                        ps, J6 = self._sweep_currents(g, ss[g] * phi_new[g] + q)
                        facs[g] = self._cmfd_factor(g, ps, J6)
                    phi_h[g] = ps
                self._sync()
                tp = time.perf_counter()
                phi_c, k_c, cmfd_ok = _cmfd_power(facs, nsf, chi, scat, phi_h, k,
                                                  xp=self.xp)
                self.t_power += time.perf_counter() - tp
                self.t_cmfd += time.perf_counter() - tc
                if cmfd_ok:
                    phi_new = phi_c
            fiss_new = sum(nsf[g] * phi_new[g] for g in range(G))
            total_new = fiss_new.sum()
            k_new = k_c if cmfd_ok else k * total_new / total
            dk = abs(k_new - k)
            rel = max(np.max(np.abs(phi_new[g] - phi[g])) /
                      max(np.max(np.abs(phi_new[g])), 1e-30) for g in range(G))
            phi, k = phi_new, k_new
            k_hist.append(k)
            if verbose:
                print(f"  outer {outer:3d}  k = {k:.7f}  dk = {dk:.2e}  rel = {rel:.2e}")
            if dk < tol_k and rel < tol_source and tol < max(1e-8, tol_k):
                converged = True
                break
            if cmfd_ok:
                # CMFD already accelerated the update; Anderson would mix in
                # stale pre-CMFD pairs, so take the iterate as-is.
                hist = []
                fiss = fiss_new
                total = total_new
            else:
                # Anderson-accelerate the fission source (the loosely-coupled
                # core has a dominance ratio near 1, so power iteration crawls).
                raw = fiss_new * (n_act / total_new)         # mean active source = 1
                if rel > 1.1 * prev_err:
                    hist = []
                hist.append((fiss_in, raw))
                if len(hist) > 6:
                    hist.pop(0)
                fiss = _anderson(hist)
                fiss *= n_act / fiss.sum()
                total = fiss.sum()
            prev_rel = max(rel, dk)
            prev_err = rel
        flux = np.stack([phi[g].reshape(self.grid.shape) for g in range(G)])
        return SNResult(k_eff=k, flux=flux, converged=converged,
                        outer_iterations=outer,
                        solve_seconds=time.perf_counter() - t0,
                        n_ordinates=self.M, k_history=k_hist,
                        n_sweeps=self._sweep_count - sweeps0)
