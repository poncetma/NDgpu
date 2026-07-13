"""Simplified, human-friendly front ends for building and running a reactor.

The full solver classes are flexible but ask you to build grids, hand-assemble
material-map arrays, and read a terse ``Result``. These builders wrap the common
cases and return a :class:`ReactorResult` that prints a transparent report
(k_eff, reactivity, where the neutrons go, flux peaking, per-material power):

* :class:`Model`       -- a rectangular Cartesian core, painted with boxes in cm
                          (diffusion or SP3; 1-D / 2-D / 3-D; forward or adjoint).
* :class:`MeshModel`   -- an arbitrary unstructured mesh (a Gmsh file or an
                          assembled :class:`~ndgpu.Mesh`), 2-D or 3-D, materials
                          assigned by physical region or mesh tag.
* :class:`HexLattice`  -- a hexagonal lattice of assemblies on the body-fitted
                          triangular solver (diffusion or SP3).

Materials are ordinary :class:`~ndgpu.Material` objects, so defining new cross
sections is unchanged; these classes only remove the grid/array bookkeeping.

Example -- a reflected Cartesian core::

    import ndgpu
    m = ndgpu.Model(size=(120, 120), cells=(60, 60))     # 2-D, cm
    m.fill(reflector).add_box(fuel, x=(30, 90), y=(30, 90)).set_boundary("vacuum")
    print(m.run())
"""

from __future__ import annotations

import numpy as np

from .grid import Grid
from .hexraster import rasterize_hex_sites
from .materials import Kinetics, Material
from .mesh import UnstructuredDiffusionSolver, read_gmsh
from .operator import face_alpha
from .solver import DiffusionEigenSolver, SP3EigenSolver
from .transient import TransientSolver
from .tri import TriDiffusionEigenSolver, TriGrid, TriSP3EigenSolver

_STRUCTURED = {"diffusion": DiffusionEigenSolver, "sp3": SP3EigenSolver}
_TRI = {"diffusion": TriDiffusionEigenSolver, "sp3": TriSP3EigenSolver}
_AXES = ("x", "y", "z")


def _albedo(spec) -> float:
    """Robin coefficient for a boundary name/albedo (zero-flux -> a large finite alpha)."""
    a = face_alpha(spec)
    return 1e8 if a == float("inf") else float(a)


# ---------------------------------------------------------------------------
# Shared diagnostics + report
# ---------------------------------------------------------------------------
def _diagnostics(materials, cell_material, flux_flat, cell_volume, active):
    """Reaction-rate diagnostics from the scalar flux, restricted to active cells.

    Returns (production, absorption, material_rows). Rates are formed straight
    from the cross sections and the physical scalar flux (so this works for
    diffusion or SP3), volume-weighted for non-uniform meshes.
    """
    G = flux_flat.shape[0]
    ncells = flux_flat.shape[1]
    sa = np.array([m.sigma_a for m in materials])          # (M, G)
    nf = np.array([m.nu_sigma_f for m in materials])
    cm = np.asarray(cell_material).reshape(-1)
    vol = np.broadcast_to(np.asarray(cell_volume, float), (ncells,))
    act = (np.ones(ncells, bool) if active is None
           else np.asarray(active).reshape(-1).astype(bool))
    w = flux_flat * vol * act                              # (G, ncells) volume-flux weight

    production = float(sum((nf[cm, g] * w[g]).sum() for g in range(G)))
    absorption = float(sum((sa[cm, g] * w[g]).sum() for g in range(G)))

    total_vol = float((vol * act).sum())
    rows = []
    for idx, mat in enumerate(materials):
        sel = act & (cm == idx)
        if not sel.any():
            continue
        volf = float((vol * sel).sum()) / total_vol
        fis = float(sum(nf[idx, g] * (flux_flat[g] * vol * sel).sum() for g in range(G)))
        rows.append([mat.name or f"material {idx}", volf, fis, 0.0])
    for r in rows:
        r[3] = r[2] / production if production > 0 else 0.0
    return production, absorption, rows


class ReactorResult:
    """A solved reactor with a human-readable report.

    Printing it (or :meth:`summary`) shows k_eff, reactivity, the converged
    neutron balance, flux peaking and per-material power. Numeric fields
    (:attr:`k_eff`, :attr:`flux`, :attr:`leakage_fraction`, ...) are plain values.
    """

    def __init__(self, *, k_eff, converged, outer_iterations, inner_iterations,
                 solve_seconds, device, method, flux, materials, cell_material,
                 cell_volume=1.0, active=None, geometry_line="", boundary_line="",
                 adjoint=False):
        self.k_eff = float(k_eff)
        self.reactivity_pcm = (1.0 - 1.0 / self.k_eff) * 1e5
        self.converged = bool(converged)
        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.solve_seconds = solve_seconds
        self.device = device
        self.method = method
        self.adjoint = adjoint
        self.flux = flux                          # (G, *shape) numpy
        self.n_groups = flux.shape[0]
        self._geometry_line = geometry_line
        self._boundary_line = boundary_line

        flat = flux.reshape(self.n_groups, -1)    # (G, ncells)
        act = (np.ones(flat.shape[1], bool) if active is None
               else np.asarray(active).reshape(-1).astype(bool))
        prod, absorb, rows = _diagnostics(materials, cell_material, flat, cell_volume, act)
        loss = prod / self.k_eff                  # neutrons available per generation
        self.absorbed_fraction = absorb / loss if loss > 0 else float("nan")
        self.leakage_fraction = 1.0 - self.absorbed_fraction
        thermal = flat[-1][act]
        self.peaking = float(thermal.max() / thermal.mean()) if thermal.size else float("nan")
        self._material_stats = rows

    def summary(self) -> str:
        kind = "adjoint (importance) solution" if self.adjoint else "solution"
        status = (f"converged in {self.outer_iterations} outer / "
                  f"{self.inner_iterations} inner iterations, {self.solve_seconds:.2f} s"
                  if self.converged else "DID NOT CONVERGE")
        lines = [
            f"NDgpu reactor {kind}",
            "=" * (14 + len(kind)),
            f"  geometry    : {self._geometry_line}",
            f"  groups      : {self.n_groups}     method: {self.method}     device: {self.device}",
            f"  boundary    : {self._boundary_line}",
            "",
            f"  k_eff       : {self.k_eff:.6f}",
            f"  reactivity  : {self.reactivity_pcm:+.0f} pcm   (rho = (k-1)/k)",
            f"  status      : {status}",
            "",
        ]
        if self.adjoint:
            lines.append("  (adjoint flux is neutron importance, not a physical flux)")
            lines.append(f"  importance peaking (thermal): peak / average = {self.peaking:.2f}")
            return "\n".join(lines)
        lines += [
            "  where the fission neutrons go (per neutron produced / k):",
            f"    absorbed  : {self.absorbed_fraction * 100:5.1f} %",
            f"    leaked    : {self.leakage_fraction * 100:5.1f} %",
            "",
            f"  flux peaking (thermal group): peak / average = {self.peaking:.2f}",
        ]
        if len(self._material_stats) > 1:
            lines.append("")
            lines.append(f"  {'material':<16}{'volume':>9}{'fission':>10}")
            for name, volf, _fis, fisf in self._material_stats:
                lines.append(f"    {name:<14}{volf * 100:7.1f}%{fisf * 100:9.1f}%")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        s = "converged" if self.converged else "NOT CONVERGED"
        tag = "adjoint " if self.adjoint else ""
        return (f"ReactorResult({tag}k_eff={self.k_eff:.6f}, {self.reactivity_pcm:+.0f} pcm, "
                f"{s}, {self.method} on {self.device})")


class ModelResult(ReactorResult):
    """:class:`ReactorResult` for a structured :class:`Model` (keeps ``.model``)."""

    def __init__(self, model, result, method, adjoint=False):
        self.model = model
        super().__init__(
            k_eff=result.k_eff, converged=result.converged,
            outer_iterations=result.outer_iterations,
            inner_iterations=result.inner_iterations,
            solve_seconds=result.solve_seconds, device=result.device, method=method,
            flux=result.flux_numpy, materials=model._materials,
            cell_material=model._map, adjoint=adjoint,
            geometry_line=model._geometry_line(), boundary_line=model._boundary_line())


_SPARK = " ▁▂▃▄▅▆▇█"


class TransientModelResult:
    """The outcome of :meth:`Model.transient`: the initial steady state plus the
    power history.

    Every transient begins from a steady-state solve, so that solution is kept
    here as :attr:`steady` (a :class:`ModelResult`) and its eigenvalue as
    :attr:`k0`; the perturbation drives the power away from ``P(0) = 1``. The
    time and power arrays (:attr:`times`, :attr:`power`) are plain NumPy.
    """

    def __init__(self, model, tres, kinetics):
        self.model = model
        self.kinetics = kinetics
        self.k0 = float(tres.k0)
        self.times = np.asarray(tres.times)
        self.power = np.asarray(tres.power)
        self.device = tres.device
        self.solve_seconds = tres.solve_seconds
        self.total_inner_iterations = tres.total_inner_iterations
        self.flux = tres.flux_numpy                    # final scalar flux
        self.steady = ModelResult(model, tres.steady, "diffusion")
        ip = int(np.argmax(self.power))
        self.peak_power, self.peak_time = float(self.power[ip]), float(self.times[ip])
        self.final_power, self.final_time = float(self.power[-1]), float(self.times[-1])
        self.dt = float(self.times[1] - self.times[0]) if self.times.size > 1 else 0.0

    def _sparkline(self, width=48):
        p = self.power
        idx = np.linspace(0, len(p) - 1, min(width, len(p))).round().astype(int)
        s = p[idx]
        lo, hi = float(s.min()), float(s.max())
        lvl = (np.zeros(len(s), int) if hi - lo < 1e-12
               else np.clip(((s - lo) / (hi - lo) * 8).astype(int), 0, 8))
        return "".join(_SPARK[i] for i in lvl)

    def summary(self) -> str:
        kin = self.kinetics
        fam = "family" if kin.n_families == 1 else "families"
        lines = [
            "NDgpu reactor transient",
            "=======================",
            f"  geometry    : {self.model._geometry_line()}",
            f"  groups      : {self.steady.n_groups}     boundary: {self.model._boundary_line()}",
            f"  kinetics    : {kin.n_families} delayed {fam}, beta = {kin.beta_total * 1e5:.0f} pcm",
            "",
            f"  initial steady state : k0 = {self.k0:.6f}  "
            f"(fission source normalised by k0; unperturbed -> P/P0 stays 1)",
            f"  time span            : 0 -> {self.final_time:g} s, dt = {self.dt:g} s, "
            f"{self.times.size - 1} steps, {self.solve_seconds:.2f} s on {self.device}",
            "",
            "  power P/P0:",
            f"    peak      : {self.peak_power:.4f} at t = {self.peak_time:g} s",
            f"    final     : {self.final_power:.4f} at t = {self.final_time:g} s",
            f"    trace     : {self._sparkline()}",
            f"                [{self.power.min():.3g} .. {self.power.max():.3g}] over 0..{self.final_time:g} s",
            "",
            "  (.steady holds the full t=0 solution report)",
        ]
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return (f"TransientModelResult(k0={self.k0:.6f}, P(end)={self.final_power:.4f} P0, "
                f"peak {self.peak_power:.4f} at t={self.peak_time:g}s, on {self.device})")


# ---------------------------------------------------------------------------
# Structured Cartesian model
# ---------------------------------------------------------------------------
class Model:
    """A rectangular reactor defined in centimetres and painted with materials.

    size  : physical extent, ``(Lx,)``, ``(Lx, Ly)`` or ``(Lx, Ly, Lz)`` in cm;
            its length sets the dimensionality. cells : cells per axis.

    Call :meth:`fill` (the background material), then any :meth:`add_box` /
    :meth:`set_boundary`, then :meth:`run`. Methods return ``self`` for chaining.
    """

    def __init__(self, size, cells):
        size = tuple(float(s) for s in np.atleast_1d(size))
        cells = tuple(int(c) for c in np.atleast_1d(cells))
        if len(size) != len(cells):
            raise ValueError("size and cells must have the same length")
        if not 1 <= len(size) <= 3:
            raise ValueError("size must be 1, 2 or 3 numbers (1D/2D/3D)")
        if any(s <= 0 for s in size) or any(c < 1 for c in cells):
            raise ValueError("sizes must be positive and cell counts >= 1")
        self.ndim = len(size)
        self.size = size
        self.cells = cells
        self._shape = tuple(list(cells) + [1] * (3 - self.ndim))
        self._size3 = tuple(list(size) + [1.0] * (3 - self.ndim))
        self._materials: list[Material] = []
        self._map = np.zeros(self._shape, dtype=np.int64)
        self._bc = ["vacuum"] * self.ndim
        self._kinetics = None
        self._centers = [(np.arange(n) + 0.5) * (L / n)
                         for n, L in zip(self._shape, self._size3)]

    def _index(self, material):
        for i, m in enumerate(self._materials):
            if m is material:
                return i
        self._materials.append(material)
        return len(self._materials) - 1

    def fill(self, material) -> "Model":
        """Set the background material filling the whole core (index 0)."""
        if self._materials:
            self._materials[0] = material
        else:
            self._materials.append(material)
        self._map[...] = 0
        return self

    def add_box(self, material, x=None, y=None, z=None) -> "Model":
        """Paint an axis-aligned box (ranges in cm; ``None`` spans the whole core)."""
        if not self._materials:
            raise ValueError("call fill() to set the background material first")
        idx = self._index(material)
        mask = np.ones(self._shape, dtype=bool)
        for ax, rng in enumerate((x, y, z)):
            if rng is None:
                continue
            if ax >= self.ndim:
                raise ValueError(f"cannot set {_AXES[ax]} range on a {self.ndim}D model")
            lo, hi = rng
            sel = (self._centers[ax] >= lo) & (self._centers[ax] <= hi)
            shape = [1, 1, 1]
            shape[ax] = self._shape[ax]
            mask &= sel.reshape(shape)
        self._map[mask] = idx
        return self

    def set_boundary(self, spec=None, *, x=None, y=None, z=None) -> "Model":
        """Boundary conditions by name (``"vacuum"``/``"reflective"``/``"zero-flux"``/
        albedo). ``spec`` sets every real axis; ``x``/``y``/``z`` override one axis
        (a face spec or a ``(lo, hi)`` pair). Collapsed axes stay reflective."""
        if spec is not None:
            self._bc = [spec] * self.ndim
        for ax, val in enumerate((x, y, z)):
            if val is None:
                continue
            if ax >= self.ndim:
                raise ValueError(f"cannot set {_AXES[ax]} boundary on a {self.ndim}D model")
            self._bc[ax] = val
        return self

    @property
    def material_map(self) -> np.ndarray:
        """Per-cell material index, squeezed to the model's dimension."""
        return self._map.reshape(self.cells).copy()

    @property
    def materials(self) -> list:
        """Materials in index order (index 0 is the background fill)."""
        return list(self._materials)

    def _full_bc(self):
        return tuple(self._bc[ax] if ax < self.ndim else "reflective" for ax in range(3))

    def _geometry_line(self):
        dims = " x ".join(f"{s:g}" for s in self.size) + " cm"
        cells = " x ".join(str(c) for c in self.cells)
        return f"{dims},  {cells} cells ({int(np.prod(self.cells)):,})  [{self.ndim}D]"

    def _boundary_line(self):
        parts = []
        for ax in range(self.ndim):
            spec = self._bc[ax]
            parts.append(f"{_AXES[ax]}: {spec[0]}/{spec[1]}" if isinstance(spec, (list, tuple))
                         else f"{_AXES[ax]}: {spec}")
        if self.ndim < 3:
            parts.append("collapsed axes reflective")
        return " | ".join(parts)

    def run(self, method: str = "diffusion", device: str = "auto", adjoint: bool = False,
            tol_k: float = 1e-6, tol_source: float = 1e-5, **solve_kw) -> ModelResult:
        """Solve the k-eigenvalue problem and return a :class:`ModelResult`.

        method : ``"diffusion"`` or ``"sp3"``. device : ``"auto"``/``"cpu"``/``"gpu"``.
        adjoint : solve the adjoint (importance) problem instead of the forward one.
        """
        if not self._materials:
            raise ValueError("empty model: call fill() with a material first")
        if method not in _STRUCTURED:
            raise ValueError(f"method must be one of {sorted(_STRUCTURED)}, got {method!r}")
        groups = self._materials[0].n_groups
        if any(m.n_groups != groups for m in self._materials):
            raise ValueError("all materials must have the same number of groups")

        grid = Grid(shape=self._shape, size=self._size3)
        material = self._materials[0] if len(self._materials) == 1 else self._materials
        kw = dict(bc=self._full_bc(), device=device)
        if len(self._materials) > 1:
            kw["material_map"] = self._map
        solver = _STRUCTURED[method](grid, material, **kw)
        res = solver.solve(tol_k=tol_k, tol_source=tol_source, adjoint=adjoint, **solve_kw)
        return ModelResult(self, res, method, adjoint=adjoint)

    def set_kinetics(self, velocities, beta, decay, chi_delayed=None) -> "Model":
        """Attach point-kinetics data for :meth:`transient` (one velocity per group,
        one beta/decay per delayed-neutron family)."""
        self._kinetics = Kinetics(velocities=velocities, beta=beta, decay=decay,
                                  chi_delayed=chi_delayed)
        return self

    def transient(self, t_end: float, dt: float, kinetics=None, materials_at=None,
                  at=None, device: str = "auto", **solve_kw) -> TransientModelResult:
        """Run a time-dependent solve, starting from this model's steady state.

        A steady eigenvalue solve is always performed first; its eigenvalue k0
        normalises the fission source (the standard critical adjustment), so the
        t=0 state is an exact equilibrium -- an unperturbed run stays at P/P0 = 1
        and the power moves only in response to the perturbation. k0 and the full
        initial solution are returned on the result (``.k0``, ``.steady``).

        kinetics     : :class:`~ndgpu.Kinetics`, or set it earlier with
                       :meth:`set_kinetics`.
        materials_at : optional ``t -> list[Material]`` returning the materials
                       (in this model's index order) at time t -- how the cross
                       sections evolve (a rod ramp, a temperature feedback...).
                       Return the *same* Material objects while nothing changes so
                       operators are only rebuilt when they must be. Omit for an
                       unperturbed run (which must stay at P/P0 = 1).
        at           : advanced escape hatch -- a full ``t -> (materials,
                       material_map)`` callback (lets the geometry move too).
        """
        if not self._materials:
            raise ValueError("empty model: call fill() first")
        kin = kinetics if kinetics is not None else self._kinetics
        if kin is None:
            raise ValueError("no kinetics: use set_kinetics(...) or transient(kinetics=...)")
        groups = self._materials[0].n_groups
        if any(m.n_groups != groups for m in self._materials):
            raise ValueError("all materials must have the same number of groups")
        if len(kin.velocities) != groups:
            raise ValueError(f"kinetics.velocities must have {groups} entries (one per group)")

        grid = Grid(shape=self._shape, size=self._size3)
        if at is not None:
            problem_at = at
        else:
            base, mp, cache = list(self._materials), self._map, {}

            def problem_at(t):
                mats = materials_at(t) if materials_at is not None else base
                key = tuple(id(m) for m in mats)
                if key not in cache:               # keep only the latest so operators
                    cache.clear()                  # are rebuilt exactly when materials change
                    cache[key] = list(mats)
                return cache[key], mp

        solver = TransientSolver(grid, problem_at, kin, bc=self._full_bc(), device=device)
        tres = solver.solve(t_end=t_end, dt=dt, **solve_kw)
        return TransientModelResult(self, tres, kin)


# ---------------------------------------------------------------------------
# Unstructured mesh model
# ---------------------------------------------------------------------------
class MeshModel:
    """An unstructured-mesh reactor: assign materials to the cells of a Gmsh mesh.

    mesh : a path to a Gmsh ``.msh`` file, or an assembled :class:`~ndgpu.Mesh`
           (2-D triangles/quads or 3-D tets/hexes/prisms).

    Assign with :meth:`fill` (all cells), then :meth:`assign` (by mesh tag or a
    boolean/callable selector) and :meth:`add_box` (by cell centroid), then
    :meth:`set_boundary` and :meth:`run`. Solves with the matrix-free
    finite-volume diffusion solver on CPU or GPU.
    """

    def __init__(self, mesh):
        self.mesh = read_gmsh(mesh) if isinstance(mesh, str) else mesh
        self._materials: list[Material] = []
        self._cellmat = np.zeros(self.mesh.n_cells, dtype=np.int64)
        self._bc = "vacuum"
        self._dim = self.mesh.coords.shape[1]

    def _index(self, material):
        for i, m in enumerate(self._materials):
            if m is material:
                return i
        self._materials.append(material)
        return len(self._materials) - 1

    def fill(self, material) -> "MeshModel":
        """Set the material for every cell (index 0)."""
        if self._materials:
            self._materials[0] = material
        else:
            self._materials.append(material)
        self._cellmat[...] = 0
        return self

    def assign(self, material, tag=None, where=None) -> "MeshModel":
        """Assign ``material`` to a subset of cells.

        tag   : match the mesh's per-cell tag (Gmsh physical id).
        where : a boolean mask over cells, or a callable ``centroid -> bool``
                receiving each cell centroid (an (x, y[, z]) array).
        """
        if not self._materials:
            raise ValueError("call fill() first")
        idx = self._index(material)
        if tag is not None:
            sel = np.asarray(self.mesh.cell_tag) == tag
        elif where is not None:
            sel = (np.array([bool(where(c)) for c in self.mesh.centroid])
                   if callable(where) else np.asarray(where, bool))
        else:
            raise ValueError("assign needs either tag= or where=")
        self._cellmat[sel] = idx
        return self

    def add_box(self, material, x=None, y=None, z=None) -> "MeshModel":
        """Assign ``material`` to cells whose centroid falls in the given box (cm)."""
        if not self._materials:
            raise ValueError("call fill() first")
        idx = self._index(material)
        c = self.mesh.centroid
        sel = np.ones(self.mesh.n_cells, bool)
        for ax, rng in enumerate((x, y, z)):
            if rng is None:
                continue
            if ax >= self._dim:
                raise ValueError(f"mesh is {self._dim}D; no {_AXES[ax]} axis")
            sel &= (c[:, ax] >= rng[0]) & (c[:, ax] <= rng[1])
        self._cellmat[sel] = idx
        return self

    def set_boundary(self, spec) -> "MeshModel":
        """Set the boundary condition on every boundary face (name or albedo)."""
        self._bc = spec
        return self

    def _geometry_line(self):
        return (f"{self.mesh.n_cells:,} cells, {len(self.mesh.faces):,} interior faces, "
                f"volume {self.mesh.area.sum():.3g} cm^{self._dim}  [{self._dim}D unstructured]")

    def run(self, device: str = "auto", tol_k: float = 1e-6, tol_source: float = 1e-5,
            **solve_kw) -> ReactorResult:
        """Solve on the unstructured mesh and return a :class:`ReactorResult`."""
        if not self._materials:
            raise ValueError("empty model: call fill() first")
        groups = self._materials[0].n_groups
        if any(m.n_groups != groups for m in self._materials):
            raise ValueError("all materials must have the same number of groups")
        solver = UnstructuredDiffusionSolver(self.mesh, self._materials, self._cellmat,
                                             alpha_boundary=_albedo(self._bc), device=device)
        res = solver.solve(tol_k=tol_k, tol_source=tol_source, **solve_kw)
        return ReactorResult(
            k_eff=res.k_eff, converged=res.converged, outer_iterations=res.outer_iterations,
            inner_iterations=res.inner_iterations, solve_seconds=res.solve_seconds,
            device=res.device, method="diffusion", flux=res.flux, materials=self._materials,
            cell_material=self._cellmat, cell_volume=self.mesh.area,
            geometry_line=self._geometry_line(), boundary_line=f"all faces: {self._bc}")


# ---------------------------------------------------------------------------
# Triangular hex-lattice model
# ---------------------------------------------------------------------------
class HexLattice:
    """A hexagonal lattice of assemblies on the body-fitted triangular solver.

    pitch  : centre-to-centre hex spacing, cm. refine : triangles per hex is
             ``6 * refine**2`` (>= 2 for a usable mesh).

    Place assemblies with :meth:`set_site` (axial coordinates ``(R, C)``), set the
    lattice-edge boundary, then :meth:`run` with diffusion or SP3 transport.
    """

    def __init__(self, pitch: float, refine: int = 4):
        if pitch <= 0 or refine < 1:
            raise ValueError("pitch must be positive and refine >= 1")
        self.pitch = float(pitch)
        self.refine = int(refine)
        self._sites: dict = {}
        self._bc = "vacuum"

    def set_site(self, rc, material) -> "HexLattice":
        """Place ``material`` at hex site ``rc = (R, C)`` (axial coordinates)."""
        self._sites[tuple(rc)] = material
        return self

    def set_boundary(self, spec) -> "HexLattice":
        """Set the boundary condition on the lattice edge (name or albedo)."""
        self._bc = spec
        return self

    def run(self, method: str = "diffusion", device: str = "auto", adjoint: bool = False,
            tol_k: float = 1e-6, tol_source: float = 1e-5, **solve_kw) -> ReactorResult:
        """Rasterize the lattice and solve; returns a :class:`ReactorResult`."""
        if not self._sites:
            raise ValueError("place at least one assembly with set_site()")
        if method not in _TRI:
            raise ValueError(f"method must be one of {sorted(_TRI)}, got {method!r}")
        uniq, id_of = [], {}
        for mat in self._sites.values():
            if id(mat) not in id_of:
                id_of[id(mat)] = len(uniq) + 1              # 1-based; 0 is void
                uniq.append(mat)
        groups = uniq[0].n_groups
        if any(m.n_groups != groups for m in uniq):
            raise ValueError("all assemblies must have the same number of groups")
        void = Material(name="void", diffusion=[1.0] * groups, sigma_a=[0.0] * groups,
                        nu_sigma_f=[0.0] * groups)
        materials = [void] + uniq

        site_material = {rc: id_of[id(mat)] for rc, mat in self._sites.items()}
        raster = rasterize_hex_sites(site_material, self.pitch, self.refine)
        grid = TriGrid(shape=raster.material_map.shape, side=raster.side)
        active = raster.material_map > 0
        solver = _TRI[method](grid, materials, raster.material_map, active=active,
                              mask_bc=self._bc, device=device)
        res = solver.solve(tol_k=tol_k, tol_source=tol_source, adjoint=adjoint, **solve_kw)

        ntri = int(active.sum())
        geo = (f"{len(self._sites)} hex sites, pitch {self.pitch:g} cm, "
               f"{6 * self.refine ** 2} triangles/hex, {ntri:,} active triangles  "
               f"[triangular]")
        return ReactorResult(
            k_eff=res.k_eff, converged=res.converged, outer_iterations=res.outer_iterations,
            inner_iterations=res.inner_iterations, solve_seconds=res.solve_seconds,
            device=res.device, method=method, flux=res.flux_numpy, materials=materials,
            cell_material=raster.material_map, active=active, adjoint=adjoint,
            geometry_line=geo, boundary_line=f"lattice edge: {self._bc}")
