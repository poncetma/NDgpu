"""Simplified, human-friendly front ends for building and running a reactor.

The full solver classes are flexible but ask you to build grids, hand-assemble
material-map arrays, and read a terse ``Result``. These builders wrap the common
cases and return a :class:`ReactorResult` that prints a transparent report
(k_eff, reactivity, where the neutrons go, flux peaking, per-material power):

* :class:`Model`       -- a rectangular Cartesian core, painted with boxes in cm
                          (diffusion or SP3; 1-D / 2-D / 3-D; forward or adjoint).
* :class:`MeshModel`   -- a restricted TPFA-compatible mesh (Gmsh 2.2 ASCII or
                          an assembled :class:`~ndgpu.Mesh`), 2-D or 3-D,
                          materials assigned by centroid region or cell tag.
* :class:`HexLattice`  -- a hexagonal/prismatic geometry builder whose
                          :meth:`~HexLattice.build` returns a reusable
                          :class:`TriReactor` for steady, transient and coupled
                          triangular-grid calculations.

Materials are ordinary :class:`~ndgpu.Material` objects, so defining new cross
sections is unchanged; these classes only remove the grid/array bookkeeping.

Example -- a reflected Cartesian core::

    import ndgpu
    m = ndgpu.Model(size=(120, 120), cells=(60, 60))     # 2-D, cm
    m.fill(reflector).add_box(fuel, x=(30, 90), y=(30, 90)).set_boundary("vacuum")
    print(m.run())
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .grid import Grid
from .hexraster import hex_site_xy, rasterize_hex_sites
from .materials import Kinetics, Material
from .mesh import UnstructuredDiffusionSolver, read_gmsh
from .stencil import face_alpha
from .solver import (DiffusionEigenSolver, SDP1EigenSolver, SDP2EigenSolver,
                     SDP3EigenSolver, SP1EigenSolver, SP3EigenSolver,
                     SP5EigenSolver, SP7EigenSolver)
from .transient import TransientSolver
from .tri import (TriDiffusionEigenSolver, TriGrid, TriSDP1EigenSolver,
                  TriSDP2EigenSolver, TriSDP3EigenSolver, TriSP1EigenSolver,
                  TriSP3EigenSolver, TriSP5EigenSolver, TriSP7EigenSolver)

_STRUCTURED = {"diffusion": DiffusionEigenSolver, "sp1": SP1EigenSolver,
               "sp3": SP3EigenSolver, "sp5": SP5EigenSolver,
               "sp7": SP7EigenSolver, "sdp1": SDP1EigenSolver,
               "sdp2": SDP2EigenSolver, "sdp3": SDP3EigenSolver}
_TRI = {"diffusion": TriDiffusionEigenSolver, "sp1": TriSP1EigenSolver,
        "sp3": TriSP3EigenSolver, "sp5": TriSP5EigenSolver,
        "sp7": TriSP7EigenSolver, "sdp1": TriSDP1EigenSolver,
        "sdp2": TriSDP2EigenSolver, "sdp3": TriSDP3EigenSolver}
_AXES = ("x", "y", "z")


def _albedo(spec) -> float:
    """Robin coefficient for a boundary name/albedo (zero-flux -> a large finite alpha)."""
    a = face_alpha(spec)
    return 1e8 if a == float("inf") else float(a)


# ---------------------------------------------------------------------------
# Shared diagnostics + report
# ---------------------------------------------------------------------------
def _diagnostics(materials, cell_material, flux_flat, cell_volume, active,
                 mix_material=None, mix_weight=None):
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
    vf = flux_flat * vol * act                             # (G, ncells) volume-flux weight
    if mix_material is None:
        mm = np.zeros(ncells, dtype=np.int64)
        mw = np.zeros(ncells)
        mixed = np.zeros(ncells, dtype=bool)
    else:
        mm = np.asarray(mix_material).reshape(-1)
        mw = np.asarray(mix_weight, dtype=float).reshape(-1)
        if mm.shape != cm.shape or mw.shape != cm.shape:
            raise ValueError("mix_material/mix_weight must match the material map")
        mixed = mm >= 0
        mm = np.where(mixed, mm, 0)
    base_frac = np.where(mixed, 1.0 - mw, 1.0)
    mix_frac = np.where(mixed, mw, 0.0)

    production = 0.0
    absorption = 0.0
    for g in range(G):
        nfg = nf[cm, g] * base_frac + nf[mm, g] * mix_frac
        sag = sa[cm, g] * base_frac + sa[mm, g] * mix_frac
        production += float((nfg * vf[g]).sum())
        absorption += float((sag * vf[g]).sum())

    total_vol = float((vol * act).sum())
    rows = []
    for idx, mat in enumerate(materials):
        frac = act * (base_frac * (cm == idx) + mix_frac * (mm == idx))
        if not np.any(frac > 0.0):
            continue
        volf = float((vol * frac).sum()) / total_vol
        fis = float(sum(nf[idx, g] * (flux_flat[g] * vol * frac).sum()
                        for g in range(G)))
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
                 adjoint=False, absorption_correction=0.0,
                 mix_material=None, mix_weight=None):
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
        prod, absorb, rows = _diagnostics(
            materials, cell_material, flat, cell_volume, act,
            mix_material=mix_material, mix_weight=mix_weight)
        absorb += absorption_correction           # volume-mixed sub-cell absorbers (drum arcs)
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

        method : ``"diffusion"``, ``"sp3"``, or the simplified double-PN
        approximations ``"sdp1"``/``"sdp2"``/``"sdp3"`` (equal-cost to
        SP3/SP5/SP7, more accurate than SPN in strongly heterogeneous media).
        device : ``"auto"``/``"cpu"``/``"gpu"``.
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
    """A steady TPFA reactor model on a compatible unstructured mesh.

    mesh : a Gmsh 2.2 ASCII path, or an assembled :class:`~ndgpu.Mesh` of
           first-order 2-D triangles/quads or 3-D tets/hexes/prisms. Imported
           meshes must independently satisfy the TPFA geometry requirements;
           this is not a general CAD-mesh solver.

    Assign with :meth:`fill` (all cells), then :meth:`assign` (by mesh tag or a
    boolean/callable selector) and :meth:`add_box` (by cell centroid), then
    :meth:`set_boundary` and :meth:`run`. Solves with the matrix-free
    finite-volume diffusion solver on one CPU process or one GPU. One boundary
    condition is applied to every exterior face.
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
# Reusable triangular-grid reactor model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeedbackSpec:
    """Temperature feedback data for one named reactor material.

    ``reference_temperature`` is the cross-section tabulation temperature in K;
    ``doppler`` is a scalar or one value per energy group in K**-1/2; and
    ``expansion`` is the optional frozen-geometry density coefficient in 1/K.
    The high-level API expands these named specifications into the material-order
    arrays consumed by :class:`~ndgpu.ThermalFeedback`.
    """

    reference_temperature: float
    doppler: object = 0.0
    expansion: float | None = None


def hex_ring(radius: int) -> list[tuple[int, int]]:
    """Axial ``(R, C)`` coordinates on one hexagonal ring.

    Ring zero contains only ``(0, 0)``; ring ``r`` contains ``6*r`` sites.
    """
    radius = int(radius)
    if radius < 0:
        raise ValueError("hex ring radius must be non-negative")
    if radius == 0:
        return [(0, 0)]
    return [(R, C) for R in range(-radius, radius + 1)
            for C in range(-radius, radius + 1)
            if max(abs(R), abs(C), abs(-R - C)) == radius]


def hex_disk(radius: int) -> list[tuple[int, int]]:
    """Axial coordinates in a complete hexagonal disk through ``radius``."""
    radius = int(radius)
    if radius < 0:
        raise ValueError("hex disk radius must be non-negative")
    return [(R, C) for R in range(-radius, radius + 1)
            for C in range(-radius, radius + 1)
            if max(abs(R), abs(C), abs(-R - C)) <= radius]


class TriReactor:
    """A reusable reactor definition on :class:`~ndgpu.TriGrid`.

    This is the user-facing bridge between geometry construction and solver
    classes. It accepts any already-rasterized triangular/prismatic problem,
    while :meth:`HexLattice.build` creates one without exposing array plumbing.
    The same object can run steady diffusion/transport approximations, neutron
    kinetics, steady thermal coupling, and coupled transients.

    ``total_power`` configured through :meth:`configure_thermal` is the power in
    this modeled domain. For a 2-D slice, supply the slice power (W per modeled
    thickness), not the full 3-D plant rating.
    """

    def __init__(self, grid, materials, material_map, *, active=None,
                 bc="reflective", mask_bc="vacuum", mix_material=None,
                 mix_weight=None, kinetics=None, raster=None,
                 name="tri-grid reactor", geometry_line=None):
        if not isinstance(grid, TriGrid):
            raise TypeError("TriReactor needs a TriGrid")
        self.grid = grid
        self.materials = list(materials)
        if not self.materials:
            raise ValueError("TriReactor needs at least one material")
        self.material_map = np.asarray(material_map, dtype=np.int64)
        if self.material_map.shape != tuple(grid.shape):
            raise ValueError(f"material_map shape {self.material_map.shape} != "
                             f"grid shape {tuple(grid.shape)}")
        self.active = (np.ones(grid.shape, dtype=bool) if active is None else
                       np.asarray(active, dtype=bool))
        if self.active.shape != tuple(grid.shape):
            raise ValueError("active mask must match the grid shape")
        used = self.material_map
        if used.size and (used.min() < 0 or used.max() >= len(self.materials)):
            raise ValueError("material_map contains an out-of-range material index")
        if (mix_material is None) != (mix_weight is None):
            raise ValueError("mix_material and mix_weight must be supplied together")
        self.mix_material = (None if mix_material is None else
                             np.asarray(mix_material, dtype=np.int64))
        self.mix_weight = (None if mix_weight is None else
                           np.asarray(mix_weight, dtype=float))
        if self.mix_material is not None:
            if (self.mix_material.shape != tuple(grid.shape) or
                    self.mix_weight.shape != tuple(grid.shape)):
                raise ValueError("volume-mix arrays must match the grid shape")
            has_mix = self.mix_material >= 0
            if (np.any(self.mix_weight < 0.0) or np.any(self.mix_weight > 1.0) or
                    np.any(has_mix & (self.mix_material >= len(self.materials)))):
                raise ValueError("invalid volume-mix material index or weight")
        groups = self.materials[0].n_groups
        if any(m.n_groups != groups for m in self.materials):
            raise ValueError("all materials must have the same number of groups")
        self.bc, self.mask_bc = bc, mask_bc
        self.kinetics = kinetics
        self.raster = raster
        self.name = str(name)
        self._geometry = geometry_line or (
            f"{int(self.active.sum()):,} active triangular cells")
        self._thermal_config = None

    @property
    def n_groups(self):
        return self.materials[0].n_groups

    @property
    def shape(self):
        return tuple(self.grid.shape)

    @property
    def is_3d(self):
        return len(self.shape) == 4

    def _boundary_line(self):
        z_spec = (self.bc[2] if self.is_3d and isinstance(self.bc, (tuple, list))
                  and len(self.bc) == 3 else self.bc)
        z = f" | z faces: {z_spec}" if self.is_3d else ""
        return f"core edge: {self.mask_bc}{z}"

    def set_kinetics(self, kinetics=None, *, velocities=None, beta=None,
                     decay=None, chi_delayed=None) -> "TriReactor":
        """Attach delayed-neutron data for :meth:`transient` and coupling."""
        if kinetics is not None and any(x is not None for x in
                                        (velocities, beta, decay, chi_delayed)):
            raise ValueError("pass a Kinetics object or its component arrays, not both")
        if kinetics is None:
            if velocities is None or beta is None or decay is None:
                raise ValueError("velocities, beta and decay are required")
            kinetics = Kinetics(velocities, beta, decay, chi_delayed)
        if not isinstance(kinetics, Kinetics):
            raise TypeError("kinetics must be a Kinetics object")
        if kinetics.velocities.shape[-1] != self.n_groups:
            raise ValueError(f"kinetics needs {self.n_groups} velocities per row")
        self.kinetics = kinetics
        return self

    def with_materials(self, materials, *, name=None) -> "TriReactor":
        """Return this geometry with a replacement material list.

        The order and length must be unchanged. This is the direct hand-off for
        ``SphResult.corrected_materials`` or a new cross-section library while
        preserving the validated mesh, controls, kinetics, and thermal setup.
        """
        materials = list(materials)
        if len(materials) != len(self.materials):
            raise ValueError("replacement materials must preserve count/order")
        out = TriReactor(
            self.grid, materials, self.material_map, active=self.active,
            bc=self.bc, mask_bc=self.mask_bc,
            mix_material=self.mix_material, mix_weight=self.mix_weight,
            kinetics=self.kinetics, raster=self.raster,
            name=self.name if name is None else name,
            geometry_line=self._geometry)
        if self._thermal_config is not None:
            out._thermal_config = dict(self._thermal_config)
        return out

    def steady(self, method="diffusion", device="auto", adjoint=False,
               tol_k=1e-6, tol_source=1e-5, solver_kwargs=None,
               **solve_kwargs) -> ReactorResult:
        """Solve a steady k-eigenvalue problem on this reusable geometry."""
        method = method.lower()
        if method not in _TRI:
            raise ValueError(f"method must be one of {sorted(_TRI)}, got {method!r}")
        init = dict(active=self.active, mask_bc=self.mask_bc, bc=self.bc,
                    mix_material=self.mix_material, mix_weight=self.mix_weight,
                    device=device)
        init.update(solver_kwargs or {})
        solver = _TRI[method](self.grid, self.materials, self.material_map, **init)
        res = solver.solve(tol_k=tol_k, tol_source=tol_source,
                           adjoint=adjoint, **solve_kwargs)
        out = ReactorResult(
            k_eff=res.k_eff, converged=res.converged,
            outer_iterations=res.outer_iterations,
            inner_iterations=res.inner_iterations,
            solve_seconds=res.solve_seconds, device=res.device, method=method,
            flux=res.flux_numpy, materials=self.materials,
            cell_material=self.material_map, cell_volume=self.grid.cell_volume,
            active=self.active, adjoint=adjoint,
            geometry_line=self._geometry, boundary_line=self._boundary_line(),
            mix_material=self.mix_material, mix_weight=self.mix_weight)
        out.reactor = self
        out.raw = res
        return out

    run = steady

    def transient(self, t_end, dt, *, kinetics=None, materials_at=None,
                  state_at=None, device="auto", precond_degree=0,
                  precond_dtype=None, linear_solver="cg", **solve_kwargs):
        """Run neutron kinetics, optionally changing materials or full states.

        ``materials_at(t)`` returns materials in this reactor's stable index
        order. ``state_at(t)`` may instead return another :class:`TriReactor`
        with the same grid shape, enabling cached control-geometry frames.
        """
        from .tri import TriGroupOperator

        kin = self.kinetics if kinetics is None else kinetics
        if kin is None:
            raise ValueError("no kinetics: call set_kinetics() or pass kinetics=")
        if materials_at is not None and state_at is not None:
            raise ValueError("use materials_at or state_at, not both")
        base = self
        cache = {}

        def problem_at(t):
            if state_at is not None:
                state = state_at(t)
                if not isinstance(state, TriReactor):
                    raise TypeError("state_at must return a TriReactor")
                if state.shape != base.shape:
                    raise ValueError("state_at returned a different grid shape")
                return (state.materials, state.material_map,
                        state.mix_material, state.mix_weight)
            mats = self.materials if materials_at is None else list(materials_at(t))
            if len(mats) != len(self.materials):
                raise ValueError("materials_at must preserve material count/order")
            key = tuple(id(m) for m in mats)
            if key not in cache:
                cache.clear()
                cache[key] = mats
            return (cache[key], self.material_map,
                    self.mix_material, self.mix_weight)

        solver = TransientSolver(
            self.grid, problem_at, kin, bc=self.bc, active=self.active,
            mask_bc=self.mask_bc, mix_material=self.mix_material,
            mix_weight=self.mix_weight, group_operator=TriGroupOperator,
            eig_solver=TriDiffusionEigenSolver, precond_degree=precond_degree,
            precond_dtype=precond_dtype, linear_solver=linear_solver,
            device=device)
        result = solver.solve(t_end=t_end, dt=dt, **solve_kwargs)
        result.reactor = self
        return result

    def _ordered_thermal_materials(self, spec):
        from .thermal import ThermalMaterial

        if not isinstance(spec, Mapping):
            values = list(spec)
            if len(values) != len(self.materials):
                raise ValueError("thermal material list must align with materials")
            return values
        names = [m.name for m in self.materials]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError("named thermal mapping needs unique non-empty material names")
        used = set(np.unique(self.material_map[self.active]).tolist())
        if self.mix_material is not None:
            used.update(np.unique(self.mix_material[
                self.active & (self.mix_material >= 0) & (self.mix_weight > 0)]).tolist())
        out = []
        for i, name in enumerate(names):
            if name in spec:
                out.append(spec[name])
            elif i not in used:
                out.append(ThermalMaterial(1e-12, name=f"inactive {name}"))
            else:
                raise ValueError(f"no thermal material supplied for active {name!r}")
        return out

    def _feedback_from_mapping(self, specs, ambient_temperature):
        from .feedback import ThermalFeedback

        names = [m.name for m in self.materials]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError("named feedback mapping needs unique non-empty material names")
        rows, t_ref, expansion = [], [], []
        any_groupwise = False
        any_expansion = False
        for name in names:
            item = specs.get(name, FeedbackSpec(ambient_temperature))
            if isinstance(item, Mapping):
                item = FeedbackSpec(**item)
            if not isinstance(item, FeedbackSpec):
                raise TypeError("feedback mapping values must be FeedbackSpec or dict")
            row = np.atleast_1d(np.asarray(item.doppler, dtype=float))
            if row.size not in (1, self.n_groups):
                raise ValueError(f"feedback for {name!r} needs scalar or "
                                 f"{self.n_groups}-group doppler")
            any_groupwise |= row.size > 1
            rows.append(row)
            t_ref.append(float(item.reference_temperature))
            expansion.append(0.0 if item.expansion is None else float(item.expansion))
            any_expansion |= item.expansion is not None
        doppler = (np.vstack([np.broadcast_to(r, (self.n_groups,)) for r in rows])
                   if any_groupwise else np.array([r[0] for r in rows]))
        return ThermalFeedback(t_ref, doppler,
                               np.array(expansion) if any_expansion else None)

    def configure_thermal(self, thermal_materials, *, total_power, feedback=None,
                          thermal_bc="adiabatic", thermal_mask_bc="adiabatic",
                          ambient_temperature=300.0, warm_start=True,
                          eigen_kwargs=None) -> "TriReactor":
        """Attach conduction and feedback data used by coupled calculations.

        ``thermal_materials`` may be a list in solver material order or a
        ``{material_name: ThermalMaterial}`` mapping. ``feedback`` may be an
        existing feedback object or a named mapping of :class:`FeedbackSpec`.
        Missing feedback entries mean zero feedback; inactive padding materials
        need not be present in the thermal mapping.
        """
        from .feedback import ThermalFeedback

        thermal = self._ordered_thermal_materials(thermal_materials)
        if feedback is None:
            feedback = ThermalFeedback(
                np.full(len(self.materials), float(ambient_temperature)),
                np.zeros(len(self.materials)))
        elif isinstance(feedback, Mapping):
            feedback = self._feedback_from_mapping(feedback, ambient_temperature)
        if (hasattr(feedback, "n_materials") and
                feedback.n_materials != len(self.materials)):
            raise ValueError("feedback material count does not match reactor materials")
        if not np.isfinite(total_power) or total_power < 0.0:
            raise ValueError("total_power must be finite and non-negative")
        self._thermal_config = dict(
            thermal_materials=thermal, total_power=float(total_power),
            feedback=feedback, thermal_bc=thermal_bc,
            thermal_mask_bc=thermal_mask_bc,
            ambient_temperature=float(ambient_temperature),
            warm_start=bool(warm_start),
            eigen_kwargs=dict(eigen_kwargs or {"tol_k": 1e-9,
                                               "tol_source": 1e-8}))
        return self

    def coupling_context(self, *, device="auto", dtype=np.float64, **overrides):
        """Create the low-level :class:`CouplingContext` for advanced use."""
        from .coupling import CouplingContext

        if self._thermal_config is None:
            raise ValueError("thermal physics is not configured; call configure_thermal()")
        kw = dict(self._thermal_config)
        kw.update(overrides)
        return CouplingContext(
            grid=self.grid, materials=self.materials,
            material_map=self.material_map, active=self.active,
            mask_bc=self.mask_bc, bc=self.bc,
            mix_material=self.mix_material, mix_weight=self.mix_weight,
            kinetics=self.kinetics, device=device, dtype=dtype, **kw)

    def coupled_steady(self, *, device="auto", context_kwargs=None,
                       **solve_kwargs):
        """Solve the self-consistent steady neutronics/thermal state."""
        from .coupling import CoupledSolver

        ctx = self.coupling_context(device=device, **(context_kwargs or {}))
        return CoupledSolver(ctx).solve(**solve_kwargs)

    def coupled_transient(self, t_end, dt, *, device="auto", problem_at=None,
                          state_at=None, **kwargs):
        """Run the GPU-resident coupled transient through the public model API."""
        from .coupling import coupled_transient

        if self.kinetics is None:
            raise ValueError("no kinetics: call set_kinetics() first")
        if problem_at is not None and state_at is not None:
            raise ValueError("use problem_at or state_at, not both")
        if state_at is not None:
            def problem_at(t):
                state = state_at(t)
                if not isinstance(state, TriReactor):
                    raise TypeError("state_at must return a TriReactor")
                if state.shape != self.shape:
                    raise ValueError("state_at returned a different grid shape")
                return (state.materials, state.material_map,
                        state.mix_material, state.mix_weight)
        ctx = self.coupling_context(device=device)
        return coupled_transient(ctx, t_end=t_end, dt=dt,
                                 problem_at=problem_at, **kwargs)

    def quasistatic_transient(self, t_end, dt, *, shape_dt=2.0,
                              device="auto", problem_at=None, state_at=None,
                              **kwargs):
        """Run an improved quasi-static coupled transient.

        The adjoint-weighted population amplitude and precursors advance every
        ``dt`` while the full spatial shape is corrected every ``shape_dt``;
        the returned power includes the shape's adjoint-weighted fission-power
        factor. Time-dependent IQS is the default; ``shape_method="adiabatic"``
        selects instantaneous eigen shapes. Residual thresholds can force early
        corrections or full-diffusion fallback intervals for rapid localized
        motion; ``iqs_predictor_tol`` adaptively shortens subsequent shape
        intervals after a large amplitude-predictor disagreement.
        ``adjoint_residual_tol`` can refresh a reused importance shape before
        its configured ``adjoint_every`` maximum age.
        """
        from .quasistatic import quasistatic_coupled_transient

        if self.kinetics is None:
            raise ValueError("no kinetics: call set_kinetics() first")
        if problem_at is not None and state_at is not None:
            raise ValueError("use problem_at or state_at, not both")
        if state_at is not None:
            def problem_at(t):
                state = state_at(t)
                if not isinstance(state, TriReactor):
                    raise TypeError("state_at must return a TriReactor")
                if state.shape != self.shape:
                    raise ValueError("state_at returned a different grid shape")
                return (state.materials, state.material_map,
                        state.mix_material, state.mix_weight)
        ctx = self.coupling_context(device=device)
        return quasistatic_coupled_transient(
            ctx, t_end=t_end, dt=dt, shape_dt=shape_dt,
            problem_at=problem_at, **kwargs)


def _drum_absorber_mix(raster, pitch, drums, samples):
    """Per-cell volume fraction of each drum's absorber arc (mix_material/weight).

    ``drums`` is a list of dicts with ``rc``, ``inner``, ``outer``, ``arc_half``
    (radians), ``arc_az`` (radians) and ``absorber_id``. For every triangle near a
    drum the fraction of its area inside that drum's annular sector -- radius in
    (inner, outer], azimuth within +-arc_half of the (rotated) arc centre -- is
    estimated by barycentric sub-sampling. The fraction is non-zero even where the
    thin arc only partly crosses a cell, and varies smoothly as the drum rotates,
    so it drives the volume-mix (mix a fraction of the absorber into the drum-body
    cell) rather than a staircase. Returns (mix_material, mix_weight) shaped like
    the material map (sentinel mix_material = -1 where there is no absorber).
    """
    mmap = raster.material_map
    ni, nj, _ = mmap.shape
    weight = np.zeros((ni, nj, 2))
    mix = np.full((ni, nj, 2), -1, dtype=np.int64)
    owner = np.full((ni, nj, 2), -1, dtype=np.int32)
    centres = np.array([hex_site_xy(rc[0], rc[1], pitch) for d in drums for rc in (d["rc"],)])
    outers = np.array([d["outer"] for d in drums])
    n = int(samples)
    if n < 1:
        raise ValueError("samples must be >= 1")
    # Barycentres of n**2 equal-area sub-triangles. Equally weighting a point
    # lattice over-represents the parent edges and biases thin annular arcs.
    bary = []
    for i in range(n):
        for j in range(n - i):
            u, v = (i + 1.0 / 3.0) / n, (j + 1.0 / 3.0) / n
            bary.append((1.0 - u - v, u, v))
            if i + j < n - 1:
                u, v = (i + 2.0 / 3.0) / n, (j + 2.0 / 3.0) / n
                bary.append((1.0 - u - v, u, v))
    bary = np.asarray(bary)
    for a in range(ni):
        for b in range(nj):
            for t in (0, 1):
                if mmap[a, b, t] == 0:
                    continue
                V = raster.cell_vertices(a, b, t)
                cx, cy = V.mean(0)
                d2 = (centres[:, 0] - cx) ** 2 + (centres[:, 1] - cy) ** 2
                d = int(d2.argmin())
                reach = outers[d] + raster.side
                if d2[d] > reach * reach:
                    continue
                dm = drums[d]
                owner[a, b, t] = d
                pts = bary @ V
                dx, dy = pts[:, 0] - centres[d, 0], pts[:, 1] - centres[d, 1]
                rr = np.hypot(dx, dy)
                dphi = (np.arctan2(dy, dx) - dm["arc_az"] + np.pi) % (2 * np.pi) - np.pi
                inside = (rr > dm["inner"]) & (rr <= dm["outer"]) & (np.abs(dphi) <= dm["arc_half"])
                f = float(inside.mean())
                if f > 0.0:
                    weight[a, b, t] = f
                    mix[a, b, t] = dm["absorber_id"]

    # Preserve each drum's analytic absorber inventory independently at every
    # angle. The same correction is used by the HP-MR benchmark path.
    cell_area = math.sqrt(3.0) / 4.0 * raster.side**2
    for drum_index, drum in enumerate(drums):
        selected = (owner == drum_index) & (weight > 0.0)
        current = cell_area * float(np.sum(weight[selected]))
        if current <= 0.0:
            continue
        target = drum["arc_half"] * (
            drum["outer"]**2 - drum["inner"]**2)
        delta = target - current
        if delta <= 0.0:
            weight[selected] *= target / current
            continue
        capacity = cell_area * float(np.sum(1.0 - weight[selected]))
        if capacity > 0.0:
            weight[selected] += min(delta / capacity, 1.0) * (
                1.0 - weight[selected])
    np.clip(weight, 0.0, 1.0, out=weight)
    return mix, weight


# ---------------------------------------------------------------------------
# Triangular hex-lattice model
# ---------------------------------------------------------------------------
class HexLattice:
    """A hexagonal lattice of assemblies on the body-fitted triangular solver.

    pitch  : centre-to-centre hex spacing, cm. refine : triangles per hex is
             ``6 * refine**2`` (>= 2 for a usable mesh).

    Place assemblies with :meth:`set_site` (axial coordinates ``(R, C)``) or, for a
    control drum, :meth:`set_drum` (a hex with a rotatable absorber arc), set the
    lattice-edge boundary, then :meth:`run` with diffusion or SP3 transport.
    """

    def __init__(self, pitch: float, refine: int = 4):
        if pitch <= 0 or refine < 1:
            raise ValueError("pitch must be positive and refine >= 1")
        self.pitch = float(pitch)
        self.refine = int(refine)
        self._sites: dict = {}
        self._drums: dict = {}
        self._bc = "vacuum"
        self._height = None
        self._nz = None
        self._z_bc = "vacuum"
        self._axial_regions = []
        self._kinetics = None

    def set_site(self, rc, material) -> "HexLattice":
        """Place ``material`` at hex site ``rc = (R, C)`` (axial coordinates)."""
        rc = tuple(rc)
        if len(rc) != 2 or any(int(v) != v for v in rc):
            raise ValueError("a hex site must be two integer axial coordinates")
        self._sites[tuple(map(int, rc))] = material
        return self

    def set_sites(self, sites, material) -> "HexLattice":
        """Place one material at every coordinate in ``sites``."""
        for rc in sites:
            self.set_site(rc, material)
        return self

    def set_ring(self, radius, material) -> "HexLattice":
        """Fill one complete hex ring; shorthand for ``set_sites(hex_ring(...))``."""
        return self.set_sites(hex_ring(radius), material)

    def set_disk(self, radius, material) -> "HexLattice":
        """Fill a complete disk through ``radius`` (later calls may overwrite)."""
        return self.set_sites(hex_disk(radius), material)

    def set_drum(self, rc, body, absorber, inner_radius, outer_radius, arc_deg,
                 angle_deg=0.0) -> "HexLattice":
        """Place a control drum at hex site ``rc``: a ``body`` assembly carrying a
        rotatable ``absorber`` arc, volume-mixed by area fraction.

        The absorber occupies the annular sector between ``inner_radius`` and
        ``outer_radius`` (cm from the hex centre) spanning ``arc_deg`` degrees.
        ``angle_deg`` rotates it about the hex centre, measured from the inward
        radial direction: 0 points at the core centre (inserted), 180 away from
        the core (withdrawn). Any number of drums may be placed, each rotated
        independently -- the drum-worth curve is swept by rerunning with different
        ``angle_deg``.
        """
        rc = tuple(rc)
        inner_radius, outer_radius, arc_deg = (float(inner_radius),
                                                float(outer_radius),
                                                float(arc_deg))
        if len(rc) != 2 or any(int(v) != v for v in rc):
            raise ValueError("a drum site must be two integer axial coordinates")
        rc = tuple(map(int, rc))
        if inner_radius < 0.0 or outer_radius <= inner_radius:
            raise ValueError("drum radii need 0 <= inner_radius < outer_radius")
        if not 0.0 < arc_deg <= 360.0:
            raise ValueError("drum arc_deg must lie in (0, 360]")
        self._sites[rc] = body
        self._drums[rc] = dict(absorber=absorber, inner=inner_radius,
                               outer=outer_radius,
                               arc_half=math.radians(arc_deg) / 2.0,
                               angle_deg=float(angle_deg))
        return self

    def set_drum_angle(self, rc, angle_deg) -> "HexLattice":
        """Rotate an already-defined control drum without redefining its arc."""
        rc = tuple(rc)
        if rc not in self._drums:
            raise KeyError(f"no control drum at site {rc}")
        self._drums[rc]["angle_deg"] = float(angle_deg)
        return self

    def extrude(self, height, nz, *, boundary="vacuum") -> "HexLattice":
        """Extrude the radial lattice into ``nz`` uniform triangular prisms.

        ``boundary`` applies to the lower/upper axial faces. Radial leakage is
        still controlled by :meth:`set_boundary`.
        """
        height, nz = float(height), int(nz)
        if height <= 0.0 or nz < 1:
            raise ValueError("extrusion height must be positive and nz >= 1")
        self._height, self._nz, self._z_bc = height, nz, boundary
        return self

    def add_axial_region(self, material, z, *, replace=None) -> "HexLattice":
        """Paint a material over an axial interval after extrusion.

        ``replace`` optionally supplies one material object or an iterable to
        replace. Omit it to paint every active assembly in the interval. This is
        the convenient way to add top/bottom reflectors while leaving radial
        reflectors or full-height control drums unchanged.
        """
        if self._height is None:
            raise ValueError("call extrude() before adding axial regions")
        lo, hi = map(float, z)
        if not 0.0 <= lo < hi <= self._height:
            raise ValueError("axial region must lie inside 0 <= z <= height")
        if replace is None or isinstance(replace, Material):
            replace = None if replace is None else (replace,)
        else:
            replace = tuple(replace)
        self._axial_regions.append((material, (lo, hi), replace))
        return self

    def set_kinetics(self, kinetics=None, **components) -> "HexLattice":
        """Attach kinetics that :meth:`build` copies to the TriReactor."""
        if kinetics is None:
            kinetics = Kinetics(**components)
        elif components:
            raise ValueError("pass a Kinetics object or component arrays, not both")
        if not isinstance(kinetics, Kinetics):
            raise TypeError("kinetics must be a Kinetics object")
        self._kinetics = kinetics
        return self

    def set_boundary(self, spec) -> "HexLattice":
        """Set the boundary condition on the lattice edge (name or albedo)."""
        self._bc = spec
        return self

    def build(self, samples: int = 8, name="hex-lattice reactor") -> TriReactor:
        """Rasterize once and return a reusable :class:`TriReactor`.

        The returned model exposes its grid/maps for inspection and supports
        steady, transient, thermal, and coupled calculations. ``samples`` sets
        the control-drum arc area-fraction resolution.
        """
        if not self._sites:
            raise ValueError("place at least one assembly with set_site()")
        uniq, id_of = [], {}

        def register(mat):
            if id(mat) not in id_of:
                id_of[id(mat)] = len(uniq) + 1              # 1-based; 0 is void
                uniq.append(mat)
            return id_of[id(mat)]

        for mat in self._sites.values():
            register(mat)
        for dm in self._drums.values():
            register(dm["absorber"])
        for mat, _z, _replace in self._axial_regions:
            register(mat)
        groups = uniq[0].n_groups
        if any(m.n_groups != groups for m in uniq):
            raise ValueError("all assemblies must have the same number of groups")
        void = Material(name="void", diffusion=[1.0] * groups, sigma_a=[0.0] * groups,
                        nu_sigma_f=[0.0] * groups)
        materials = [void] + uniq

        site_material = {rc: id_of[id(mat)] for rc, mat in self._sites.items()}
        raster = rasterize_hex_sites(site_material, self.pitch, self.refine)
        mix_material = mix_weight = None
        if self._drums:
            specs = []
            for rc, dm in self._drums.items():
                x, y = hex_site_xy(rc[0], rc[1], self.pitch)
                specs.append(dict(rc=rc, inner=dm["inner"], outer=dm["outer"],
                                  arc_half=dm["arc_half"],
                                  arc_az=math.atan2(y, x) + math.radians(180.0 + dm["angle_deg"]),
                                  absorber_id=id_of[id(dm["absorber"])]))
            mix_material, mix_weight = _drum_absorber_mix(raster, self.pitch, specs, samples)
        mmap = raster.material_map
        if self._height is None:
            grid = TriGrid(shape=mmap.shape, side=raster.side)
            active = mmap > 0
            bc = "reflective"
            dimension = "2-D triangular"
        else:
            nz = self._nz
            mmap = np.repeat(mmap[..., None], nz, axis=3)
            mix_material = (None if mix_material is None else
                            np.repeat(mix_material[..., None], nz, axis=3))
            mix_weight = (None if mix_weight is None else
                          np.repeat(mix_weight[..., None], nz, axis=3))
            zc = (np.arange(nz) + 0.5) * self._height / nz
            for mat, (lo, hi), replace in self._axial_regions:
                zsel = (zc >= lo) & (zc <= hi if hi == self._height else zc < hi)
                if replace is None:
                    radial = mmap > 0
                else:
                    missing = [m.name for m in replace if id(m) not in id_of]
                    if missing:
                        raise ValueError("axial replace materials are not present "
                                         f"in the radial lattice: {missing}")
                    replace_ids = [id_of[id(m)] for m in replace]
                    radial = np.isin(mmap, replace_ids)
                mmap[radial & zsel[None, None, None, :]] = id_of[id(mat)]
            grid = TriGrid(shape=mmap.shape, side=raster.side,
                           height=self._height)
            active = mmap > 0
            bc = ("reflective", "reflective", self._z_bc)
            dimension = f"3-D triangular prisms, {nz} axial layers"

        ntri = int(active.sum())
        drum_note = f", {len(self._drums)} drums" if self._drums else ""
        geo = (f"{len(self._sites)} hex sites{drum_note}, pitch {self.pitch:g} cm, "
               f"{6 * self.refine ** 2} triangles/hex, {ntri:,} active cells "
               f"[{dimension}]")
        return TriReactor(
            grid, materials, mmap, active=active, bc=bc, mask_bc=self._bc,
            mix_material=mix_material, mix_weight=mix_weight,
            kinetics=self._kinetics, raster=raster, name=name,
            geometry_line=geo)

    def run(self, method: str = "diffusion", device: str = "auto", adjoint: bool = False,
            tol_k: float = 1e-6, tol_source: float = 1e-5, samples: int = 8,
            **solve_kw) -> ReactorResult:
        """Backward-compatible shorthand for ``build(samples).steady(...)``."""
        return self.build(samples=samples).steady(
            method=method, device=device, adjoint=adjoint,
            tol_k=tol_k, tol_source=tol_source, **solve_kw)
