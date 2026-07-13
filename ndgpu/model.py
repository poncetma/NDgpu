"""A simplified, human-friendly front end for building and running a reactor.

The full solver classes (:class:`~ndgpu.DiffusionEigenSolver`, the SP3 and
triangular variants) are flexible but ask you to build a ``Grid``, hand-assemble
a ``material_map`` array, and read a terse ``Result``. :class:`Model` wraps the
common case -- a rectangular core, defined in centimetres, painted with named
materials, with boundary conditions given by name -- and returns a
:class:`ModelResult` that prints a transparent, human-readable report (k_eff,
reactivity, where the neutrons go, flux peaking, per-material power).

Example -- a bare two-group cube::

    import ndgpu
    m = ndgpu.Model(size=(90, 90, 90), cells=(30, 30, 30))
    m.fill(ndgpu.PWR_TWO_GROUP)
    m.set_boundary("vacuum")
    print(m.run())

Example -- a reflected core, built up region by region::

    m = ndgpu.Model(size=(120, 120), cells=(60, 60))     # 2D
    m.fill(reflector)                                     # background
    m.add_box(fuel, x=(30, 90), y=(30, 90))              # central fuel block
    m.set_boundary("vacuum")
    result = m.run(method="diffusion")
    print(result)                                        # human-readable summary
    result.flux                                          # (G, nx, ny, nz) array

Materials are ordinary :class:`~ndgpu.Material` objects, so defining new cross
sections is exactly as before. This module only removes the grid/array bookkeeping.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid
from .materials import Material
from .solver import DiffusionEigenSolver, SP3EigenSolver

_SOLVERS = {"diffusion": DiffusionEigenSolver, "sp3": SP3EigenSolver}
_AXES = ("x", "y", "z")


class Model:
    """A rectangular reactor defined in centimetres and painted with materials.

    size  : physical extent, ``(Lx,)``, ``(Lx, Ly)`` or ``(Lx, Ly, Lz)`` in cm.
            Its length sets the dimensionality (1D / 2D / 3D).
    cells : number of cells per axis, same length as ``size``.

    Build it by calling :meth:`fill` (the background material), then any number
    of :meth:`add_box` (region overrides) and :meth:`set_boundary`, then
    :meth:`run`. The methods return ``self`` so they can be chained.
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
        # pad to 3D internally; collapsed axes are one cell of nominal 1 cm.
        self._shape = tuple(list(cells) + [1] * (3 - self.ndim))
        self._size3 = tuple(list(size) + [1.0] * (3 - self.ndim))
        self._materials: list[Material] = []
        self._map = np.zeros(self._shape, dtype=np.int64)
        self._bc = ["vacuum"] * self.ndim               # real axes; collapsed -> reflective
        self._centers = [(np.arange(n) + 0.5) * (L / n)
                         for n, L in zip(self._shape, self._size3)]

    # -- construction -------------------------------------------------------
    def _index(self, material: Material) -> int:
        for i, m in enumerate(self._materials):
            if m is material:
                return i
        self._materials.append(material)
        return len(self._materials) - 1

    def fill(self, material: Material) -> "Model":
        """Set the background material filling the whole core (index 0)."""
        if self._materials:
            self._materials[0] = material
        else:
            self._materials.append(material)
        self._map[...] = 0
        return self

    def add_box(self, material: Material, x=None, y=None, z=None) -> "Model":
        """Paint an axis-aligned box with ``material``.

        Each of ``x``, ``y``, ``z`` is a ``(lo, hi)`` range in cm; an axis left
        as ``None`` spans the whole core. Cells whose centre falls in every given
        range are assigned ``material`` (later calls overwrite earlier ones).
        """
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
        """Set boundary conditions by name.

        ``spec`` applies one face spec to every real axis; ``x``/``y``/``z``
        override a single axis (a face spec, or a ``(lo, hi)`` pair for the two
        faces of that axis). A face spec is ``"vacuum"``, ``"reflective"``,
        ``"zero-flux"``, or a non-negative albedo. Collapsed axes of a 1D/2D
        model are always reflective.
        """
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
        """The per-cell material index array, squeezed to the model's dimension.

        Handy for inspecting what :meth:`add_box` painted; index ``i`` refers to
        ``self.materials[i]``.
        """
        return self._map.reshape(self.cells).copy()

    @property
    def materials(self) -> list:
        """The materials in index order (index 0 is the background fill)."""
        return list(self._materials)

    # -- running ------------------------------------------------------------
    def _full_bc(self):
        # real axes use the user's spec; collapsed axes are reflective (no leakage).
        return tuple(self._bc[ax] if ax < self.ndim else "reflective" for ax in range(3))

    def run(self, method: str = "diffusion", device: str = "auto",
            tol_k: float = 1e-6, tol_source: float = 1e-5, **solve_kw) -> "ModelResult":
        """Solve the k-eigenvalue problem and return a :class:`ModelResult`.

        method : ``"diffusion"`` (default) or ``"sp3"`` (simplified P3 transport).
        device : ``"auto"`` (GPU if present), ``"cpu"`` or ``"gpu"``.
        """
        if not self._materials:
            raise ValueError("empty model: call fill() with a material first")
        if method not in _SOLVERS:
            raise ValueError(f"method must be one of {sorted(_SOLVERS)}, got {method!r}")
        groups = self._materials[0].n_groups
        if any(m.n_groups != groups for m in self._materials):
            raise ValueError("all materials must have the same number of groups")

        grid = Grid(shape=self._shape, size=self._size3)
        bc = self._full_bc()
        material = self._materials[0] if len(self._materials) == 1 else self._materials
        kw = dict(bc=bc, device=device)
        if len(self._materials) > 1:
            kw["material_map"] = self._map
        solver = _SOLVERS[method](grid, material, **kw)
        res = solver.solve(tol_k=tol_k, tol_source=tol_source, **solve_kw)
        return ModelResult(self, res, method)


def _reaction_rates(materials, material_map, flux):
    """Total fission-production and absorption rates from the scalar flux.

    Both are formed directly from the cross sections and the (physical) scalar
    flux, so this is method-agnostic -- it works for diffusion or SP3 alike. The
    global balance then fixes leakage = production/k - absorption, which is the
    net current lost through the boundary.
    """
    G = flux.shape[0]
    sa = np.array([m.sigma_a for m in materials])          # (M, G)
    nf = np.array([m.nu_sigma_f for m in materials])
    per_cell_a = sa[material_map]                          # (nx, ny, nz, G)
    per_cell_f = nf[material_map]
    absorption = float(sum((per_cell_a[..., g] * flux[g]).sum() for g in range(G)))
    production = float(sum((per_cell_f[..., g] * flux[g]).sum() for g in range(G)))
    return production, absorption


class ModelResult:
    """The outcome of :meth:`Model.run`, with a human-readable report.

    Printing it (or calling :meth:`summary`) shows k_eff, reactivity, the
    converged neutron balance, flux peaking and per-material power. The numeric
    fields (:attr:`k_eff`, :attr:`flux`, ...) are plain floats / NumPy arrays.
    """

    def __init__(self, model: Model, result, method: str):
        self.model = model
        self.method = method
        self.k_eff = float(result.k_eff)
        self.reactivity_pcm = (1.0 - 1.0 / self.k_eff) * 1e5
        self.converged = bool(result.converged)
        self.outer_iterations = result.outer_iterations
        self.inner_iterations = result.inner_iterations
        self.solve_seconds = result.solve_seconds
        self.device = result.device
        self.flux = result.flux_numpy            # (G, nx, ny, nz)
        self.n_groups = self.flux.shape[0]

        prod, absorb = _reaction_rates(model._materials, model._map, self.flux)
        loss = prod / self.k_eff                 # neutrons available per generation
        leak = loss - absorb                     # balance: production/k = absorption + leakage
        self.absorbed_fraction = absorb / loss
        self.leakage_fraction = leak / loss
        # thermal-group (last) peak-to-average flux
        thermal = self.flux[-1]
        self.peaking = float(thermal.max() / thermal.mean())
        self._material_stats = self._per_material()

    def _per_material(self):
        mmap = self.model._map
        mats = self.model._materials
        total_cells = mmap.size
        total_fis = 0.0
        rows = []
        for idx, mat in enumerate(mats):
            mask = mmap == idx
            if not mask.any():
                continue
            nsf = mat.nu_sigma_f
            fis = float(sum(nsf[g] * self.flux[g][mask].sum() for g in range(self.n_groups)))
            rows.append([mat.name or f"material {idx}", mask.sum() / total_cells, fis])
            total_fis += fis
        for r in rows:
            r.append(r[2] / total_fis if total_fis > 0 else 0.0)
        return rows

    # -- reporting ----------------------------------------------------------
    def _geometry_line(self):
        dims = " x ".join(f"{s:g}" for s in self.model.size) + " cm"
        cells = " x ".join(str(c) for c in self.model.cells)
        ncell = int(np.prod(self.model.cells))
        return f"{dims},  {cells} cells ({ncell:,})  [{self.model.ndim}D]"

    def _boundary_line(self):
        parts = []
        for ax in range(self.model.ndim):
            spec = self.model._bc[ax]
            if isinstance(spec, (list, tuple)):
                parts.append(f"{_AXES[ax]}: {spec[0]}/{spec[1]}")
            else:
                parts.append(f"{_AXES[ax]}: {spec}")
        if self.model.ndim < 3:
            parts.append("collapsed axes reflective")
        return " | ".join(parts)

    def summary(self) -> str:
        status = (f"converged in {self.outer_iterations} outer / "
                  f"{self.inner_iterations} inner iterations, {self.solve_seconds:.2f} s"
                  if self.converged else "DID NOT CONVERGE")
        lines = [
            "NDgpu reactor solution",
            "======================",
            f"  geometry    : {self._geometry_line()}",
            f"  groups      : {self.n_groups}     method: {self.method}     device: {self.device}",
            f"  boundary    : {self._boundary_line()}",
            "",
            f"  k_eff       : {self.k_eff:.6f}",
            f"  reactivity  : {self.reactivity_pcm:+.0f} pcm   (rho = (k-1)/k)",
            f"  status      : {status}",
            "",
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
        return (f"ModelResult(k_eff={self.k_eff:.6f}, {self.reactivity_pcm:+.0f} pcm, "
                f"{s}, {self.method} on {self.device})")
