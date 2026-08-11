"""Steady heat conduction with a volumetric sink -- the thermal half of a
coupled neutronics/thermal calculation.

    -div(k(r) grad T)  +  h(r) (T - T_sink(r))  =  q'''(r)

with a Robin surface law ``-k dT/dn = alpha (T_s - T_inf)``.

The `h` term is how a heat-pipe reactor gets rid of its heat: rather than a
coolant channel carrying enthalpy out of the domain, each fuel assembly is
pierced by heat pipes that hold their working fluid at a nearly uniform
evaporator temperature and absorb power in proportion to the local
solid-to-pipe temperature difference. Homogenized over an assembly that is a
volumetric conductance `h` (W/cm^3/K) to a fixed `T_sink`. It is also what
makes the problem well posed on an otherwise adiabatic core: without a sink or
a lossy boundary, a steady state with an internal source does not exist.

**No new discretization.** This is the operator the diffusion solver already
builds -- ``-div(D grad .) + Sigma_r`` -- read with `D -> k` and
`Sigma_r -> h`. So conduction inherits, for free and bit-for-bit, the
harmonic-mean face coefficients (exact for piecewise-constant k), the
non-rectangular `active` mask, the Robin boundary machinery, the triangular and
extruded-prism meshes, the matrix-free CPU/GPU stencil and the CG solve. The
physics that is genuinely new here is three lines: the source, the sink and the
ambient boundary term.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .backend import asnumpy, device_name, get_backend, synchronize
from .blend import MaterialBlend
from .grid import Grid
from .linalg import get_linear_solver, neumann_preconditioner
from .stencil import BC_REFLECTIVE, BC_ZERO_FLUX, GroupOperator, face_alpha, normalize_bc
from .tri import TriGrid, TriGroupOperator


@dataclass
class ThermalMaterial:
    """Thermal constants for one material region.

    conductivity     : k, W/(cm K).
    sink_coeff       : h, W/(cm^3 K) -- the volumetric conductance to the heat
                       pipes, zero in a region they do not pass through.
    sink_temperature : T_sink, K -- the heat-pipe evaporator temperature.
                       Irrelevant where sink_coeff is zero.
    heat_capacity    : rho*cp, J/(cm^3 K). Only the TRANSIENT solver reads it;
                       it is what sets how fast the region can actually change
                       temperature, and therefore whether a power excursion
                       shows up as heat or is ridden out by the thermal mass.
    """

    conductivity: float
    sink_coeff: float = 0.0
    sink_temperature: float = 0.0
    heat_capacity: float = 0.0
    name: str = ""


@dataclass
class ThermalResult:
    temperature: object          # (*grid.shape), on the solve device
    iterations: int
    solve_seconds: float
    device: str
    source_watts: float          # integral of q''' over the core
    sink_watts: float            # heat removed by the volumetric sink
    leakage_watts: float         # heat lost through Robin surfaces
    storage_watts: float = 0.0   # rate of change of stored energy (transient)

    @property
    def temperature_numpy(self) -> np.ndarray:
        return asnumpy(self.temperature)

    @property
    def balance_residual(self) -> float:
        """Relative closure of the discrete energy balance.

        source == sink + leakage is an *exact* identity of the discretization
        (the interior face couplings telescope on summation, because the
        operator is symmetric), not an approximation that improves with mesh.
        A nonzero value means a sign error, a dropped metric weight or a
        mis-scaled boundary term -- none of which stop CG from converging.
        """
        if not all(np.isfinite(x) for x in (
                self.source_watts, self.sink_watts,
                self.leakage_watts, self.storage_watts)):
            return float("nan")
        residual = abs(self.source_watts - self.sink_watts
                       - self.leakage_watts - self.storage_watts)
        # Normalize by the largest flow, not by the source: at zero power the
        # source is zero while heat still moves between the pipes and the
        # vessel, and dividing by the source would either blow up or (worse)
        # silently report an absolute number on a relative scale.
        scale = max(abs(self.source_watts), abs(self.sink_watts),
                    abs(self.leakage_watts), abs(self.storage_watts))
        return residual / scale if scale > 0.0 else residual

    def __repr__(self):
        T = self.temperature_numpy
        return (f"ThermalResult(T = {T.min():.1f} .. {T.max():.1f} K, "
                f"mean {T.mean():.1f} K, {self.iterations} CG iterations, "
                f"balance {self.balance_residual:.1e}, "
                f"{self.solve_seconds:.3f} s, {self.device})")


#: Boundary-condition names in thermal language, mapped onto the operator's
#: neutronics vocabulary. A bare number is a surface heat-transfer coefficient
#: alpha in W/(cm^2 K), which is exactly what the Robin face term wants.
_THERMAL_BC = {
    "adiabatic": BC_REFLECTIVE,     # alpha = 0, zero heat flux
    "reflective": BC_REFLECTIVE,    # the repo's own name for the same thing
    "isothermal": BC_ZERO_FLUX,     # alpha -> infinity, surface pinned to T_inf
    "zero-flux": BC_ZERO_FLUX,      # ditto (the name refers to the neutron flux)
}


def _thermal_bc(spec):
    """Translate a thermal boundary spec into the operator's vocabulary.

    Rejects ``"vacuum"``: it is the Marshak neutron albedo alpha = 1/2, a
    number with no thermal meaning at all. Silently accepting it would put a
    0.5 W/(cm^2 K) film coefficient on the surface and look plausible.
    """
    if isinstance(spec, str):
        key = spec.lower()
        if key == "vacuum":
            raise ValueError(
                "'vacuum' is a neutron boundary condition (albedo 1/2) and has "
                "no thermal meaning; use 'adiabatic', 'isothermal', or a "
                "numeric surface heat-transfer coefficient in W/(cm^2 K)")
        if key not in _THERMAL_BC:
            raise ValueError(
                f"unknown thermal boundary {spec!r}; use 'adiabatic', "
                f"'isothermal', or a numeric heat-transfer coefficient")
        return _THERMAL_BC[key]
    if isinstance(spec, (tuple, list)):
        return type(spec)(_thermal_bc(s) for s in spec)
    alpha = float(spec)
    if alpha < 0.0:
        raise ValueError("a surface heat-transfer coefficient must be >= 0")
    return alpha


class ConductionSolver:
    """Steady conduction on the neutronics mesh.

    grid              : ``Grid`` (Cartesian or cylindrical r-z) or ``TriGrid``.
    thermal_materials : list of :class:`ThermalMaterial`, indexed by
                        ``material_map`` -- the SAME map the neutronics uses.
    bc                : outer box faces. Thermal names ('adiabatic',
                        'isothermal') or a numeric heat-transfer coefficient,
                        per-face like the neutronics bc. NOTE a ``TriGrid``
                        reads this for its z faces only; the in-plane core
                        surface is governed by ``mask_bc``.
    mask_bc           : the law on faces where an active cell meets an excised
                        one -- the real core surface for a body-fitted mesh.
    ambient_temperature : T_inf, K, seen by every Robin surface.
    mix_material / mix_weight : the solver's two-material volume blend, so a
                        control-drum arc is the same fraction of B4C to heat as
                        it is to neutrons.

    Conductivity blends *arithmetically* across a mixed cell, unlike the
    diffusion coefficient's harmonic blend. Both are right: the operator
    already takes the harmonic mean ACROSS faces (conduction in series, the
    direction heat crosses a boundary), while two materials sharing a cell
    conduct in parallel, whose correct average is the volume-weighted
    arithmetic one. It also keeps a void-adjacent cell finite.
    """

    def __init__(self, grid, thermal_materials, material_map=None, *,
                 bc="adiabatic", active=None, mask_bc="adiabatic",
                 ambient_temperature=300.0, mix_material=None, mix_weight=None,
                 op_cls=None, device="auto", dtype=np.float64,
                 linear_solver="cg", time_step=None, precond_degree=0):
        self.grid = grid
        self.xp = xp = get_backend(device)
        self.device = device_name(xp)
        self.dtype = np.dtype(dtype)
        self.ambient_temperature = float(ambient_temperature)
        self._linsolve = get_linear_solver(linear_solver)
        self.precond_degree = int(precond_degree)
        if self.precond_degree < 0:
            raise ValueError("precond_degree must be non-negative")

        mats = ([thermal_materials] if isinstance(thermal_materials, ThermalMaterial)
                else list(thermal_materials))
        blend = MaterialBlend(xp, grid.shape, material_map, len(mats),
                              dtype=self.dtype, mix_material=mix_material,
                              mix_weight=mix_weight)
        self.blend = blend

        k = blend.field(np.array([m.conductivity for m in mats], dtype=float))
        h = blend.field(np.array([m.sink_coeff for m in mats], dtype=float))
        t_sink = blend.field(np.array([m.sink_temperature for m in mats], dtype=float))
        rho_cp = blend.field(np.array([m.heat_capacity for m in mats], dtype=float))
        if float(xp.min(k)) <= 0.0:
            raise ValueError("conductivity must be positive in every cell "
                             "(give a void region a small positive k)")
        if float(xp.min(h)) < 0.0:
            raise ValueError("sink_coeff must be non-negative")

        self.active = None if active is None else xp.asarray(active).astype(bool)
        if self.active is not None:
            # Excised cells take no source and no sink. The operator gives them
            # a unit diagonal and no couplings, so they end up at exactly T_inf
            # -- inert, and they drop out of the energy balance identically.
            h = xp.where(self.active, h, 0.0)
            rho_cp = xp.where(self.active, rho_cp, 0.0)
        self.k, self.h, self.t_sink = k, h, t_sink
        self.rho_cp = rho_cp

        # Backward Euler in one line of algebra: rho*cp dT/dt costs the
        # operator a diagonal term rho*cp/dt and the source a rho*cp/dt * T_old.
        # That is the SAME shape as the heat-pipe sink h(T - T_sink), so a
        # transient step reuses the steady operator untouched -- with dt fixed
        # it is even the same factorization-free stencil, built once.
        self.time_step = None if time_step is None else float(time_step)
        if self.time_step is not None and self.time_step <= 0.0:
            raise ValueError("time_step must be positive")
        self._capacity_over_dt = (0.0 if self.time_step is None
                                  else rho_cp / self.time_step)
        removal = h if self.time_step is None else h + self._capacity_over_dt

        bc = _thermal_bc(bc)
        mask_bc = _thermal_bc(mask_bc)
        if op_cls is None:
            op_cls = TriGroupOperator if isinstance(grid, TriGrid) else GroupOperator
        self.op = op_cls(xp, grid, k, removal, bc=bc, active=self.active,
                         mask_bc=mask_bc)
        self._precond = neumann_preconditioner(
            self.op.apply, self.op.inv_diag, self.precond_degree)
        # Discontinuity factors make the two sides of a face carry different
        # weights, which breaks the symmetry CG assumes AND puts a constant
        # field outside the leakage null space -- so the boundary identity
        # below would pick up a spurious interior source. Only reachable
        # through a custom op_cls; caught rather than silently mis-solved.
        a, b = getattr(self.op, "a_hyp", None), getattr(self.op, "b_hyp", None)
        if a is not None and b is not None and not bool(xp.all(a == b)):
            raise ValueError(
                "conduction needs a symmetric operator, but this one carries "
                "discontinuity factors (a neutronics equivalence device): they "
                "break both the CG solve and the constant-field identity the "
                "ambient boundary term relies on")

        # The boundary source, extracted from the operator rather than rebuilt.
        # On a CONSTANT field the interior face couplings telescope to zero, so
        # A.1 = h*w + (Robin surface terms) exactly -- subtracting the sink
        # leaves the surface terms alone, whatever the geometry (Cartesian,
        # cylindrical-weighted, triangular, extruded) and whichever faces are
        # exposed. That avoids re-deriving robin_face_term here and getting the
        # metric weights subtly wrong.
        w = getattr(self.op, "rhs_weight", None)
        self._w = 1.0 if w is None else w
        # The identity below subtracts the operator's FULL removal, which in a
        # transient includes the capacity term -- otherwise the leftover would
        # be mistaken for a boundary source and grow as dt shrinks.
        self._removal = removal
        # An r-z row is weighted by the CELL RADIUS, not by the true annular
        # volume 2*pi*r*dr*dz: the neutronics only ever forms ratios, so the
        # 2*pi and the dummy y extent cancel and nobody noticed. Absolute watts
        # do not have that luxury -- and neither does the power density, which
        # divides a rated wattage by this volume, so getting it wrong scales
        # the whole temperature field. One constant factor fixes both.
        met = getattr(grid, "cylindrical_metrics", lambda: None)()
        self._geom_scale = 1.0 if met is None else 2.0 * math.pi / grid.spacing[1]
        self._bnd = (self.op.apply(xp.ones(grid.shape, dtype=self.dtype))
                     - self._removal * self._w)

        if self._is_singular():
            raise ValueError(
                "the conduction problem is singular: no cell has a heat sink "
                "(sink_coeff) and no surface loses heat (every boundary is "
                "adiabatic), so a steady state with an internal source does "
                "not exist. Give the heat somewhere to go.")

    def _is_singular(self):
        """No sink anywhere and no lossy surface => a constant shift is a null
        vector and there is no steady state. (Excised cells are excluded: their
        unit diagonal is bookkeeping, not physics.)

        A transient is never singular: the capacity term rho*cp/dt is itself a
        positive diagonal, and physically the answer to "there is nowhere for
        the heat to go" is that the core just keeps heating up -- which is a
        perfectly well-posed thing to march.
        """
        if self.time_step is not None and float(self.xp.max(self.rho_cp)) > 0.0:
            return False
        xp = self.xp
        bnd = (self._bnd if self.active is None
               else xp.where(self.active, self._bnd, 0.0))
        return float(xp.max(self.h)) <= 0.0 and float(xp.max(xp.abs(bnd))) <= 0.0

    @property
    def cell_volume(self):
        """True per-cell volume in cm^3.

        Scalar on Cartesian and triangular meshes; on r-z it is the annulus
        ``2 pi r dr dz``, per cell. Pass this to
        :func:`ndgpu.power.power_density` so the rated wattage is divided by a
        physical volume.
        """
        return self.grid.cell_volume * self._w * self._geom_scale

    def step(self, power_density, temperature, rtol=1e-12, maxiter=20000,
             check_every=1, diagnostics=True,
             synchronize_timing=True) -> ThermalResult:
        """Advance one backward-Euler step of ``time_step`` from ``temperature``.

        Unconditionally stable, so the step size is chosen by how fast the
        answer moves rather than by a stability limit -- which matters here
        because the neutronics step is set by prompt kinetics (milliseconds)
        while the thermal response is seconds, and an explicit scheme would
        force the slow physics onto the fast clock.
        """
        if self.time_step is None:
            raise ValueError("step() needs a time_step; construct the solver "
                             "with time_step=dt (or call solve() for steady)")
        return self.solve(
            power_density, rtol=rtol, maxiter=maxiter,
            check_every=check_every, diagnostics=diagnostics,
            synchronize_timing=synchronize_timing,
            t0=temperature, _previous=temperature)

    def solve(self, power_density, rtol=1e-12, maxiter=20000, t0=None,
              _previous=None, check_every=1, diagnostics=True,
              synchronize_timing=True) -> ThermalResult:
        """Solve for the temperature field given a volumetric source in W/cm^3.

        ``check_every`` spaces the Krylov convergence reductions, which avoids
        a device-to-host pipeline stall on every iteration on GPU.  Setting
        ``diagnostics=False`` skips the four global reductions used by the
        exact energy-balance check; the corresponding result fields and
        ``balance_residual`` are NaN.  ``synchronize_timing=False`` avoids the
        two timing-only stream synchronizations when an enclosing driver uses
        CUDA events or does not need an isolated thermal wall time.
        """
        xp = self.xp
        check_every = int(check_every)
        if check_every < 1:
            raise ValueError("check_every must be positive")
        q = xp.asarray(power_density, dtype=self.dtype)
        if q.shape != tuple(self.grid.shape):
            raise ValueError(f"power density shape {q.shape} != grid shape "
                             f"{tuple(self.grid.shape)}")
        if self.active is not None:
            q = xp.where(self.active, q, 0.0)

        source = q + self.h * self.t_sink
        if _previous is not None:
            source = source + self._capacity_over_dt * xp.asarray(
                _previous, dtype=self.dtype)
        elif self.time_step is not None:
            raise ValueError("a transient solver needs the previous "
                             "temperature; call step(q, T_old)")
        rhs = self._w * source + self.ambient_temperature * self._bnd
        x0 = (xp.full(self.grid.shape, self.ambient_temperature, dtype=self.dtype)
              if t0 is None else xp.asarray(t0, dtype=self.dtype).copy())

        if synchronize_timing:
            synchronize(xp)
        t_start = time.perf_counter()
        T, iters = self._linsolve(self.op.apply, rhs, x0, self.op.inv_diag, xp,
                                  rtol=rtol, maxiter=maxiter,
                                  precond=self._precond,
                                  check_every=check_every)
        if synchronize_timing:
            synchronize(xp)
            seconds = time.perf_counter() - t_start
        else:
            # A host timer cannot attribute asynchronous GPU work without a
            # synchronization.  The coupled driver uses CUDA events instead.
            seconds = (time.perf_counter() - t_start
                       if xp is np else float("nan"))

        if diagnostics:
            source, sink, leak, storage = self.energy_balance(T, q, _previous)
        else:
            source = sink = leak = storage = float("nan")
        return ThermalResult(temperature=T, iterations=iters,
                             solve_seconds=seconds, device=self.device,
                             source_watts=source, sink_watts=sink,
                             leakage_watts=leak, storage_watts=storage)

    def energy_balance(self, temperature, power_density, previous=None):
        """(source, sink, leakage, storage) in watts.

        Summing the discrete equations over all cells makes the interior face
        couplings cancel in pairs, leaving fission heat in = heat-pipe removal
        + surface loss. An exact identity, so it is a true check rather than a
        convergence estimate.
        """
        xp = self.xp
        T = xp.asarray(temperature, dtype=self.dtype)
        q = xp.asarray(power_density, dtype=self.dtype)
        dV = self.grid.cell_volume * self._geom_scale
        vol = dV * self._w
        source = float(xp.sum(vol * q))
        sink = float(xp.sum(vol * self.h * (T - self.t_sink)))
        # _bnd already carries the geometry's metric weight (it came out of the
        # operator), so it pairs with the bare cell volume, not the weighted one.
        leak = float(xp.sum(dV * self._bnd * (T - self.ambient_temperature)))
        # Transient: what did not leave went into raising the temperature.
        storage = 0.0
        if previous is not None and self.time_step is not None:
            prev = xp.asarray(previous, dtype=self.dtype)
            storage = float(xp.sum(vol * self._capacity_over_dt * (T - prev)))
        return source, sink, leak, storage
