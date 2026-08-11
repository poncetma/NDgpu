"""Conduction solver against exact solutions and exact invariants.

The sinked conduction equation has a closed form on a slab, and it is a
hyperbolic one -- not a polynomial the finite-volume scheme could reproduce by
accident -- so it tests the discretization rather than flattering it. The
energy balance is an exact identity of the discretization, which makes it the
sharpest single check available: a sign error, a dropped metric weight or a
mis-scaled boundary term all break it while CG still converges happily.
"""

import numpy as np
import pytest

from ndgpu import Grid
from ndgpu.thermal import ConductionSolver, ThermalMaterial
from ndgpu.tri import TriGrid

# One physically-scaled parameter set, shared by the analytic tests, chosen so
# the diffusion length L_d = sqrt(k/h) = 4.32 cm is comparable to the slab
# thickness: the profile is then genuinely curved over the domain.
K, H, Q = 0.25, 0.0134, 0.672          # W/cm/K, W/cm^3/K, W/cm^3
T_SINK, LENGTH = 750.0, 10.0


def _slab(n, bc, ambient, size=LENGTH):
    grid = Grid(shape=(n, 1, 1), size=(size, 1.0, 1.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=H, sink_temperature=T_SINK)
    solver = ConductionSolver(grid, mat, bc=bc, ambient_temperature=ambient)
    res = solver.solve(np.full(grid.shape, Q))
    x = grid.cell_centers(0)
    return x, res.temperature_numpy[:, 0, 0], res, solver


def _dirichlet_exact(x, t_wall):
    """-k T'' + h(T - T_sink) = q, T'(0) = 0, T(L) = t_wall."""
    ld = np.sqrt(K / H)
    particular = T_SINK + Q / H
    return particular + (t_wall - particular) * np.cosh(x / ld) / np.cosh(LENGTH / ld)


def _robin_exact(x, alpha, t_inf):
    """Same, with -k T'(L) = alpha (T(L) - T_inf) at the outer face."""
    ld = np.sqrt(K / H)
    particular = T_SINK + Q / H
    amp = (-alpha * (particular - t_inf)
           / (np.sqrt(K * H) * np.sinh(LENGTH / ld)
              + alpha * np.cosh(LENGTH / ld)))
    return particular + amp * np.cosh(x / ld)


def _order(errors):
    """Observed convergence order between successive mesh doublings."""
    return [np.log2(a / b) for a, b in zip(errors[:-1], errors[1:])]


def test_slab_with_imposed_surface_temperature_is_second_order():
    t_wall = 600.0
    errs, span = [], None
    for n in (40, 80, 160):
        x, T, _, _ = _slab(n, bc=(("adiabatic", "isothermal"),
                                  "adiabatic", "adiabatic"), ambient=t_wall)
        exact = _dirichlet_exact(x, t_wall)
        span = np.ptp(exact)
        errs.append(np.max(np.abs(T - exact)))
    # Quoted relative to the temperature SWING, not to an absolute kelvin: the
    # profile spans ~161 K, so a fixed 1e-3 K bar would be a 6e-6 relative
    # demand that only a much finer mesh could meet, and would say nothing
    # about the scheme. The order is the real evidence.
    assert errs[-1] / span < 1e-4
    assert all(1.9 < p < 2.1 for p in _order(errs)), _order(errs)


def test_slab_with_convective_surface_is_second_order():
    """The Robin branch -- the one the HP-MR core boundary actually uses, and
    the one that exercises the ambient source term at a lossy face."""
    alpha, t_inf = 0.05, 400.0
    errs, span = [], None
    for n in (40, 80, 160):
        x, T, _, _ = _slab(n, bc=(("adiabatic", alpha), "adiabatic", "adiabatic"),
                           ambient=t_inf)
        exact = _robin_exact(x, alpha, t_inf)
        span = np.ptp(exact)
        errs.append(np.max(np.abs(T - exact)))
    assert errs[-1] / span < 1e-4
    assert all(1.9 < p < 2.1 for p in _order(errs)), _order(errs)


def test_pure_sink_with_no_conduction_is_exact():
    """h(T - T_sink) = q with an adiabatic box has the algebraic solution
    T = T_sink + q/h in every cell, to round-off -- no discretization involved."""
    grid = Grid(shape=(5, 4, 3), size=(10.0, 8.0, 6.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=H, sink_temperature=T_SINK)
    res = ConductionSolver(grid, mat, bc="adiabatic").solve(np.full(grid.shape, Q))
    np.testing.assert_allclose(res.temperature_numpy, T_SINK + Q / H,
                               rtol=1e-11, atol=1e-9)


def test_uniform_ambient_is_reproduced_exactly():
    """q = 0 and T_sink = T_inf must give T == T_inf everywhere, including on
    excised cells. This is what pins the constant-field boundary identity: any
    error in the ambient source shows up here as a deviation from a constant."""
    grid = Grid(shape=(9, 7, 5), size=(9.0, 7.0, 5.0))
    active = np.ones(grid.shape, dtype=bool)
    active[0, :, :] = False
    active[4, 3, 2] = False                       # an interior void, too
    t_inf = 823.0
    mats = [ThermalMaterial(conductivity=0.4, sink_coeff=H, sink_temperature=t_inf),
            ThermalMaterial(conductivity=1.1, sink_coeff=0.0, sink_temperature=0.0)]
    mmap = np.zeros(grid.shape, dtype=int)
    mmap[3:6] = 1
    res = ConductionSolver(grid, mats, mmap, bc=(0.03, "adiabatic", 0.2),
                           active=active, mask_bc=0.07,
                           ambient_temperature=t_inf).solve(np.zeros(grid.shape))
    np.testing.assert_allclose(res.temperature_numpy, t_inf, rtol=0, atol=1e-9)


def test_energy_balance_closes_on_a_heterogeneous_masked_core():
    grid = Grid(shape=(12, 10, 6), size=(24.0, 20.0, 12.0))
    rng = np.random.default_rng(3)
    active = np.ones(grid.shape, dtype=bool)
    active[0], active[-1], active[:, 0], active[:, -1] = False, False, False, False
    mats = [ThermalMaterial(conductivity=0.25, sink_coeff=H, sink_temperature=T_SINK),
            ThermalMaterial(conductivity=0.9, sink_coeff=0.0, sink_temperature=0.0),
            ThermalMaterial(conductivity=0.2, sink_coeff=0.0, sink_temperature=0.0)]
    mmap = rng.integers(0, 3, size=grid.shape)
    q = np.where(mmap == 0, Q, 0.0)
    res = ConductionSolver(grid, mats, mmap, bc=(0.01, 0.01, "adiabatic"),
                           active=active, mask_bc=0.02,
                           ambient_temperature=450.0).solve(q)
    assert res.balance_residual < 1e-10
    assert res.source_watts == pytest.approx(
        float(np.sum(np.where(active, q, 0.0)) * grid.cell_volume), rel=1e-12)


def test_tri_prism_axial_profile_matches_the_slab_solution():
    """The triangular-prism path against the SAME closed form: with the
    in-plane faces adiabatic the problem is one-dimensional in z, so a correct
    tri-z coupling and z-face Robin closure must reproduce the slab cosh."""
    alpha, t_inf = 0.05, 400.0
    errs, span = [], None
    for nz in (20, 40, 80):
        grid = TriGrid(shape=(4, 4, 2, nz), side=3.0, height=LENGTH)
        mat = ThermalMaterial(conductivity=K, sink_coeff=H, sink_temperature=T_SINK)
        solver = ConductionSolver(grid, mat, bc=("adiabatic", "adiabatic",
                                                 ("adiabatic", alpha)),
                                  mask_bc="adiabatic", ambient_temperature=t_inf)
        res = solver.solve(np.full(grid.shape, Q))
        T = res.temperature_numpy
        # in-plane adiabatic => every column identical
        assert np.ptp(T, axis=(0, 1, 2)).max() < 1e-9
        assert res.balance_residual < 1e-10
        z = (np.arange(nz) + 0.5) * grid.dz
        exact = _robin_exact(z, alpha, t_inf)
        span = np.ptp(exact)
        errs.append(np.max(np.abs(T[0, 0, 0] - exact)))
    assert errs[-1] / span < 1e-4
    assert all(1.9 < p < 2.1 for p in _order(errs)), _order(errs)


def test_hpmr_core_energy_balance_closes():
    """The real body-fitted triangular core, drum arcs volume-mixed: the
    balance identity must hold on the jagged active mask too, where no closed
    form exists."""
    from ndgpu.benchmarks.hpmr import build_hpmr2d

    p = build_hpmr2d(refine=4, drum_angle_deg=90.0, absorber="polar")
    tm = [ThermalMaterial(conductivity=1e-6),                       # void
          ThermalMaterial(0.25, sink_coeff=H, sink_temperature=T_SINK),  # fuel
          ThermalMaterial(0.30), ThermalMaterial(0.90),
          ThermalMaterial(0.90), ThermalMaterial(0.20)]
    q = np.where(p.material_map == 1, Q, 0.0)
    solver = ConductionSolver(p.grid, tm, p.material_map, active=p.active,
                              mask_bc=1e-3, ambient_temperature=400.0,
                              mix_material=p.mix_material,
                              mix_weight=p.mix_weight)
    res = solver.solve(q)
    assert res.balance_residual < 1e-10
    assert res.sink_watts > 0.0 and res.leakage_watts > 0.0
    T = res.temperature_numpy[p.active]
    assert T.max() > T_SINK              # the fuel runs above the heat pipes
    assert T.min() > 400.0               # ...and everything above ambient


def test_cylindrical_volumes_are_true_annuli():
    """On r-z the operator weights rows by the cell RADIUS, not by the annular
    volume -- fine for neutronics, which only forms ratios, but the thermal
    side divides a rated wattage by this volume, so a constant factor off
    scales the whole temperature field."""
    grid = Grid(shape=(24, 1, 12), size=(30.0, 1.0, 20.0), geometry="cylindrical")
    mat = ThermalMaterial(conductivity=K, sink_coeff=H, sink_temperature=T_SINK)
    solver = ConductionSolver(grid, mat, bc=(("adiabatic", 0.05), "adiabatic",
                                             "adiabatic"),
                              ambient_temperature=400.0)
    dr, _, dz = grid.spacing
    rc = np.asarray(grid.cylindrical_metrics()[0])
    np.testing.assert_allclose(np.asarray(solver.cell_volume),
                               2.0 * np.pi * rc * dr * dz, rtol=1e-13)
    # And they sum to the cylinder: sum_i 2 pi r_i dr = pi R^2 exactly for
    # midpoint radii, times the full height.
    total = float(np.sum(np.broadcast_to(np.asarray(solver.cell_volume),
                                         grid.shape)))
    assert total == pytest.approx(np.pi * 30.0**2 * 20.0, rel=1e-12)

    res = solver.solve(np.full(grid.shape, Q))
    assert res.balance_residual < 1e-10


def test_cylindrical_pure_sink_limit_is_exact():
    grid = Grid(shape=(16, 1, 8), size=(30.0, 1.0, 20.0), geometry="cylindrical")
    mat = ThermalMaterial(conductivity=K, sink_coeff=H, sink_temperature=T_SINK)
    res = ConductionSolver(grid, mat, bc="adiabatic").solve(np.full(grid.shape, Q))
    np.testing.assert_allclose(res.temperature_numpy, T_SINK + Q / H,
                               rtol=1e-10, atol=1e-7)


RHO_CP = 3.0        # J/(cm^3 K), graphite-ish


def test_transient_relaxes_to_the_steady_state_exponentially():
    """Lumped limit: no conduction gradient (uniform everything, adiabatic
    box), so every cell obeys rho*cp dT/dt = q - h(T - T_sink) exactly and the
    solution is a single exponential with time constant rho*cp/h. Backward
    Euler is 1st order in dt, so the error must halve when dt does."""
    grid = Grid(shape=(4, 3, 2), size=(4.0, 3.0, 2.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=H, sink_temperature=T_SINK,
                          heat_capacity=RHO_CP)
    tau = RHO_CP / H
    t_inf_val = T_SINK + Q / H              # the steady state it relaxes to
    T0, t_end = 600.0, 100.0

    def exact(t):
        return t_inf_val + (T0 - t_inf_val) * np.exp(-t / tau)

    errs = []
    for dt in (2.0, 1.0, 0.5):
        solver = ConductionSolver(grid, mat, bc="adiabatic", time_step=dt)
        T = np.full(grid.shape, T0)
        for _ in range(int(round(t_end / dt))):
            T = solver.step(np.full(grid.shape, Q), T).temperature_numpy
        errs.append(abs(float(T.mean()) - exact(t_end)))
    order = [np.log2(a / b) for a, b in zip(errs[:-1], errs[1:])]
    assert all(0.85 < p < 1.15 for p in order), (errs, order)


def test_transient_reaches_the_steady_solution_for_large_time():
    """Marched far enough, the transient must land on what the steady solver
    returns for the same problem -- the two share every coefficient, so any
    disagreement is a bug in the capacity term, not physics."""
    grid = Grid(shape=(16, 1, 1), size=(LENGTH, 1.0, 1.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=H, sink_temperature=T_SINK,
                          heat_capacity=RHO_CP)
    bc = (("adiabatic", 0.05), "adiabatic", "adiabatic")
    steady = ConductionSolver(grid, mat, bc=bc, ambient_temperature=400.0)
    ref = steady.solve(np.full(grid.shape, Q)).temperature_numpy

    tr = ConductionSolver(grid, mat, bc=bc, ambient_temperature=400.0, time_step=5.0)
    T = np.full(grid.shape, 500.0)
    for _ in range(4000):                     # 20,000 s >> rho*cp/h = 224 s
        res = tr.step(np.full(grid.shape, Q), T)
        T = res.temperature_numpy
    np.testing.assert_allclose(T, ref, rtol=1e-9, atol=1e-6)
    # Looser than the steady bar (1e-10) on purpose. The transient right-hand
    # side is dominated by rho*cp/dt * T_old -- here ~600x the fission source --
    # so the linear solve's relative tolerance leaves a proportionally larger
    # absolute residual, which the balance then reports against the much
    # smaller source. That is the CG tolerance showing through, not a defect in
    # the identity, and asking below it would just be measuring solver noise.
    assert res.balance_residual < 1e-8        # storage ~ 0 at equilibrium


def test_transient_energy_balance_includes_storage():
    """Mid-transient the books only close if the stored energy is counted."""
    grid = Grid(shape=(10, 8, 4), size=(20.0, 16.0, 8.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=H, sink_temperature=T_SINK,
                          heat_capacity=RHO_CP)
    solver = ConductionSolver(grid, mat, bc=0.01, ambient_temperature=400.0,
                              time_step=1.0)
    T = np.full(grid.shape, 500.0)
    res = solver.step(np.full(grid.shape, Q), T)
    assert res.storage_watts > 0.0            # a cold core is soaking heat up
    assert res.balance_residual < 1e-10
    # Without the storage term the balance would be badly broken -- check the
    # test is actually testing something.
    naive = abs(res.source_watts - res.sink_watts - res.leakage_watts)
    assert naive / abs(res.source_watts) > 0.1


def test_coupling_controls_skip_diagnostics_and_reach_the_linear_solver(monkeypatch):
    """The fast coupled path must not pay for balance reductions or hide the
    residual-check/preconditioner controls before they reach Krylov."""
    grid = Grid(shape=(4, 3, 2), size=(4.0, 3.0, 2.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=H,
                          sink_temperature=T_SINK, heat_capacity=RHO_CP)
    solver = ConductionSolver(grid, mat, bc="adiabatic", time_step=0.5,
                              precond_degree=1)
    captured = {}

    def fake_linsolve(apply, rhs, x0, inv_diag, xp, **kwargs):
        captured.update(kwargs)
        return x0, 3

    monkeypatch.setattr(solver, "_linsolve", fake_linsolve)
    monkeypatch.setattr(
        solver, "energy_balance",
        lambda *a, **k: pytest.fail("diagnostics=False computed a balance"))
    res = solver.step(np.full(grid.shape, Q), np.full(grid.shape, 700.0),
                      rtol=1e-7, check_every=5, diagnostics=False,
                      synchronize_timing=False)
    assert captured["rtol"] == 1e-7
    assert captured["check_every"] == 5
    assert captured["precond"] is solver._precond
    assert res.iterations == 3
    assert np.isnan(res.balance_residual)


def test_thermal_check_cadence_must_be_positive():
    grid = Grid(shape=(2, 1, 1), size=(2.0, 1.0, 1.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=H)
    solver = ConductionSolver(grid, mat, bc="adiabatic")
    with pytest.raises(ValueError, match="check_every"):
        solver.solve(np.full(grid.shape, Q), check_every=0)


def test_transient_needs_a_previous_temperature():
    grid = Grid(shape=(4, 1, 1), size=(4.0, 1.0, 1.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=H, heat_capacity=RHO_CP)
    solver = ConductionSolver(grid, mat, bc="adiabatic", time_step=1.0)
    with pytest.raises(ValueError, match="previous temperature"):
        solver.solve(np.full(grid.shape, Q))


def test_an_adiabatic_transient_with_no_sink_is_well_posed():
    """Steady would be singular -- nowhere for the heat to go -- but the
    transient answer is simply that the core keeps heating, at q/(rho*cp)."""
    grid = Grid(shape=(4, 4, 4), size=(4.0, 4.0, 4.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=0.0, heat_capacity=RHO_CP)
    dt = 0.5
    solver = ConductionSolver(grid, mat, bc="adiabatic", time_step=dt)
    T = np.full(grid.shape, 700.0)
    for _ in range(20):
        T = solver.step(np.full(grid.shape, Q), T).temperature_numpy
    assert float(T.mean()) == pytest.approx(700.0 + Q / RHO_CP * (20 * dt), rel=1e-9)


def test_vacuum_boundary_is_rejected():
    grid = Grid(shape=(4, 1, 1), size=(4.0, 1.0, 1.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=H)
    with pytest.raises(ValueError, match="no thermal meaning"):
        ConductionSolver(grid, mat, bc="vacuum")


def test_singular_problem_is_rejected():
    """No sink and no lossy surface: there is no steady state, and the operator
    is singular. Better a message than a silently divergent CG."""
    grid = Grid(shape=(4, 4, 4), size=(4.0, 4.0, 4.0))
    mat = ThermalMaterial(conductivity=K, sink_coeff=0.0)
    with pytest.raises(ValueError, match="singular"):
        ConductionSolver(grid, mat, bc="adiabatic")
