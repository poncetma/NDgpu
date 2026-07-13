"""The simplified Model front end.

Model is a thin builder over the structured eigensolver: it must produce exactly
the same eigenvalue as the equivalent low-level call (it only removes the
grid/array bookkeeping), paint regions where asked, and report a neutron balance
that closes. These tests pin all three, plus the dimensionality handling and the
human-readable summary.
"""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, Grid, HexLattice, Material, MeshModel,
                   Model, ONE_GROUP_DEMO, PWR_TWO_GROUP, k_bare_box)
from ndgpu.mesh import UnstructuredDiffusionSolver, assemble_mesh_3d
from ndgpu.hexraster import rasterize_hex_sites
from ndgpu.tri import TriDiffusionEigenSolver, TriGrid

_REFLECTOR = Material(name="reflector", diffusion=[1.13, 0.16], sigma_a=[0.0004, 0.0197],
                      nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.0494], [0.0, 0.0]])
_FUEL = Material(name="fuel", diffusion=[1.2627, 0.3543], sigma_a=[0.01207, 0.1210],
                 nu_sigma_f=[0.008476, 0.18514], sigma_s=[[0.0, 0.02619], [0.0, 0.0]],
                 chi=[1.0, 0.0])
TOL = dict(tol_k=1e-8, tol_source=1e-7)


def test_model_reproduces_low_level_solver_homogeneous():
    # The wrapper must not change the physics: same grid, material and BC give
    # the identical eigenvalue as building the solver by hand.
    size, cells = (90.0, 90.0, 90.0), (30, 30, 30)
    k_model = Model(size=size, cells=cells).fill(_FUEL).set_boundary("vacuum").run(**TOL).k_eff
    k_low = DiffusionEigenSolver(Grid(shape=cells, size=size), _FUEL, bc="vacuum",
                                 device="cpu").solve(**TOL).k_eff
    assert k_model == pytest.approx(k_low, rel=0, abs=1e-9)


def test_model_reproduces_low_level_solver_heterogeneous():
    # Central fuel block in a reflector, built with add_box, must match the same
    # problem assembled from an explicit material_map.
    size, cells = (120.0, 120.0, 20.0), (24, 24, 4)
    model = (Model(size=size, cells=cells).fill(_REFLECTOR)
             .add_box(_FUEL, x=(30, 90), y=(30, 90)).set_boundary("vacuum"))
    k_model = model.run(**TOL).k_eff

    grid = Grid(shape=cells, size=size)
    xc, yc = grid.cell_centers(0), grid.cell_centers(1)
    mmap = np.zeros(cells, dtype=np.int64)
    sel = ((xc >= 30) & (xc <= 90))[:, None, None] & ((yc >= 30) & (yc <= 90))[None, :, None]
    mmap[np.broadcast_to(sel, cells)] = 1
    k_low = DiffusionEigenSolver(grid, [_REFLECTOR, _FUEL], material_map=mmap,
                                 bc="vacuum", device="cpu").solve(**TOL).k_eff
    assert k_model == pytest.approx(k_low, rel=0, abs=1e-9)


def test_add_box_paints_the_right_cells():
    model = Model(size=(100.0, 100.0), cells=(10, 10)).fill(_REFLECTOR)
    model.add_box(_FUEL, x=(50, 100))                 # right half, all y
    mmap = model.material_map                          # (10, 10), squeezed to 2D
    assert mmap.shape == (10, 10)
    assert np.all(mmap[:5, :] == 0)                    # left half is background
    assert np.all(mmap[5:, :] == 1)                    # right half is fuel
    assert model.materials[1] is _FUEL


def test_zero_flux_bare_cube_converges_to_analytic():
    # With the Dirichlet (zero-flux) boundary the bare cube matches the exact
    # analytic k_bare_box, so the wrapper's geometry/BC plumbing is correct.
    size = (90.0, 90.0, 90.0)
    k = Model(size=size, cells=(32, 32, 32)).fill(_FUEL).set_boundary("zero-flux").run(**TOL).k_eff
    assert k == pytest.approx(k_bare_box(_FUEL, size), abs=2e-4)


def test_neutron_balance_and_material_report_are_consistent():
    model = (Model(size=(120.0, 120.0, 120.0), cells=(24, 24, 24)).fill(_REFLECTOR)
             .add_box(_FUEL, x=(30, 90), y=(30, 90), z=(30, 90)).set_boundary("vacuum"))
    res = model.run(**TOL)
    # absorbed + leaked account for every produced neutron
    assert res.absorbed_fraction + res.leakage_fraction == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < res.leakage_fraction < 1.0
    # only the fuel fissions: it should carry ~100% of the fission rate
    rows = {name: fisf for name, _v, _f, fisf in res._material_stats}
    assert rows["fuel"] == pytest.approx(1.0, abs=1e-6)
    assert rows["reflector"] == pytest.approx(0.0, abs=1e-6)


def test_dimensionality_and_flux_shape():
    r1 = Model(size=(100.0,), cells=(40,)).fill(ONE_GROUP_DEMO).run(**TOL)
    assert r1.flux.shape == (1, 40, 1, 1) and r1.model.ndim == 1
    r2 = Model(size=(80.0, 80.0), cells=(20, 20)).fill(_FUEL).run(**TOL)
    assert r2.flux.shape == (2, 20, 20, 1) and r2.model.ndim == 2


def test_summary_is_human_readable():
    res = Model(size=(90.0, 90.0, 90.0), cells=(16, 16, 16)).fill(_FUEL).run(**TOL)
    text = res.summary()
    assert str(res) == text
    for token in ("k_eff", "reactivity", "pcm", "absorbed", "leaked", "peaking"):
        assert token in text
    assert f"{res.k_eff:.6f}" in text


def test_helpful_errors():
    with pytest.raises(ValueError):
        Model(size=(10.0, 10.0), cells=(4,))          # length mismatch
    with pytest.raises(ValueError):
        Model(size=(10.0,), cells=(4,)).run()         # empty (no fill)
    with pytest.raises(ValueError):
        Model(size=(10.0,), cells=(4,)).fill(_FUEL).add_box(_REFLECTOR, y=(0, 5))  # no y axis
    with pytest.raises(ValueError):
        (Model(size=(10.0,), cells=(4,)).fill(ONE_GROUP_DEMO)  # mixed group counts
         .add_box(_FUEL, x=(0, 5)).run())


# --- structured adjoint --------------------------------------------------
def test_adjoint_run_matches_forward_eigenvalue():
    build = lambda: Model(size=(90.0, 90.0, 90.0), cells=(16, 16, 16)).fill(_FUEL).set_boundary("vacuum")
    fwd = build().run(**TOL)
    adj = build().run(adjoint=True, **TOL)
    assert adj.adjoint and not fwd.adjoint
    assert adj.k_eff == pytest.approx(fwd.k_eff, abs=1e-6)
    text = adj.summary()
    assert "adjoint" in text and "importance" in text
    assert "where the fission neutrons go" not in text   # physical balance suppressed


# --- unstructured MeshModel ----------------------------------------------
def _hex_mesh(n, L):
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


def test_meshmodel_reproduces_low_level_solver():
    # The mesh wrapper only wires up the solver; it must give the identical k as
    # calling UnstructuredDiffusionSolver by hand on the same painted mesh.
    mesh = _hex_mesh(12, 120.0)
    mm = (MeshModel(mesh).fill(_REFLECTOR)
          .add_box(_FUEL, x=(30, 90), y=(30, 90), z=(30, 90)).set_boundary("vacuum"))
    k_model = mm.run(**TOL).k_eff

    cm = np.zeros(mesh.n_cells, np.int64)
    c = mesh.centroid
    sel = ((c[:, 0] >= 30) & (c[:, 0] <= 90) & (c[:, 1] >= 30) & (c[:, 1] <= 90)
           & (c[:, 2] >= 30) & (c[:, 2] <= 90))
    cm[sel] = 1
    k_low = UnstructuredDiffusionSolver(mesh, [_REFLECTOR, _FUEL], cm,
                                        alpha_boundary=0.5).solve(**TOL).k_eff
    assert k_model == pytest.approx(k_low, rel=0, abs=1e-9)


def test_meshmodel_assign_by_tag_and_balance():
    # Tag half the cells; assign() by tag must paint exactly those, and the
    # volume-weighted balance and per-material report must be consistent.
    dx = 10.0
    nid, coords, cells, tags = {}, [], [], []

    def gid(i, j, k):
        if (i, j, k) not in nid:
            nid[(i, j, k)] = len(coords); coords.append((i * dx, j * dx, k * dx))
        return nid[(i, j, k)]

    for i in range(8):
        for j in range(4):
            for k in range(4):
                cells.append((gid(i, j, k), gid(i + 1, j, k), gid(i + 1, j + 1, k), gid(i, j + 1, k),
                              gid(i, j, k + 1), gid(i + 1, j, k + 1), gid(i + 1, j + 1, k + 1), gid(i, j + 1, k + 1)))
                tags.append(1 if i < 4 else 2)          # left half tag 1, right half tag 2
    mesh = assemble_mesh_3d(coords, cells, tags)
    mm = MeshModel(mesh).fill(_REFLECTOR).assign(_FUEL, tag=1).set_boundary("vacuum")
    res = mm.run(**TOL)
    assert res.absorbed_fraction + res.leakage_fraction == pytest.approx(1.0, abs=1e-9)
    rows = {name: (volf, fisf) for name, volf, _f, fisf in res._material_stats}
    assert rows["fuel"][0] == pytest.approx(0.5, abs=1e-9)       # tag 1 is half the cells
    assert rows["fuel"][1] == pytest.approx(1.0, abs=1e-6)       # only the fuel fissions


# --- triangular HexLattice -----------------------------------------------
def test_hexlattice_reproduces_low_level_tri_solver():
    lat = HexLattice(pitch=20.0, refine=3).set_boundary("vacuum")
    lat.set_site((0, 0), _FUEL)
    for rc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        lat.set_site(rc, _REFLECTOR)
    k_model = lat.run(method="diffusion", **TOL).k_eff

    # rebuild the same raster/solver by hand: ids 1=fuel, 2=reflector, 0=void
    void = Material(name="void", diffusion=[1.0, 1.0], sigma_a=[0.0, 0.0], nu_sigma_f=[0.0, 0.0])
    site_mat = {(0, 0): 1, (1, 0): 2, (-1, 0): 2, (0, 1): 2, (0, -1): 2}
    raster = rasterize_hex_sites(site_mat, 20.0, 3)
    grid = TriGrid(shape=raster.material_map.shape, side=raster.side)
    k_low = TriDiffusionEigenSolver(grid, [void, _FUEL, _REFLECTOR], raster.material_map,
                                    active=raster.material_map > 0,
                                    mask_bc="vacuum").solve(**TOL).k_eff
    assert k_model == pytest.approx(k_low, rel=0, abs=1e-9)


def test_hexlattice_sp3_differs_from_diffusion():
    lat = HexLattice(pitch=18.0, refine=3).set_boundary("vacuum").set_site((0, 0), _FUEL)
    for rc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        lat.set_site(rc, _REFLECTOR)
    kd = lat.run(method="diffusion", **TOL).k_eff
    ks = lat.run(method="sp3", **TOL).k_eff
    assert abs(ks - kd) * 1e5 > 20.0                    # SP3 is a distinct transport solution


# --- transients ----------------------------------------------------------
from ndgpu import Kinetics                                  # noqa: E402
from ndgpu.transient import TransientSolver                 # noqa: E402

_KIN = dict(velocities=[1.0e7, 3.0e5], beta=[0.0065], decay=[0.08])


def _transient_core():
    return (Model(size=(200.0, 200.0, 200.0), cells=(10, 10, 10))
            .fill(_FUEL).set_boundary("vacuum").set_kinetics(**_KIN))


def test_transient_unperturbed_stays_at_equilibrium():
    # The steady eigenvalue k0 normalises the fission source, so with no change
    # the reactor sits exactly at P/P0 = 1 for the whole transient.
    res = _transient_core().transient(t_end=1.0, dt=0.05)
    assert np.allclose(res.power, 1.0, atol=1e-6)
    assert res.k0 == pytest.approx(res.steady.k_eff, abs=1e-12)   # steady exposed on result


def test_transient_reproduces_low_level_solver():
    # A thermal-absorption step; Model.transient must give the same power trace
    # as driving TransientSolver directly with the equivalent problem_at.
    hot = Material(name="fuel", diffusion=_FUEL.diffusion, sigma_a=[0.01207, 0.1240],
                   nu_sigma_f=_FUEL.nu_sigma_f, sigma_s=_FUEL.sigma_s, chi=_FUEL.chi)
    mats_at = lambda t: [_FUEL if t <= 0 else hot]
    res = _transient_core().transient(t_end=0.6, dt=0.02, materials_at=mats_at)

    grid = Grid(shape=(10, 10, 10), size=(200.0, 200.0, 200.0))
    kin = Kinetics(**_KIN)
    tres = TransientSolver(grid, lambda t: ([_FUEL if t <= 0 else hot], None), kin,
                           bc="vacuum", device="cpu").solve(t_end=0.6, dt=0.02)
    assert np.allclose(res.power, tres.power, rtol=1e-6, atol=1e-9)


def test_transient_rod_insertion_lowers_power_positive_raises_it():
    def step(dsig):                                          # change thermal absorption
        pert = Material(name="fuel", diffusion=_FUEL.diffusion,
                        sigma_a=[0.01207, 0.1210 + dsig], nu_sigma_f=_FUEL.nu_sigma_f,
                        sigma_s=_FUEL.sigma_s, chi=_FUEL.chi)
        return lambda t: [_FUEL if t <= 0 else pert]
    down = _transient_core().transient(t_end=1.0, dt=0.02, materials_at=step(+0.0020))
    up = _transient_core().transient(t_end=1.0, dt=0.02, materials_at=step(-0.0003))
    assert down.final_power < 0.95 and down.power.min() > 0.0    # rod in -> falls, stays physical
    assert up.final_power > 1.05                                 # positive rho -> rises


def test_transient_requires_kinetics_and_reports():
    with pytest.raises(ValueError):
        Model(size=(50.0,), cells=(5,)).fill(_FUEL).transient(t_end=0.1, dt=0.05)
    res = _transient_core().transient(t_end=0.2, dt=0.05)
    text = res.summary()
    for token in ("transient", "k0", "beta", "power P/P0", "steady"):
        assert token in text
