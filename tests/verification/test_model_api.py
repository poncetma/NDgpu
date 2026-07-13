"""The simplified Model front end.

Model is a thin builder over the structured eigensolver: it must produce exactly
the same eigenvalue as the equivalent low-level call (it only removes the
grid/array bookkeeping), paint regions where asked, and report a neutron balance
that closes. These tests pin all three, plus the dimensionality handling and the
human-readable summary.
"""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, Grid, Material, Model, ONE_GROUP_DEMO,
                   PWR_TWO_GROUP, k_bare_box)

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
