"""Per-material kinetics and per-family delayed spectra in TransientSolver.

Consistency checks: per-material tables with identical rows must reproduce the
global-kinetics solution exactly (same math, field instead of scalar), the
unperturbed state must stay an exact equilibrium even when the delayed
spectrum differs from the material (cumulative) spectrum, and the mix-array
(pin-resolved) path must hold equilibrium too.
"""

import numpy as np
import pytest

from ndgpu import (Grid, Kinetics, Material, TransientSDP1Solver,
                   TransientSolver)

# Two-region, two-group toy core (TWIGL-flavoured numbers).
_SEED = dict(diffusion=[1.4, 0.4], sigma_a=[0.010, 0.150],
             nu_sigma_f=[0.007, 0.200], sigma_s=[[0.0, 0.01], [0.0, 0.0]],
             chi=[1.0, 0.0])
_BLANKET = dict(diffusion=[1.3, 0.5], sigma_a=[0.008, 0.050],
                nu_sigma_f=[0.003, 0.060], sigma_s=[[0.0, 0.01], [0.0, 0.0]],
                chi=[1.0, 0.0])
GRID = Grid(shape=(10, 10, 1), size=(80.0, 80.0, 8.0))
V, LAM = [1.0e7, 2.0e5], [0.08, 0.4]
BETA = [0.005, 0.0025]


def _two_region():
    mats = [Material(name="seed", **_SEED), Material(name="blanket", **_BLANKET)]
    mmap = np.zeros(GRID.shape, dtype=np.int64)
    mmap[4:, :, :] = 1
    pert = [Material(name="seed*", **{**_SEED, "sigma_a": [0.010, 0.1495]}),
            mats[1]]
    problem_at = lambda t: ((mats if t <= 0 else pert), mmap)
    return mats, mmap, problem_at


def test_uniform_per_material_rows_match_global_kinetics():
    """Identical rows for every material = global data; traces must agree to
    the arithmetic noise of field-vs-scalar evaluation."""
    _, _, problem_at = _two_region()
    kin_g = Kinetics(velocities=V, beta=BETA, decay=LAM)
    kin_m = Kinetics(velocities=np.tile(V, (2, 1)), beta=np.tile(BETA, (2, 1)),
                     decay=LAM)
    kw = dict(t_end=0.1, dt=0.01)
    p_g = TransientSolver(GRID, problem_at, kin_g, device="cpu").solve(**kw).power
    p_m = TransientSolver(GRID, problem_at, kin_m, device="cpu").solve(**kw).power
    assert p_g[-1] > 1.01  # the step actually inserted reactivity
    assert np.allclose(p_g, p_m, rtol=0, atol=1e-9), np.max(np.abs(p_g - p_m))


def test_per_family_chi_delayed_identical_rows_match_single_spectrum():
    _, _, problem_at = _two_region()
    chi_d = [0.9, 0.1]
    kin_1 = Kinetics(velocities=V, beta=BETA, decay=LAM, chi_delayed=chi_d)
    kin_2 = Kinetics(velocities=V, beta=BETA, decay=LAM,
                     chi_delayed=np.tile(chi_d, (2, 1)))
    kw = dict(t_end=0.1, dt=0.01)
    p_1 = TransientSolver(GRID, problem_at, kin_1, device="cpu").solve(**kw).power
    p_2 = TransientSolver(GRID, problem_at, kin_2, device="cpu").solve(**kw).power
    assert np.allclose(p_1, p_2, rtol=0, atol=1e-9), np.max(np.abs(p_1 - p_2))


def test_equilibrium_exact_for_distinct_delayed_spectrum():
    """chi_delayed != chi: the cumulative-spectrum treatment (w_fis,g =
    chi_g - sum_i chi_d,ig beta_i/(1+lam_i dt)) keeps the unperturbed state an
    exact equilibrium; the naive chi_g[(1-beta)+omega] lumping drifts by
    ~beta * (delayed thermal share) on the first step."""
    mats, mmap, _ = _two_region()
    static = lambda t: (mats, mmap)
    kin = Kinetics(velocities=np.tile(V, (2, 1)),
                   beta=[[0.005, 0.0025], [0.006, 0.002]], decay=LAM,
                   chi_delayed=[[0.95, 0.05], [0.85, 0.15]])
    res = TransientSolver(GRID, static, kin, device="cpu").solve(t_end=0.1, dt=0.02)
    assert np.allclose(res.power, 1.0, atol=1e-6), res.power


def test_equilibrium_with_mix_arrays():
    """mix_material/mix_weight threading: a partially-covered absorber cell
    pattern (as in pin-resolved geometry) must also hold the steady state."""
    mats, mmap, _ = _two_region()
    mix_material = np.full(GRID.shape, -1, dtype=np.int64)
    mix_weight = np.zeros(GRID.shape)
    mix_material[3:5, 3:5, :] = 1      # blend some seed cells with blanket
    mix_weight[3:5, 3:5, :] = 0.37
    static = lambda t: (mats, mmap)
    kin = Kinetics(velocities=np.tile(V, (2, 1)),
                   beta=[[0.005, 0.0025], [0.006, 0.002]], decay=LAM,
                   chi_delayed=[[0.95, 0.05], [0.85, 0.15]])
    res = TransientSolver(GRID, static, kin, device="cpu",
                          mix_material=mix_material,
                          mix_weight=mix_weight).solve(t_end=0.1, dt=0.02)
    assert np.allclose(res.power, 1.0, atol=1e-6), res.power


def test_per_material_kinetics_follow_a_moving_material_map():
    """A map change must remap 1/v and beta, not just the cross sections.

    The two materials have identical diffusion data, so before the fix the
    moving-map result was bit-identical to the static one: only the stale
    kinetic fields could distinguish the cases.
    """
    grid = Grid(shape=(8, 1, 1), size=(80.0, 1.0, 1.0))
    base = Material(diffusion=[1.3], sigma_a=[0.03], nu_sigma_f=[0.04])
    pert = Material(diffusion=[1.3], sigma_a=[0.029], nu_sigma_f=[0.04])
    mats0, mats1 = [base, base], [pert, pert]
    mmap0 = np.zeros(grid.shape, dtype=np.int64)
    mmap1 = mmap0.copy()
    mmap1[:2] = 1
    kin = Kinetics(velocities=[[1.0e7], [1.0e5]],
                   beta=[[0.0065], [0.0065]], decay=[0.08])

    static = lambda t: (mats0, mmap0) if t <= 0.1 else (mats1, mmap0)
    moving = lambda t: (mats0, mmap0) if t <= 0.1 else (mats1, mmap1)
    kw = dict(t_end=0.2, dt=0.02)
    p_static = TransientSolver(grid, static, kin, device="cpu").solve(**kw).power
    p_moving = TransientSolver(grid, moving, kin, device="cpu").solve(**kw).power
    assert abs(p_moving[-1] - p_static[-1]) > 1e-6


def test_per_material_kinetics_accept_static_blend_from_problem_at():
    """A four-element, but static, problem specification is valid input."""
    mats, mmap, _ = _two_region()
    mix_material = np.full(GRID.shape, -1, dtype=np.int64)
    mix_weight = np.zeros(GRID.shape)
    kin = Kinetics(velocities=np.tile(V, (2, 1)), beta=np.tile(BETA, (2, 1)),
                   decay=LAM)
    res = TransientSolver(
        GRID, lambda _t: (mats, mmap, mix_material, mix_weight), kin,
        device="cpu").solve(t_end=0.1, dt=0.02)
    assert np.allclose(res.power, 1.0, atol=1e-6)


def test_sdpn_solvers_reject_per_material_kinetics():
    _, _, problem_at = _two_region()
    kin = Kinetics(velocities=np.tile(V, (2, 1)), beta=np.tile(BETA, (2, 1)),
                   decay=LAM)
    with pytest.raises(NotImplementedError):
        TransientSDP1Solver(GRID, problem_at, kin, device="cpu")
