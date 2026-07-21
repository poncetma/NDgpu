"""OECD/NEA C5G7-TD (2D time-dependent C5G7) with diffusion kinetics.

Checks the case construction (bank splitting, perturbation laws, consistency
with the steady C5G7 builder at t = 0), the transient physics of the rod
exercises (prompt drop bounded by the rodded static eigenvalue, spec-mandated
worth ordering of the TD1 cases), and the TD3 moderator-density cases.
Runtime is kept down with cells_per_pin=1 and short horizons; the full-length
traces against the FEMFFUSION SP1 reference live in the example/benchmark
scripts, not here.
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, TransientSolver
from ndgpu.benchmarks import build_c5g7_2d, build_c5g7_td
from ndgpu.benchmarks.c5g7_td import CASES, _rod_fraction, _water_factor


def test_case_table_and_perturbation_laws():
    assert len(CASES) == 17                       # 5 + 5 + 3 + 4
    # TD0: 10% step, half out at 1 s, out after 2 s.
    assert _rod_fraction(0, 0.0) == 0.0
    assert _rod_fraction(0, 0.5) == 0.10
    assert _rod_fraction(0, 1.5) == 0.05
    assert _rod_fraction(0, 2.5) == 0.0
    # TD1/TD2 ramps peak at 1%/10% at t = 1 s and vanish at 2 s.
    assert _rod_fraction(1, 1.0) == pytest.approx(0.01)
    assert _rod_fraction(2, 1.0) == pytest.approx(0.10)
    assert _rod_fraction(1, 1.5) == pytest.approx(0.005)
    assert _rod_fraction(2, 3.0) == 0.0
    # TD3 density dip hits omega at 1 s and recovers by 2 s.
    assert _water_factor(0.8, 1.0) == pytest.approx(0.8)
    assert _water_factor(0.8, 0.5) == pytest.approx(0.9)
    assert _water_factor(0.8, 2.0) == 1.0


def test_geometry_bank_split():
    prob = build_c5g7_td("TD0-5", cells_per_pin=1)
    counts = np.bincount(prob.pin_map.ravel(), minlength=11)
    assert all(counts[3 + b] == 24 for b in range(1, 5))   # 24 GTs per bank
    assert counts[8] == 4                                  # fission chambers
    assert counts[:4].sum() == 4 * 264                     # fuel pins
    assert counts[9] == 0                                  # core water: hom map
    assert counts[10] == 51 * 51 - 4 * 289                 # reflector
    # rodding bank 1 must change exactly the bank-1 guide-tube material
    m0, _ = prob.problem_at(0.0)
    prob1 = build_c5g7_td("TD0-1", cells_per_pin=1)
    m1, _ = prob1.problem_at(0.5)
    changed = [k for k in range(11)
               if not np.allclose(m0[k].sigma_a, m1[k].sigma_a)]
    assert changed == [4]


def test_t0_matches_steady_c5g7_builder():
    """At t=0 the TD problem is exactly the steady C5G7 problem (same XS,
    finer material split), so the eigenvalues must coincide."""
    kw = dict(tol_k=1e-8, tol_source=1e-7)
    td = build_c5g7_td("TD1-1", cells_per_pin=1)
    mats, mmap = td.problem_at(0.0)
    k_td = DiffusionEigenSolver(td.grid, mats, mmap, bc=td.bc,
                                device="cpu").solve(**kw).k_eff
    st = build_c5g7_2d(cells_per_pin=1)
    k_st = DiffusionEigenSolver(st.grid, st.materials, st.material_map,
                                bc=st.bc, device="cpu").solve(**kw).k_eff
    assert k_td == pytest.approx(k_st, abs=1e-9), (k_td, k_st)


def test_td0_1_prompt_drop_bracketed_by_static_worth():
    """The 10% step insertion of bank 1: after a few prompt-adjustment steps
    the power sits between the asymptotic prompt-jump estimate and 1, and the
    rodded static configuration reproduces the inserted worth."""
    prob = build_c5g7_td("TD0-1", cells_per_pin=1)
    mats1, mmap = prob.problem_at(0.5)
    k1 = DiffusionEigenSolver(prob.grid, mats1, mmap, bc=prob.bc,
                              device="cpu").solve(tol_k=1e-8,
                                                  tol_source=1e-7).k_eff
    res = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                          bc=prob.bc, device="cpu").solve(t_end=0.2, dt=0.02)
    rho = (k1 - res.k0) / (k1 * res.k0)      # static worth, negative
    assert rho < -0.02                        # a strong (multi-$) insertion
    # prompt-jump approximation with beta_eff in the 0.003-0.007 span of the
    # fuels: P ~ beta/(beta - rho)
    lo, hi = (b / (b - rho) for b in (0.003, 0.007))
    p = res.power[-1]
    assert 0.8 * lo < p < 1.5 * hi, (p, lo, hi)
    assert np.all(np.diff(res.power) <= 1e-12)   # monotone decay, no bounce


def test_td1_worth_ordering_at_peak_insertion():
    """Spec Sec 2.2.2: increasing maximum inserted reactivity orders the TD1
    cases TD1-3 < TD1-2 < TD1-1 < TD1-4 < TD1-5, so the power at the 1 s peak
    orders the opposite way."""
    p_at_1s = {}
    for case in ("TD1-1", "TD1-2", "TD1-3", "TD1-4", "TD1-5"):
        prob = build_c5g7_td(case, cells_per_pin=1)
        res = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                              bc=prob.bc, device="cpu").solve(t_end=1.0, dt=0.1)
        p_at_1s[case] = res.power[-1]
    p = p_at_1s
    assert p["TD1-3"] > p["TD1-2"] > p["TD1-1"] > p["TD1-4"] > p["TD1-5"], p


def test_td3_density_dip_and_recovery():
    """TD3-4 (omega = 0.80): power dips while the moderator thins (less
    moderation -> less thermal fission), then recovers toward but below 1
    after the density is restored (delayed-precursor deficit)."""
    prob = build_c5g7_td("TD3-4", cells_per_pin=1)
    res = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                          bc=prob.bc, device="cpu").solve(t_end=3.0, dt=0.1)
    p_at = lambda t: res.power[np.searchsorted(res.times, t)]
    assert p_at(1.0) < 0.9                    # significant dip at peak
    assert p_at(1.0) == res.power.min() or p_at(1.1) == res.power.min()
    assert p_at(3.0) > p_at(1.0) + 0.05       # recovery after 2 s
    assert p_at(3.0) < 1.0
