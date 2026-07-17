"""ANL-7416 Problem 8-A1: 2D (r-z) delayed supercritical transient.

The first cylindrical-geometry benchmark in the suite (Benchmark Source
Situation 8 of the Argonne Code Center Benchmark Problem Book, ANL-7416
Suppl. 2, 1977). Problem definition, cross sections and the published
reference values live in ndgpu.benchmarks.anl_bss8.

What "agreement" means here: the three published solutions (TWODTA, TWODQD,
ADEP) all share one spatial discretization -- mesh-point (vertex-centered)
5-point finite differences on the Delta_r = 8 cm x Delta_z = 18.75 cm mesh --
so their 1% mutual spread does not include spatial error. Re-implementing
that scheme on this problem reproduces their k0 to ~1 pcm and their power
trace to ~3%, but its coarse-mesh ramp worth (0.398 $) sits well below the
mesh-converged value (0.4195 $), while the cell-centered FV stencil on the
same coarse mesh overshoots it (0.454 $). A delayed-supercritical power
excursion amplifies a worth difference strongly, so the k-eigenvalue is
validated tightly against the book while the power trace is validated in two
parts: ramp phase within a few %, tail against the published value with the
documented discretization band, plus a tight regression pin on our own
mesh-converged answer.
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, TransientSolver
from ndgpu.benchmarks import build_anl8a1
from ndgpu.benchmarks.anl_bss8 import K_REFERENCE, P_REFERENCE

K_EIGENSOLVE_REFERENCE = 0.867053   # 8-A1-1's unadjusted eigensolve (same mesh)


def test_region_layout_tiles_the_reactor():
    prob = build_anl8a1()
    mmap = prob.material_map[:, 0, :]
    assert mmap.shape == (30, 28)
    # Region 16 is the outer fuel ring: r 200-240, z 75-450 (25 x 20 cells).
    assert np.all(mmap[25:, 4:24] == 15)
    # Full-radius axial reflector bands at both ends (regions 1/2 and 14/15).
    for cols, mat in ((slice(0, 2), 14), (slice(2, 4), 13),
                      (slice(24, 26), 1), (slice(26, 28), 0)):
        assert np.all(mmap[:, cols] == mat)
    # Perturbed regions: 3 and 7 innermost (r < 40), 11 wide (r < 120).
    assert np.all(mmap[:5, 18:24] == 2)     # region 3
    assert np.all(mmap[:5, 10:18] == 6)     # region 7
    assert np.all(mmap[:15, 4:10] == 10)    # region 11


def test_initial_keff_matches_book():
    # Benchmark mesh (30 x 28): the book's unadjusted eigensolve is 0.867053;
    # cell-centered FV lands 75 pcm above it.
    prob = build_anl8a1(perturbed=False)
    mats, mmap = prob.problem_at(0.0)
    res = DiffusionEigenSolver(prob.grid, mats, mmap, bc=prob.bc,
                               device="cpu").solve(tol_k=1e-8, tol_source=1e-7)
    assert res.converged
    assert res.k_eff == pytest.approx(K_EIGENSOLVE_REFERENCE, abs=1.5e-3)

    # Mesh-refined (120 x 112): converges to within ~40 pcm of the published
    # k's (K_REFERENCE 0.866901 / ADEP 0.866861), i.e. the FV and mesh-point
    # discretizations agree in the limit.
    prob4 = build_anl8a1(refine=4, perturbed=False)
    mats, mmap = prob4.problem_at(0.0)
    res4 = DiffusionEigenSolver(prob4.grid, mats, mmap, bc=prob4.bc,
                                device="cpu").solve(tol_k=1e-8, tol_source=1e-7)
    assert res4.k_eff == pytest.approx(K_REFERENCE, abs=5e-4)
    # and refinement moves k toward the reference
    assert abs(res4.k_eff - K_REFERENCE) < abs(res.k_eff - K_REFERENCE)


def test_transient_exhibit_a():
    # 2x the benchmark mesh, dt = 0.02 s (backward Euler is dt-converged at
    # this step: dt = 0.005 changes the trace by < 0.2%).
    prob = build_anl8a1(refine=2)
    solver = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                             bc=prob.bc, device="cpu")
    res = solver.solve(t_end=4.0, dt=0.02)
    assert res.k0 == pytest.approx(K_REFERENCE, abs=1e-3)

    p = {t: float(res.power[np.searchsorted(res.times, t)]) for t in P_REFERENCE}

    # Ramp phase: the excursion has not yet amplified the worth difference.
    for t in (0.2, 0.4, 0.6):
        assert p[t] == pytest.approx(P_REFERENCE[t], rel=0.03), (t, p[t])

    # Full trace against Exhibit A within the documented discretization band:
    # the published tail carries the coarse vertex-scheme worth (-5% vs
    # converged), our refine=2 mesh carries +2%; both compound over 4 s.
    for t, ref in P_REFERENCE.items():
        assert p[t] == pytest.approx(ref, rel=0.13), (t, p[t], ref)
    # The excursion must sit above the published trace (coarse-mesh vertex
    # worth is below the converged worth), not below it.
    assert p[4.0] > P_REFERENCE[4.0]

    # Physics shape: monotone delayed-supercritical rise, ramp kink at t = 1,
    # asymptotic-ish period consistent with the frozen worth.
    assert np.all(np.diff(res.power) > 0)
    growth = np.diff(np.log(res.power)) / np.diff(res.times)
    i1 = np.searchsorted(res.times, 1.0)
    assert growth[:i1 - 1].max() > 2.0 * growth[-1]  # ramp >> settled growth
    omega_tail = float(np.log(p[4.0] / p[3.0]))
    assert 0.10 < omega_tail < 0.20

    # Regression pin on this solver's own answer at this mesh/dt, so any
    # future change in worth or kinetics shows up hard. (Mesh-converged:
    # 2.905 at refine=4, 2.887 at refine=8; see the benchmark docstring.)
    assert p[4.0] == pytest.approx(2.979, rel=0.01)


@pytest.mark.parametrize("name", ["bicgstab", "gmres"])
def test_keff_via_divergence_form(name):
    # The same benchmark solved *without* the volume-weighted SPD trick: the
    # natural divergence-form r-z stencil is non-symmetric, so CG is out
    # (the solver refuses it) and GMRES/BiCGStab carry the solve. Same
    # discrete equations, so k must match the weighted-CG answer to solver
    # noise, and the book value to the same tolerance as the SPD path.
    prob = build_anl8a1(perturbed=False)
    mats, mmap = prob.problem_at(0.0)
    with pytest.raises(ValueError, match="gmres.*bicgstab"):
        DiffusionEigenSolver(prob.grid, mats, mmap, bc=prob.bc, device="cpu",
                             symmetric_operator=False)
    k_cg = DiffusionEigenSolver(prob.grid, mats, mmap, bc=prob.bc, device="cpu"
                                ).solve(tol_k=1e-8, tol_source=1e-7).k_eff
    res = DiffusionEigenSolver(prob.grid, mats, mmap, bc=prob.bc, device="cpu",
                               symmetric_operator=False, linear_solver=name
                               ).solve(tol_k=1e-8, tol_source=1e-7)
    assert res.converged
    assert res.k_eff == pytest.approx(k_cg, abs=1e-7)
    assert res.k_eff == pytest.approx(K_EIGENSOLVE_REFERENCE, abs=1.5e-3)


def test_transient_via_divergence_form():
    # Full 4 s excursion on the benchmark mesh through the non-symmetric
    # path (BiCGStab; GMRES is covered by the eigensolve above). The two
    # operator forms are row-rescalings of the same equations, so the traces
    # agree to the cross-solver Krylov noise floor (~1e-4 over 200 steps).
    prob = build_anl8a1()
    traces = {}
    for kw in (dict(),
               dict(symmetric_operator=False, linear_solver="bicgstab")):
        s = TransientSolver(prob.grid, prob.problem_at, prob.kinetics,
                            bc=prob.bc, device="cpu", **kw)
        res = s.solve(t_end=4.0, dt=0.02)
        assert res.k0 == pytest.approx(K_REFERENCE, abs=1.5e-3)
        traces[bool(kw)] = res.power
    assert np.allclose(traces[True], traces[False], rtol=3e-4)
