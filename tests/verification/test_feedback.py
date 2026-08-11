"""Temperature feedback: the hook is inert when off, and the physics has the
right sign, magnitude and structure when on.

The single most likely way to get a feedback law wrong is a sign, and the
second is to scale ``removal`` instead of adding to absorption -- which is
invisible in k but wrong in the spectrum. Both are pinned here.
"""

import numpy as np
import pytest

from ndgpu import DiffusionEigenSolver, Grid, Material
from ndgpu.feedback import (ThermalFeedback, calibrate, measure_pcm_per_K,
                            scale_to, uniform)
from ndgpu.solver import Fields
from ndgpu.backend import get_backend

T_REF = 800.0


def _materials():
    fuel = Material(name="fuel", diffusion=[1.26, 0.354], sigma_a=[0.0121, 0.121],
                    nu_sigma_f=[0.0085, 0.185], sigma_s=[[0.0, 0.0262], [0.0, 0.0]],
                    chi=[1.0, 0.0])
    refl = Material(name="refl", diffusion=[1.15, 0.90], sigma_a=[2e-4, 5e-3],
                    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.045], [0.0, 0.0]])
    return [refl, fuel]


def _problem(n=16):
    grid = Grid(shape=(n, n, 1), size=(120.0, 120.0, 1.0))
    mmap = np.zeros(grid.shape, dtype=int)
    mmap[2:-2, 2:-2] = 1
    return grid, mmap


def _feedback(doppler=1e-3, expansion=None):
    """Fuel-only response: material 0 is the reflector and must not react."""
    return ThermalFeedback(t_ref=[T_REF, T_REF],
                           doppler=[0.0, doppler],
                           expansion=None if expansion is None else [0.0, expansion])


def _factory(grid, mmap):
    def build(xs_update):
        return DiffusionEigenSolver(grid, _materials(), mmap, bc="vacuum",
                                    device="cpu", xs_update=xs_update)
    return build


def _solve(grid, mmap, xs_update=None):
    return _factory(grid, mmap)(xs_update).solve(tol_k=1e-10, tol_source=1e-9)


# -- the hook is inert when it should be ---------------------------------------

def test_no_hook_and_zero_feedback_are_bit_identical():
    grid, mmap = _problem()
    xp = get_backend("cpu")
    plain = Fields(xp, grid, _materials(), mmap, np.float64)
    zero = Fields(xp, grid, _materials(), mmap, np.float64,
                  xs_update=uniform(_feedback(doppler=0.0), grid.shape, T_REF))
    for name in ("sigma_a", "removal", "sigma_t", "diffusion", "nu_sigma_f", "chi"):
        for a, b in zip(getattr(plain, name), getattr(zero, name)):
            assert np.array_equal(a, b), name


def test_temperature_at_the_reference_is_a_no_op():
    """sqrt(T) - sqrt(T_ref) vanishes at T_ref, so even a large coefficient
    must leave k exactly where it was."""
    grid, mmap = _problem()
    base = _solve(grid, mmap)
    at_ref = _solve(grid, mmap, uniform(_feedback(doppler=5e-2), grid.shape, T_REF))
    assert at_ref.k_eff == pytest.approx(base.k_eff, abs=2e-8)


def test_sigma_a_plus_out_scatter_equals_removal_before_and_after():
    """The invariant that makes additive Doppler correct: feedback must move
    absorption and removal by the SAME amount, leaving out-scatter alone."""
    grid, mmap = _problem(n=8)
    xp = get_backend("cpu")
    mats = _materials()
    out_scatter = np.array([m.sigma_s.sum(axis=1) - np.diag(m.sigma_s) for m in mats])

    for T in (T_REF, 1200.0):
        f = Fields(xp, grid, mats, mmap, np.float64,
                   xs_update=uniform(_feedback(doppler=2e-3), grid.shape, T))
        for g in range(2):
            expected = f.sigma_a[g] + out_scatter[:, g][mmap]
            np.testing.assert_allclose(f.removal[g], expected, rtol=0, atol=1e-15)


def test_only_the_named_materials_respond():
    grid, mmap = _problem(n=8)
    xp = get_backend("cpu")
    plain = Fields(xp, grid, _materials(), mmap, np.float64)
    hot = Fields(xp, grid, _materials(), mmap, np.float64,
                 xs_update=uniform(_feedback(doppler=2e-3), grid.shape, 1200.0))
    reflector = mmap == 0
    for g in range(2):
        np.testing.assert_array_equal(hot.sigma_a[g][reflector],
                                      plain.sigma_a[g][reflector])
        assert np.all(hot.sigma_a[g][~reflector] > plain.sigma_a[g][~reflector])


# -- the physics ---------------------------------------------------------------

def test_doppler_coefficient_is_negative():
    grid, mmap = _problem()
    alpha = measure_pcm_per_K(_factory(grid, mmap), grid.shape, _feedback(),
                              t_cal=T_REF, dT=50.0, tol_k=1e-10, tol_source=1e-9)
    assert alpha < 0.0, f"hotter fuel must lose reactivity, got {alpha:+.3f} pcm/K"


def test_reactivity_is_linear_in_the_coefficient_amplitude():
    """What justifies fitting the amplitude from a single probe."""
    grid, mmap = _problem()
    fac, shape = _factory(grid, mmap), grid.shape
    kw = dict(t_cal=T_REF, dT=50.0, tol_k=1e-10, tol_source=1e-9)
    a1 = measure_pcm_per_K(fac, shape, _feedback(doppler=1e-3), **kw)
    a2 = measure_pcm_per_K(fac, shape, _feedback(doppler=2e-3), **kw)
    assert a2 / a1 == pytest.approx(2.0, rel=0.02)


def test_calibration_hits_the_requested_coefficient():
    grid, mmap = _problem()
    target = -2.5
    cal, measured = calibrate(_factory(grid, mmap), grid.shape, _feedback(),
                              target_pcm_per_K=target, t_cal=T_REF, dT=50.0,
                              tol_k=1e-10, tol_source=1e-9)
    assert measured == pytest.approx(target, rel=0.01)
    assert cal.doppler[1, 0] > 0.0            # a positive c_D on the fuel...
    assert cal.doppler[0, 0] == 0.0           # ...and still nothing on the reflector


def test_positive_target_is_refused():
    grid, mmap = _problem(n=8)
    with pytest.raises(ValueError, match="must be negative"):
        calibrate(_factory(grid, mmap), grid.shape, _feedback(),
                  target_pcm_per_K=+1.0)


def test_doppler_and_expansion_are_separable_and_additive():
    """Each mechanism alone, then both: the total must be the sum to within the
    genuine second-order cross term, not to machine precision."""
    grid, mmap = _problem()
    fac, shape = _factory(grid, mmap), grid.shape
    kw = dict(t_cal=T_REF, dT=50.0, tol_k=1e-10, tol_source=1e-9)
    a_dop = measure_pcm_per_K(fac, shape, _feedback(doppler=1e-3), **kw)
    a_exp = measure_pcm_per_K(fac, shape, _feedback(doppler=0.0, expansion=1e-4), **kw)
    a_both = measure_pcm_per_K(fac, shape, _feedback(doppler=1e-3, expansion=1e-4), **kw)
    assert a_exp < 0.0
    assert a_both == pytest.approx(a_dop + a_exp, rel=0.05)


def test_expansion_scales_every_cross_section_and_inverts_the_diffusion_coefficient():
    grid, mmap = _problem(n=8)
    xp = get_backend("cpu")
    beta, dT = 1e-4, 200.0
    f_expect = 1.0 - beta * dT
    plain = Fields(xp, grid, _materials(), mmap, np.float64)
    hot = Fields(xp, grid, _materials(), mmap, np.float64,
                 xs_update=uniform(_feedback(doppler=0.0, expansion=beta),
                                   grid.shape, T_REF + dT))
    fuel = mmap == 1
    for g in range(2):
        np.testing.assert_allclose(hot.nu_sigma_f[g][fuel],
                                   plain.nu_sigma_f[g][fuel] * f_expect, rtol=1e-13)
        np.testing.assert_allclose(hot.diffusion[g][fuel],
                                   plain.diffusion[g][fuel] / f_expect, rtol=1e-13)
        np.testing.assert_allclose(hot.chi[g][fuel], plain.chi[g][fuel], rtol=0)
    np.testing.assert_allclose(hot.sigma_s[0][1][fuel],
                               plain.sigma_s[0][1][fuel] * f_expect, rtol=1e-13)


def test_per_group_doppler_is_honoured_and_a_wrong_width_is_refused():
    """A (M, G) coefficient lets the resonance range carry the effect while the
    thermal groups do not. A width that is neither 1 nor G would otherwise be
    clamped to the last column, silently mis-weighting every group past it."""
    grid, mmap = _problem(n=8)
    xp = get_backend("cpu")
    fast_only = ThermalFeedback(t_ref=[T_REF, T_REF],
                                doppler=[[0.0, 0.0], [2e-3, 0.0]])
    f = Fields(xp, grid, _materials(), mmap, np.float64,
               xs_update=uniform(fast_only, grid.shape, 1400.0))
    plain = Fields(xp, grid, _materials(), mmap, np.float64)
    fuel = mmap == 1
    assert np.all(f.sigma_a[0][fuel] > plain.sigma_a[0][fuel])       # fast group
    np.testing.assert_array_equal(f.sigma_a[1][fuel], plain.sigma_a[1][fuel])

    wrong = ThermalFeedback(t_ref=[T_REF, T_REF],
                            doppler=[[0.0, 0.0, 0.0], [1e-3, 1e-3, 1e-3]])
    with pytest.raises(ValueError, match="one per group"):
        Fields(xp, grid, _materials(), mmap, np.float64,
               xs_update=uniform(wrong, grid.shape, 1000.0))


def test_runaway_expansion_is_refused():
    grid, mmap = _problem(n=8)
    xp = get_backend("cpu")
    with pytest.raises(ValueError, match="linearization"):
        Fields(xp, grid, _materials(), mmap, np.float64,
               xs_update=uniform(_feedback(doppler=0.0, expansion=1e-2),
                                 grid.shape, T_REF + 500.0))


# -- the hook reaches the other geometries and angular approximations ----------

def test_tabulated_feedback_reproduces_the_library_at_its_nodes():
    """Interpolating the library's branches must return the library's own
    numbers at the tabulation points -- if it does not, the interpolation is
    reading the wrong axis or the wrong node."""
    from ndgpu import TriDiffusionEigenSolver
    from ndgpu.benchmarks.hpmr import (_ASSEMBLY_VOLUME_FRACTIONS,
                                       _XS_FUEL_COMPACT, build_hpmr2d,
                                       hpmr_materials_builtin)
    from ndgpu.benchmarks.hpmr_assembly import PIN_XS_IDS, pin_materials_builtin
    from ndgpu.benchmarks.hpmr_thermal import (hpmr_endfb8_builtin,
                                               hpmr_tabulated_feedback)
    from ndgpu.griffin_xs import volume_homogenize

    fb = hpmr_tabulated_feedback()
    p = build_hpmr2d(refine=2, drum_angle_deg=180.0, absorber="polar",
                     materials=hpmr_endfb8_builtin())
    common = dict(bc=p.bc, active=p.active, mask_bc=p.mask_bc,
                  mix_material=p.mix_material, mix_weight=p.mix_weight,
                  device="cpu")

    # The 800 K node is what the unfed materials already are, so the hook must
    # be a no-op there -- the sharpest single check available without the XML.
    t_ref = 800.0
    plain = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                    **common).solve(tol_k=1e-9, tol_source=1e-8)
    at_node = TriDiffusionEigenSolver(
        p.grid, p.materials, p.material_map,
        xs_update=fb.hook(np.full(p.grid.shape, t_ref)),
        **common).solve(tol_k=1e-9, tol_source=1e-8)
    assert at_node.k_eff == pytest.approx(plain.k_eff, abs=1e-7)


def test_tabulated_feedback_is_monotonic_and_negative():
    from ndgpu import TriDiffusionEigenSolver
    from ndgpu.benchmarks.hpmr import build_hpmr2d
    from ndgpu.benchmarks.hpmr_thermal import (hpmr_endfb8_builtin,
                                               hpmr_tabulated_feedback)

    fb = hpmr_tabulated_feedback()
    p = build_hpmr2d(refine=2, drum_angle_deg=180.0, absorber="polar",
                     materials=hpmr_endfb8_builtin())
    ks = []
    for T in fb.temperatures:
        r = TriDiffusionEigenSolver(
            p.grid, p.materials, p.material_map, bc=p.bc, active=p.active,
            mask_bc=p.mask_bc, mix_material=p.mix_material,
            mix_weight=p.mix_weight, device="cpu",
            xs_update=fb.hook(np.full(p.grid.shape, float(T)))
        ).solve(tol_k=1e-9, tol_source=1e-8)
        ks.append(r.k_eff)
    ks = np.array(ks)
    assert np.all(np.diff(ks) < 0), ks           # hotter is always less reactive
    alpha = 1e5 * (1 / ks[0] - 1 / ks[-1]) / (fb.temperatures[-1] - fb.temperatures[0])
    assert -6.0 < alpha < -1.5, f"{alpha:+.2f} pcm/K is outside the plausible band"


def test_tabulated_feedback_clamps_outside_the_evaluated_range():
    """Extrapolating a resonance-broadening curve past the evaluated range is
    not supported by the data, and a coupling iterate can transiently overshoot,
    so values outside the nodes must clamp rather than run away."""
    from ndgpu.backend import get_backend
    from ndgpu.benchmarks.hpmr import build_hpmr2d
    from ndgpu.benchmarks.hpmr_thermal import (hpmr_endfb8_builtin,
                                               hpmr_tabulated_feedback)
    from ndgpu.solver import Fields

    fb = hpmr_tabulated_feedback()
    p = build_hpmr2d(refine=2, drum_angle_deg=180.0, absorber="polar",
                     materials=hpmr_endfb8_builtin())
    xp = get_backend("cpu")
    lo, hi = float(fb.temperatures[0]), float(fb.temperatures[-1])
    common = dict(mix_material=p.mix_material, mix_weight=p.mix_weight)

    def build(T):
        return Fields(xp, p.grid, p.materials, p.material_map, np.float64,
                      xs_update=fb.hook(np.full(p.grid.shape, T)), **common)

    for inside, outside in ((lo, lo - 500.0), (hi, hi + 500.0)):
        a, b = build(inside), build(outside)
        for g in range(a.n_groups):
            np.testing.assert_allclose(a.sigma_a[g], b.sigma_a[g], rtol=1e-13)


def test_hook_is_inherited_by_the_triangular_and_sp3_solvers():
    from ndgpu.benchmarks.hpmr import build_hpmr2d
    from ndgpu import TriDiffusionEigenSolver, TriSP3EigenSolver

    p = build_hpmr2d(refine=2, drum_angle_deg=180.0, absorber="polar")
    fb = ThermalFeedback(t_ref=[T_REF] * len(p.materials),
                         doppler=[0.0, 5e-3] + [0.0] * (len(p.materials) - 2))
    hook = uniform(fb, p.grid.shape, 1400.0)
    common = dict(active=p.active, mask_bc=p.mask_bc,
                  mix_material=p.mix_material, mix_weight=p.mix_weight,
                  device="cpu")
    for cls in (TriDiffusionEigenSolver, TriSP3EigenSolver):
        cold = cls(p.grid, p.materials, p.material_map, **common).solve(
            tol_k=1e-9, tol_source=1e-8)
        hot = cls(p.grid, p.materials, p.material_map, xs_update=hook,
                  **common).solve(tol_k=1e-9, tol_source=1e-8)
        assert hot.k_eff < cold.k_eff, cls.__name__
