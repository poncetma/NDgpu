"""2D HP-MR core model: geometry rasterization, symmetry, drum worth.

Cross sections are placeholders, so these tests pin behaviour (counts,
symmetry, orderings, mesh stability), not absolute reactivity.
"""

import numpy as np
import pytest

from ndgpu.benchmarks.hpmr import (DRUM_ABSORBER, FUEL, MATERIAL_NAMES,
                                   _placeholder_materials, build_hpmr2d)
from ndgpu.tri import TriDiffusionEigenSolver

TOL = dict(tol_k=1e-8, tol_source=1e-7)


def _k(refine, drum_angle_deg, materials=None):
    p = build_hpmr2d(refine=refine, drum_angle_deg=drum_angle_deg,
                     materials=materials)
    res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                  active=p.active, mask_bc=p.mask_bc,
                                  device="cpu").solve(**TOL)
    assert res.converged
    return res.k_eff


def test_geometry_counts():
    r = 4
    p = build_hpmr2d(refine=r)
    # 30 assemblies, each exactly 6 r^2 body-fitted triangles
    assert int((p.material_map == FUEL).sum()) == 30 * 6 * r * r
    # some absorber is rasterized on every drum
    assert int((p.material_map == DRUM_ABSORBER).sum()) >= 12
    # the active mask keeps the required one-cell void border
    act = p.active
    assert not (act[0].any() or act[-1].any() or act[:, 0].any() or act[:, -1].any())


def test_refine_below_absorber_resolution_raises():
    with pytest.raises(ValueError, match="refine >= 4"):
        build_hpmr2d(refine=2)


def test_wrong_material_count_raises():
    with pytest.raises(ValueError, match="expected 6 materials"):
        build_hpmr2d(refine=4, materials=_placeholder_materials()[:3])


def test_drum_rotation_worth_is_monotone():
    # Convention: 0 = arcs at the core centre (inserted), 180 = outward
    # (withdrawn). Withdrawing the arcs removes negative reactivity, so k rises
    # monotonically: k(0) < k(90) < k(180).
    k0, k90, k180 = (_k(4, a) for a in (0.0, 90.0, 180.0))
    assert k0 < k90 < k180
    worth_pcm = (1 / k0 - 1 / k180) * 1e5    # magnitude of the insertion worth
    assert worth_pcm > 500          # all-drums worth is far above mesh noise


def test_single_drum_mirror_symmetry():
    # One corner drum inserted vs its diametral opposite: identical cores up
    # to the lattice point symmetry, so identical k.
    a, b = np.zeros(12), np.zeros(12)
    a[0], b[3] = 180.0, 180.0
    assert _k(4, a) == pytest.approx(_k(4, b), abs=1e-10)


def test_scalar_angle_broadcasts():
    assert _k(4, 90.0) == pytest.approx(_k(4, np.full(12, 90.0)), abs=1e-12)


def test_k_stable_under_refinement():
    # Drums-out (angle 180, withdrawn) k moves < 100 pcm from refine 4 to 6.
    assert abs(_k(4, 180.0) - _k(6, 180.0)) < 1e-3


def _k_polar(refine, drum_angle_deg, samples=10):
    p = build_hpmr2d(refine=refine, drum_angle_deg=drum_angle_deg,
                     absorber="polar", samples=samples)
    res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                  active=p.active, mask_bc=p.mask_bc,
                                  mix_material=p.mix_material,
                                  mix_weight=p.mix_weight, device="cpu").solve(**TOL)
    assert res.converged
    return res.k_eff


def test_polar_builds_below_raster_floor():
    # The volume-mixing path represents the arc as an area fraction, so it works
    # below the raster's refine>=4 floor (where the annulus is sub-cell).
    p = build_hpmr2d(refine=2, drum_angle_deg=120.0, absorber="polar")
    assert p.mix_material is not None and p.mix_weight is not None
    assert p.mix_material.shape == p.material_map.shape
    w = p.mix_weight
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert int((w > 0).sum()) >= 12            # absorber present on every drum
    # weight>0 exactly where a mix material is set
    assert np.array_equal((w > 0), (p.mix_material == DRUM_ABSORBER))


def test_polar_worth_ordering_and_symmetry():
    k0, k90, k180 = (_k_polar(4, a) for a in (0.0, 90.0, 180.0))
    assert k0 < k90 < k180                     # 0 = inserted (arcs at core) = min k
    assert (1 / k0 - 1 / k180) * 1e5 > 500
    a, b = np.zeros(12), np.zeros(12)
    a[0], b[3] = 180.0, 180.0                  # a drum and its diametral opposite
    assert _k_polar(4, a) == pytest.approx(_k_polar(4, b), abs=1e-9)


def test_polar_worth_curve_smoother_than_raster():
    # Sweep the drum angle finely at a coarse mesh, where the raster staircases:
    # some adjacent angles flip no cells (dead steps) while others jump. The
    # polar area fraction changes every step, so its largest single-step
    # reactivity jump is smaller than the raster's.
    angles = np.arange(80.0, 141.0, 10.0)      # 60 deg sweep, steepest region
    def worth(fn):
        k = np.array([fn(4, a) for a in angles])
        return (1.0 / k - 1.0 / k[0]) * 1e5
    raster_steps = np.abs(np.diff(worth(_k)))
    polar_steps = np.abs(np.diff(worth(_k_polar)))
    assert polar_steps.max() < raster_steps.max()


def test_3d_polar_absorber_extrudes_from_2d():
    from ndgpu.benchmarks.hpmr import build_hpmr3d
    p2 = build_hpmr2d(refine=4, drum_angle_deg=120.0, absorber="polar")
    p3 = build_hpmr3d(refine=4, nz=10, drum_angle_deg=120.0, absorber="polar")
    assert p3.mix_material is not None
    assert p3.mix_material.shape == p3.material_map.shape
    mw = p3.mix_weight
    # drums run the full height, so the mix is z-invariant and every layer is
    # exactly the 2D mix.
    assert np.all(mw == mw[..., :1])
    assert np.array_equal(mw[..., 0], p2.mix_weight)
    assert np.array_equal(p3.mix_material[..., 0], p2.mix_material)


def test_3d_polar_worth_resolved_at_coarse_mesh():
    # At refine 4 the raster under-resolves the drum worth (~1950 pcm); the
    # polar mix recovers the bulk of it (~2650 pcm) even on this coarse mesh.
    from ndgpu.benchmarks.hpmr import build_hpmr3d

    def k3(ang):
        p = build_hpmr3d(refine=4, nz=10, drum_angle_deg=ang, absorber="polar")
        r = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                    active=p.active, mask_bc=p.mask_bc, bc=p.bc,
                                    mix_material=p.mix_material,
                                    mix_weight=p.mix_weight, device="cpu").solve(
            tol_k=1e-6, tol_source=1e-5)
        assert r.converged
        return r.k_eff

    k_in, k_out = k3(0.0), k3(180.0)             # 0 = inserted, 180 = withdrawn
    assert k_out > k_in
    assert (1 / k_in - 1 / k_out) * 1e5 > 2000   # above the raster's coarse value


def test_3d_geometry_counts():
    from ndgpu.benchmarks.hpmr import AXIAL_REFLECTOR, build_hpmr3d
    r, nz = 4, 10
    p2, p3 = build_hpmr2d(refine=r), build_hpmr3d(refine=r, nz=nz)
    assert p3.grid.shape == p2.grid.shape + (nz,)
    assert p3.grid.dz == pytest.approx(20.0)
    n_fuel_2d = int((p2.material_map == FUEL).sum())
    # fuel spans the central 160/200 of the height; axial Be replaces fuel and
    # the central cell in the outer layers
    assert int((p3.material_map == FUEL).sum()) == n_fuel_2d * (nz * 8 // 10)
    assert int((p3.material_map == AXIAL_REFLECTOR).sum()) == \
        (n_fuel_2d + int((p2.material_map == 2).sum())) * (nz * 2 // 10)
    # drums and their arcs run the full height
    assert int((p3.material_map == DRUM_ABSORBER).sum()) == \
        int((p2.material_map == DRUM_ABSORBER).sum()) * nz


def test_3d_nz_alignment_and_material_count_raise():
    from ndgpu.benchmarks.hpmr import build_hpmr3d, _placeholder_materials
    with pytest.raises(ValueError, match="multiple of 10"):
        build_hpmr3d(refine=4, nz=12)
    with pytest.raises(ValueError, match="expected 7 materials"):
        build_hpmr3d(refine=4, nz=10, materials=_placeholder_materials())


def test_3d_axial_leakage_worth_and_symmetry():
    from ndgpu.benchmarks.hpmr import build_hpmr3d
    solve = dict(tol_k=1e-7, tol_source=1e-6)

    def k3(angle):
        p = build_hpmr3d(refine=4, nz=10, drum_angle_deg=angle)
        res = TriDiffusionEigenSolver(p.grid, p.materials, p.material_map,
                                      active=p.active, mask_bc=p.mask_bc,
                                      bc=p.bc, device="cpu").solve(**solve)
        assert res.converged
        return p, res

    p_out, r_out = k3(180.0)                     # 180 = arcs out (withdrawn)
    _, r_in = k3(0.0)                            # 0 = arcs at the core (inserted)
    # axial leakage: 3D k below the 2D radial slice, but well above shutdown
    assert r_out.k_eff < _k(4, 180.0)
    assert r_out.k_eff > 1.0
    # drums still worth thousands of pcm in 3D
    assert (1 / r_in.k_eff - 1 / r_out.k_eff) * 1e5 > 500
    # thermal flux in the fuel: symmetric about the midplane, peaked there
    fuel = p_out.material_map == FUEL
    prof = (r_out.flux_numpy[1] * fuel).sum(axis=(0, 1, 2))
    prof = prof[prof > 0]
    np.testing.assert_allclose(prof, prof[::-1], rtol=1e-6)
    assert prof.argmax() in (len(prof) // 2 - 1, len(prof) // 2)


def test_custom_materials_swap_in():
    # The materials hook accepts a replacement set ordered as MATERIAL_NAMES
    # (this is how SPH-corrected FEMFFUSION constants will enter).
    mats = _placeholder_materials()
    assert len(mats) == len(MATERIAL_NAMES)
    k_ref = _k(4, 0.0)
    from ndgpu.materials import Material
    f = mats[FUEL]
    mats[FUEL] = Material(name="fuel+10%", diffusion=f.diffusion,
                          sigma_a=f.sigma_a, nu_sigma_f=1.1 * f.nu_sigma_f,
                          sigma_s=f.sigma_s, chi=f.chi)
    assert _k(4, 0.0, materials=mats) > k_ref
