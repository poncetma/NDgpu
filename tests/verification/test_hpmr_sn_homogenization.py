"""Transport-weighted homogenization of the HP-MR assembly.

``hpmr_sn_homogenization`` solves the pin-resolved assembly with tri-S_N and
collapses that one flux either over the whole assembly (constants for
``build_hpmr2d``) or per pin type (each clad sharing its pin's region, so the
homogenization performs the clad smearing).

The homogenizer itself is already pinned by ``test_sph.py``; what is at risk here
is the *region mapping*. If a clad were attached to the wrong pin, or the mixed
component's region silently dropped, the constants would still look plausible and
only the region volumes would betray it -- so that is what these check.
"""

import numpy as np
import pytest

from ndgpu.benchmarks.hpmr import hpmr_sn_homogenization


@pytest.fixture(scope="module")
def ref():
    # refine 4 keeps the S_N solve near 30 s while still resolving the pins.
    return hpmr_sn_homogenization(refine=4, device="cpu")


@pytest.mark.slow
def test_transport_reference_is_sane(ref):
    assert 1.0 < ref["k_inf"] < 1.5, ref["k_inf"]
    assert len(ref["pin_materials"]) == 4
    assert {m.n_groups for m in ref["pin_materials"]} == {ref["assembly"].n_groups}
    assert ref["flux"].shape[0] == ref["assembly"].n_groups


@pytest.mark.slow
def test_region_volumes_reproduce_the_pin_type_composition(ref):
    """A mis-mapped clad shows up here and essentially nowhere else."""
    vol = np.asarray(ref["region_volume"])
    frac = vol / vol.sum()
    # graphite, fuel, moderator+clad, heat pipe+wall
    for got, want, name in zip(frac, (0.3501, 0.3193, 0.1158, 0.2148),
                               ("graphite", "fuel", "moderator_pin", "hp_pin")):
        assert abs(got - want) < 0.03, (name, got, want)


@pytest.mark.slow
def test_flux_weighting_differs_from_volume_weighting(ref):
    """Otherwise the transport solve bought nothing.

    The clads sit in a depressed flux, so weighting by it must move the pin
    constants measurably away from the area-weighted mix of the same materials.
    """
    from ndgpu.benchmarks.hpmr_assembly import (R_HP, R_HP_SHELL, R_MOD,
                                                R_MOD_SHELL,
                                                pin_materials_builtin)
    from ndgpu.griffin_xs import volume_homogenize

    _, _, mod, mod_shell, hp, hp_shell = pin_materials_builtin()

    def area_mix(inner, outer, r_in, r_out, name):
        f = r_in ** 2 / r_out ** 2
        return volume_homogenize({0: inner, 1: outer}, {0: f, 1: 1.0 - f},
                                 name=name)

    pairs = ((2, area_mix(mod, mod_shell, R_MOD, R_MOD_SHELL, "m")),
             (3, area_mix(hp, hp_shell, R_HP, R_HP_SHELL, "h")))
    for i, vw in pairs:
        a_fw = np.asarray(ref["pin_materials"][i].sigma_a, dtype=float)
        a_vw = np.asarray(vw.sigma_a, dtype=float)
        rel = np.max(np.abs(a_fw - a_vw) / np.maximum(a_vw, 1e-30))
        assert rel > 1e-3, (i, rel)


@pytest.mark.slow
def test_region_flux_is_not_flat(ref):
    """The whole premise: pin types see materially different spectra, so a
    volume-weighted collapse discards real information."""
    rf = np.asarray(ref["region_flux"])
    thermal = rf[:, -1]
    assert thermal.max() / thermal.min() > 1.2, thermal
