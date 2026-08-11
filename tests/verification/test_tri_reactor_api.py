"""End-to-end tests for the public triangular reactor-design API.

The model is deliberately not HP-MR: a seven-site reflected test reactor proves
that geometry, kinetics and thermal coupling no longer depend on benchmark
builders or hand-assembled solver arguments.
"""

import numpy as np
import pytest

from ndgpu import (FeedbackSpec, HexLattice, Kinetics, Material,
                   ThermalMaterial, TriReactor, hex_disk, hex_ring)


FUEL = Material(
    name="test fuel", diffusion=[1.25, 0.35], sigma_a=[0.012, 0.120],
    nu_sigma_f=[0.0085, 0.185],
    sigma_s=[[0.0, 0.026], [0.0, 0.0]], chi=[1.0, 0.0])
REFLECTOR = Material(
    name="test reflector", diffusion=[1.15, 0.25], sigma_a=[0.001, 0.020],
    nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.035], [0.0, 0.0]])
AXIAL = Material(
    name="axial reflector", diffusion=REFLECTOR.diffusion,
    sigma_a=REFLECTOR.sigma_a, nu_sigma_f=REFLECTOR.nu_sigma_f,
    sigma_s=REFLECTOR.sigma_s)
KINETICS = Kinetics(velocities=[1.0e7, 3.0e5], beta=[0.0065], decay=[0.08])
TOL = dict(tol_k=1e-8, tol_source=1e-7)


def seven_site_lattice(refine=2):
    return (HexLattice(pitch=18.0, refine=refine)
            .set_disk(1, REFLECTOR)
            .set_site((0, 0), FUEL)
            .set_boundary("vacuum"))


def test_hex_coordinate_helpers_and_reusable_build():
    assert hex_ring(0) == [(0, 0)]
    assert len(hex_ring(3)) == 18
    assert len(hex_disk(3)) == 37
    assert set(hex_ring(2)) <= set(hex_disk(2))

    lattice = seven_site_lattice()
    reactor = lattice.build(name="seven-site demonstrator")
    assert isinstance(reactor, TriReactor)
    assert reactor.name == "seven-site demonstrator"
    assert reactor.shape == reactor.material_map.shape
    assert int(reactor.active.sum()) == 7 * 6 * lattice.refine**2

    # HexLattice.run remains exactly the convenience shorthand over build/run.
    built = reactor.steady(**TOL)
    shortcut = lattice.run(**TOL)
    assert built.k_eff == pytest.approx(shortcut.k_eff, abs=1e-12)
    assert built.reactor is reactor
    assert built.raw.flux_numpy.shape == built.flux.shape


def test_extrusion_and_axial_material_painting():
    reactor = (seven_site_lattice(refine=1)
               .extrude(height=40.0, nz=4, boundary="vacuum")
               .add_axial_region(AXIAL, z=(0.0, 10.0), replace=FUEL)
               .add_axial_region(AXIAL, z=(30.0, 40.0), replace=FUEL)
               .build())
    assert reactor.is_3d
    assert reactor.shape[-1] == 4
    fuel_id = next(i for i, m in enumerate(reactor.materials) if m is FUEL)
    axial_id = next(i for i, m in enumerate(reactor.materials) if m is AXIAL)
    assert not np.any(reactor.material_map[..., 0] == fuel_id)
    assert np.any(reactor.material_map[..., 0] == axial_id)
    assert np.any(reactor.material_map[..., 1] == fuel_id)
    assert np.any(reactor.material_map[..., 2] == fuel_id)
    assert not np.any(reactor.material_map[..., 3] == fuel_id)

    result = reactor.steady(**TOL)
    assert result.flux.shape == (2,) + reactor.shape
    assert result.converged


def test_tri_reactor_neutron_transient_and_material_replacement():
    reactor = seven_site_lattice(refine=1).set_kinetics(KINETICS).build()
    result = reactor.transient(t_end=0.10, dt=0.05)
    np.testing.assert_allclose(result.power, 1.0, atol=2e-6)
    assert result.reactor is reactor

    replacement = Material(
        name=FUEL.name, diffusion=FUEL.diffusion,
        sigma_a=FUEL.sigma_a * [1.0, 1.001],
        nu_sigma_f=FUEL.nu_sigma_f, sigma_s=FUEL.sigma_s, chi=FUEL.chi)
    materials = [replacement if m is FUEL else m for m in reactor.materials]
    changed = reactor.with_materials(materials)
    assert changed.grid is reactor.grid
    np.testing.assert_array_equal(changed.material_map, reactor.material_map)
    assert changed.kinetics is reactor.kinetics
    assert changed.steady(**TOL).k_eff < reactor.steady(**TOL).k_eff


def test_custom_reactor_runs_coupled_transient_without_benchmark_helpers():
    reactor = seven_site_lattice(refine=1).set_kinetics(KINETICS).build()
    reactor.configure_thermal(
        {
            "test fuel": ThermalMaterial(
                conductivity=0.25, sink_coeff=0.02,
                sink_temperature=650.0, heat_capacity=2.5,
                name="fuel solid"),
            "test reflector": ThermalMaterial(
                conductivity=0.35, heat_capacity=2.0,
                name="reflector solid"),
        },
        total_power=100.0,
        feedback={"test fuel": FeedbackSpec(
            reference_temperature=700.0, doppler=[0.0, 0.0])},
        thermal_mask_bc=0.01, ambient_temperature=400.0)

    context = reactor.coupling_context(device="cpu")
    assert context.kinetics is KINETICS
    assert len(context.thermal_materials) == len(reactor.materials)
    # The inactive raster padding was filled automatically; users only supplied
    # thermal data for materials that physically occur in the active core.
    assert context.thermal_materials[0].name.startswith("inactive")

    result = reactor.coupled_transient(
        t_end=0.10, dt=0.05, dt_thermal=0.10, device="cpu", profile=True)
    np.testing.assert_allclose(result.power, 1.0, atol=2e-8)
    assert result.counters["neutronics_steps"] == 2
    assert result.counters["thermal_steps"] == 1
    assert np.all(np.isfinite(result.temperature))

    qs = reactor.quasistatic_transient(
        t_end=0.10, dt=0.05, dt_thermal=0.10, shape_dt=0.10,
        device="cpu", profile=True)
    np.testing.assert_allclose(qs.power, 1.0, atol=2e-8)
    assert qs.counters["shape_updates"] == 1
    assert qs.counters["iqs_shape_solves"] == 1
    assert np.all(np.isfinite(qs.temperature))


def test_named_thermal_mapping_rejects_missing_active_material():
    reactor = seven_site_lattice(refine=1).build()
    with pytest.raises(ValueError, match="test reflector"):
        reactor.configure_thermal(
            {"test fuel": ThermalMaterial(0.2)}, total_power=10.0)
