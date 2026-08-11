"""Build a new prismatic reactor through the public Python API.

This example contains no benchmark-specific builder or hand-assembled material
map. It makes a 19-assembly hex core, extrudes it, adds axial reflectors, solves
the hot/cold neutronics, and runs a short coupled perturbation.

    python examples/custom_tri_reactor.py              # CPU fallback
    python examples/custom_tri_reactor.py --device gpu # CuPy/CUDA
"""

import argparse

import numpy as np

import ndgpu


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--device", default="auto")
args = ap.parse_args()

# Homogenized assembly constants: fast -> thermal group order, cm / 1/cm.
fuel = ndgpu.Material(
    name="fuel", diffusion=[1.25, 0.35], sigma_a=[0.012, 0.120],
    nu_sigma_f=[0.0085, 0.185],
    sigma_s=[[0.0, 0.026], [0.0, 0.0]], chi=[1.0, 0.0])
radial_reflector = ndgpu.Material(
    name="radial reflector", diffusion=[1.15, 0.25],
    sigma_a=[0.001, 0.020], nu_sigma_f=[0.0, 0.0],
    sigma_s=[[0.0, 0.035], [0.0, 0.0]])
axial_reflector = ndgpu.Material(
    name="axial reflector", diffusion=[1.10, 0.23],
    sigma_a=[0.0015, 0.023], nu_sigma_f=[0.0, 0.0],
    sigma_s=[[0.0, 0.033], [0.0, 0.0]])

# Later paint calls overwrite earlier ones: disk 2 is the reflector background,
# disk 1 is seven fuel assemblies. Each hex gets 6*refine^2 triangles.
model = (
    ndgpu.HexLattice(pitch=18.0, refine=2)
    .set_disk(2, radial_reflector)
    .set_disk(1, fuel)
    .set_boundary("vacuum")
    .extrude(height=100.0, nz=10, boundary="vacuum")
    .add_axial_region(axial_reflector, z=(0.0, 10.0), replace=fuel)
    .add_axial_region(axial_reflector, z=(90.0, 100.0), replace=fuel)
    .set_kinetics(velocities=[1.0e7, 3.0e5], beta=[0.0065], decay=[0.08])
    .build(name="19-assembly demonstration core")
)

print(model.steady(device=args.device, tol_k=1e-8, tol_source=1e-7))

# Thermal data and feedback are keyed by Material.name, not internal integer
# indices. Inactive triangular padding is filled automatically.
model.configure_thermal(
    {
        "fuel": ndgpu.ThermalMaterial(
            conductivity=0.25, sink_coeff=0.015,
            sink_temperature=650.0, heat_capacity=2.5),
        "radial reflector": ndgpu.ThermalMaterial(
            conductivity=0.35, heat_capacity=2.0),
        "axial reflector": ndgpu.ThermalMaterial(
            conductivity=0.32, heat_capacity=2.1),
    },
    total_power=50_000.0,                         # W in this full 3-D model
    feedback={
        "fuel": ndgpu.FeedbackSpec(
            reference_temperature=700.0,
            doppler=[2.0e-4, 0.0]),
    },
    thermal_mask_bc=0.01, ambient_temperature=400.0,
)

# A state callback may return another TriReactor with the same mesh. Here it is
# a simple absorption step; cached control-drum frames use the same interface.
hot_fuel = ndgpu.Material(
    name="fuel", diffusion=fuel.diffusion,
    sigma_a=fuel.sigma_a * np.array([1.0, 1.002]),
    nu_sigma_f=fuel.nu_sigma_f, sigma_s=fuel.sigma_s, chi=fuel.chi)
perturbed = model.with_materials(
    [hot_fuel if m is fuel else m for m in model.materials])
state_at = lambda t: model if t <= 0.0 else perturbed

transient = model.coupled_transient(
    t_end=0.20, dt=0.05, dt_thermal=0.10,
    device=args.device, state_at=state_at, profile=True)
print("\n", transient)
print("  final P/P0:", transient.power[-1])
print("  thermal steps:", transient.counters["thermal_steps"])
print("  phase seconds:", transient.phase_seconds)
