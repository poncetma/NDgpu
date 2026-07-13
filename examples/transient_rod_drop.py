"""A control-rod drop transient via the Model API.

Every transient starts from a steady state, and Model.transient makes that
explicit: it solves the t=0 equilibrium first (available afterwards as
result.steady) and then marches in time. Here a central channel holds coolant
while the reactor sits critical; at t = 0.5 s a control rod drops into the
channel (the channel material becomes a strong absorber), inserting negative
reactivity, and the power falls.

The perturbation is described by materials_at(t): a function returning the
material list at time t (in the model's index order). Returning the *same*
objects while nothing changes lets the solver reuse its operators.

Usage: python examples/transient_rod_drop.py [cpu|gpu|auto]
"""

import sys

import ndgpu
from ndgpu import Material

device = sys.argv[1] if len(sys.argv) > 1 else "auto"

fuel = Material(name="fuel", diffusion=[1.26, 0.35], sigma_a=[0.011, 0.10],
                nu_sigma_f=[0.007, 0.175], sigma_s=[[0.0, 0.032], [0.0, 0.0]], chi=[1, 0])
coolant = Material(name="coolant", diffusion=[1.5, 1.1], sigma_a=[0.0004, 0.008],
                   nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.05], [0.0, 0.0]])
rod = Material(name="control rod", diffusion=[1.2, 0.30], sigma_a=[0.02, 0.45],
               nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.02], [0.0, 0.0]])

# Core with a central channel (coolant at t=0); rod drops in at t = 0.5 s.
model = (
    ndgpu.Model(size=(160, 160, 160), cells=(16, 16, 16))
    .fill(fuel)
    .add_box(coolant, x=(60, 100), y=(60, 100))         # the rod channel
    .set_boundary("vacuum")
    .set_kinetics(velocities=[1.0e7, 3.0e5], beta=[0.0065], decay=[0.08])
)

drop_time = 0.5
materials_at = lambda t: [fuel, rod if t >= drop_time else coolant]

print(f"Control-rod drop at t = {drop_time:g} s, on {device}\n")
result = model.transient(t_end=3.0, dt=0.02, materials_at=materials_at, device=device)

print("Initial steady state:")
print("  k_eff =", f"{result.steady.k_eff:.6f}",
      f"({result.steady.reactivity_pcm:+.0f} pcm before critical adjustment)\n")
print(result)
