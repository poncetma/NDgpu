"""Define a reactor from scratch with the simplified Model API.

This is the "bring your own reactor" example: custom two-group materials with
your own cross sections, a geometry painted region by region in centimetres, and
boundary conditions by name -- then one call to run() and a human-readable
report. A fuel block sits in a graphite-like reflector; run it at a few core
sizes to watch the reflector savings and the leakage fraction change.

Usage: python examples/reflected_core.py [fuel_half_width_cm] [cpu|gpu|auto]
"""

import sys

import ndgpu
from ndgpu import Material

half = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0   # fuel block half-width
device = sys.argv[2] if len(sys.argv) > 2 else "auto"

# 1. Materials -- macroscopic two-group constants, all in 1/cm (D in cm).
fuel = Material(
    name="fuel",
    diffusion=[1.26, 0.35],
    sigma_a=[0.012, 0.121],
    nu_sigma_f=[0.0085, 0.185],
    sigma_s=[[0.0, 0.026], [0.0, 0.0]],     # fast -> thermal down-scatter
    chi=[1.0, 0.0],                          # all fission neutrons born fast
)
reflector = Material(
    name="reflector",
    diffusion=[1.15, 0.90],                  # good moderator: low absorption, scatters
    sigma_a=[0.0002, 0.0050],
    nu_sigma_f=[0.0, 0.0],                   # non-fissile
    sigma_s=[[0.0, 0.045], [0.0, 0.0]],
)

# 2. Geometry -- a 120 cm cube, reflector everywhere, fuel block in the centre.
box = 120.0
lo, hi = box / 2 - half, box / 2 + half
model = (
    ndgpu.Model(size=(box, box, box), cells=(40, 40, 40))
    .fill(reflector)
    .add_box(fuel, x=(lo, hi), y=(lo, hi), z=(lo, hi))
    .set_boundary("vacuum")                  # leak from the outer surface
)

# 3. Run and report.
print(f"Fuel block {2 * half:.0f} cm across in a {box:.0f} cm reflector, on {device}\n")
result = model.run(device=device)
print(result)
