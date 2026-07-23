"""The HP-MR microreactor -- core AND control drums -- via the HexLattice API.

The ANL/INL heat-pipe microreactor is a hexagonal lattice of assemblies (a
central assembly, a ring of fuel, a beryllium reflector) with twelve control
drums around the periphery. Each drum is a beryllium cylinder carrying a thin
B4C absorber arc that rotates to change reactivity. All of this maps onto
ndgpu.HexLattice: set_site for the assemblies, and set_drum for the drums, whose
arc is volume-mixed by area fraction and rotated with the angle_deg argument.

Sweeping every drum's angle from 0 (arc inserted, facing the core) to 180 (arc
withdrawn, facing outward) gives the drum-worth curve -- the reactivity the drums
can hold down.

Usage: python examples/hpmr_hexlattice.py [refine] [cpu|gpu|auto]
"""

import sys

import ndgpu
from ndgpu.benchmarks.hpmr import (_placeholder_materials, _FUEL_SITES, _BE_SITES,
                                   _DRUM_SITES, PITCH, DRUM_ABSORBER_INNER,
                                   DRUM_RADIUS, DRUM_ARC_HALF_DEG, CENTRAL, FUEL,
                                   BE_REFLECTOR, DRUM_BE, DRUM_ABSORBER)

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
device = sys.argv[2] if len(sys.argv) > 2 else "auto"

mats = _placeholder_materials()
central, fuel, be = mats[CENTRAL], mats[FUEL], mats[BE_REFLECTOR]
drum_body, b4c = mats[DRUM_BE], mats[DRUM_ABSORBER]


def hpmr(drum_angle_deg):
    """Build the HP-MR with every control drum rotated to the given angle."""
    lat = ndgpu.HexLattice(pitch=PITCH, refine=refine).set_boundary("vacuum")
    lat.set_site((0, 0), central)
    for s in _FUEL_SITES:
        lat.set_site(s, fuel)
    for s in _BE_SITES:
        lat.set_site(s, be)
    for s in _DRUM_SITES:
        lat.set_drum(s, body=drum_body, absorber=b4c,
                     inner_radius=DRUM_ABSORBER_INNER, outer_radius=DRUM_RADIUS,
                     arc_deg=2 * DRUM_ARC_HALF_DEG, angle_deg=drum_angle_deg)
    return lat.run(device=device, tol_k=1e-9, tol_source=1e-8, samples=10)


print(f"HP-MR via HexLattice, refine={refine}, {len(_DRUM_SITES)} control drums, on {device}\n")
print(f"  {'drum angle':>10}  {'k_eff':>9}  {'reactivity vs inserted (0 deg) (pcm)':>36}")
res0 = hpmr(0.0)
k0 = res0.k_eff
for angle in (0.0, 45.0, 90.0, 135.0, 180.0):
    r = hpmr(angle)
    worth = (1.0 / r.k_eff - 1.0 / k0) * 1e5
    print(f"  {angle:>10.0f}  {r.k_eff:>9.6f}  {worth:>30.0f}")

print(f"\n  total drum worth (0 -> 180 deg): "
      f"{(1.0 / hpmr(180.0).k_eff - 1.0 / k0) * 1e5:+.0f} pcm\n")
print("Fully-withdrawn state report:")
print(res0)
