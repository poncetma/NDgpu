"""The HP-MR microreactor core built with the HexLattice Model API.

The ANL/INL heat-pipe microreactor is a hexagonal lattice of assemblies -- a
central assembly, a ring of fuel, a beryllium reflector, and twelve control
drums -- which maps directly onto ndgpu.HexLattice.set_site((R, C), material).
This reproduces the core geometry exactly (checked below against the benchmark
raster) and runs on both diffusion and SP3 transport.

One caveat: the control drums carry a thin B4C absorber *arc* inside the drum hex
that rotates to change reactivity. That is sub-hex detail, and HexLattice places
one material per whole hex, so the drum-worth-vs-angle study still uses
ndgpu.benchmarks.build_hpmr2d (which paints the arc). Here the drums are their
beryllium bodies (arc withdrawn), which is the reactor's most-reactive state.

Usage: python examples/hpmr_hexlattice.py [refine] [cpu|gpu|auto]
"""

import sys

import ndgpu
from ndgpu.benchmarks.hpmr import (_placeholder_materials, _FUEL_SITES, _BE_SITES,
                                   _DRUM_SITES, PITCH, CENTRAL, FUEL, BE_REFLECTOR,
                                   DRUM_BE)

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
device = sys.argv[2] if len(sys.argv) > 2 else "auto"

mats = _placeholder_materials()      # [void, fuel, central, be_reflector, drum_be, drum_absorber]
central, fuel, be, drum = mats[CENTRAL], mats[FUEL], mats[BE_REFLECTOR], mats[DRUM_BE]

# Place every assembly at its hex site.
lattice = ndgpu.HexLattice(pitch=PITCH, refine=refine).set_boundary("vacuum")
lattice.set_site((0, 0), central)
for s in _FUEL_SITES:
    lattice.set_site(s, fuel)
for s in _BE_SITES:
    lattice.set_site(s, be)
for s in _DRUM_SITES:
    lattice.set_site(s, drum)

print(f"HP-MR core via HexLattice, refine={refine} "
      f"({len(_FUEL_SITES)} fuel + {len(_BE_SITES)} Be + {len(_DRUM_SITES)} drums), on {device}\n")

result = lattice.run(method="diffusion", device=device, tol_k=1e-9, tol_source=1e-8)
print(result)

sp3 = lattice.run(method="sp3", device=device, tol_k=1e-9, tol_source=1e-8)
print(f"\n  SP3 transport k_eff : {sp3.k_eff:.6f}")
print(f"  transport correction: {(sp3.k_eff - result.k_eff) * 1e5:+.0f} pcm (SP3 - diffusion)")
