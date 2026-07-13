"""A hexagonal assembly lattice on the body-fitted triangular solver.

Demonstrates ndgpu.HexLattice: place assemblies at hex sites (axial (R, C)
coordinates), pick diffusion or SP3 transport, and get the human-readable
report. Here a fissile assembly is surrounded by a ring of reflector
assemblies; running it with both methods shows the transport (SP3) correction
that a small, leaky cluster brings out.

Usage: python examples/hex_lattice.py [refine] [cpu|gpu|auto]
"""

import sys

import ndgpu
from ndgpu import Material

refine = int(sys.argv[1]) if len(sys.argv) > 1 else 4
device = sys.argv[2] if len(sys.argv) > 2 else "auto"

fuel = Material(name="fuel", diffusion=[1.26, 0.35], sigma_a=[0.012, 0.121],
                nu_sigma_f=[0.0085, 0.185], sigma_s=[[0.0, 0.026], [0.0, 0.0]], chi=[1, 0])
reflector = Material(name="reflector", diffusion=[1.15, 0.90], sigma_a=[0.0002, 0.005],
                     nu_sigma_f=[0.0, 0.0], sigma_s=[[0.0, 0.045], [0.0, 0.0]])

# A central fuel assembly with its six nearest hex neighbours as reflector.
lattice = ndgpu.HexLattice(pitch=20.0, refine=refine).set_boundary("vacuum")
lattice.set_site((0, 0), fuel)
for rc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
    lattice.set_site(rc, reflector)

print(f"7-assembly hex cluster, refine={refine}, on {device}\n")
diffusion = lattice.run(method="diffusion", device=device)
print(diffusion)

sp3 = lattice.run(method="sp3", device=device)
print(f"\n  SP3 transport k_eff : {sp3.k_eff:.6f}")
print(f"  transport correction: {(sp3.k_eff - diffusion.k_eff) * 1e5:+.0f} pcm "
      f"(SP3 - diffusion)")
