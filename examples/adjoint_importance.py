"""Adjoint (importance) solve via the Model API, and what it's good for.

Passing adjoint=True to Model.run solves the adjoint k-eigenproblem: the same
eigenvalue as the forward problem, but the "flux" is the neutron *importance* --
how much a neutron born at each point contributes to the chain reaction. In this
thermal reactor the importance is peaked in the thermal group: a neutron becomes
most valuable once it has slowed down, because that is where it actually drives
fission (nu*Sigma_f is far larger thermal than fast here).

The adjoint is the weighting function for first-order perturbation theory and
adjoint-weighted kinetics (beta_eff, Lambda); here we just show that forward and
adjoint eigenvalues agree and print the importance report.

Usage: python examples/adjoint_importance.py [n_cells_per_axis] [cpu|gpu|auto]
"""

import sys

import ndgpu
from ndgpu import PWR_TWO_GROUP

n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
device = sys.argv[2] if len(sys.argv) > 2 else "auto"

core = lambda: (ndgpu.Model(size=(80, 80, 80), cells=(n, n, n))
                .fill(PWR_TWO_GROUP).set_boundary("vacuum"))

forward = core().run(device=device)
adjoint = core().run(adjoint=True, device=device)

print(adjoint)
print(f"\n  forward k_eff : {forward.k_eff:.6f}")
print(f"  adjoint k_eff : {adjoint.k_eff:.6f}   (must match the forward value)")
print(f"  difference    : {(adjoint.k_eff - forward.k_eff) * 1e5:+.2f} pcm")

# The importance is thermal-peaked (mirror image of the physical flux, which is
# fast-driven at birth); a thermalized neutron is worth more here.
fast_imp = adjoint.flux[0]
thermal_imp = adjoint.flux[1]
ratio = thermal_imp.flat[thermal_imp.size // 2] / fast_imp.flat[fast_imp.size // 2]
print(f"\n  adjoint thermal/fast importance ratio at core centre: {ratio:.2f}  "
      f"(>1: a thermalized neutron is worth more)")
