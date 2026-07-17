"""ANL-7416 Benchmark Problem 8-A1: 2D (r-z) delayed supercritical transient.

Benchmark Source Situation 8 of the Argonne Code Center Benchmark Problem
Book (ANL-7416, Supplement 2, June 1977, pp. 182-186; submitted by
H. L. Dodds, Jr.). A large thermal reactor as a body of revolution:
radius 240 cm, height 525 cm, two-group diffusion, zero-flux external
boundaries, six delayed precursor families.

Sixteen material regions (r x z bands, dimensions in cm; region numbers from
the book's figure, z here measured from the *bottom* -- the book draws z
downward, the problem's physics does not care):

    z 487.5-525.0 : region 1  (r 0-240)                       "reflector A"
    z 450.0-487.5 : region 2  (r 0-240)                       "reflector B"
    z 337.5-450.0 : regions 3 (0-40), 4 (40-120), 5 (120-160), 6 (160-200)
    z 187.5-337.5 : regions 7 (0-40), 8 (40-120), 9 (120-160), 10 (160-200)
    z  75.0-337.5+: region 16 (200-240) spans z 75-450 (outer fuel ring)
    z  75.0-187.5 : regions 11 (0-120), 12 (120-160), 13 (160-200)
    z  37.5- 75.0 : region 14 (r 0-240)  (same material as 2)
    z   0.0- 37.5 : region 15 (r 0-240)  (same material as 1)

Transient (Problem 8-A1): the total group-2 cross section Sigma_2 ramps
linearly for one second and then holds -- x(1 + 0.03 t) in regions 3 and 7,
x(1 - 0.03 t) in region 11 -- with production cross sections divided by the
initial k_eff (critical adjustment) and precursors initially in equilibrium.
Only the absorption part of Sigma_2 is perturbed here (nu Sigma_f is held),
the standard reading shared with the TWIGL-family benchmarks: the ramp
represents control poison. Net reactivity is positive (region 11 is much
larger than 3 + 7): a delayed supercritical power rise to ~2.7x P0 at 4 s.

Reference solutions in the book, all on the Delta_r = 8 cm x Delta_z =
18.75 cm mesh (30 x 28), mesh-point (vertex-centered) 5-point finite
differences:

  8-A1-1  TWODTA  (implicit FD)   k_eff = 0.866901 (eigensolve: 0.867053)
  8-A1-2  TWODQD  (quasistatic)   power from 8-A1-1's k
  8-A1-3  ADEP    (ADE explicit)  k_eff = 0.866861

P_REFERENCE below is Exhibit A of 8-A1-1 at Delta_t = 0.001 s; at t = 4 s
the three solutions report 2.659 / 2.661 / 2.683.

Erratum: the book prints D_1 = "1.2997+1" (12.997 cm) for region 16, an
order of magnitude above every other fast-group D (all ~1.3 cm). That is a
typo for 1.2997+0: with 1.2997 cm this solver reproduces the reference
eigenvalue on the benchmark mesh to ~75 pcm (0.86780 vs the reported
eigensolve 0.867053), while the as-printed 12.997 cm gives k ~ 0.855,
~1200 pcm from every reference solution. 1.2997 cm is used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Kinetics, Material

# Same six-family delayed data as the Langenbuch problem from the same book.
ANL8A1_KINETICS = Kinetics(
    velocities=[1.0e7, 2.2e5],
    beta=[2.47e-4, 1.384e-3, 1.222e-3, 2.645e-3, 8.32e-4, 1.69e-4],
    decay=[1.27e-2, 3.17e-2, 1.15e-1, 3.11e-1, 1.40, 3.87],
)

K_REFERENCE = 0.866901          # 8-A1-1 (TWODTA); 8-A1-3 (ADEP): 0.866861
P_REFERENCE = {                 # Exhibit A of 8-A1-1, Delta_t = 0.001 s
    0.2: 1.039, 0.4: 1.108, 0.6: 1.209, 0.8: 1.360, 1.0: 1.596,
    1.2: 1.752, 1.4: 1.828, 1.6: 1.895, 1.8: 1.960, 2.0: 2.027,
    3.0: 2.335, 4.0: 2.659,
}
P4_REFERENCE_SPREAD = (2.659, 2.683)   # t = 4 s across solutions 8-A1-1..3

# Discretization context for P_REFERENCE (see dev-refs/anl8a1_vertex_scheme.py
# for the re-derivation): all three published solutions share one spatial
# scheme -- vertex-centered box integration on the 30 x 28 mesh -- whose ramp
# worth is 0.398 $, below the mesh-converged 0.4195 $; the cell-centered FV
# stencil on the same coarse mesh gives 0.454 $, above it. The published
# excursion tail therefore sits *below* the mesh-converged answer: this
# solver's converged trace (refine=8, dt = 0.02 backward Euler) is
#   {0.2: 1.041, 0.4: 1.114, 0.6: 1.222, 0.8: 1.386, 1.0: 1.648,
#    1.2: 1.821, 1.4: 1.910, 1.6: 1.987, 1.8: 2.062, 2.0: 2.135,
#    3.0: 2.500, 4.0: 2.887}
# i.e. ~+8.5% at 4 s, exactly what the +5.4% worth difference compounds to.

# Initial two-group constants, book p. 184: region -> (D1, D2, Sigma_1,
# Sigma_2, nuSigma_f1, nuSigma_f2, Sigma_1->2). Sigma_1 includes the
# downscatter (Sigma_1 = Sigma_c1 + Sigma_f1 + Sigma_1->2); Sigma_2 is total
# group-2 absorption. Units cm / 1/cm.
_XS = {
    1:  (1.0684, 0.32051, 2.8e-2, 3.3e-3, 0.0, 0.0, 2.6e-2),
    2:  (1.3495, 0.87032, 1.201e-2, 1.9e-2, 0.0, 0.0, 1.2e-2),
    3:  (1.3052, 0.88857, 1.0475e-2, 1.3063e-2, 1.1776e-3, 1.3268e-2, 8.0351e-3),
    5:  (1.3052, 0.88857, 1.0475e-2, 1.2623e-2, 1.1776e-3, 1.3268e-2, 8.0351e-3),
    6:  (1.3052, 0.88857, 1.0475e-2, 1.2183e-2, 1.1776e-3, 1.3268e-2, 8.0351e-3),
    7:  (1.3052, 0.88857, 1.0475e-2, 1.3453e-2, 1.1776e-3, 1.3268e-2, 8.0351e-3),
    9:  (1.3052, 0.88857, 1.0475e-2, 1.2973e-2, 1.1776e-3, 1.3268e-2, 8.0351e-3),
    10: (1.3052, 0.88857, 1.0475e-2, 1.2933e-2, 1.1776e-3, 1.3268e-2, 8.0351e-3),
    # D1 printed as "1.2997+1" in the book -- a typo for 1.2997 (see erratum
    # note in the module docstring).
    16: (1.2997, 0.87951, 1.0470e-2, 1.3065e-2, 1.2875e-3, 1.4246e-2, 7.9061e-3),
}
_XS[4] = _XS[11] = _XS[3]
_XS[12] = _XS[5]
_XS[13] = _XS[6]
_XS[8] = _XS[7]
_XS[14] = _XS[2]
_XS[15] = _XS[1]

# Regions whose Sigma_2 ramps during the transient: region -> rate (1/s),
# applied as Sigma_2(t) = Sigma_2(0) * (1 + rate * min(t, 1)).
_RAMP = {3: +0.03, 7: +0.03, 11: -0.03}

# (r_lo, r_hi, z_lo, z_hi) extent of each region in cm, z from the bottom.
_LAYOUT = [
    (1, 0, 240, 487.5, 525.0),
    (2, 0, 240, 450.0, 487.5),
    (3, 0, 40, 337.5, 450.0), (4, 40, 120, 337.5, 450.0),
    (5, 120, 160, 337.5, 450.0), (6, 160, 200, 337.5, 450.0),
    (7, 0, 40, 187.5, 337.5), (8, 40, 120, 187.5, 337.5),
    (9, 120, 160, 187.5, 337.5), (10, 160, 200, 187.5, 337.5),
    (11, 0, 120, 75.0, 187.5), (12, 120, 160, 75.0, 187.5),
    (13, 160, 200, 75.0, 187.5),
    (16, 200, 240, 75.0, 450.0),
    (14, 0, 240, 37.5, 75.0),
    (15, 0, 240, 0.0, 37.5),
]


def _material(region: int, sigma2_factor: float = 1.0) -> Material:
    D1, D2, s1, s2, nf1, nf2, s12 = _XS[region]
    name = f"region {region}"
    if sigma2_factor != 1.0:
        name += f" (Sigma_2 x {sigma2_factor:.6f})"
    return Material(
        name=name,
        diffusion=[D1, D2],
        sigma_a=[s1 - s12, s2 * sigma2_factor],
        nu_sigma_f=[nf1, nf2],
        sigma_s=[[0.0, s12], [0.0, 0.0]],
        chi=[1.0, 0.0],
    )


@dataclass
class Anl8A1Problem:
    grid: Grid
    material_map: np.ndarray
    kinetics: Kinetics
    bc: tuple
    problem_at: object  # callable t -> (materials, material_map)


def build_anl8a1(refine: int = 1, perturbed: bool = True) -> Anl8A1Problem:
    """Assemble Problem 8-A1 on the benchmark mesh (or a refinement of it).

    refine : cells per benchmark mesh interval; refine=1 is the 30 x 28
        (Delta_r = 8 cm, Delta_z = 18.75 cm) mesh of the reference solutions.
    perturbed : False freezes the t=0 cross sections (steady state only).
    """
    nr, nz = 30 * refine, 28 * refine
    grid = Grid(shape=(nr, 1, nz), size=(240.0, 1.0, 525.0),
                geometry="cylindrical")

    # Region map from cell centers; material index = region number - 1.
    r = grid.cell_centers(0)
    z = grid.cell_centers(2)
    Rc, Zc = np.meshgrid(r, z, indexing="ij")
    mmap = np.full((nr, 1, nz), -1, dtype=np.int64)
    for region, r0, r1, z0, z1 in _LAYOUT:
        inside = (Rc >= r0) & (Rc < r1) & (Zc >= z0) & (Zc < z1)
        mmap[:, 0, :][inside] = region - 1
    assert not np.any(mmap < 0), "region layout does not tile the domain"

    base = [_material(region) for region in range(1, 17)]
    cache: dict[float, list] = {}

    def problem_at(t: float):
        tc = min(float(t), 1.0) if perturbed else 0.0
        if tc not in cache:
            mats = list(base)
            for region, rate in _RAMP.items():
                mats[region - 1] = _material(region, 1.0 + rate * tc)
            cache[tc] = mats
        return cache[tc], mmap

    bc = (("reflective", "zero-flux"), "reflective", "zero-flux")
    return Anl8A1Problem(grid=grid, material_map=mmap,
                         kinetics=ANL8A1_KINETICS, bc=bc,
                         problem_at=problem_at)
