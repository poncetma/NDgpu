"""The TWIGL seed-blanket benchmark, built with the Model API and checked
against the published literature values.

TWIGL (Hageman & Yasinsky, 1969) is a two-group, 80 x 80 cm quarter core with
three rectangular regions: a blanket, an L-shaped seed, and a central perturbable
seed. It is reflective on the inner faces and zero-flux on the outer ones, with
one delayed-neutron family. It has published values for both the static
eigenvalue and the transient power, so it exercises the whole Model API -- steady
solve, region painting, boundary conditions, kinetics, and the transient -- and
lets us confirm correctness against the reference.

Usage: python examples/twigl_benchmark.py [cells_per_8cm] [step|ramp] [cpu|gpu|auto]
"""

import sys

import numpy as np

import ndgpu
from ndgpu import Material

r = int(sys.argv[1]) if len(sys.argv) > 1 else 4
kind = sys.argv[2] if len(sys.argv) > 2 else "ramp"
device = sys.argv[3] if len(sys.argv) > 3 else "auto"

# --- cross sections (Hageman & Yasinsky 1969; FEMFFUSION 2D_TWIGL) -----------
seed_xs = dict(diffusion=1 / (3 * np.array([0.238095, 0.83333])),
               sigma_a=[0.010, 0.150], nu_sigma_f=[0.007, 0.200],
               sigma_s=[[0.0, 0.01], [0.0, 0.0]], chi=[1, 0])
blanket_xs = dict(diffusion=1 / (3 * np.array([0.25641, 0.66667])),
                  sigma_a=[0.008, 0.050], nu_sigma_f=[0.003, 0.060],
                  sigma_s=[[0.0, 0.01], [0.0, 0.0]], chi=[1, 0])

blanket = Material(name="blanket", **blanket_xs)
seed = Material(name="seed", **seed_xs)
seed_c = Material(name="seed (perturbable)", **seed_xs)   # distinct object -> its own region

# --- geometry: three rectangular regions in an 80 cm quarter core ------------
model = (
    ndgpu.Model(size=(80, 80), cells=(10 * r, 10 * r))
    .fill(blanket)
    .add_box(seed, x=(0, 24), y=(24, 56)).add_box(seed, x=(24, 56), y=(0, 24))   # L-shaped seed
    .add_box(seed_c, x=(24, 56), y=(24, 56))                                     # central seed
    .set_boundary(x=("reflective", "zero-flux"), y=("reflective", "zero-flux"))
    .set_kinetics(velocities=[1.0e7, 2.0e5], beta=[0.0075], decay=[0.08])
)

# --- static eigenvalue -------------------------------------------------------
k = model.run(device=device, tol_k=1e-8, tol_source=1e-7).k_eff
print(f"TWIGL static k_eff : {k:.5f}   (published 0.91321, {(k - 0.91321) * 1e5:+.0f} pcm)\n")

# --- transient ---------------------------------------------------------------
def perturbed(f):
    return Material(name="seed (perturbed)", **{**seed_xs, "sigma_a": [0.010, 0.150 * f]})

def materials_at(t):
    if kind == "step":
        f = 0.976667 if t > 0.0 else 1.0
    else:                                   # ramp: -11.667% over the first 0.2 s
        f = 1.0 - 0.11667 * min(max(t, 0.0), 0.2)
    return [blanket, seed, perturbed(round(f, 12))]

result = model.transient(t_end=0.5, dt=0.005, materials_at=materials_at, device=device)
print(result)

at = lambda tt: result.power[np.argmin(np.abs(result.times - tt))]
ref = {"step": (2.06, 2.13), "ramp": (1.31, 2.11)}[kind]
print(f"\n  {kind} transient vs published:")
print(f"    P(0.1 s) = {at(0.1):.2f}   (published {ref[0]})")
print(f"    P(0.5 s) = {at(0.5):.2f}   (published {ref[1]})")
