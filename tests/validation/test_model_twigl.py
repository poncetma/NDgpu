"""The TWIGL benchmark built through the Model API, checked against literature.

This is an end-to-end validation of the high-level Model API -- region painting,
boundary conditions, the steady solve, kinetics and the transient -- against the
published TWIGL seed-blanket references (Hageman & Yasinsky, 1969): the static
eigenvalue and the transient power at 0.1 s / 0.5 s for the step and ramp
perturbations. The cross sections match the FEMFFUSION 2D_TWIGL example, the same
data the low-level benchmark (ndgpu.benchmarks.twigl) is transcribed from, so a
mismatch would flag a wrapper bug rather than a data disagreement.
"""
import numpy as np
import pytest

import ndgpu
from ndgpu import Material

_SEED = dict(diffusion=1 / (3 * np.array([0.238095, 0.83333])),
             sigma_a=[0.010, 0.150], nu_sigma_f=[0.007, 0.200],
             sigma_s=[[0.0, 0.01], [0.0, 0.0]], chi=[1, 0])
_BLANKET = dict(diffusion=1 / (3 * np.array([0.25641, 0.66667])),
                sigma_a=[0.008, 0.050], nu_sigma_f=[0.003, 0.060],
                sigma_s=[[0.0, 0.01], [0.0, 0.0]], chi=[1, 0])


def _twigl(r=4):
    blanket = Material(name="blanket", **_BLANKET)
    seed = Material(name="seed", **_SEED)
    seed_c = Material(name="seed-c", **_SEED)         # distinct object -> its own region
    model = (ndgpu.Model(size=(80, 80), cells=(10 * r, 10 * r))
             .fill(blanket)
             .add_box(seed, x=(0, 24), y=(24, 56)).add_box(seed, x=(24, 56), y=(0, 24))
             .add_box(seed_c, x=(24, 56), y=(24, 56))
             .set_boundary(x=("reflective", "zero-flux"), y=("reflective", "zero-flux"))
             .set_kinetics(velocities=[1.0e7, 2.0e5], beta=[0.0075], decay=[0.08]))
    return model, blanket, seed


def test_twigl_static_eigenvalue_matches_literature():
    # Published static k = 0.91321; the FV diffusion value converges to it from
    # below as the mesh refines (2nd order).
    k_coarse = _twigl(2)[0].run(tol_k=1e-9, tol_source=1e-8).k_eff
    k_fine = _twigl(8)[0].run(tol_k=1e-9, tol_source=1e-8).k_eff
    assert k_fine == pytest.approx(0.91321, abs=1e-4)     # within 10 pcm at 1 cm cells
    assert abs(k_fine - 0.91321) < abs(k_coarse - 0.91321)  # refining approaches the reference


@pytest.mark.parametrize("kind,p01,p05", [("step", 2.06, 2.13), ("ramp", 1.31, 2.11)])
def test_twigl_transient_power_matches_literature(kind, p01, p05):
    model, blanket, seed = _twigl(4)

    def perturbed(f):
        return Material(name="pert", **{**_SEED, "sigma_a": [0.010, 0.150 * f]})

    def materials_at(t):
        f = (0.976667 if (kind == "step" and t > 0) else
             1.0 - 0.11667 * min(max(t, 0.0), 0.2))
        return [blanket, seed, perturbed(round(f, 12))]

    res = model.transient(t_end=0.5, dt=0.005, materials_at=materials_at)
    at = lambda tt: res.power[np.argmin(np.abs(res.times - tt))]
    assert at(0.1) == pytest.approx(p01, abs=0.02)
    assert at(0.5) == pytest.approx(p05, abs=0.02)
    assert res.k0 == pytest.approx(res.steady.k_eff, abs=1e-12)   # transient carries its steady state
