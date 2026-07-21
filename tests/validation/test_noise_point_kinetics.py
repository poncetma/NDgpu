"""Frequency-domain neutron noise validated against point-kinetics theory.

For a homogeneous, fully reflected (leakage-free) reactor the fundamental mode
is spatially flat, so a spatially uniform absorption fluctuation drives a flat
response whose complex amplitude is *exactly* the zero-power reactor transfer
function times the reactivity of the perturbation. In one energy group point
kinetics is exact (no spectral distortion), so the neutron-noise solver must
reproduce delta-phi/phi_0 = G(w) * delta-rho to solver precision across the
whole frequency range -- pinning the i w/v term, the delayed-neutron feedback
folded into chi_eff(w), and the noise-source construction.
"""

import numpy as np
import pytest

from ndgpu import (Grid, Kinetics, Material, NoiseSolver, NoiseSource,
                   zero_power_transfer_function)


def _critical_one_group():
    """One-group homogeneous reactor, reflective on all faces. nuSigma_f =
    Sigma_a gives k_inf = 1 and, with no leakage, an exactly critical, flat
    fundamental -- the setting where point kinetics is rigorously exact."""
    mat = Material(name="crit", diffusion=[1.2], sigma_a=[0.020],
                   nu_sigma_f=[0.020])
    grid = Grid(shape=(3, 3, 1), size=(15.0, 15.0, 15.0))
    return mat, grid, "reflective"


@pytest.mark.parametrize("angular", ["diffusion", "sp3", "sp5"])
@pytest.mark.parametrize("families", [
    ([0.0065], [0.0784]),                         # one delayed family
    ([0.00021, 0.00142, 0.00127, 0.00257, 0.00075, 0.00027],
     [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01]),  # six-family U-235
])
def test_zero_power_transfer_function(families, angular):
    beta, decay = families
    mat, grid, bc = _critical_one_group()
    kin = Kinetics(velocities=[2.2e5], beta=beta, decay=decay)
    ns = NoiseSolver(grid, mat, kinetics=kin, bc=bc, angular=angular)
    assert abs(ns.k_eff - 1.0) < 1e-9
    Lam = ns.generation_time()

    # Reactivity of a uniform absorption bump, in the critically-scaled (F/k)
    # normalization the noise solver uses: k * (1/k - 1/k'). Finite difference
    # on the eigenvalue -- solver-agnostic, so it also covers the SPN moment
    # solvers (first_order_reactivity handles only scalar-flux solvers). Both
    # the reactivity (exactly -eps/Sigma_a for this 1-group reflective medium)
    # and the noise are linear in eps, so it cancels in the ratio below; eps is
    # only kept large enough (1e-3) for the eigenvalue change to be resolvable.
    eps = 1e-3
    matp = Material(name="p", diffusion=[1.2], sigma_a=[0.020 + eps],
                    nu_sigma_f=[0.020])
    k_pert = NoiseSolver(grid, matp, kinetics=kin, bc=bc, angular=angular).k_eff
    d_rho = ns.k_eff * (1.0 / ns.k_eff - 1.0 / k_pert)

    src = NoiseSource(d_sigma_a=[eps])
    for f in (0.02, 0.2, 2.0, 20.0, 200.0):
        w = 2.0 * np.pi * f
        res = ns.solve(src, w, tol=1e-11)
        assert res.converged
        rel = res.relative()[0]
        amp = complex(np.mean(rel))
        # Response is spatially flat (no spurious spatial mode).
        assert float(np.max(np.abs(rel - amp))) / abs(amp) < 1e-9
        predicted = zero_power_transfer_function(w, kin, Lam) * d_rho
        assert abs(amp / predicted - 1.0) < 1e-6      # magnitude and phase


def test_high_frequency_prompt_asymptote():
    """Well above the delayed-neutron break frequencies the response follows the
    prompt transfer function G -> 1/(i w Lambda): |delta-phi/phi| ~ |d_rho|/(w
    Lambda), a slope of -1 in log-log. Checks that chi_eff(w) sheds the delayed
    feedback at high frequency."""
    mat, grid, bc = _critical_one_group()
    kin = Kinetics(velocities=[2.2e5], beta=[0.0065], decay=[0.0784])
    ns = NoiseSolver(grid, mat, kinetics=kin, bc=bc)
    Lam = ns.generation_time()
    eps = 1e-6
    src = NoiseSource(d_sigma_a=[eps])

    f_hi = [1.0e3, 1.0e4]
    amps = []
    for f in f_hi:
        res = ns.solve(src, 2.0 * np.pi * f, tol=1e-11)
        amps.append(abs(complex(np.mean(res.relative()[0]))))
        # Prompt asymptote |d_rho|/(w Lambda) (d_rho magnitude cancels in slope).
    slope = np.log(amps[1] / amps[0]) / np.log(f_hi[1] / f_hi[0])
    assert abs(slope + 1.0) < 1e-3                     # -1 decade per decade


def test_localized_source_global_to_local_transition():
    """A localized absorber in a heterogeneous near-critical core: the complex
    fixed point converges at every frequency (Anderson near criticality), and
    the response shape follows the static fundamental at low frequency (global
    point-kinetics response) but progressively localizes around the
    perturbation as the frequency rises -- the hallmark global-to-local
    transition of neutron noise."""
    from ndgpu.benchmarks import build_twigl
    p = build_twigl("none", cells_per_8cm=2)
    mats, mmap = p.problem_at(0.0)
    ns = NoiseSolver(p.grid, mats, mmap, kinetics=p.kinetics, bc=p.bc)
    nx, ny, _ = p.grid.shape
    damp = np.zeros(p.grid.shape, dtype=complex)
    damp[nx // 2, ny // 2, 0] = 5e-4
    src = NoiseSource(d_sigma_a=[np.zeros(p.grid.shape), damp])
    phi0 = np.asarray(ns.flux0[1]).ravel()

    def shape_similarity(res):
        d = np.abs(np.asarray(res.d_flux_numpy()[1]).ravel())
        return float(np.dot(d, phi0) / (np.linalg.norm(d) * np.linalg.norm(phi0)))

    cos = {}
    for f in (0.01, 1.0, 100.0):
        res = ns.solve(src, 2.0 * np.pi * f, tol=1e-8)
        assert res.converged
        cos[f] = shape_similarity(res)

    assert cos[0.01] > 0.99                 # low frequency: fundamental shape
    assert cos[100.0] < cos[0.01] - 0.02    # localizes as frequency rises
