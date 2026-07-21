"""Cross-check ndgpu's frequency-domain noise solver against FEMFFUSION.

Replicates FEMFFUSION's own noise regression case (test/1D_noise_SPN: a 1D
two-group slab, 1 Hz absorption fluctuation in one cell) and compares the
complex flux noise field by field against FEMFFUSION's output, for both angular
approximations:

  - diffusion vs FEMFFUSION's diffusion noise;
  - SP3 (with the moment-coupled Marshak vacuum) vs FEMFFUSION's Full_SPN N=3.

The two codes share the noise formulation (critical adjustment, omega = 2*pi*f,
the identical chi_eff(w), and -- for SP3 -- the coupled Marshak vacuum boundary),
so they differ only in spatial discretization (ndgpu cell-centred finite volume
vs FEMFFUSION continuous-Galerkin finite elements). At the reference 60-cell mesh
they agree to a fraction of a percent, and the difference falls at ~second order
under refinement (verified separately at 120 cells: delta-phi L2 diffusion
4.1e-3 -> 1.0e-3, SP3 3.3e-3 -> 0.8e-3).
"""

import numpy as np
import pytest

from ndgpu import NoiseSolver, NoiseSource
from ndgpu.benchmarks import build_femffusion_1d_noise


@pytest.mark.parametrize("angular", ["diffusion", "sp3"])
def test_femffusion_1d_2group_noise(angular):
    bench = build_femffusion_1d_noise(cells=60, angular=angular)
    ns = NoiseSolver(bench.grid, bench.materials, bench.material_map,
                     kinetics=bench.kinetics, bc=bench.bc, angular=bench.angular,
                     marshak_vacuum=bench.marshak_vacuum)

    # Static eigenvalue matches FEMFFUSION to ~1 pcm.
    assert abs(ns.k_eff - bench.k_eff_ref) < 5e-5

    res = ns.solve(NoiseSource(d_sigma_a=bench.d_sigma_a),
                   2.0 * np.pi * bench.frequency_hz, tol=1e-10)
    assert res.converged

    # delta-phi scales with the static-flux normalization, which each code sets
    # independently; match it once (least squares on the group-1 static flux)
    # so the absolute complex noise can be compared directly.
    phi1 = np.asarray(ns.flux0[0]).ravel()
    scale = np.dot(bench.static_flux_ref, phi1) / np.dot(phi1, phi1)
    for g in range(2):
        d = scale * np.asarray(res.d_flux[g]).ravel()
        ref = bench.d_flux_ref[g]
        rel_l2 = np.linalg.norm(d - ref) / np.linalg.norm(ref)
        assert rel_l2 < 1e-2                        # FV vs FE, 60-cell mesh
        # Peak amplitude and phase at the perturbation cell -- the source
        # singularity, where the FV/FE difference is largest (it falls to
        # <~1% at 120 cells); the smooth-field L2 above is the tight metric.
        ip = int(np.argmax(np.abs(ref)))
        assert abs(abs(d[ip]) / abs(ref[ip]) - 1.0) < 0.03
        assert abs(np.degrees(np.angle(d[ip] / ref[ip]))) < 0.1
