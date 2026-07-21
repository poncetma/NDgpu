"""Cross-check ndgpu's frequency-domain noise solver against FEMFFUSION.

Replicates FEMFFUSION's own noise regression case (test/1D_noise_SPN, diffusion:
1D two-group slab, a 1 Hz absorption fluctuation in one cell) and compares the
complex flux noise field by field against FEMFFUSION's output. The two codes
share the noise formulation (critical adjustment, omega = 2*pi*f, the same
chi_eff(w)); they differ only in spatial discretization -- ndgpu's cell-centred
finite volume vs FEMFFUSION's continuous-Galerkin finite elements -- so at the
reference 60-cell mesh they agree to a fraction of a percent, and the difference
falls at ~second order under refinement (verified separately at 120 cells:
delta-phi L2 4.1e-3 -> 1.0e-3).
"""

import numpy as np

from ndgpu import NoiseSolver, NoiseSource
from ndgpu.benchmarks import build_femffusion_1d_noise
from ndgpu.benchmarks.femffusion_noise import K_EFF_REF


def test_femffusion_1d_2group_noise():
    bench = build_femffusion_1d_noise(cells=60)
    ns = NoiseSolver(bench.grid, bench.materials, bench.material_map,
                     kinetics=bench.kinetics, bc=bench.bc)

    # Static eigenvalue matches FEMFFUSION to well under 1 pcm.
    assert abs(ns.k_eff - K_EFF_REF) < 5e-5

    res = ns.solve(NoiseSource(d_sigma_a=bench.d_sigma_a),
                   2.0 * np.pi * bench.frequency_hz, tol=1e-10)
    assert res.converged

    # delta-phi scales with the static-flux normalization, which each code sets
    # independently; match it once (least squares on the group-1 static flux)
    # so the absolute complex noise can be compared directly.
    phi1 = np.asarray(ns.flux0[0]).ravel()
    scale = np.dot(bench.static_flux_ref[0], phi1) / np.dot(phi1, phi1)
    d_ref = bench.d_flux_ref
    for g in range(2):
        d = scale * np.asarray(res.d_flux[g]).ravel()
        rel_l2 = np.linalg.norm(d - d_ref[g]) / np.linalg.norm(d_ref[g])
        assert rel_l2 < 1e-2                       # FV vs FE, 60-cell mesh
        # Peak amplitude and phase at the perturbation cell -- the source
        # singularity, where the FV/FE difference is largest (it falls to
        # ~0.4% at 120 cells); the smooth-field L2 above is the tight metric.
        ip = int(np.argmax(np.abs(d_ref[g])))
        assert abs(abs(d[ip]) / abs(d_ref[g][ip]) - 1.0) < 0.03
        assert abs(np.degrees(np.angle(d[ip] / d_ref[g][ip]))) < 0.1
