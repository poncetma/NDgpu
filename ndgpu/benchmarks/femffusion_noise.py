"""1D two-group neutron-noise cross-check against FEMFFUSION.

Replicates FEMFFUSION's own noise regression case ``test/1D_noise_SPN`` (the
diffusion variant) so ndgpu's frequency-domain solver can be compared field by
field against an independent code:

  - 300 cm homogeneous slab, 60 cells of 5 cm, vacuum (Marshak) on both ends;
  - two energy groups, one delayed family;
  - a stationary absorption fluctuation delta-Sigma_a in the single cell 24
    (both groups), oscillating at 1 Hz.

Geometry, cross sections, kinetics and the perturbation are transcribed from
the FEMFFUSION repository (https://github.com/Zonni/FEMFFUSION,
test/1D_noise_SPN/{1D.xsec, 1D.dxs, 1D.dyn.prm}). The reference flux noise
:data:`DPHI1_REF` / :data:`DPHI2_REF` is FEMFFUSION's output for that input
(diffusion, continuous-Galerkin FE degree 1), sampled at the 60 cell centres.

Both codes critically adjust the base fission (nu_Sigma_f / k_eff), share the
omega = 2*pi*f convention, and build the same effective fission spectrum
chi_eff,g(w) = (1-beta) chi_p,g + sum_p chi_d,p,g beta_p lambda_p/(lambda_p+iw),
so the two solutions differ only by discretization (finite volume vs finite
element) -- a difference that converges away at second order under refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Kinetics, Material

# --- FEMFFUSION test/1D_noise_SPN cross sections (2 group) -------------------
_SIGMA_TR = np.array([4.67810e-01, 1.09620e-01])
_XS = dict(
    diffusion=1.0 / (3.0 * _SIGMA_TR),
    sigma_a=[1.17659e-02, 1.07186e-01],
    nu_sigma_f=[5.62285e-03, 1.45865e-01],
    sigma_s=[[0.0, 1.60795e-02], [0.0, 0.0]],
    chi=[1.0, 0.0],
)
KINETICS = Kinetics(velocities=[1.25e7, 2.5e5], beta=[0.0065], decay=[0.0784])
FREQUENCY_HZ = 1.0
PERTURBED_CELL = 24                       # 1-based, as in the .xsec material map
DELTA_SIGMA_A = [2.3532e-06, 2.1437e-05]  # per group, real (from 1D.dxs)

# FEMFFUSION reference: eigenvalue, static flux and complex flux noise per group
# at the 60 cell centres (test/1D_noise_SPN/1D_noise_diffusion.out).
K_EFF_REF = 0.982765
PHI1_REF = np.array([2.682485e+00, 5.788550e+00, 8.668114e+00, 1.147263e+01, 1.423399e+01, 1.695461e+01, 1.962971e+01, 2.225282e+01, 2.481718e+01, 2.731605e+01, 2.974285e+01, 3.209117e+01, 3.435483e+01, 3.652784e+01, 3.860449e+01, 4.057928e+01, 4.244701e+01, 4.420275e+01, 4.584187e+01, 4.736005e+01, 4.875327e+01, 5.001786e+01, 5.115049e+01, 5.214817e+01, 5.300826e+01, 5.372850e+01, 5.430698e+01, 5.474219e+01, 5.503296e+01, 5.517854e+01, 5.517854e+01, 5.503296e+01, 5.474218e+01, 5.430698e+01, 5.372850e+01, 5.300826e+01, 5.214817e+01, 5.115049e+01, 5.001786e+01, 4.875327e+01, 4.736005e+01, 4.584187e+01, 4.420275e+01, 4.244701e+01, 4.057928e+01, 3.860449e+01, 3.652784e+01, 3.435483e+01, 3.209117e+01, 2.974284e+01, 2.731605e+01, 2.481718e+01, 2.225282e+01, 1.962970e+01, 1.695460e+01, 1.423399e+01, 1.147263e+01, 8.668114e+00, 5.788549e+00, 2.682485e+00])
PHI2_REF = np.array([5.200310e-01, 8.951276e-01, 1.303714e+00, 1.717717e+00, 2.129376e+00, 2.535958e+00, 2.935982e+00, 3.328293e+00, 3.711831e+00, 4.085578e+00, 4.448546e+00, 4.799778e+00, 5.138346e+00, 5.463358e+00, 5.773955e+00, 6.069319e+00, 6.348670e+00, 6.611271e+00, 6.856429e+00, 7.083497e+00, 7.291877e+00, 7.481018e+00, 7.650422e+00, 7.799642e+00, 7.928283e+00, 8.036007e+00, 8.122529e+00, 8.187621e+00, 8.231111e+00, 8.252885e+00, 8.252885e+00, 8.231111e+00, 8.187621e+00, 8.122529e+00, 8.036007e+00, 7.928283e+00, 7.799641e+00, 7.650422e+00, 7.481018e+00, 7.291877e+00, 7.083497e+00, 6.856429e+00, 6.611270e+00, 6.348670e+00, 6.069319e+00, 5.773955e+00, 5.463358e+00, 5.138346e+00, 4.799778e+00, 4.448546e+00, 4.085578e+00, 3.711831e+00, 3.328293e+00, 2.935982e+00, 2.535958e+00, 2.129376e+00, 1.717717e+00, 1.303714e+00, 8.951276e-01, 5.200310e-01])
DPHI1_REF = np.array([-2.896649e-03+1.360443e-04j, -6.260824e-03+2.935652e-04j, -9.402035e-03+4.396284e-04j, -1.249405e-02+5.819602e-04j, -1.558158e-02+7.222086e-04j, -1.867745e-02+8.605156e-04j, -2.178659e-02+9.966486e-04j, -2.491198e-02+1.130279e-03j, -2.805616e-02+1.261050e-03j, -3.122152e-02+1.388596e-03j, -3.441048e-02+1.512543e-03j, -3.762545e-02+1.632508e-03j, -4.086883e-02+1.748104e-03j, -4.414309e-02+1.858933e-03j, -4.745069e-02+1.964592e-03j, -5.079412e-02+2.064669e-03j, -5.417586e-02+2.158741e-03j, -5.759834e-02+2.246377e-03j, -6.106362e-02+2.327136e-03j, -6.457223e-02+2.400564e-03j, -6.811845e-02+2.466195e-03j, -7.167107e-02+2.523548e-03j, -7.509581e-02+2.572131e-03j, -7.698441e-02+2.611488e-03j, -7.603004e-02+2.641388e-03j, -7.353249e-02+2.661975e-03j, -7.090783e-02+2.673620e-03j, -6.829152e-02+2.676729e-03j, -6.571561e-02+2.671697e-03j, -6.318655e-02+2.658905e-03j, -6.070455e-02+2.638718e-03j, -5.826824e-02+2.611490e-03j, -5.587593e-02+2.577562e-03j, -5.352584e-02+2.537264e-03j, -5.121623e-02+2.490913e-03j, -4.894535e-02+2.438819e-03j, -4.671149e-02+2.381280e-03j, -4.451297e-02+2.318584e-03j, -4.234815e-02+2.251013e-03j, -4.021537e-02+2.178838e-03j, -3.811303e-02+2.102324e-03j, -3.603955e-02+2.021724e-03j, -3.399336e-02+1.937287e-03j, -3.197292e-02+1.849252e-03j, -2.997670e-02+1.757852e-03j, -2.800320e-02+1.663314e-03j, -2.605091e-02+1.565858e-03j, -2.411836e-02+1.465700e-03j, -2.220408e-02+1.363052e-03j, -2.030664e-02+1.258121e-03j, -1.842458e-02+1.151111e-03j, -1.655650e-02+1.042223e-03j, -1.470096e-02+9.316533e-04j, -1.285653e-02+8.195939e-04j, -1.102173e-02+7.062291e-04j, -9.194791e-03+5.917219e-04j, -7.372792e-03+4.761577e-04j, -5.548167e-03+3.593165e-04j, -3.694522e-03+2.397508e-04j, -1.709315e-03+1.110551e-04j])
DPHI2_REF = np.array([-5.636859e-04+2.652009e-05j, -9.718581e-04+4.563955e-05j, -1.419516e-03+6.647105e-05j, -1.877830e-03+8.759202e-05j, -2.339933e-03+1.086107e-04j, -2.804390e-03+1.293901e-04j, -3.271113e-03+1.498561e-04j, -3.740345e-03+1.699497e-04j, -4.212412e-03+1.896151e-04j, -4.687666e-03+2.087965e-04j, -5.166463e-03+2.274376e-04j, -5.649164e-03+2.454812e-04j, -6.136133e-03+2.628690e-04j, -6.627738e-03+2.795416e-04j, -7.124350e-03+2.954382e-04j, -7.626344e-03+3.104971e-04j, -8.134104e-03+3.246550e-04j, -8.648034e-03+3.378470e-04j, -9.168609e-03+3.500069e-04j, -9.696572e-03+3.610666e-04j, -1.023375e-02+3.709562e-04j, -1.078630e-02+3.796030e-04j, -1.137795e-02+3.869296e-04j, -1.172390e-02+3.928554e-04j, -1.151859e-02+3.973373e-04j, -1.106587e-02+4.004054e-04j, -1.065257e-02+4.021273e-04j, -1.025500e-02+4.025679e-04j, -9.867068e-03+4.017868e-04j, -9.487059e-03+3.998410e-04j, -9.114333e-03+3.967854e-04j, -8.748522e-03+3.926731e-04j, -8.389329e-03+3.875553e-04j, -8.036481e-03+3.814815e-04j, -7.689710e-03+3.744994e-04j, -7.348754e-03+3.666554e-04j, -7.013357e-03+3.579942e-04j, -6.683267e-03+3.485592e-04j, -6.358235e-03+3.383926e-04j, -6.038015e-03+3.275351e-04j, -5.722366e-03+3.160262e-04j, -5.411049e-03+3.039041e-04j, -5.103830e-03+2.912061e-04j, -4.800477e-03+2.779681e-04j, -4.500760e-03+2.642251e-04j, -4.204453e-03+2.500111e-04j, -3.911333e-03+2.353593e-04j, -3.621175e-03+2.203021e-04j, -3.333763e-03+2.048713e-04j, -3.048877e-03+1.890980e-04j, -2.766302e-03+1.730128e-04j, -2.485824e-03+1.566457e-04j, -2.207233e-03+1.400264e-04j, -1.930321e-03+1.231842e-04j, -1.654894e-03+1.061486e-04j, -1.380807e-03+8.895202e-05j, -1.108114e-03+7.163926e-05j, -8.376589e-04+5.430636e-05j, -5.734940e-04+3.725751e-05j, -3.326309e-04+2.163845e-05j])


@dataclass
class NoiseBenchmark:
    grid: Grid
    materials: list
    material_map: np.ndarray
    kinetics: Kinetics
    bc: tuple
    d_sigma_a: list          # per-group complex fields (grid shape)
    frequency_hz: float
    static_flux_ref: list    # FEMFFUSION static flux per group (for normalization)
    d_flux_ref: list         # FEMFFUSION complex flux noise per group


def build_femffusion_1d_noise(cells: int = 60) -> NoiseBenchmark:
    """Assemble the FEMFFUSION 1D 2-group noise case on ``cells`` cells.

    The reference field is FEMFFUSION's 60-cell output, so pass cells=60 for a
    like-for-like comparison; finer meshes converge to the same continuum
    solution (the FV/FE difference falls at second order)."""
    n = cells
    grid = Grid(shape=(n, 1, 1), size=(300.0, 5.0, 5.0))
    xc = (np.arange(n) + 0.5) * (300.0 / n)
    # Perturbed physical region = 1-based cell 24 of the 60-cell mesh: [115, 120] cm.
    lo = (PERTURBED_CELL - 1) * 5.0
    reg = (xc >= lo) & (xc < lo + 5.0)
    mmap = np.zeros((n, 1, 1), dtype=np.int64)
    mmap[reg, 0, 0] = 1
    mats = [Material(name="unperturbed", **_XS), Material(name="perturbed", **_XS)]

    d_sigma_a = []
    for g in range(2):
        field = np.zeros((n, 1, 1), dtype=complex)
        field[reg, 0, 0] = DELTA_SIGMA_A[g]
        d_sigma_a.append(field)

    bc = (("vacuum", "vacuum"), "reflective", "reflective")
    return NoiseBenchmark(
        grid=grid, materials=mats, material_map=mmap, kinetics=KINETICS, bc=bc,
        d_sigma_a=d_sigma_a, frequency_hz=FREQUENCY_HZ,
        static_flux_ref=[PHI1_REF, PHI2_REF], d_flux_ref=[DPHI1_REF, DPHI2_REF])
