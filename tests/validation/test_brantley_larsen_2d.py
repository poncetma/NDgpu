"""SPN / SDPN validation against Carreno et al. (2024) Table 6 / Fig. 3.

The 2D one-group Brantley-Larsen problem: three fuel bars (1 cm wide, 9 cm tall)
at x in [1,2],[4,5],[7,8], y in [0,9], in a 10x10 cm moderator square;
reflective at x=0 and y=0, vacuum at x=10 and y=10. Every material interface is
grid-aligned, so ndgpu's 2nd-order FV converges cheaply and the eigenvalue is a
*physical* check of the angular coefficients -- unlike the k_infinity / dense
tests, which only pin internal consistency. This problem caught a sign typo in
the paper's SDP2 c^(1) closure coefficient (Appendix A.3).

ndgpu sits a uniform +45..+105 pcm above the paper for every method (its
per-moment Robin vacuum vs the paper's coupled Marshak boundary), and reproduces
the full ordering, including the headline SDPN-beats-SPN-at-equal-DoF result.
"""

import numpy as np
import pytest

from ndgpu import (Grid, Material, DiffusionEigenSolver, SP3EigenSolver,
                   SP5EigenSolver, SP7EigenSolver, SDP1EigenSolver,
                   SDP2EigenSolver, SDP3EigenSolver)

PAPER = {"SP1": 0.77680, "SP3": 0.79904, "SP5": 0.80280, "SP7": 0.80354,
         "SDP1": 0.80161, "SDP2": 0.80373, "SDP3": 0.80402}
SOLVER = {"SP1": DiffusionEigenSolver, "SP3": SP3EigenSolver,
          "SP5": SP5EigenSolver, "SP7": SP7EigenSolver, "SDP1": SDP1EigenSolver,
          "SDP2": SDP2EigenSolver, "SDP3": SDP3EigenSolver}

FUEL = Material(name="fuel", diffusion=[1.0 / 4.5], sigma_a=[0.15],
                nu_sigma_f=[0.24], total=[1.5], chi=[1.0])
MOD = Material(name="mod", diffusion=[1.0 / 3.0], sigma_a=[0.07],
               nu_sigma_f=[0.0], total=[1.0], chi=[1.0])


def _problem(n=80):
    h = 10.0 / n
    xc = (np.arange(n) + 0.5) * h
    in_bar = ((xc > 1) & (xc < 2)) | ((xc > 4) & (xc < 5)) | ((xc > 7) & (xc < 8))
    mmap = np.where(in_bar[:, None] & (xc < 9.0)[None, :], 0, 1).astype(int)
    grid = Grid(shape=(n, n, 1), size=(10.0, 10.0, h))
    bc = (("reflective", "vacuum"), ("reflective", "vacuum"), "reflective")
    return grid, mmap[:, :, None], bc


@pytest.fixture(scope="module")
def keff():
    grid, mmap, bc = _problem(80)
    out = {}
    for name, cls in SOLVER.items():
        r = cls(grid, [FUEL, MOD], material_map=mmap, bc=bc,
                device="cpu").solve(tol_k=1e-8, tol_source=1e-7)
        assert r.converged, (name, r)
        out[name] = r.k_eff
    return out


@pytest.mark.parametrize("name", list(PAPER))
def test_keff_matches_paper(keff, name):
    """Each method within ~150 pcm of Table 6 (the residual is the consistent
    vacuum-BC offset). A gross coefficient error -- e.g. the SDP2 c^(1) typo --
    shows here as a >1000 pcm miss."""
    assert keff[name] == pytest.approx(PAPER[name], abs=1.5e-3), \
        f"{name}: {keff[name]:.5f} vs paper {PAPER[name]:.5f}"


def test_offset_is_uniform(keff):
    """The ndgpu-minus-paper gap is the same small positive vacuum-BC offset for
    every method (no method is an outlier)."""
    d = np.array([(keff[n] - PAPER[n]) * 1e5 for n in PAPER])
    assert np.all(d > 0) and np.all(d < 150), dict(zip(PAPER, d))
    assert d.max() - d.min() < 80          # tight spread => consistent physics


def test_hierarchy_ordering(keff):
    """SPN and SDPN both converge monotonically upward toward the transport k."""
    assert keff["SP1"] < keff["SP3"] < keff["SP5"] < keff["SP7"]
    assert keff["SDP1"] < keff["SDP2"] < keff["SDP3"]


def test_sdpn_beats_spn_at_matched_dofs(keff):
    """The paper's headline: at equal degrees of freedom the simplified double-PN
    eigenvalue is closer to the transport reference than SPN."""
    assert keff["SDP1"] > keff["SP3"]      # 2-moment blocks
    assert keff["SDP2"] > keff["SP5"]      # 3-moment blocks
    assert keff["SDP3"] > keff["SP7"]      # 4-moment blocks


# --- coupled Marshak vacuum boundary (matches the paper's exact BC) ------------

def test_marshak_diagonal_g_reduces_to_per_moment():
    """The coupled-boundary machinery with a diagonal g = 1/2 I is identical to
    the default per-moment Robin vacuum -- a correctness check on the K formula
    (K reduces to robin_face_term when the moments decouple)."""
    grid, mmap, bc = _problem(48)
    k_pm = SP5EigenSolver(grid, [FUEL, MOD], material_map=mmap, bc=bc,
                          device="cpu").solve(tol_k=1e-8, tol_source=1e-7).k_eff

    class SP5diag(SP5EigenSolver):
        _g_coeffs = {2: [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]}

    k_diag = SP5diag(grid, [FUEL, MOD], material_map=mmap, bc=bc, device="cpu",
                     marshak_vacuum=True).solve(tol_k=1e-8, tol_source=1e-7).k_eff
    assert k_diag == pytest.approx(k_pm, abs=1e-6)


def test_marshak_converges_to_paper():
    """The exact coupled Marshak boundary makes SPN/SDPN converge to the paper's
    Table 6 k_eff (per-moment converges tens of pcm off). Checked by 2nd-order
    Richardson from two meshes, for a representative SP and SDP method."""
    from ndgpu.solver import SPNEigenSolver, SDPNEigenSolver

    class SP3g(SPNEigenSolver):
        _order = 1

    class SDP1g(SDPNEigenSolver):
        _order = 1

    for cls, paper in [(SP3g, PAPER["SP3"]), (SDP1g, PAPER["SDP1"])]:
        ks = []
        for n in (120, 240):
            grid, mmap, bc = _problem(n)
            ks.append(cls(grid, [FUEL, MOD], material_map=mmap, bc=bc,
                          device="cpu", marshak_vacuum=True).solve(
                tol_k=1e-9, tol_source=1e-8).k_eff)
        kinf = ks[1] + (ks[1] - ks[0]) / 3.0        # h -> 0, 2nd order (ratio 2)
        assert kinf == pytest.approx(paper, abs=1e-4), \
            f"{cls.__name__}: extrapolated {kinf:.5f} vs paper {paper:.5f}"
