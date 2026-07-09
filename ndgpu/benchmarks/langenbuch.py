"""3D Langenbuch (LMW) operational transient benchmark.

Small 3D LWR (Langenbuch, Maurer & Werner; ANL-7416 Suppl. 2 collection):
220 x 220 x 200 cm, 11 x 11 x 10 cells of 20 cm, two groups, six delayed
precursor families, zero-flux boundaries. Two control-rod banks move during
the transient:

  - bank 1 (4 rods), initially half inserted (tips at z = 100 cm), withdraws
    at 3 cm/s from t = 0 until fully out at t = 26.7 s;
  - bank 2 (5 rods), initially withdrawn, inserts at 3 cm/s from t = 7.5 s
    to t = 47.5 s (tips from 180 down to 60 cm).

Power first rises with the bank-1 withdrawal, then falls as bank 2 drives the
core subcritical — the classic delayed-critical operational transient.

Rods hang from z = 180 cm (the top 20 cm layer is reflector, with permanently
rodded guide positions). A rodded cell takes the "absorber" composition of
its base material; the cell containing a rod tip gets a volume-weighted mix
(fraction of the cell height covered). Data transcribed from the FEMFFUSION
repository (examples/3D_Langenbuch, https://github.com/Zonni/FEMFFUSION).
The four excised corner cells of the octagonal plant map are filled with
reflector (the reference excludes them; the difference sits deep in the
reflector next to a zero-flux boundary).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid
from ..materials import Kinetics, Material

LANGENBUCH_KINETICS = Kinetics(
    velocities=[1.25e7, 2.5e5],
    beta=[0.000247, 0.0013845, 0.001222, 0.0026455, 0.000832, 0.000169],
    decay=[0.0127, 0.0317, 0.115, 0.311, 1.4, 3.87],
)

# Literature reference: relative power peaks ~1.6x near t = 21 s
# (bank-1 withdrawal), then falls as bank 2 inserts.
PEAK_REFERENCE = (21.0, 1.6)

# (sigma_tr, sigma_a, nu_sigma_f) per group; sigma_12 is the downscatter.
_XS = {
    "reflector":       dict(tr=[0.20397003, 1.26261670], sa=[0.00266057, 0.04936351],
                            nsf=[0.0, 0.0], s12=0.02759693),
    "fuel 2":          dict(tr=[0.23381787, 0.95082160], sa=[0.01099263, 0.09925634],
                            nsf=[0.00750328, 0.1378004], s12=0.01717768),
    "fuel 1":          dict(tr=[0.23409670, 0.93552546], sa=[0.01040206, 0.08766217],
                            nsf=[0.00647769, 0.1127328], s12=0.01755550),
    "fuel 1 rodded":   dict(tr=[0.23409670, 0.93552546], sa=[0.01095206, 0.09146217],
                            nsf=[0.00647769, 0.1127328], s12=0.01755550),
    "reflector rodded": dict(tr=[0.20397003, 1.26261670], sa=[0.00321050, 0.05316351],
                             nsf=[0.0, 0.0], s12=0.02759693),
}
# indices in the material list
REFLECTOR, FUEL2, FUEL1, FUEL1_ROD, REFLECTOR_ROD = range(5)

# Rod plan positions (0-indexed row, col in the 11 x 11 map) per bank.
BANK_POSITIONS = {
    1: [(2, 5), (5, 2), (5, 8), (8, 5)],
    2: [(3, 3), (3, 7), (5, 5), (7, 3), (7, 7)],
}
ROD_TOP = 180.0


def rod_tip(bank: int, t: float) -> float:
    """Height of the rod bottom tip (cm); the rod occupies [tip, 180]."""
    if bank == 1:
        return min(100.0 + 3.0 * max(t, 0.0), 180.0)
    if t <= 7.5:
        return 180.0
    return max(180.0 - 3.0 * (t - 7.5), 60.0)


def _material(name: str, xs, f_rod: float = 0.0, rodded=None) -> Material:
    """Build a Material, optionally mixed with its rodded composition."""
    mix = lambda key: (1 - f_rod) * np.asarray(xs[key]) + f_rod * np.asarray(rodded[key]) \
        if f_rod else np.asarray(xs[key])
    tr, sa, nsf, s12 = mix("tr"), mix("sa"), mix("nsf"), mix("s12")
    return Material(name=name, diffusion=1.0 / (3.0 * tr), sigma_a=sa,
                    nu_sigma_f=nsf, sigma_s=[[0.0, float(s12)], [0.0, 0.0]],
                    chi=[1.0, 0.0], total=tr)


@dataclass
class LangenbuchProblem:
    grid: Grid
    kinetics: Kinetics
    bc: str
    problem_at: object  # callable t -> (materials, material_map)


def build_langenbuch(refine: int = 1) -> LangenbuchProblem:
    """Assemble the 3D Langenbuch core at 20/refine cm resolution."""
    r = refine
    grid = Grid(shape=(11 * r, 11 * r, 10 * r), size=(220.0, 220.0, 200.0))
    base_mats = [_material(k, v) for k, v in _XS.items()]

    # Base map (no moving rods): bottom & top layers reflector, fuel-2 ring
    # around a fuel-1 zone in planes 2-9, rod guides in the top layer rodded.
    plan = np.full((11, 11), REFLECTOR, dtype=np.int64)
    plan[1:10, 2:9] = FUEL2
    plan[2:9, 1:10] = FUEL2
    plan[2:9, 3:8] = FUEL1
    plan[3:8, 2:9] = FUEL1
    base = np.empty((11, 11, 10), dtype=np.int64)
    base[:, :, 0] = REFLECTOR
    base[:, :, 9] = REFLECTOR
    for positions in BANK_POSITIONS.values():
        for (i, j) in positions:
            base[i, j, 9] = REFLECTOR_ROD
    for k in range(1, 9):
        base[:, :, k] = plan

    # Refined base map and axial cell edges.
    base = np.kron(base, np.ones((r, r, r), dtype=np.int64))
    dz = 20.0 / r
    z_lo = np.arange(10 * r) * dz

    cache: dict[tuple, tuple] = {}

    def problem_at(t: float):
        tips = (round(rod_tip(1, t), 9), round(rod_tip(2, t), 9))
        if tips in cache:
            return cache[tips]
        mats = list(base_mats)
        mmap = base.copy()
        for bank, tip in zip((1, 2), tips):
            # Rod coverage fraction of each axial cell, [0, 1].
            frac = np.clip((z_lo + dz - tip) / dz, 0.0, 1.0)
            frac[z_lo >= ROD_TOP] = 0.0  # above the guide top: handled by base map
            partial_idx = {}  # axial index -> material index of the mix
            for k in np.nonzero((frac > 0) & (frac < 1))[0]:
                mats.append(_material(f"fuel 1 rodded f={frac[k]:.4f}",
                                      _XS["fuel 1"], f_rod=float(frac[k]),
                                      rodded=_XS["fuel 1 rodded"]))
                partial_idx[int(k)] = len(mats) - 1
            for (i, j) in BANK_POSITIONS[bank]:
                for k in np.nonzero(frac > 0)[0]:
                    cells = np.s_[i * r:(i + 1) * r, j * r:(j + 1) * r, k]
                    if not np.all(mmap[cells] == FUEL1):
                        continue  # only fuel-1 positions are rodded
                    mmap[cells] = FUEL1_ROD if frac[k] >= 1.0 else partial_idx[int(k)]
        cache[tips] = (mats, mmap)
        return cache[tips]

    return LangenbuchProblem(grid=grid, kinetics=LANGENBUCH_KINETICS,
                             bc="zero-flux", problem_at=problem_at)
