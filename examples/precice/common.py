"""Shared setup for the two preCICE participants.

Deliberately thin, and deliberately the ONLY place either participant builds
anything. Both processes must construct a bit-identical problem -- same mesh,
same drum angle, same active mask, same ravel order -- because the
nearest-neighbour mapping between them is exact only for identical vertex sets.
Building it twice from one function is what keeps that true.

Neither participant contains physics: they call ``neutronics_step`` and
``thermal_step`` from :mod:`ndgpu.coupling`, the same functions the internal
:class:`~ndgpu.coupling.CoupledSolver` calls. That is the whole point of the
cross-verification -- if each script re-implemented its own half, the two
couplings agreeing would only show that the same author made the same
assumptions twice.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from ndgpu.benchmarks.hpmr import build_hpmr2d, build_hpmr3d
from ndgpu.benchmarks.hpmr_thermal import build_hpmr_coupling, hpmr_endfb8_builtin
from ndgpu.coupling import coupling_vertices


def parse_args(participant: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"ndgpu preCICE participant: {participant}")
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--nz", type=int, default=0,
                    help="axial layers; 0 (default) = the 2D radial core")
    ap.add_argument("--drum-deg", type=float, default=180.0)
    ap.add_argument("--groups", choices=("2", "11"), default="11")
    ap.add_argument("--power", type=float, default=None,
                    help="rated thermal power in W (default: the design 2 MWt)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                                     "precice-config.xml"))
    ap.add_argument("--csv", default=None,
                    help="write the per-iteration trace here")
    ap.add_argument("--warm-start", action="store_true",
                    help="reuse the previous flux. OFF by default: it makes "
                         "the coupled map depend on its own history, which "
                         "breaks the lockstep comparison with the internal "
                         "driver (and preCICE's checkpoints restore to the "
                         "start of a window, not the previous iteration).")
    return ap.parse_args()


def build_context(args):
    """The problem and its coupling context -- identical in both processes."""
    materials = hpmr_endfb8_builtin(three_d=args.nz > 0) if args.groups == "11" else None
    if args.nz > 0:
        problem = build_hpmr3d(refine=args.refine, nz=args.nz,
                               drum_angle_deg=args.drum_deg,
                               absorber="polar", materials=materials)
    else:
        problem = build_hpmr2d(refine=args.refine, drum_angle_deg=args.drum_deg,
                               absorber="polar", materials=materials)
    kw = {} if args.power is None else {"power_w": args.power}
    ctx = build_hpmr_coupling(problem, warm_start=args.warm_start, **kw)
    coords, flat_idx = coupling_vertices(problem)
    return problem, ctx, coords, flat_idx


def scatter(values, flat_idx, shape, fill):
    """preCICE vertex values -> a full grid field, ``fill`` off the active set."""
    field = np.full(int(np.prod(shape)), float(fill))
    field[flat_idx] = values
    return field.reshape(shape)


def gather(field, flat_idx):
    """A full grid field -> the active-cell vertex values preCICE exchanges."""
    return np.ascontiguousarray(np.asarray(field).reshape(-1)[flat_idx])


class Trace:
    """Per-iteration record, written as CSV for the comparison test."""

    def __init__(self, path, columns):
        self.path = path
        self.columns = list(columns)
        self.rows = []

    def add(self, **kw):
        self.rows.append([kw.get(c, "") for c in self.columns])

    def write(self):
        if self.path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".",
                    exist_ok=True)
        with open(self.path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.columns)
            w.writerows(self.rows)


def check_version():
    import precice

    version = precice.get_version_information()
    if isinstance(version, bytes):
        version = version.decode()
    major = version.split(".")[0]
    if not major.startswith("3"):
        raise SystemExit(f"these participants target preCICE 3.x, found {version}")
    return version
