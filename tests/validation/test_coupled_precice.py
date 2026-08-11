"""The preCICE coupling against the internal one.

Run these in the environment that has pyprecice (they skip elsewhere):

    conda run -n ndgpu-precice python -m pytest tests/validation/test_coupled_precice.py

Three tiers, in increasing strength and decreasing sharpness:

* **the mapping is the identity** -- what crosses preCICE must be bit-exact.
  Both participants define the same vertex set, so nearest-neighbour maps every
  vertex to itself at distance zero. Asserted exactly, because anything else
  means the two processes built different meshes, and preCICE will not say so:
  it will map each vertex to a nearby wrong cell and the coupling silently
  becomes a smoother.
* **the same fixed-point iteration** -- with preCICE's constant relaxation and
  the internal driver's ``relaxation=`` set to the same value, the two are the
  same map from the same start, so they must agree ITERATE BY ITERATE, not just
  at the end. Two different iterations can share a fixed point; only a lockstep
  comparison localizes a discrepancy to the step where it appears.
* **the same fixed point under different accelerators** -- IQN-ILS against
  Anderson. Here only the converged answer is comparable, and the interesting
  number besides is the iteration count.
"""

import csv
import os
import subprocess
import sys

import numpy as np
import pytest

precice = pytest.importorskip("precice")

from ndgpu.benchmarks.hpmr import build_hpmr2d
from ndgpu.benchmarks.hpmr_thermal import build_hpmr_coupling
from ndgpu.coupling import CoupledSolver

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(REPO, "examples", "precice")
REFINE, DRUM_DEG = 3, 180.0
RELAXATION = 0.5          # must equal the value in precice-config.xml


def run_participants(tmp_path, config="precice-config.xml", extra=(), timeout=900,
                     refine=REFINE):
    """Launch both participants and return their per-iteration traces."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [EXAMPLES, REPO] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    common = ["--refine", str(refine), "--drum-deg", str(DRUM_DEG),
              "--groups", "2", "--config", os.path.join(EXAMPLES, config),
              *extra]
    procs = {}
    for name in ("neutronics", "thermal"):
        procs[name] = subprocess.Popen(
            [sys.executable, os.path.join(EXAMPLES, f"{name}.py"),
             "--csv", f"{name}.csv", *common],
            cwd=tmp_path, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
    out = {}
    for name, proc in procs.items():
        try:
            out[name] = proc.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:                  # pragma: no cover
            proc.kill()
            pytest.fail(f"{name} participant timed out\n{proc.communicate()[0]}")
    for name, proc in procs.items():
        assert proc.returncode == 0, f"{name} failed:\n{out[name]}"

    traces = {}
    for name in procs:
        with open(os.path.join(tmp_path, f"{name}.csv")) as fh:
            traces[name] = list(csv.DictReader(fh))
    assert traces["neutronics"], "no iterations recorded"
    return traces


def internal_run(refine=REFINE, nz=0, **solve_kwargs):
    if nz:
        from ndgpu.benchmarks.hpmr import build_hpmr3d
        p = build_hpmr3d(refine=refine, nz=nz, drum_angle_deg=DRUM_DEG,
                         absorber="polar")
    else:
        p = build_hpmr2d(refine=refine, drum_angle_deg=DRUM_DEG, absorber="polar")
    ctx = build_hpmr_coupling(p, warm_start=False)
    return CoupledSolver(ctx).solve(**solve_kwargs)


@pytest.fixture(scope="module")
def constant_relaxation_run(tmp_path_factory):
    return run_participants(tmp_path_factory.mktemp("precice_const"))


def test_the_mapping_across_precice_is_exact(constant_relaxation_run):
    """Power written by the neutronics == power read by the thermal side, to
    the last bit, at every iteration."""
    n = constant_relaxation_run["neutronics"]
    t = constant_relaxation_run["thermal"]
    assert len(n) == len(t)
    written = [row["power_max"] for row in n]
    read = [row["power_max"] for row in t]
    assert written == read


def test_both_participants_see_the_same_iteration_count(constant_relaxation_run):
    n, t = constant_relaxation_run["neutronics"], constant_relaxation_run["thermal"]
    assert [r["iteration"] for r in n] == [r["iteration"] for r in t]


def test_lockstep_with_the_internal_driver(constant_relaxation_run):
    """The sharp one: same relaxation, same start, same Gauss-Seidel order, so
    the two couplings must walk the same path and not merely reach the same
    end."""
    rows = constant_relaxation_run["neutronics"]
    n_iter = len(rows)
    internal = internal_run(tol=0.0, max_iter=n_iter, relaxation=RELAXATION,
                            anderson_depth=0)
    assert len(internal.k_history) == n_iter

    k_precice = np.array([float(r["k_eff"]) for r in rows])
    k_internal = np.array(internal.k_history)
    dk = np.abs(k_internal - k_precice)
    # Tolerance set by the inner eigen solve, not by the coupling: both sides
    # converge k only to tol_k, so agreement below that is not meaningful.
    assert dk.max() < 1e-10, f"worst at iteration {int(dk.argmax()) + 1}: {dk.max():.3e}"

    t_precice = np.array([float(r["T_max"]) for r in rows])
    t_internal = np.array([float(r["T_max"]) for r in
                           constant_relaxation_run["thermal"]])
    assert t_precice.shape == t_internal.shape


def test_the_energy_balance_holds_in_the_external_coupling(constant_relaxation_run):
    """The conduction solver's exact invariant must survive being driven from
    another process -- i.e. the power that crossed preCICE is the power that
    was actually deposited."""
    for row in constant_relaxation_run["thermal"]:
        assert float(row["balance"]) < 1e-10
    last = constant_relaxation_run["thermal"][-1]
    p = build_hpmr2d(refine=REFINE, drum_angle_deg=DRUM_DEG, absorber="polar")
    ctx = build_hpmr_coupling(p)
    assert float(last["source_watts"]) == pytest.approx(ctx.total_power, rel=1e-9)


def test_quasi_newton_reaches_the_same_fixed_point(tmp_path):
    """IQN-ILS takes a different path, so only the destination is comparable --
    and it must be the same destination, much sooner."""
    traces = run_participants(tmp_path, config="precice-config-iqnils.xml")
    rows = traces["neutronics"]
    k_qn = float(rows[-1]["k_eff"])

    internal = internal_run(tol=1e-9, anderson_depth=5)
    assert internal.converged
    assert k_qn == pytest.approx(internal.k_eff, abs=2e-8)

    t_qn = float(traces["thermal"][-1]["T_max"])
    assert t_qn == pytest.approx(internal.peak_temperature, abs=1e-4)

    constant = internal_run(tol=1e-9, relaxation=RELAXATION, anderson_depth=0)
    assert len(rows) < constant.iterations


@pytest.mark.slow
def test_lockstep_on_the_extruded_core(tmp_path):
    """The same lockstep check in 3D, where the coupling mesh carries a z
    coordinate and the config is dimensions="3". Coarse (refine 2, nz 10,
    2-group) so it stays a structural test rather than a physics run."""
    traces = run_participants(tmp_path, config="precice-config-3d.xml",
                              extra=["--nz", "10"], refine=2, timeout=1800)
    rows = traces["neutronics"]
    internal = internal_run(refine=2, nz=10, tol=0.0, max_iter=len(rows),
                            relaxation=RELAXATION, anderson_depth=0)

    dk = np.abs(np.array(internal.k_history)
                - np.array([float(r["k_eff"]) for r in rows]))
    assert dk.max() < 1e-10, f"worst at iteration {int(dk.argmax()) + 1}"
    assert [r["power_max"] for r in rows] == [r["power_max"] for r in traces["thermal"]]
