"""CPU benchmark of full diffusion, adiabatic QS, and time-dependent IQS.

The three two-dimensional HP-MR manoeuvres isolate different demands on the
quasi-static approximation:

* slow symmetric withdrawal: shape motion is slow compared with the macro step;
* fast symmetric withdrawal: the same worth is delivered rapidly;
* asymmetric withdrawal: four neighbouring drums create a strong flux tilt.

Each case is run with full transient diffusion, adiabatic quasi-static (QS),
time-dependent improved quasi-static (IQS), and residual-guarded IQS.  Raw
histories, summary metrics, and two figures are written to ``--output-dir``.

The default two-group constants make the complete CPU comparison practical.
They retain the real 55-site geometry, polar drum arcs, delayed kinetics,
conduction, and Doppler coupling, but are illustrative rather than predictive.
Use ``--groups 11`` for the ENDF/B-VIII-derived energy structure when the much
larger CPU cost is acceptable.

    python examples/hpmr_quasistatic_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ndgpu.benchmarks.hpmr import HPMR_KINETICS, build_hpmr2d
from ndgpu.benchmarks.hpmr_thermal import (build_hpmr_coupling,
                                           hpmr_endfb8_builtin)
from ndgpu.coupling import coupled_transient
from ndgpu.quasistatic import quasistatic_coupled_transient
from ndgpu.tri import TriDiffusionEigenSolver


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    angle_from: np.ndarray
    angle_to: np.ndarray
    t_start: float
    t_ramp: float
    t_end: float
    dt: float
    dt_thermal: float
    shape_dt: float
    n_frames: int


def scenarios(quick=False):
    base = np.full(12, 90.0)
    symmetric = np.full(12, 95.0)
    asymmetric = base.copy()
    # Four drums on the +x side.  The public HP-MR per-drum ordering is six
    # ring-3 corners followed by six ring-4 mid-edges.
    asymmetric[[0, 5, 6, 11]] += 20.0
    if quick:
        return [
            Scenario("slow_symmetric", "Slow symmetric", base, symmetric,
                     0.10, 0.40, 0.80, 0.05, 0.20, 0.20, 5),
            Scenario("fast_symmetric", "Fast symmetric", base, symmetric,
                     0.05, 0.10, 0.40, 0.025, 0.10, 0.10, 5),
            Scenario("asymmetric", "Asymmetric (four drums)", base, asymmetric,
                     0.10, 0.20, 0.60, 0.05, 0.20, 0.20, 5),
        ]
    return [
        Scenario("slow_symmetric", "Slow symmetric", base, symmetric,
                 1.0, 8.0, 12.0, 0.10, 0.50, 1.0, 17),
        Scenario("fast_symmetric", "Fast symmetric", base, symmetric,
                 0.25, 0.50, 3.0, 0.025, 0.25, 0.25, 11),
        Scenario("asymmetric", "Asymmetric (four drums)", base, asymmetric,
                 0.50, 2.0, 6.0, 0.05, 0.50, 0.50, 11),
    ]


def cached_drum_ramp(problem, case, refine, materials, samples=10):
    """Return cached per-drum polar-blend frames for a scalar or asymmetric ramp."""
    fractions = np.linspace(0.0, 1.0, case.n_frames)
    frames = []
    for fraction in fractions:
        angles = (case.angle_from
                  + fraction * (case.angle_to - case.angle_from))
        frame = build_hpmr2d(
            refine=refine, drum_angle_deg=angles, absorber="polar",
            materials=materials, samples=samples)
        frames.append((frame.mix_material, frame.mix_weight))
    fixed = problem.materials, problem.material_map

    def problem_at(t):
        if t <= case.t_start:
            index = 0
        elif t >= case.t_start + case.t_ramp:
            index = case.n_frames - 1
        else:
            fraction = (t - case.t_start) / case.t_ramp
            index = int(round((case.n_frames - 1) * fraction))
        mix_material, mix_weight = frames[index]
        return fixed[0], fixed[1], mix_material, mix_weight

    problem_at.angles = np.asarray([
        case.angle_from + f * (case.angle_to - case.angle_from)
        for f in fractions])
    return problem_at


def solve_static(problem):
    solver = TriDiffusionEigenSolver(
        problem.grid, problem.materials, problem.material_map,
        bc=problem.bc, active=problem.active, mask_bc=problem.mask_bc,
        mix_material=problem.mix_material, mix_weight=problem.mix_weight,
        device="cpu")
    return solver.solve(tol_k=1e-10, tol_source=1e-9).k_eff


def host_array(value):
    try:
        import cupy as cp
        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
    except ImportError:
        pass
    return np.asarray(value)


def normalized_flux_shape(flux, active):
    values = host_array(flux)[:, np.asarray(active, dtype=bool)].ravel()
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("final flux has no finite positive norm")
    return values / norm


MODES = (
    ("full", "Full diffusion"),
    ("adiabatic", "Adiabatic QS"),
    ("iqs", "Time-dependent IQS"),
    ("guarded_iqs", "Guarded IQS"),
)

# Calibrated against the refine-3 cached-frame residuals used below.  The soft
# residual limit shortens an IQS macro interval; the hard limit sends that fine
# interval through full transient diffusion. Predictor disagreement above 2%
# halves subsequent macro intervals. These are benchmark controls, not claimed
# universal production tolerances.
RESIDUAL_TOL = 0.012
FALLBACK_RESIDUAL = 0.020
PREDICTOR_TOL = 0.020
ADJOINT_EVERY = 6
ADJOINT_RESIDUAL_TOL = 0.005


def run_mode(problem, problem_at, case, mode):
    ctx = build_hpmr_coupling(problem, device="cpu")
    start = time.perf_counter()
    if mode == "full":
        result = coupled_transient(
            ctx, t_end=case.t_end, dt=case.dt,
            dt_thermal=case.dt_thermal, problem_at=problem_at, profile=True)
    else:
        options = dict(
            shape_dt=case.shape_dt, adjoint_every=2,
            shape_method=("adiabatic" if mode == "adiabatic" else "iqs"))
        if mode == "guarded_iqs":
            options.update(residual_tol=RESIDUAL_TOL,
                           fallback_residual=FALLBACK_RESIDUAL,
                           iqs_predictor_tol=PREDICTOR_TOL,
                           adjoint_every=ADJOINT_EVERY,
                           adjoint_residual_tol=ADJOINT_RESIDUAL_TOL)
        result = quasistatic_coupled_transient(
            ctx, t_end=case.t_end, dt=case.dt,
            dt_thermal=case.dt_thermal, problem_at=problem_at,
            profile=True, **options)
    wall = time.perf_counter() - start
    dynamic = max(wall - result.steady.seconds, 0.0)
    return result, wall, dynamic


def result_metrics(case, mode, label, result, wall, dynamic, reference,
                   active):
    if reference is None:
        max_power_error = max_temperature_error = shape_error = 0.0
        peak_power_bias = final_power_bias = 0.0
    else:
        max_power_error = float(np.max(
            np.abs(result.power - reference.power)
            / np.maximum(np.abs(reference.power), 1e-14)))
        max_temperature_error = float(np.max(
            np.abs(result.mean_temperature - reference.mean_temperature)))
        peak_power_bias = float(result.power.max() / reference.power.max() - 1.0)
        final_power_bias = float(result.power[-1] / reference.power[-1] - 1.0)
        a = normalized_flux_shape(reference.flux, active)
        b = normalized_flux_shape(result.flux, active)
        if np.dot(a, b) < 0.0:
            b = -b
        shape_error = float(np.linalg.norm(a - b))

    counters = result.counters
    residual = np.asarray(getattr(result, "shape_residual", []), dtype=float)
    reasons = Counter(getattr(result, "shape_update_reasons", []))
    return {
        "scenario": case.name,
        "scenario_label": case.label,
        "mode": mode,
        "mode_label": label,
        "steps": int(result.steps),
        "wall_seconds": float(wall),
        "steady_seconds": float(result.steady.seconds),
        "dynamic_seconds": float(dynamic),
        "march_seconds": float(result.seconds),
        "initialization_seconds": float(getattr(result, "initialization_seconds", 0.0)),
        "peak_power": float(result.power.max()),
        "final_power": float(result.power[-1]),
        "mean_temperature_rise_K": float(
            result.mean_temperature[-1] - result.mean_temperature[0]),
        "peak_temperature_K": float(result.peak_temperature.max()),
        "max_relative_power_error": max_power_error,
        "peak_power_bias": peak_power_bias,
        "final_power_bias": final_power_bias,
        "max_mean_temperature_error_K": max_temperature_error,
        "final_flux_shape_l2_error": shape_error,
        "shape_updates": int(counters.get("shape_updates", 0)),
        "iqs_shape_solves": int(counters.get("iqs_shape_solves", 0)),
        "adiabatic_shape_solves": int(counters.get("forward_shape_solves", 0)),
        "adjoint_solves": int(counters.get("adjoint_eigen_solves", 0)),
        "residual_evaluations": int(counters.get("residual_evaluations", 0)),
        "max_shape_residual": float(residual.max()) if residual.size else 0.0,
        "residual_shape_updates": int(reasons["residual"]),
        "fallback_intervals": int(counters.get("full_diffusion_fallbacks", 0)),
        "final_power_shape_factor": float(
            np.asarray(getattr(result, "power_shape_factor", [1.0]))[-1]),
        "max_shape_derivative_correction": (
            1e-6 * int(counters.get(
                "max_shape_derivative_correction_ppm", 0))),
        "iqs_corrector_substeps": int(
            counters.get("iqs_corrector_substeps", 0)),
        "iqs_precursor_shape_corrections": int(
            counters.get("iqs_precursor_shape_corrections", 0)),
        "predictor_interval_reductions": int(
            counters.get("iqs_predictor_interval_reductions", 0)),
        "predictor_interval_recoveries": int(
            counters.get("iqs_predictor_interval_recoveries", 0)),
        "adjoint_residual_evaluations": int(
            counters.get("adjoint_residual_evaluations", 0)),
        "adjoint_residual_refreshes": int(
            counters.get("adjoint_residual_refreshes", 0)),
        "max_adjoint_residual": float(
            np.max(getattr(result, "adjoint_residual", [0.0])))
            if np.size(getattr(result, "adjoint_residual", [])) else 0.0,
        "max_iqs_predictor_error": (
            1e-6 * int(counters.get("iqs_max_amplitude_error_ppm", 0))),
        "neutron_inner_iterations": int(counters.get("neutron_inner_iterations", 0)),
        "iqs_inner_iterations": int(counters.get("iqs_inner_iterations", 0)),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figures(output_dir, cases, histories, summary):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ndgpu-matplotlib")
    import matplotlib.pyplot as plt

    colors = {"full": "black", "adiabatic": "#e69f00",
              "iqs": "#0072b2", "guarded_iqs": "#009e73"}
    styles = {"full": "-", "adiabatic": "--", "iqs": "-.",
              "guarded_iqs": ":"}
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 9), sharex="row")
    for row, case in enumerate(cases):
        for mode, label in MODES:
            result = histories[(case.name, mode)]
            axes[row, 0].plot(result.times, result.power, styles[mode],
                              color=colors[mode], label=label, linewidth=1.8)
            axes[row, 1].plot(
                result.times,
                result.mean_temperature - result.mean_temperature[0],
                styles[mode], color=colors[mode], label=label, linewidth=1.8)
        axes[row, 0].set_ylabel(f"{case.label}\nP/P0")
        axes[row, 1].set_ylabel("mean fuel dT [K]")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    axes[0, 0].legend(ncol=2, fontsize=9)
    fig.suptitle("2D HP-MR coupled transient histories (CPU, two groups)")
    fig.tight_layout()
    fig.savefig(output_dir / "histories.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.23
    x = np.arange(len(cases))
    reference_dynamic = {
        row["scenario"]: row["dynamic_seconds"]
        for row in summary if row["mode"] == "full"}
    for j, (mode, label) in enumerate(MODES[1:]):
        selected = [next(row for row in summary
                         if row["scenario"] == case.name and row["mode"] == mode)
                    for case in cases]
        speedup = [reference_dynamic[row["scenario"]] / row["dynamic_seconds"]
                   for row in selected]
        error = [100.0 * row["max_relative_power_error"] for row in selected]
        offset = (j - 1) * width
        axes[0].bar(x + offset, speedup, width, color=colors[mode], label=label)
        axes[1].bar(x + offset, error, width, color=colors[mode], label=label)
    for ax in axes:
        ax.set_xticks(x, [case.label for case in cases], rotation=12, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].axhline(1.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("dynamic-time speedup vs full diffusion")
    axes[1].set_ylabel("maximum power-history error [%]")
    axes[0].legend(fontsize=9)
    fig.suptitle("Quasi-static CPU performance/accuracy trade-off")
    fig.tight_layout()
    fig.savefig(output_dir / "tradeoffs.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refine", type=int, default=3)
    parser.add_argument("--groups", choices=("2", "11"), default="2")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmark-results/hpmr-quasistatic-cpu"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = scenarios(args.quick)
    materials = hpmr_endfb8_builtin() if args.groups == "11" else None
    summary = []
    history_rows = []
    histories = {}
    case_metadata = []
    benchmark_start = time.perf_counter()

    print(f"HP-MR quasi-static CPU benchmark: refine={args.refine}, "
          f"groups={args.groups}")
    for case in cases:
        print(f"\n[{case.label}] building {case.n_frames} cached drum frames")
        problem = build_hpmr2d(
            refine=args.refine, drum_angle_deg=case.angle_from,
            absorber="polar", materials=materials)
        problem_at = cached_drum_ramp(
            problem, case, args.refine, materials)
        endpoint = build_hpmr2d(
            refine=args.refine, drum_angle_deg=case.angle_to,
            absorber="polar", materials=materials)
        k_from, k_to = solve_static(problem), solve_static(endpoint)
        worth_pcm = 1e5 * (1.0 / k_from - 1.0 / k_to)
        dollars = worth_pcm / (1e5 * float(HPMR_KINETICS.beta.sum()))
        case_metadata.append({
            "name": case.name, "label": case.label,
            "angle_from_deg": case.angle_from.tolist(),
            "angle_to_deg": case.angle_to.tolist(),
            "t_start": case.t_start, "t_ramp": case.t_ramp,
            "t_end": case.t_end, "dt": case.dt,
            "dt_thermal": case.dt_thermal, "shape_dt": case.shape_dt,
            "n_frames": case.n_frames, "k_from": k_from, "k_to": k_to,
            "static_worth_pcm": worth_pcm, "static_worth_dollars": dollars,
            "active_cells": int(np.count_nonzero(problem.active)),
        })
        print(f"  cold static worth: {worth_pcm:+.1f} pcm ({dollars:+.3f} $)")

        reference = None
        for mode, label in MODES:
            print(f"  {label:<20}", end="", flush=True)
            result, wall, dynamic = run_mode(problem, problem_at, case, mode)
            histories[(case.name, mode)] = result
            metrics = result_metrics(
                case, mode, label, result, wall, dynamic, reference,
                problem.active)
            if mode == "full":
                reference = result
            summary.append(metrics)
            print(f" {wall:7.2f} s; peak={result.power.max():.4f}; "
                  f"dPmax={100*metrics['max_relative_power_error']:.3f}%; "
                  f"fallbacks={metrics['fallback_intervals']}")
            for index, t in enumerate(result.times):
                history_rows.append({
                    "scenario": case.name, "mode": mode,
                    "time_s": float(t), "power_ratio": float(result.power[index]),
                    "mean_temperature_K": float(result.mean_temperature[index]),
                    "peak_temperature_K": float(result.peak_temperature[index]),
                })

    by_case = {row["scenario"]: row["dynamic_seconds"]
               for row in summary if row["mode"] == "full"}
    for row in summary:
        row["dynamic_speedup_vs_full"] = (
            by_case[row["scenario"]] / row["dynamic_seconds"])
    total_seconds = time.perf_counter() - benchmark_start
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(), "processor": platform.processor(),
        "python": platform.python_version(), "refine": args.refine,
        "groups": int(args.groups), "quick": args.quick,
        "guarded_iqs": {"residual_tol": RESIDUAL_TOL,
                        "fallback_residual": FALLBACK_RESIDUAL,
                        "predictor_tol": PREDICTOR_TOL,
                        "adjoint_every": ADJOINT_EVERY,
                        "adjoint_residual_tol": ADJOINT_RESIDUAL_TOL},
        "total_benchmark_seconds": total_seconds,
        "cases": case_metadata,
    }
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "histories.csv", history_rows)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    make_figures(args.output_dir, cases, histories, summary)

    fallback_total = sum(row["fallback_intervals"] for row in summary
                         if row["mode"] == "guarded_iqs")
    residual_total = sum(row["residual_evaluations"] for row in summary
                         if row["mode"] == "guarded_iqs")
    is_reference_configuration = (
        args.refine == 3 and args.groups == "2" and not args.quick)
    if is_reference_configuration and (residual_total == 0 or fallback_total == 0):
        raise RuntimeError("guarded IQS did not exercise residual evaluation "
                           "and at least one full-diffusion fallback")
    print(f"\nCompleted in {total_seconds:.1f} s; artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
