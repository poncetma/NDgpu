"""Emit a structured corrected-LRA tolerance and matched-BE benchmark.

The default is the 3.75 cm finite-volume comparison used by Steps 07f/07g.
It is intentionally a long CPU benchmark; use ``--refine 1`` for a smoke run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json

from ndgpu.benchmarks import run_lra2d_cpu


_METRIC_KEYS = (
    "first_peak_time_s", "first_peak_power_w_cm3",
    "second_peak_power_w_cm3", "power_at_3s_w_cm3",
    "mean_temperature_at_3s_k", "peak_temperature_at_3s_k",
)


def _parse_tolerances(value):
    values = [float(item) for item in value.split(",")]
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("tolerances must be positive")
    return values


def _summary(result, rtol, order):
    transient = result.transient
    accepted_outer = sum(transient.step_iterations)
    rejected_outer = sum(transient.rejected_step_iterations)
    return {
        "method": "backward-euler" if order == 1 else f"automatic-bdf{order}",
        "rtol": rtol,
        "steps": len(transient.times) - 1,
        "rejected_steps": transient.rejected_steps,
        "fgmres_applications": accepted_outer + rejected_outer,
        "inner_iterations": transient.total_inner_iterations,
        "feedback_iterations": (sum(transient.feedback_iterations)
                                + sum(transient.rejected_feedback_iterations)),
        "solve_seconds": transient.solve_seconds,
        "order_counts": dict(sorted(Counter(transient.time_orders).items())),
        **result.metrics(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", type=int, default=4)
    ap.add_argument("--bdf-order", type=int, default=5, choices=range(2, 7))
    ap.add_argument("--bdf-rtols", type=_parse_tolerances,
                    default=_parse_tolerances("1e-3,1e-4,1e-5"))
    ap.add_argument("--reference-rtol", type=float, default=1e-6)
    ap.add_argument("--backward-euler-rtol", type=float, default=1e-4)
    ap.add_argument("--initial-dt", type=float, default=1e-6)
    ap.add_argument("--min-dt", type=float, default=1e-7)
    ap.add_argument("--max-dt", type=float, default=0.1)
    ap.add_argument("--rejection-strategy", choices=("half", "error"),
                    default="half")
    ap.add_argument("--reject-max-factor", type=float, default=0.5)
    args = ap.parse_args()

    cache = {}

    def run(order, rtol, automatic):
        key = order, rtol, automatic
        if key not in cache:
            result = run_lra2d_cpu(
                refine=args.refine, dt=args.initial_dt, bdf_order=order,
                implicit_feedback=True, thermal_zones="assembly",
                adaptive_bdf={
                    "rtol": rtol, "min_dt": args.min_dt,
                    "max_dt": args.max_dt, "automatic_order": automatic,
                    "rejection_strategy": args.rejection_strategy,
                    "reject_max_factor": args.reject_max_factor,
                },
            )
            cache[key] = _summary(result, rtol, order)
        return cache[key]

    reference = run(args.bdf_order, args.reference_rtol, True)
    bdf_rows = [run(args.bdf_order, rtol, True) for rtol in args.bdf_rtols]
    backward_euler = run(1, args.backward_euler_rtol, False)

    def errors(row):
        return {key: (row[key] - reference[key]) / reference[key]
                for key in _METRIC_KEYS}

    for row in bdf_rows:
        row["relative_error_vs_reference"] = errors(row)
    backward_euler["relative_error_vs_reference"] = errors(backward_euler)

    # Pair BE with the BDF row having the closest absolute first-peak error.
    peak_key = "first_peak_power_w_cm3"
    be_peak_error = abs(backward_euler["relative_error_vs_reference"][peak_key])
    matched = min(bdf_rows, key=lambda row: abs(
        abs(row["relative_error_vs_reference"][peak_key]) - be_peak_error))
    payload = {
        "case": {
            "refine": args.refine,
            "spatial_discretization": "cell-centred finite volume",
            "implicit_feedback": True,
            "thermal_zones": "assembly",
            "initial_dt_s": args.initial_dt,
            "min_dt_s": args.min_dt,
            "max_dt_s": args.max_dt,
            "rejection_strategy": args.rejection_strategy,
            "reject_max_factor": args.reject_max_factor,
        },
        "temporal_reference": reference,
        "automatic_bdf": bdf_rows,
        "backward_euler": backward_euler,
        "matched_first_peak_comparison": {
            "bdf_rtol": matched["rtol"],
            "backward_euler_rtol": backward_euler["rtol"],
            "bdf_first_peak_relative_error":
                matched["relative_error_vs_reference"][peak_key],
            "backward_euler_first_peak_relative_error":
                backward_euler["relative_error_vs_reference"][peak_key],
            "wall_speedup": (backward_euler["solve_seconds"]
                             / matched["solve_seconds"]),
            "fgmres_work_ratio": (backward_euler["fgmres_applications"]
                                   / matched["fgmres_applications"]),
            "inner_work_ratio": (backward_euler["inner_iterations"]
                                 / matched["inner_iterations"]),
            "accepted_step_ratio": backward_euler["steps"] / matched["steps"],
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
