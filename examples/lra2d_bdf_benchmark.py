"""Run the CPU LRA-2D coupled BDF comparison.

Examples
--------
Quick assembly-mesh comparison::

    python examples/lra2d_bdf_benchmark.py --refine 1 --dt 0.01

Paper-comparison neutronics mesh (3.75 cm) with assembly thermal zones::

    python examples/lra2d_bdf_benchmark.py --refine 4 --dt 0.01

Run standard backward Euler under the same controls as the selected BDF::

    python examples/lra2d_bdf_benchmark.py --order 5 \
        --implicit-feedback --compare-backward-euler
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from ndgpu.benchmarks import lra2d_static_keff, run_lra2d_cpu
from ndgpu.benchmarks.lra import (CHEREZOV_BY_FEM_ORDER, CHEREZOV_FEMCORE,
                                  K_REFERENCE, TRANSIENT_REFERENCE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", type=int, default=1,
                    help="FV cells per 15 cm assembly edge")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--order", type=int, default=5, choices=range(1, 7))
    ap.add_argument("--tol-step", type=float, default=2e-7)
    buckling = ap.add_mutually_exclusive_group()
    buckling.add_argument("--axial-buckling", "--literal-buckling",
                          dest="axial_buckling", action="store_true",
                          help="include the specified D*Bz^2 leakage (default)")
    buckling.add_argument("--no-axial-buckling", dest="axial_buckling",
                          action="store_false",
                          help="diagnostic: omit the specified axial leakage")
    ap.set_defaults(axial_buckling=True)
    ap.add_argument("--lagged-feedback", action="store_true")
    ap.add_argument("--implicit-feedback", action="store_true")
    ap.add_argument("--thermal-zones", choices=("assembly", "cell"),
                    default="assembly")
    ap.add_argument("--control-worth-scale", type=float, default=1.0,
                    help="diagnostic multiplier on region-R absorption change")
    ap.add_argument("--compare-backward-euler", action="store_true",
                    help="also run BDF1/backward Euler with identical controls")
    ap.add_argument("--backward-euler-rtol", type=float,
                    help="adaptive BE tolerance for a matched-error comparison")
    ap.add_argument("--adaptive-rtol", type=float,
                    help="enable experimental adaptive BDF state control")
    ap.add_argument("--adaptive-min-dt", type=float, default=1e-6)
    ap.add_argument("--adaptive-max-dt", type=float, default=0.1)
    ap.add_argument("--adaptive-rejection-strategy", choices=("half", "error"),
                    default="half")
    ap.add_argument("--adaptive-reject-max-factor", type=float, default=0.5)
    order_control = ap.add_mutually_exclusive_group()
    order_control.add_argument("--automatic-order", action="store_true",
                               help="select q-1/q/q+1 by safe width proposal")
    order_control.add_argument("--fixed-max-order", dest="automatic_order",
                               action="store_false")
    ap.set_defaults(automatic_order=False)
    ap.add_argument("--include-history", action="store_true",
                    help="include accepted time/width/order/error arrays in JSON")
    ap.add_argument("--output", type=Path,
                    help="also write the JSON payload to this path")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress stdout when --output is used")
    ap.add_argument("--include-static-endpoints", action="store_true",
                    help="also solve and report the raw rods-out eigenvalue")
    ap.add_argument("--cherezov-controls", action="store_true",
                    help="use the paper's RTOL, h0, feedback, and BDF5 controls")
    args = ap.parse_args()

    if args.cherezov_controls:
        args.dt = 1e-6
        args.order = 5
        args.adaptive_rtol = 1e-5
        args.adaptive_min_dt = min(args.adaptive_min_dt, 1e-6)
        args.adaptive_max_dt = 0.1
        args.implicit_feedback = True
        args.lagged_feedback = False
        args.thermal_zones = "assembly"
        args.automatic_order = True
        args.adaptive_rejection_strategy = "half"

    def run(order, adaptive_rtol=None):
        effective_rtol = (args.adaptive_rtol if adaptive_rtol is None
                          else adaptive_rtol)
        adaptive = (None if effective_rtol is None else {
            "rtol": effective_rtol,
            "min_dt": args.adaptive_min_dt,
            "max_dt": args.adaptive_max_dt,
            "rejection_strategy": args.adaptive_rejection_strategy,
            "reject_max_factor": args.adaptive_reject_max_factor,
            "automatic_order": args.automatic_order and order > 1,
        })
        return run_lra2d_cpu(
            refine=args.refine, dt=args.dt, bdf_order=order,
            axial_buckling=args.axial_buckling, tol_step=args.tol_step,
            predict_feedback=not args.lagged_feedback,
            implicit_feedback=args.implicit_feedback,
            thermal_zones=args.thermal_zones,
            control_worth_scale=args.control_worth_scale,
            adaptive_bdf=adaptive,
        )

    def metrics(result):
        accepted_outer = sum(result.transient.step_iterations)
        rejected_outer = sum(result.transient.rejected_step_iterations)
        accepted_feedback = sum(result.transient.feedback_iterations)
        rejected_feedback = sum(
            result.transient.rejected_feedback_iterations)
        return {
            "k0": result.transient.k0,
            "rods_in": result.transient.k0,
            "solve_seconds": result.transient.solve_seconds,
            "steps": len(result.transient.times) - 1,
            "fgmres_applications": accepted_outer + rejected_outer,
            "accepted_fgmres_applications": accepted_outer,
            "rejected_fgmres_applications": rejected_outer,
            "inner_iterations": result.transient.total_inner_iterations,
            "feedback_iterations": accepted_feedback + rejected_feedback,
            "accepted_feedback_iterations": accepted_feedback,
            "rejected_feedback_iterations": rejected_feedback,
            "max_feedback_iterations": max(
                result.transient.feedback_iterations, default=0),
            "rejected_steps": result.transient.rejected_steps,
            "min_step_s": min(result.transient.step_widths, default=args.dt),
            "max_step_s": max(result.transient.step_widths, default=args.dt),
            "mean_step_s": (sum(result.transient.step_widths)
                            / max(len(result.transient.step_widths), 1)),
            "order_counts": dict(sorted(Counter(
                result.transient.time_orders).items())),
            "next_order_counts": dict(sorted(Counter(
                result.transient.next_time_orders).items())),
            **result.metrics(),
        }

    result = run(args.order)
    ndgpu_metrics = metrics(result)
    include_static_endpoints = (args.include_static_endpoints
                                or args.cherezov_controls)
    if include_static_endpoints:
        ndgpu_metrics["rods_out"] = lra2d_static_keff(
            refine=args.refine, control="out",
            axial_buckling=args.axial_buckling,
            control_worth_scale=args.control_worth_scale,
        )

    def relative_differences(reference):
        return {
            key: ((ndgpu_metrics[key] - value) / value)
            for key, value in reference.items()
            if key in ndgpu_metrics and value != 0.0
        }

    payload = {
        "case": {
            "refine": args.refine, "dt_s": args.dt,
            "bdf_order": args.order, "coupling": result.coupling,
            "axial_buckling": args.axial_buckling,
            "control_worth_scale": args.control_worth_scale,
            "adaptive_rtol": args.adaptive_rtol,
            "adaptive_rejection_strategy": args.adaptive_rejection_strategy,
            "adaptive_reject_max_factor": args.adaptive_reject_max_factor,
            "cherezov_controls": args.cherezov_controls,
            "static_endpoints": include_static_endpoints,
            "automatic_order_selection": args.automatic_order,
            "spatial_discretization": "cell-centred finite volume",
        },
        "ndgpu": ndgpu_metrics,
        "anl_reference": {**K_REFERENCE, **TRANSIENT_REFERENCE},
        "cherezov_femcore": CHEREZOV_FEMCORE,
        "cherezov_fem_order_1": CHEREZOV_BY_FEM_ORDER[1],
        "relative_difference_vs_anl": relative_differences(
            {**K_REFERENCE, **TRANSIENT_REFERENCE}),
        "relative_difference_vs_cherezov_femcore": relative_differences(
            CHEREZOV_FEMCORE),
        "relative_difference_vs_cherezov_fem_order_1": relative_differences(
            CHEREZOV_BY_FEM_ORDER[1]),
        "comparison_note": (
            "refine=1 FV is a low-order spatial comparison and should be read "
            "against the paper's FEM-order ladder; refine>1 is a FV mesh "
            "convergence study, not a reproduction of fourth-order FEM. "
            "The JSON order counts expose automatic q-1/q/q+1 behavior."
        ),
    }
    if args.include_history:
        payload["adaptive_history"] = {
            "times_s": result.transient.times.tolist(),
            "power_w_cm3": result.average_power_w_cm3.tolist(),
            "mean_temperature_k": result.average_temperature_k.tolist(),
            "peak_assembly_temperature_k": (
                result.peak_assembly_temperature_k.tolist()),
            "peak_temperature_k": result.peak_temperature_k.tolist(),
            "widths_s": result.transient.step_widths,
            "orders": result.transient.time_orders,
            "accepted_errors": result.transient.local_errors,
            "rejected_errors": result.transient.rejected_errors,
            "rejected_times_s": result.transient.rejected_times,
            "rejected_widths_s": result.transient.rejected_step_widths,
            "rejected_orders": result.transient.rejected_time_orders,
            "candidate_order_errors": result.transient.order_candidate_errors,
            "selected_next_orders": result.transient.next_time_orders,
        }
    if args.compare_backward_euler and args.order != 1:
        be_rtol = (args.adaptive_rtol if args.backward_euler_rtol is None
                   else args.backward_euler_rtol)
        backward_euler = run(1, adaptive_rtol=be_rtol)
        primary, baseline = payload["ndgpu"], metrics(backward_euler)
        baseline["adaptive_rtol"] = be_rtol
        payload["backward_euler"] = baseline
        payload["comparison_to_backward_euler"] = {
            "bdf_adaptive_rtol": args.adaptive_rtol,
            "backward_euler_adaptive_rtol": be_rtol,
            "wall_speedup": baseline["solve_seconds"]
                            / primary["solve_seconds"],
            "fgmres_work_ratio": (baseline["fgmres_applications"]
                                   / primary["fgmres_applications"]),
            "first_peak_power_difference_w_cm3": (
                primary["first_peak_power_w_cm3"]
                - baseline["first_peak_power_w_cm3"]),
            "first_peak_time_difference_s": (
                primary["first_peak_time_s"]
                - baseline["first_peak_time_s"]),
            "power_at_3s_difference_w_cm3": (
                primary["power_at_3s_w_cm3"]
                - baseline["power_at_3s_w_cm3"]),
        }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not args.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
