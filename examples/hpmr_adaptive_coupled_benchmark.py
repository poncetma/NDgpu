"""Adaptive-BDF gate for coupled HP-MR drum transients.

The fine backward-Euler reference and adaptive candidate use the same
weight-interpolated polar drum trajectory and the same thermal-exchange times.
Use ``--thermal-mass-scale 0.005`` as an explicit feedback stress test; the
default retains the physical HP-MR thermal mass.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from hpmr_bdf_cpu_benchmark import _problem_at, _trajectory_frames
from ndgpu import asnumpy
from ndgpu.benchmarks.hpmr_thermal import build_hpmr_coupling
from ndgpu.benchmarks.hpmr_transient_bench import build_case
from ndgpu.coupling import coupled_transient
from ndgpu.thermal import ThermalMaterial


CASE_DEFINITIONS = {
    "slow-symmetric": (0.8, False),
    "fast-symmetric": (0.08, False),
    "fast-asymmetric": (0.08, True),
}


def _scaled_thermal_mass(ctx, factor):
    if factor == 1.0:
        return ctx
    materials = [
        ThermalMaterial(m.conductivity, m.sink_coeff, m.sink_temperature,
                        m.heat_capacity * factor, m.name)
        for m in ctx.thermal_materials
    ]
    return replace(ctx, thermal_materials=materials, _thermal=None)


def _normalized_flux(result, active):
    values = asnumpy(result.flux)[:, np.asarray(active, dtype=bool)].ravel()
    return values / np.linalg.norm(values)


def _run(ctx, problem_at, duration, initial_steps, thermal_width,
         *, adaptive, rtol, control_intervals, scheme="bdf5"):
    transient = {
        "time_scheme": scheme if adaptive else "bdf1",
        "step_solver": "monolithic",
        "tol_step": 1e-8,
        "multigroup_kwargs": {
            "scatter_sweeps": 3, "rtol": 1e-9,
            "inner_rtol": 1e-3, "precond_degree": 1,
        },
    }
    if adaptive:
        transient.update(
            bdf_restart_times=(duration * np.arange(1, control_intervals)
                               / control_intervals),
            adaptive_bdf={
                "rtol": rtol,
                "min_dt": duration / 64 / 4096,
                "max_dt": duration / control_intervals,
                "automatic_order": scheme != "bdf1",
                "rejection_strategy": "error",
                "reject_max_factor": 0.5,
            })
    return coupled_transient(
        ctx, t_end=duration, dt=duration / initial_steps,
        dt_thermal=thermal_width, problem_at=problem_at,
        transient_kwargs=transient, profile=True)


def benchmark(*, refine=2, groups="2", cases=("slow-symmetric",),
              reference_steps=64, initial_steps=8, control_intervals=4,
              rtol=1e-3, samples=10, thermal_mass_scale=1.0,
              candidate_schemes=("bdf5",)):
    problem, _ = build_case(refine, nz=0, groups=groups)
    ctx = _scaled_thermal_mass(
        build_hpmr_coupling(problem, device="cpu"), thermal_mass_scale)
    rows = []
    for case in cases:
        duration, asymmetric = CASE_DEFINITIONS[case]
        frames = _trajectory_frames(
            problem, refine, control_intervals + 1, asymmetric,
            samples=samples)
        problem_at = _problem_at(problem, frames, duration)
        thermal_width = duration / control_intervals
        reference = _run(
            ctx, problem_at, duration, reference_steps, thermal_width,
            adaptive=False, rtol=rtol, control_intervals=control_intervals)
        for candidate_scheme in candidate_schemes:
            candidate = _run(
                ctx, problem_at, duration, initial_steps, thermal_width,
                adaptive=True, rtol=rtol,
                control_intervals=control_intervals,
                scheme=candidate_scheme)

            power_on_reference = np.interp(
                reference.times, candidate.times, candidate.power)
            thermal_times = np.linspace(
                0.0, duration, control_intervals + 1)

            def at_thermal_times(result, values):
                indices = []
                for time in thermal_times:
                    index = int(np.argmin(np.abs(result.times - time)))
                    if not np.isclose(result.times[index], time, rtol=1e-12,
                                      atol=1e-14 * max(1.0, duration)):
                        raise RuntimeError(
                            f"thermal exchange {time:g} is absent from time grid")
                    indices.append(index)
                return np.asarray(values)[indices]

            reference_mean_t = at_thermal_times(
                reference, reference.mean_temperature)
            candidate_mean_t = at_thermal_times(
                candidate, candidate.mean_temperature)
            reference_peak_t = at_thermal_times(
                reference, reference.peak_temperature)
            candidate_peak_t = at_thermal_times(
                candidate, candidate.peak_temperature)
            flux_error = np.linalg.norm(
                _normalized_flux(candidate, problem.active)
                - _normalized_flux(reference, problem.active))
            rows.append({
                "case": case,
                "refine": refine,
                "groups": int(problem.materials[0].n_groups),
                "thermal_mass_scale": thermal_mass_scale,
                "reference_steps": reference.steps,
                "reference_seconds": reference.seconds,
                "reference_inner_iterations": reference.counters[
                    "neutron_inner_iterations"],
                "reference_outer_iterations": reference.counters[
                    "neutron_total_outer_iterations"],
                "adaptive_rtol": rtol,
                "adaptive_scheme": candidate.time_scheme,
                "adaptive_steps": candidate.steps,
                "adaptive_rejected_steps": candidate.rejected_steps,
                "adaptive_seconds": candidate.seconds,
                "adaptive_inner_iterations": candidate.counters[
                    "neutron_inner_iterations"],
                "adaptive_outer_iterations": candidate.counters[
                    "neutron_total_outer_iterations"],
                "adaptive_rejected_outer_iterations": candidate.counters[
                    "neutron_rejected_outer_iterations"],
                "adaptive_order_counts": {
                    str(order): candidate.time_orders.count(order)
                    for order in sorted(set(candidate.time_orders))
                },
                "max_power_relative_error": float(np.max(
                    np.abs(power_on_reference - reference.power)
                    / np.maximum(np.abs(reference.power), 1e-14))),
                "max_mean_temperature_error_k": float(np.max(np.abs(
                    candidate_mean_t - reference_mean_t))),
                "max_peak_temperature_error_k": float(np.max(np.abs(
                    candidate_peak_t - reference_peak_t))),
                "final_flux_shape_relative_error": float(flux_error),
                "final_mean_temperature_k": float(
                    candidate.mean_temperature[-1]),
                "mean_temperature_rise_k": float(
                    candidate.mean_temperature[-1]
                    - candidate.mean_temperature[0]),
                "wall_speedup": reference.seconds / candidate.seconds,
                "inner_work_ratio": (
                    reference.counters["neutron_inner_iterations"]
                    / candidate.counters["neutron_inner_iterations"]),
                "outer_work_ratio": (
                    reference.counters["neutron_total_outer_iterations"]
                    / candidate.counters["neutron_total_outer_iterations"]),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--groups", choices=("2", "11"), default="2")
    parser.add_argument("--cases", nargs="+", choices=tuple(CASE_DEFINITIONS),
                        default=["slow-symmetric"])
    parser.add_argument("--reference-steps", type=int, default=64)
    parser.add_argument("--initial-steps", type=int, default=8)
    parser.add_argument("--control-intervals", type=int, default=4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--candidate-schemes", nargs="+",
                        choices=("bdf1", "bdf2", "bdf3", "bdf4", "bdf5",
                                 "bdf6"), default=["bdf5"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--thermal-mass-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = benchmark(
        refine=args.refine, groups=args.groups, cases=tuple(args.cases),
        reference_steps=args.reference_steps, initial_steps=args.initial_steps,
        control_intervals=args.control_intervals, rtol=args.rtol,
        samples=args.samples, thermal_mass_scale=args.thermal_mass_scale,
        candidate_schemes=tuple(args.candidate_schemes))
    payload = json.dumps(rows, indent=2) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
