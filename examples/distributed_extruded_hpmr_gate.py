"""Real-MPI correctness gate for locally refined extruded HP-MR diffusion."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
from mpi4py import MPI

from ndgpu import (DistributedContext,
                   DistributedExtrudedMeshDiffusionEigenSolver,
                   DistributedExtrudedMeshTransientSolver,
                   ExtrudedMeshDiffusionEigenSolver,
                   ExtrudedMeshTransientSolver, asnumpy)
from ndgpu.benchmarks import (HPMR_KINETICS, build_hpmr3d_local,
                              with_hpmr3d_local_drum_angle)
from ndgpu.benchmarks.hpmr_thermal import (hpmr_endfb8_builtin,
                                           hpmr_kinetics_11g)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument(
        "--communication",
        choices=("auto", "cpu-mpi", "host-staged", "cuda-aware"),
        default="auto")
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--local-levels", type=int, default=1)
    parser.add_argument("--nz", type=int, default=10)
    parser.add_argument("--groups", choices=("2", "11"), default="2")
    parser.add_argument("--initial-angle", type=float, default=90.0)
    parser.add_argument("--final-angle", type=float, default=89.0)
    parser.add_argument("--tol-step", type=float, default=1e-8)
    parser.add_argument("--max-sweeps", type=int, default=1000)
    parser.add_argument(
        "--step-solver", choices=("fixed-point", "monolithic"),
        default="fixed-point")
    return parser.parse_args()


def relative_l2(actual, expected):
    denominator = np.linalg.norm(expected.ravel())
    return float(np.linalg.norm((actual - expected).ravel()) / denominator)


def main():
    args = parse_args()
    communicator = MPI.COMM_WORLD
    context = DistributedContext.from_mpi(
        communicator, device=args.device, communication=args.communication,
        batched_halos=True)
    if context.size < 2:
        raise RuntimeError("extruded HPMR gate requires at least two MPI ranks")
    if context.size > args.nz:
        raise RuntimeError("MPI rank count cannot exceed axial layer count")

    materials = (hpmr_endfb8_builtin(three_d=True)
                 if args.groups == "11" else None)
    kinetics = hpmr_kinetics_11g() if args.groups == "11" else HPMR_KINETICS
    build = dict(
        refine=args.refine, nz=args.nz,
        drum_refine_levels=args.local_levels,
        materials=materials, absorber="polar", samples=0)
    initial = build_hpmr3d_local(
        drum_angle_deg=args.initial_angle, **build)
    perturbed = with_hpmr3d_local_drum_angle(
        initial, args.final_angle, samples=build["samples"])

    def problem_at(time):
        problem = initial if time == 0.0 else perturbed
        return (problem.materials, problem.material_map,
                problem.mix_material, problem.mix_weight)

    common = dict(
        bc=initial.bc, active=initial.active, mask_bc=initial.mask_bc)
    steady_kwargs = dict(
        tol_k=1e-10, tol_source=1e-10, inner_rtol_floor=1e-12,
        verbose=context.rank == 0)
    distributed_steady = DistributedExtrudedMeshDiffusionEigenSolver(
        initial.grid, initial.materials, initial.material_map,
        context=context, mix_material=initial.mix_material,
        mix_weight=initial.mix_weight, **common).solve(**steady_kwargs)
    gathered_initial_flux = distributed_steady.gather_flux(root=0)

    solver = DistributedExtrudedMeshTransientSolver(
        initial.grid, problem_at, kinetics, context=context, **common)
    context.reset_communication_stats()
    communicator.Barrier()
    started = perf_counter()
    step_kwargs = dict(
        t_end=0.01, dt=0.01, initial_steady=distributed_steady,
        tol_step=args.tol_step, max_sweeps=args.max_sweeps,
        verbose=context.rank == 0)
    if args.step_solver == "fixed-point":
        step_kwargs.update(
            anderson_depth=1, rebalance=True,
            linsolve_kwargs={"single_reduction": True})
    else:
        step_kwargs.update(
            step_solver="monolithic",
            multigroup_kwargs={
                "rtol": min(0.1 * args.tol_step, 1e-10),
                "inner_rtol": 1e-3,
                "precond_degree": 1,
            })
    result = solver.solve(**step_kwargs)
    communicator.Barrier()
    distributed_seconds = communicator.reduce(
        perf_counter() - started, op=MPI.MAX, root=0)
    gathered_flux = result.gather_flux(root=0)
    gathered_precursors = result.gather_precursors(root=0)
    communication = communicator.gather(
        context.communication_stats(), root=0)
    telemetry = communicator.gather({
        "rank": context.rank,
        "hostname": context.hostname,
        "device": context.device_identity,
        "owned_layers": [solver.partition.start, solver.partition.stop],
        "local_shape": solver.partition.local_shape,
    }, root=0)

    passed = True
    failure = ""
    summary = None
    if context.rank == 0:
        try:
            reference_steady = ExtrudedMeshDiffusionEigenSolver(
                initial.grid, initial.materials, initial.material_map,
                device=args.device, mix_material=initial.mix_material,
                mix_weight=initial.mix_weight, **common).solve(
                    tol_k=1e-10, tol_source=1e-10,
                    inner_rtol_floor=1e-12, verbose=True)
            reference_kwargs = dict(step_kwargs)
            reference_kwargs["initial_steady"] = reference_steady
            reference_kwargs["verbose"] = True
            reference = ExtrudedMeshTransientSolver(
                initial.grid, problem_at, kinetics,
                device=args.device, **common).solve(**reference_kwargs)

            flux_error = relative_l2(
                np.asarray(asnumpy(gathered_flux)),
                np.asarray(asnumpy(reference.flux)))
            precursor_error = relative_l2(
                np.asarray(asnumpy(gathered_precursors)),
                np.asarray(asnumpy(reference.precursors)))
            power_error = float(np.max(np.abs(result.power - reference.power)))
            k_error = abs(distributed_steady.k_eff - reference_steady.k_eff)
            initial_flux_error = relative_l2(
                np.asarray(asnumpy(gathered_initial_flux)),
                np.asarray(asnumpy(reference_steady.flux)))
            summary = {
                "status": "passed",
                "mpi_size": context.size,
                "device": args.device,
                "communication_mode": context.communication_mode,
                "global_refine": args.refine,
                "local_levels": args.local_levels,
                "effective_drum_refine": initial.drum_refine,
                "nz": args.nz,
                "groups": args.groups,
                "step_solver": args.step_solver,
                "radial_cells": initial.grid.mesh.n_cells,
                "spatial_cells": initial.grid.n_cells,
                "unknowns": initial.grid.n_cells * len(initial.materials[0].chi),
                "distributed_k": distributed_steady.k_eff,
                "serial_k": reference_steady.k_eff,
                "eigenvalue_error": k_error,
                "initial_flux_relative_l2_error": initial_flux_error,
                "maximum_power_error": power_error,
                "final_flux_relative_l2_error": flux_error,
                "final_precursor_relative_l2_error": precursor_error,
                "distributed_step_iterations": result.step_iterations,
                "serial_step_iterations": reference.step_iterations,
                "distributed_seconds": distributed_seconds,
                "communication": communication,
                "rank_telemetry": telemetry,
            }
            print("NDGPU_DISTRIBUTED_EXTRUDED_HPMR_COMPARISON "
                  + json.dumps(summary), flush=True)
            if not distributed_steady.converged or not reference_steady.converged:
                raise AssertionError("strict steady solve did not converge")
            if k_error > 2e-8:
                raise AssertionError(f"eigenvalue error {k_error} exceeds 2e-8")
            if initial_flux_error > 2e-7:
                raise AssertionError(
                    "initial flux relative error exceeds 2e-7")
            if power_error > 2e-7:
                raise AssertionError(f"power error {power_error} exceeds 2e-7")
            if flux_error > 2e-7 or precursor_error > 2e-7:
                raise AssertionError(
                    "final flux or precursor relative error exceeds 2e-7")
        except Exception as exc:
            passed = False
            failure = f"{type(exc).__name__}: {exc}"

    passed, failure = communicator.bcast((passed, failure), root=0)
    if not passed:
        raise AssertionError(f"root comparison failed: {failure}")
    if context.rank == 0:
        print("NDGPU_DISTRIBUTED_EXTRUDED_HPMR_GATE "
              + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
