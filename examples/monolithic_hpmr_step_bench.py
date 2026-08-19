"""CPU comparison of fixed-point and experimental monolithic HP-MR steps.

This is a development gate, not yet a public transient option.  It constructs
the exact backward-Euler block system after analytic precursor elimination,
solves it with flexible GMRES and an inexact energy-group sweep, and compares
against :class:`ndgpu.TransientSolver` from the same normalized steady state.

Run from the repository root::

    python examples/monolithic_hpmr_step_bench.py --refine 1
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from ndgpu.benchmarks.hpmr_transient_bench import (SIGMA_A_SCALE, build_case,
                                                    scale_absorption)
from ndgpu.linalg import fgmres
from ndgpu.multigroup import (AdjointCoarseCorrectionPreconditioner,
                              BlockJacobiPreconditioner,
                              EnergyModeCoarseCorrectionPreconditioner,
                              EnergyGroupGaussSeidelPreconditioner,
                              GalerkinCMFDPreconditioner,
                              MultigroupStepOperator,
                              SourceShapeCoarseCorrectionPreconditioner,
                              SpatialCoarseSpace)
from ndgpu.solver import Fields
from ndgpu.transient import TransientSolver
from ndgpu.tri import TriDiffusionEigenSolver, TriGroupOperator


def _delayed_terms(fields, kinetics, source, dt):
    """Return transient emission weights and fixed delayed source by group."""
    beta = [float(value) for value in kinetics.beta]
    decay = [float(value) for value in kinetics.decay]
    coefficients = [beta[i] / (1.0 + decay[i] * dt)
                    for i in range(kinetics.n_families)]
    precursors = [(beta[i] / decay[i]) * source
                  for i in range(kinetics.n_families)]
    decayed = [(decay[i] / (1.0 + decay[i] * dt)) * precursors[i]
               for i in range(kinetics.n_families)]
    G = fields.n_groups
    chi_d = kinetics.chi_delayed
    if chi_d is not None and chi_d.ndim == 2:
        weights, fixed = [], []
        for g in range(G):
            weights.append(fields.chi[g] - sum(
                chi_d[i, g] * coefficients[i]
                for i in range(kinetics.n_families)))
            fixed.append(sum(chi_d[i, g] * decayed[i]
                             for i in range(kinetics.n_families)))
        return weights, fixed
    decayed_sum = sum(decayed)
    if chi_d is not None:
        return ([fields.chi[g] - chi_d[g] * sum(coefficients)
                 for g in range(G)],
                [chi_d[g] * decayed_sum for g in range(G)])
    return ([fields.chi[g] * (1.0 - sum(coefficients)) for g in range(G)],
            [fields.chi[g] * decayed_sum for g in range(G)])


def benchmark(refine=1, dt=0.02, fixed_tol=1e-8, krylov_tol=1e-9,
              scatter_sweeps=1, include_fission=False, precond_degree=1,
              coarse_correction=False, spatial_cmfd=False, cmfd_factor=3,
              cmfd_mode="multiplicative", cmfd_labels="material",
              cmfd_smoother="group", energy_correction=False,
              energy_order="forward", source_correction=False,
              source_bins=1, source_labels="material", energy_anderson=0,
              energy_relaxation=1.0, inner_rtol=1e-3,
              inner_fixed_iterations=0, inner_fixed_relaxations=0):
    p, kinetics = build_case(refine=refine, nz=0, groups="11")
    perturbed = scale_absorption(p.materials, SIGMA_A_SCALE)

    steady_solver = TriDiffusionEigenSolver(
        p.grid, p.materials, p.material_map, bc=p.bc, active=p.active,
        mask_bc=p.mask_bc, mix_material=p.mix_material,
        mix_weight=p.mix_weight, device="cpu", precond_degree=precond_degree)
    steady = steady_solver.solve(tol_k=1e-9, tol_source=1e-8)
    if not steady.converged:
        raise RuntimeError("initial HP-MR steady solve did not converge")
    k0 = steady.k_eff

    fields0 = steady_solver.fields
    phi0 = np.asarray(steady.flux).copy()
    source0 = fields0.fission_source(phi0) / k0
    scale = 1.0 / float(np.sum(source0))
    phi0 *= scale
    source0 *= scale

    setup_start = time.perf_counter()
    fields = Fields(np, p.grid, perturbed, p.material_map, np.float64,
                    mix_material=p.mix_material, mix_weight=p.mix_weight)
    G = fields.n_groups
    inv_vdt = [1.0 / (float(kinetics.velocities[g]) * dt) for g in range(G)]
    operators = [TriGroupOperator(
        np, p.grid, fields.diffusion[g], fields.removal[g] + inv_vdt[g],
        bc=p.bc, active=p.active, mask_bc=p.mask_bc) for g in range(G)]
    emission, delayed = _delayed_terms(fields, kinetics, source0, dt)
    block = MultigroupStepOperator(
        operators, fields.sigma_s, fields.nu_sigma_f, emission, k0,
        rhs_weight=getattr(operators[0], "rhs_weight", None))
    rhs = block.fixed_rhs(inv_vdt, phi0, delayed)
    sweep = EnergyGroupGaussSeidelPreconditioner(
        block, scatter_sweeps=scatter_sweeps,
        include_fission=include_fission, inner_rtol=inner_rtol,
        inner_maxiter=300, precond_degree=precond_degree,
        ordering=energy_order, anderson_depth=energy_anderson,
        relaxation=energy_relaxation,
        inner_fixed_iterations=inner_fixed_iterations,
        inner_fixed_relaxations=inner_fixed_relaxations)
    adjoint_mode = None
    if coarse_correction or energy_correction or source_correction:
        adjoint_solver = TriDiffusionEigenSolver(
            p.grid, p.materials, p.material_map, bc=p.bc, active=p.active,
            mask_bc=p.mask_bc, mix_material=p.mix_material,
            mix_weight=p.mix_weight, device="cpu",
            precond_degree=precond_degree)
        adjoint = adjoint_solver.solve(
            tol_k=1e-9, tol_source=1e-8, adjoint=True)
        if not adjoint.converged:
            raise RuntimeError("initial HP-MR adjoint solve did not converge")
        adjoint_mode = np.asarray(adjoint.flux)
    preconditioner = sweep
    cmfd = None
    energy_coarse = None
    source_coarse = None
    if spatial_cmfd:
        if cmfd_smoother == "jacobi":
            preconditioner = BlockJacobiPreconditioner(block.inv_diag)
        elif cmfd_smoother != "group":
            raise ValueError("cmfd_smoother must be group or jacobi")
        if cmfd_labels == "material":
            labels = np.where(p.active, p.material_map, -1)
        elif cmfd_labels == "active":
            labels = np.where(p.active, 1, -1)
        elif cmfd_labels == "none":
            labels = None
        else:
            raise ValueError("cmfd_labels must be material, active, or none")
        coarse_space = SpatialCoarseSpace(
            block.cell_shape,
            factors=(int(cmfd_factor), int(cmfd_factor), 2),
            labels=labels)
        cmfd = GalerkinCMFDPreconditioner(
            block, preconditioner, coarse_space, mode=cmfd_mode)
        preconditioner = cmfd
    setup_seconds = time.perf_counter() - setup_start

    solve_start = time.perf_counter()
    if source_correction:
        nx, ny = block.cell_shape[:2]
        ix, iy = np.indices(block.cell_shape)[:2]
        bx = np.minimum(int(source_bins) * ix // nx, int(source_bins) - 1)
        by = np.minimum(int(source_bins) * iy // ny, int(source_bins) - 1)
        regions = bx + int(source_bins) * by
        if source_labels == "material":
            material = np.asarray(p.material_map).copy()
            if p.mix_material is not None:
                mixed = np.asarray(p.mix_material)
                material = np.where(mixed >= 0, mixed, material)
            regions = regions * (int(material.max()) + 1) + material
        elif source_labels != "active":
            raise ValueError("source_labels must be material or active")
        region_ids = np.unique(regions[np.asarray(p.active)])
        spatial_modes = np.stack([
            (np.asarray(p.active) & (regions == region)).astype(np.float64)
            for region in region_ids])
        source_coarse = SourceShapeCoarseCorrectionPreconditioner(
            block, sweep, phi0, adjoint_mode, spatial_modes)
        preconditioner = source_coarse
    elif energy_correction:
        energy_coarse = EnergyModeCoarseCorrectionPreconditioner(
            block, sweep, phi0, adjoint_mode)
        preconditioner = energy_coarse
    elif adjoint_mode is not None:
        # This calibration is repeated with the latest accepted forward shape
        # by the production driver, so charge its one base sweep to the timed
        # step and to the inner-work counter. Only the fresh adjoint is startup.
        preconditioner = AdjointCoarseCorrectionPreconditioner(
            block, sweep, phi0, adjoint_mode)
    mono_flux, outer = fgmres(
        block.apply, rhs, phi0, block.inv_diag, np, precond=preconditioner,
        rtol=krylov_tol, restart=30, maxiter=300)
    mono_seconds = time.perf_counter() - solve_start
    residual = np.linalg.norm(block.apply(mono_flux) - rhs) / np.linalg.norm(rhs)
    mono_power = float(np.sum(fields.fission_source(mono_flux) / k0))

    problem_at = lambda t: ((perturbed if t >= 0.5 * dt else p.materials),
                            p.material_map, p.mix_material, p.mix_weight)
    production = TransientSolver(
        p.grid, problem_at, kinetics, bc=p.bc, active=p.active,
        mask_bc=p.mask_bc, mix_material=p.mix_material,
        mix_weight=p.mix_weight, group_operator=TriGroupOperator,
        eig_solver=TriDiffusionEigenSolver, precond_degree=precond_degree,
        device="cpu")
    fixed_start = time.perf_counter()
    reference = production.solve(
        t_end=dt, dt=dt, tol_step=fixed_tol, max_sweeps=5000,
        anderson_depth=1, rebalance=True, scatter_subsweeps=6,
        initial_steady=steady)
    fixed_seconds = time.perf_counter() - fixed_start
    fixed_flux = np.asarray(reference.flux)
    flux_error = (np.linalg.norm(mono_flux - fixed_flux)
                  / np.linalg.norm(fixed_flux))

    return dict(refine=refine, cells=int(np.count_nonzero(p.active)),
                groups=G, dof=G*int(np.count_nonzero(p.active)), k0=k0,
                fixed_seconds=fixed_seconds,
                fixed_sweeps=reference.step_iterations[0],
                fixed_inner=reference.total_inner_iterations,
                fixed_power=float(reference.power[-1]),
                monolithic_setup_seconds=setup_seconds,
                monolithic_seconds=mono_seconds,
                monolithic_outer=outer,
                monolithic_inner=sweep.stats.inner_iterations,
                monolithic_inner_by_group=tuple(
                    sweep.stats.inner_iterations_by_group),
                monolithic_group_solves=sweep.stats.group_solves,
                monolithic_power=mono_power, residual=residual,
                coarse_correction=bool(coarse_correction),
                energy_correction=bool(energy_correction),
                energy_order=energy_order,
                energy_anderson=int(energy_anderson),
                energy_relaxation=float(energy_relaxation),
                inner_rtol=(float(inner_rtol) if np.ndim(inner_rtol) == 0
                            else tuple(float(value) for value in inner_rtol)),
                inner_fixed_iterations=int(inner_fixed_iterations),
                inner_fixed_relaxations=int(inner_fixed_relaxations),
                energy_coarse_unknowns=(energy_coarse.coarse_unknowns
                                        if energy_coarse is not None else 0),
                energy_coarse_condition=(energy_coarse.condition
                                         if energy_coarse is not None else 0.0),
                energy_coarse_setup_seconds=(energy_coarse.stats.setup_seconds
                                             if energy_coarse is not None else 0.0),
                energy_coarse_storage_bytes=(energy_coarse.storage_bytes
                                            if energy_coarse is not None else 0),
                energy_coarse_applications=(energy_coarse.stats.applications
                                           if energy_coarse is not None else 0),
                source_correction=bool(source_correction),
                source_bins=(int(source_bins) if source_coarse is not None else 0),
                source_labels=(source_labels if source_coarse is not None else None),
                source_coarse_unknowns=(source_coarse.coarse_unknowns
                                        if source_coarse is not None else 0),
                source_coarse_condition=(source_coarse.condition
                                         if source_coarse is not None else 0.0),
                source_coarse_setup_seconds=(source_coarse.stats.setup_seconds
                                             if source_coarse is not None else 0.0),
                source_coarse_storage_bytes=(source_coarse.storage_bytes
                                            if source_coarse is not None else 0),
                source_coarse_applications=(source_coarse.stats.applications
                                           if source_coarse is not None else 0),
                spatial_cmfd=bool(spatial_cmfd),
                cmfd_mode=(cmfd_mode if cmfd is not None else None),
                cmfd_factor=(int(cmfd_factor) if cmfd is not None else None),
                cmfd_labels=(cmfd_labels if cmfd is not None else None),
                cmfd_smoother=(cmfd_smoother if cmfd is not None else None),
                cmfd_coarse_cells=(cmfd.coarse_cells if cmfd is not None else 0),
                cmfd_coarse_unknowns=(cmfd.coarse_unknowns
                                      if cmfd is not None else 0),
                cmfd_setup_seconds=(cmfd.stats.setup_seconds
                                    if cmfd is not None else 0.0),
                cmfd_applications=(cmfd.stats.applications
                                   if cmfd is not None else 0),
                cmfd_fine_residual_applies=(
                    cmfd.stats.fine_residual_applies
                    if cmfd is not None else 0),
                flux_error=flux_error,
                power_error=abs(mono_power - float(reference.power[-1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--fixed-tol", type=float, default=1e-8)
    parser.add_argument("--krylov-tol", type=float, default=1e-9)
    parser.add_argument("--scatter-sweeps", type=int, default=1)
    parser.add_argument("--precond-degree", type=int, default=1)
    parser.add_argument("--include-fission", action="store_true")
    parser.add_argument("--coarse-correction", action="store_true")
    parser.add_argument("--energy-correction", action="store_true")
    parser.add_argument("--energy-order",
                        choices=("forward", "reverse", "alternating"),
                        default="forward")
    parser.add_argument("--energy-anderson", type=int, choices=(0, 1, 2),
                        default=0)
    parser.add_argument("--energy-relaxation", type=float, default=1.0)
    parser.add_argument("--inner-rtol", type=float, default=1e-3)
    parser.add_argument("--inner-fixed-iterations", type=int, default=0)
    parser.add_argument("--inner-fixed-relaxations", type=int, default=0)
    parser.add_argument("--source-correction", action="store_true")
    parser.add_argument("--source-bins", type=int, default=1)
    parser.add_argument("--source-labels", choices=("material", "active"),
                        default="material")
    parser.add_argument("--spatial-cmfd", action="store_true")
    parser.add_argument("--cmfd-factor", type=int, default=3)
    parser.add_argument("--cmfd-mode",
                        choices=("multiplicative", "coarse-first", "additive"),
                        default="multiplicative")
    parser.add_argument("--cmfd-labels", choices=("material", "active", "none"),
                        default="material")
    parser.add_argument("--cmfd-smoother", choices=("group", "jacobi"),
                        default="group")
    args = parser.parse_args()
    result = benchmark(
        refine=args.refine, dt=args.dt, fixed_tol=args.fixed_tol,
        krylov_tol=args.krylov_tol, scatter_sweeps=args.scatter_sweeps,
        precond_degree=args.precond_degree,
        include_fission=args.include_fission,
        coarse_correction=args.coarse_correction,
        energy_correction=args.energy_correction,
        energy_order=args.energy_order,
        energy_anderson=args.energy_anderson,
        energy_relaxation=args.energy_relaxation,
        inner_rtol=args.inner_rtol,
        inner_fixed_iterations=args.inner_fixed_iterations,
        inner_fixed_relaxations=args.inner_fixed_relaxations,
        source_correction=args.source_correction,
        source_bins=args.source_bins, source_labels=args.source_labels,
        spatial_cmfd=args.spatial_cmfd, cmfd_factor=args.cmfd_factor,
        cmfd_mode=args.cmfd_mode, cmfd_labels=args.cmfd_labels,
        cmfd_smoother=args.cmfd_smoother)
    for key, value in result.items():
        print(f"{key:28s}: {value}")


if __name__ == "__main__":
    main()
