"""Verification of tensor-product unstructured-mesh extrusion."""

import numpy as np
import pytest

from ndgpu import PWR_TWO_GROUP
from ndgpu.extruded_mesh import (ExtrudedMeshDiffusionEigenSolver,
                                 ExtrudedMeshGrid, ExtrudedMeshGroupOperator,
                                 ExtrudedMeshTransientSolver)
from ndgpu.mesh import (UnstructuredDiffusionSolver, assemble_mesh,
                        assemble_mesh_3d)


def _base_mesh():
    coords = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.5],
        [0.0, 1.5],
        [0.8, 0.6],
    ])
    cells = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    return assemble_mesh(coords, cells, np.arange(len(cells)))


def _explicit_prisms(base, nz, height):
    nnode = len(base.coords)
    dz = height / nz
    coords = np.empty(((nz + 1) * nnode, 3))
    for level in range(nz + 1):
        block = slice(level * nnode, (level + 1) * nnode)
        coords[block, :2] = base.coords
        coords[block, 2] = level * dz
    cells = []
    tags = []
    for cell_index, triangle in enumerate(base.cells):
        for level in range(nz):
            lower = level * nnode
            upper = (level + 1) * nnode
            cells.append(tuple(node + lower for node in triangle)
                         + tuple(node + upper for node in triangle))
            tags.append(cell_index)
    return assemble_mesh_3d(coords, cells, tags)


def _explicit_apply(mesh, diffusion, removal, flux, alpha):
    diagonal = removal * mesh.area
    result = diagonal * flux
    for left, right, measure, distance in mesh.faces:
        weight = (2.0 * diffusion[left] * diffusion[right]
                  / (diffusion[left] + diffusion[right])
                  * measure / distance)
        result[left] += weight * (flux[left] - flux[right])
        result[right] += weight * (flux[right] - flux[left])
        diagonal[left] += weight
        diagonal[right] += weight
    for cell, measure, distance in mesh.bfaces:
        weight = (alpha * diffusion[cell] * measure
                  / (distance * alpha + diffusion[cell]))
        result[cell] += weight * flux[cell]
        diagonal[cell] += weight
    return result, diagonal


def test_extruded_operator_matches_explicit_conforming_prisms():
    base = _base_mesh()
    grid = ExtrudedMeshGrid(base, height=9.0, nz=3)
    explicit = _explicit_prisms(base, grid.nz, grid.height)
    rng = np.random.default_rng(20260902)
    diffusion = 0.2 + rng.random(grid.shape)
    removal = 0.01 + 0.1 * rng.random(grid.shape)
    flux = rng.random(grid.shape)

    operator = ExtrudedMeshGroupOperator(
        np, grid, diffusion, removal, bc="vacuum", mask_bc="vacuum")
    expected, expected_diagonal = _explicit_apply(
        explicit, diffusion.ravel(), removal.ravel(), flux.ravel(), alpha=0.5)

    np.testing.assert_allclose(
        grid.cell_volumes.ravel(), explicit.area, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(
        operator.diag.ravel(), expected_diagonal, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        operator.apply(flux).ravel(), expected, rtol=2e-14, atol=2e-14)

    out = np.empty_like(flux)
    assert operator.apply(flux, out=out) is out
    np.testing.assert_allclose(out.ravel(), expected, rtol=2e-14, atol=2e-14)


def test_extruded_grid_rejects_invalid_geometry():
    base = _base_mesh()
    for height, nz in ((0.0, 2), (1.0, 0), (1.0, 1.5)):
        try:
            ExtrudedMeshGrid(base, height=height, nz=nz)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid extrusion was accepted")


def test_extruded_eigen_solver_matches_explicit_prism_mesh():
    base = _base_mesh()
    grid = ExtrudedMeshGrid(base, height=9.0, nz=3)
    explicit = _explicit_prisms(base, grid.nz, grid.height)
    material_map = np.zeros(grid.shape, dtype=np.int64)

    result = ExtrudedMeshDiffusionEigenSolver(
        grid, [PWR_TWO_GROUP], material_map, bc="vacuum",
        mask_bc="vacuum", device="cpu").solve(
            tol_k=1e-9, tol_source=1e-8)
    reference = UnstructuredDiffusionSolver(
        explicit, [PWR_TWO_GROUP], material_map.ravel(),
        alpha_boundary=0.5, device="cpu").solve(
            tol_k=1e-9, tol_source=1e-8)

    assert result.converged and reference.converged
    assert result.k_eff == pytest.approx(reference.k_eff, abs=2e-9)


def test_hpmr_local_extrusion_maps_axial_reflectors_and_blends():
    from ndgpu.benchmarks.hpmr import (AXIAL_REFLECTOR, CENTRAL, FUEL,
                                       build_hpmr3d_local)

    problem = build_hpmr3d_local(
        refine=1, nz=10, drum_angle_deg=90.0,
        drum_refine_levels=1, absorber="polar", samples=0)

    assert problem.grid.shape == problem.material_map.shape
    assert problem.grid.shape == problem.active.shape
    assert problem.grid.shape == problem.mix_material.shape
    assert problem.grid.shape == problem.mix_weight.shape
    assert problem.drum_refine == 2
    assert np.all(problem.active)
    assert np.any(problem.material_map[:, 0] == AXIAL_REFLECTOR)
    assert np.any(problem.material_map[:, -1] == AXIAL_REFLECTOR)
    assert not np.any(np.isin(
        problem.material_map[:, (0, -1)], (FUEL, CENTRAL)))
    assert np.any(np.isin(problem.material_map[:, 5], (FUEL, CENTRAL)))
    np.testing.assert_array_equal(
        problem.mix_weight[:, 0], problem.mix_weight[:, -1])

    with pytest.raises(ValueError, match="20 cm axial reflectors"):
        build_hpmr3d_local(refine=1, nz=9, drum_refine_levels=0)


def test_hpmr_local_drum_rotation_reuses_topology_and_matches_rebuild():
    from ndgpu.benchmarks.hpmr import (build_hpmr3d_local,
                                       with_hpmr3d_local_drum_angle)

    initial = build_hpmr3d_local(
        refine=1, nz=10, drum_angle_deg=90.0,
        drum_refine_levels=1, absorber="polar", samples=0)
    rotated = with_hpmr3d_local_drum_angle(initial, 93.88, samples=0)
    rebuilt = build_hpmr3d_local(
        refine=1, nz=10, drum_angle_deg=93.88,
        drum_refine_levels=1, absorber="polar", samples=0)

    assert rotated.grid is initial.grid
    assert rotated.materials is initial.materials
    assert rotated.material_map is initial.material_map
    assert rotated.active is initial.active
    np.testing.assert_array_equal(rotated.mix_material, rebuilt.mix_material)
    np.testing.assert_allclose(
        rotated.mix_weight, rebuilt.mix_weight, rtol=0.0, atol=1e-15)
    assert not np.array_equal(rotated.mix_weight, initial.mix_weight)


def test_hpmr_local_extruded_transient_preserves_unperturbed_equilibrium():
    from ndgpu.benchmarks.hpmr import build_hpmr3d_local

    problem = build_hpmr3d_local(
        refine=1, nz=10, drum_angle_deg=90.0,
        drum_refine_levels=0, absorber="polar", samples=0)

    transient = ExtrudedMeshTransientSolver(
        problem.grid,
        lambda time: (problem.materials, problem.material_map,
                      problem.mix_material, problem.mix_weight),
        problem.kinetics, bc=problem.bc, active=problem.active,
        mask_bc=problem.mask_bc, device="cpu").solve(
            t_end=0.01, dt=0.01, tol_step=1e-7, rebalance=True,
            steady_kwargs={"tol_k": 1e-8, "tol_source": 1e-7})

    assert transient.steady.converged
    assert transient.power[-1] == pytest.approx(1.0, abs=2e-7)
