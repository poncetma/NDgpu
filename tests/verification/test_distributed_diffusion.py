"""Correctness gates for the dependency-safe distributed foundation."""

import numpy as np
import pytest

from ndgpu import (DiffusionEigenSolver, DistributedDiffusionEigenSolver,
                   DistributedCartesianGroupOperator,
                   DistributedTriGroupOperator,
                   DistributedTriDiffusionEigenSolver,
                   DistributedTriTransientSolver, Grid, PWR_TWO_GROUP,
                   TriDiffusionEigenSolver, TriGrid)
from ndgpu.distributed import (CartesianSlabPartition, DistributedContext,
                               DistributedResult, SerialReductions,
                               TriRowPartition, _cuda_device_identity)
from ndgpu.linalg import PCGWorkspace, pcg
from ndgpu.stencil import GroupOperator
from ndgpu.tri import TriGroupOperator


class _MirroredCommunicator:
    """Two identical CPU ranks represented inside one unit-test process."""

    def __init__(self):
        self.allreduce_calls = 0

    def Allreduce(self, send, receive, op=None):
        del op
        self.allreduce_calls += 1
        receive[...] = 2.0 * send

    def Sendrecv(self, send, *, dest, sendtag, recvbuf, source, recvtag):
        assert (dest, source, sendtag, recvtag) == (1, 1, 7, 7)
        recvbuf[...] = send


class _CompletedRequest:
    pass


class _NonblockingMPI:
    class Request:
        @staticmethod
        def Waitall(requests):
            assert requests


class _NonblockingCommunicator:
    def __init__(self, receives):
        self.receives = receives
        self.sends = []

    def Irecv(self, receive, *, source, tag):
        receive[...] = self.receives[source, tag]
        return _CompletedRequest()

    def Isend(self, send, *, dest, tag):
        self.sends.append((dest, tag, np.array(send, copy=True)))
        return _CompletedRequest()


class _CountingReductions(SerialReductions):
    def __init__(self, xp):
        super().__init__(xp)
        self.calls = {"dot": 0, "sum": 0, "dot_many": 0}

    def dot(self, left, right):
        self.calls["dot"] += 1
        return super().dot(left, right)

    def sum(self, value):
        self.calls["sum"] += 1
        return super().sum(value)

    def dot_many(self, pairs):
        self.calls["dot_many"] += 1
        return super().dot_many(pairs)


class _QueuedHaloContext(DistributedContext):
    """Serve exact neighboring planes from queued global test arrays."""

    def __init__(self, partition, global_values):
        super().__init__(
            object(), np, rank=partition.rank, size=partition.size,
            local_rank=partition.rank, communication_mode="cpu-mpi",
            hostname="in-process-test")
        self.global_values = iter(global_values)

    def exchange_halos(self, value, partition, *, tag=0):
        del tag
        global_value = next(self.global_values)
        np.testing.assert_array_equal(value, global_value[partition.owned_slice])

        def plane(index):
            sl = [slice(None)] * global_value.ndim
            sl[partition.axis] = index
            return np.array(global_value[tuple(sl)], copy=True)

        lower = None if partition.lower_rank is None else plane(partition.start - 1)
        upper = None if partition.upper_rank is None else plane(partition.stop)
        return lower, upper


def _mirrored_context():
    communicator = _MirroredCommunicator()
    context = DistributedContext(
        communicator, np, rank=0, size=2, local_rank=0,
        communication_mode="cpu-mpi", hostname="test-host")
    return context, communicator


def test_cartesian_partition_balances_longest_axis_and_gather_metadata():
    partitions = [
        CartesianSlabPartition.create((4, 10, 2), rank, 3)
        for rank in range(3)
    ]
    assert [part.axis for part in partitions] == [1, 1, 1]
    assert [(part.start, part.stop) for part in partitions] == [
        (0, 4), (4, 7), (7, 10)]
    assert partitions[1].local_shape == (4, 3, 2)
    assert partitions[1].owned_slice == (slice(None), slice(4, 7), slice(None))
    assert partitions[1].counts == (32, 24, 24)
    assert partitions[1].displacements == (0, 32, 56)
    assert partitions[1].global_index((3, 1, 1)) == (3, 5, 1)
    assert partitions[0].lower_rank is None
    assert partitions[1].lower_rank == 0 and partitions[1].upper_rank == 2
    assert partitions[2].upper_rank is None


def test_tri_partition_owns_complete_rows_and_rejects_empty_ranks():
    partition = TriRowPartition.create((11, 7, 2, 5), rank=2, size=4)
    assert (partition.start, partition.stop) == (6, 9)
    assert partition.local_shape == (3, 7, 2, 5)
    with pytest.raises(ValueError, match="triangular shape"):
        TriRowPartition.create((11, 7, 3), rank=0, size=2)
    with pytest.raises(ValueError, match="non-empty ranks"):
        TriRowPartition.create((2, 7, 2), rank=0, size=3)


def test_serial_context_and_reductions_do_not_require_mpi():
    context = DistributedContext.serial("cpu")
    values = np.arange(6.0)
    assert context.describe()["communication_mode"] == "serial"
    assert context.reductions.dot(values, values) == 55.0
    assert context.reductions.sum(values) == 15.0
    np.testing.assert_array_equal(
        context.reductions.sum_many((values, values * values)),
        np.asarray([15.0, 55.0]))
    np.testing.assert_array_equal(
        context.reductions.dot_many(((values, values), (values, values + 1))),
        np.asarray([55.0, 70.0]))
    assert isinstance(context.reductions, SerialReductions)


def test_cuda_identity_preserves_string_pci_bus_id():
    class Runtime:
        @staticmethod
        def deviceGetUuid(device_id):
            del device_id
            raise RuntimeError("UUID API unavailable")

        @staticmethod
        def deviceGetPCIBusId(device_id):
            assert device_id == 1
            return "0039:01:00.0"

    class XP:
        class cuda:
            runtime = Runtime()

    assert _cuda_device_identity(XP, 1) == "0039:01:00.0"


def test_cpu_mpi_context_reduces_and_exchanges_backend_arrays():
    context, communicator = _mirrored_context()
    values = np.arange(5.0)
    assert context.reductions.dot(values, values) == 60.0
    assert context.reductions.sum(values) == 20.0
    assert context.reductions.max(values) == 8.0
    np.testing.assert_array_equal(
        context.sendrecv(values, destination=1, source=1, tag=7), values)
    assert communicator.allreduce_calls == 3
    stats = context.communication_stats()
    assert stats["allreduce_calls"] == 3
    assert stats["allreduce_bytes"] == 3 * values.dtype.itemsize
    assert stats["sendrecv_calls"] == 1
    assert stats["sendrecv_bytes"] == values.nbytes
    assert stats["communication_seconds"] >= 0.0
    context.reset_communication_stats()
    assert context.communication_stats()["communication_seconds"] == 0.0
    assert context.communication_stats()["allreduce_calls"] == 0


@pytest.mark.parametrize(
    "rank, receive_tag, send_tag, send_index, receive_position",
    [(0, 12, 11, -1, "upper"), (1, 11, 12, 0, "lower")],
)
def test_batched_halo_exchange_posts_both_directions_once(
        rank, receive_tag, send_tag, send_index, receive_position):
    partition = TriRowPartition.create((6, 4, 2), rank=rank, size=2)
    values = np.arange(np.prod(partition.local_shape), dtype=float).reshape(
        partition.local_shape)
    neighbor = 1 - rank
    expected_receive = np.full(values.shape[1:], 70.0 + rank)
    communicator = _NonblockingCommunicator({
        (neighbor, receive_tag): expected_receive,
    })
    context = DistributedContext(
        communicator, np, rank=rank, size=2, local_rank=rank,
        communication_mode="cpu-mpi", hostname="test-host",
        batched_halos=True, _mpi=_NonblockingMPI)

    lower, upper = context.exchange_halos(values, partition, tag=11)

    received = upper if receive_position == "upper" else lower
    physical = lower if receive_position == "upper" else upper
    assert physical is None
    np.testing.assert_array_equal(received, expected_receive)
    assert len(communicator.sends) == 1
    destination, actual_tag, sent = communicator.sends[0]
    assert destination == neighbor and actual_tag == send_tag
    np.testing.assert_array_equal(sent, values[send_index])
    stats = context.communication_stats()
    assert stats["sendrecv_calls"] == 1
    assert stats["sendrecv_bytes"] == values[send_index].nbytes


@pytest.mark.parametrize("axis,size", [(0, 3), (1, 2), (2, 2)])
def test_cartesian_slab_operator_matches_serial_across_each_axis(axis, size):
    shape = (6, 5, 2)
    grid = Grid(shape=shape, size=(9.0, 11.0, 4.0))
    rng = np.random.default_rng(481 + axis)
    diffusion = 0.15 + rng.random(shape)
    removal = 0.01 + 0.2 * rng.random(shape)
    flux = rng.random(shape)
    bc = (("vacuum", "zero-flux"),
          ("reflective", 0.2),
          ("zero-flux", "vacuum"))
    reference = GroupOperator(
        np, grid, diffusion.copy(), removal.copy(), bc=bc)
    expected = reference.apply(flux)
    gathered = np.empty_like(expected)
    gathered_diag = np.empty_like(reference.diag)

    for rank in range(size):
        partition = CartesianSlabPartition.create(
            shape, rank=rank, size=size, axis=axis)
        owned = partition.owned_slice
        context = _QueuedHaloContext(partition, [diffusion, flux])
        operator = DistributedCartesianGroupOperator(
            np, grid, np.array(diffusion[owned], copy=True),
            np.array(removal[owned], copy=True), partition, context, bc=bc)
        gathered[owned] = operator.apply(np.array(flux[owned], copy=True))
        gathered_diag[owned] = operator.diag

    np.testing.assert_allclose(gathered_diag, reference.diag, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(gathered, expected, rtol=2e-15, atol=2e-15)


def test_cartesian_slab_operator_preserves_mask_boundary_across_mpi_cut():
    shape = (6, 4, 3)
    grid = Grid(shape=shape, size=(12.0, 8.0, 6.0))
    rng = np.random.default_rng(912)
    diffusion = 0.2 + rng.random(shape)
    removal = 0.03 + rng.random(shape)
    flux = rng.random(shape)
    active = np.ones(shape, dtype=bool)
    active[2, 1, 1] = False
    active[3, 2, 0] = False
    reference = GroupOperator(
        np, grid, diffusion.copy(), removal.copy(), active=active,
        bc="vacuum", mask_bc="zero-flux")
    expected = reference.apply(flux)
    gathered = np.empty_like(expected)
    gathered_diag = np.empty_like(reference.diag)

    for rank in range(2):
        partition = CartesianSlabPartition.create(
            shape, rank=rank, size=2, axis=0)
        owned = partition.owned_slice
        context = _QueuedHaloContext(partition, [diffusion, active, flux])
        operator = DistributedCartesianGroupOperator(
            np, grid, np.array(diffusion[owned], copy=True),
            np.array(removal[owned], copy=True), partition, context,
            active=np.array(active[owned], copy=True), bc="vacuum",
            mask_bc="zero-flux")
        gathered[owned] = operator.apply(np.array(flux[owned], copy=True))
        gathered_diag[owned] = operator.diag

    np.testing.assert_allclose(gathered_diag, reference.diag, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(gathered, expected, rtol=2e-15, atol=2e-15)


@pytest.mark.parametrize("extruded", [False, True])
def test_tri_row_operator_matches_serial_across_active_partition(extruded):
    shape = (8, 7, 2, 3) if extruded else (8, 7, 2)
    grid = TriGrid(shape, side=1.7, height=6.0)
    rng = np.random.default_rng(1103 + extruded)
    diffusion = 0.2 + rng.random(shape)
    removal = 0.01 + 0.2 * rng.random(shape)
    flux = rng.random(shape)
    active = np.ones(shape, dtype=bool)
    active[0] = active[-1] = False
    active[:, 0] = active[:, -1] = False
    # Exercise both connected and active-to-void faces across the row cut.
    active[3, 2, 0] = False
    active[4, 4, 1] = False
    reference = TriGroupOperator(
        np, grid, diffusion.copy(), removal.copy(), active=active,
        bc=(("reflective", "reflective"),
            ("reflective", "reflective"),
            ("vacuum", "zero-flux")),
        mask_bc="vacuum")
    expected = reference.apply(flux)
    gathered = np.empty_like(expected)
    gathered_diag = np.empty_like(reference.diag)

    for rank in range(2):
        partition = TriRowPartition.create(shape, rank=rank, size=2)
        owned = partition.owned_slice
        context = _QueuedHaloContext(partition, [diffusion, active, flux])
        operator = DistributedTriGroupOperator(
            np, grid, np.array(diffusion[owned], copy=True),
            np.array(removal[owned], copy=True), partition, context,
            active=np.array(active[owned], copy=True),
            bc=(("reflective", "reflective"),
                ("reflective", "reflective"),
                ("vacuum", "zero-flux")),
            mask_bc="vacuum")
        gathered[owned] = operator.apply(np.array(flux[owned], copy=True))
        gathered_diag[owned] = operator.diag

    np.testing.assert_allclose(gathered_diag, reference.diag, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(gathered, expected, rtol=2e-15, atol=2e-15)


def test_pcg_uses_global_reductions_without_changing_recurrence():
    matrix = np.array([
        [4.0, -1.0, 0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0, 3.0],
    ])
    rhs = np.array([15.0, 10.0, 10.0])
    inverse_diagonal = 1.0 / np.diag(matrix)
    start = np.zeros_like(rhs)
    reference, reference_iterations = pcg(
        matrix.__matmul__, rhs, start, inverse_diagonal, np, rtol=1e-13)

    context, communicator = _mirrored_context()
    solved, iterations = pcg(
        matrix.__matmul__, rhs, start, inverse_diagonal, np, rtol=1e-13,
        reductions=context.reductions)
    np.testing.assert_array_equal(solved, reference)
    assert iterations == reference_iterations
    assert communicator.allreduce_calls > iterations


def test_single_reduction_pcg_uses_one_collective_per_iteration():
    matrix = np.array([
        [4.0, -1.0, 0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0, 3.0],
    ])
    rhs = np.array([15.0, 10.0, 10.0])
    inverse_diagonal = 1.0 / np.diag(matrix)
    context, communicator = _mirrored_context()

    solved, iterations = pcg(
        matrix.__matmul__, rhs, np.zeros_like(rhs), inverse_diagonal, np,
        rtol=1e-13, reductions=context.reductions,
        single_reduction=True)

    np.testing.assert_allclose(
        solved, np.linalg.solve(matrix, rhs), rtol=2e-13, atol=2e-13)
    assert iterations > 0
    # One norm before the recurrence, one packed initialization, then one
    # packed reduction for each iteration.
    assert communicator.allreduce_calls == iterations + 2


def test_distributed_reductions_disable_pcg_graph_blocks():
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    rhs = np.array([1.0, 0.0])
    context, _ = _mirrored_context()
    workspace = PCGWorkspace.like(rhs, operator_out=True)

    def apply(value, out=None):
        result = matrix @ value
        if out is None:
            return result
        out[...] = result
        return out

    solved, _ = pcg(
        apply, rhs, np.zeros_like(rhs), 1.0 / np.diag(matrix), np,
        rtol=1e-12, check_every=2, graph_block=2, workspace=workspace,
        reductions=context.reductions)
    np.testing.assert_allclose(solved, np.linalg.solve(matrix, rhs))
    assert workspace.graph_error == (
        "CUDA graph capture is disabled for distributed reductions")


def test_size_one_power_iteration_matches_serial_reduction_order():
    grid = Grid(shape=(5, 4, 3), size=(50.0, 40.0, 30.0))
    serial_solver = DiffusionEigenSolver(grid, PWR_TWO_GROUP, device="cpu")
    provider_solver = DiffusionEigenSolver(grid, PWR_TWO_GROUP, device="cpu")

    reference = serial_solver.solve(tol_k=1e-9, tol_source=1e-8)
    reductions = _CountingReductions(np)
    result = provider_solver.solve(
        tol_k=1e-9, tol_source=1e-8, reductions=reductions)

    assert result.k_eff == reference.k_eff
    assert result.k_history == reference.k_history
    assert result.source_error_history == reference.source_error_history
    assert result.outer_iterations == reference.outer_iterations
    assert result.inner_iterations == reference.inner_iterations
    np.testing.assert_array_equal(result.flux, reference.flux)
    assert reductions.calls["dot"] > result.inner_iterations
    assert reductions.calls["sum"] > result.outer_iterations
    assert reductions.calls["dot_many"] >= result.outer_iterations


def test_size_one_distributed_cartesian_solver_matches_serial_result():
    grid = Grid(shape=(5, 4, 3), size=(50.0, 40.0, 30.0))
    reference = DiffusionEigenSolver(
        grid, PWR_TWO_GROUP, device="cpu").solve(
            tol_k=1e-9, tol_source=1e-8)
    solver = DistributedDiffusionEigenSolver(
        grid, PWR_TWO_GROUP, context=DistributedContext.serial("cpu"),
        decomposition="slab")
    result = solver.solve(tol_k=1e-9, tol_source=1e-8)

    assert isinstance(result, DistributedResult)
    assert isinstance(result.partition, CartesianSlabPartition)
    assert result.k_eff == reference.k_eff
    assert result.k_history == reference.k_history
    assert result.source_error_history == reference.source_error_history
    assert result.outer_iterations == reference.outer_iterations
    assert result.inner_iterations == reference.inner_iterations
    np.testing.assert_array_equal(result.local_flux, reference.flux)
    assert result.gather_flux(root=0) is result.local_flux
    with pytest.raises(ValueError, match="outside communicator"):
        result.gather_flux(root=1)


def test_size_one_distributed_tri_solver_matches_serial_result():
    grid = TriGrid(shape=(5, 4, 2), side=2.0)
    reference = TriDiffusionEigenSolver(
        grid, PWR_TWO_GROUP, device="cpu").solve(
            tol_k=1e-9, tol_source=1e-8)
    solver = DistributedTriDiffusionEigenSolver(
        grid, PWR_TWO_GROUP, context=DistributedContext.serial("cpu"),
        decomposition="rows")
    result = solver.solve(tol_k=1e-9, tol_source=1e-8)

    assert isinstance(result.partition, TriRowPartition)
    assert result.k_eff == reference.k_eff
    assert result.outer_iterations == reference.outer_iterations
    assert result.inner_iterations == reference.inner_iterations
    np.testing.assert_array_equal(result.local_flux, reference.flux)


def test_multi_rank_tri_solver_rejects_discontinuity_factors_early():
    context, _ = _mirrored_context()
    with pytest.raises(NotImplementedError, match="discontinuity factors"):
        DistributedTriDiffusionEigenSolver(
            object(), object(), context=context, df=object())


@pytest.mark.parametrize("nz", [0, 10])
def test_size_one_distributed_tri_transient_matches_drum_step_exactly(nz):
    from ndgpu.benchmarks import (HPMR_KINETICS, build_hpmr2d,
                                  build_hpmr3d)
    from ndgpu.transient import TransientSolver

    builder = build_hpmr3d if nz else build_hpmr2d
    build_kwargs = {"refine": 1, "absorber": "polar"}
    if nz:
        build_kwargs["nz"] = nz
    initial = builder(drum_angle_deg=120.0, **build_kwargs)
    perturbed = builder(drum_angle_deg=110.0, **build_kwargs)

    def problem_at(time):
        problem = initial if time == 0.0 else perturbed
        return (problem.materials, problem.material_map,
                problem.mix_material, problem.mix_weight)

    common = dict(
        bc=initial.bc, active=initial.active, mask_bc=initial.mask_bc)
    reference = TransientSolver(
        initial.grid, problem_at, HPMR_KINETICS, device="cpu",
        group_operator=TriGroupOperator,
        eig_solver=TriDiffusionEigenSolver, **common).solve(
            t_end=0.01, dt=0.01, tol_step=1e-6, rebalance=True,
            anderson_depth=1)
    distributed = DistributedTriTransientSolver(
        initial.grid, problem_at, HPMR_KINETICS,
        context=DistributedContext.serial("cpu"), **common).solve(
            t_end=0.01, dt=0.01, tol_step=1e-6, rebalance=True,
            anderson_depth=5)

    np.testing.assert_array_equal(distributed.power, reference.power)
    np.testing.assert_array_equal(distributed.local_flux, reference.flux)
    np.testing.assert_array_equal(
        distributed.local_precursors, reference.precursors)
    assert distributed.step_iterations == reference.step_iterations
    assert (distributed.total_inner_iterations ==
            reference.total_inner_iterations)


def test_distributed_tri_transient_reuses_compatible_initial_eigenstate():
    from ndgpu.benchmarks import HPMR_KINETICS, build_hpmr2d

    problem = build_hpmr2d(
        refine=1, drum_angle_deg=120.0, absorber="polar")
    context = DistributedContext.serial("cpu")
    common = dict(
        bc=problem.bc, active=problem.active, mask_bc=problem.mask_bc,
        mix_material=problem.mix_material, mix_weight=problem.mix_weight)
    steady = DistributedTriDiffusionEigenSolver(
        problem.grid, problem.materials, problem.material_map,
        context=context, **common).solve(tol_k=1e-9, tol_source=1e-8)
    transient = DistributedTriTransientSolver(
        problem.grid,
        lambda time: (problem.materials, problem.material_map,
                      problem.mix_material, problem.mix_weight),
        HPMR_KINETICS, context=context,
        bc=problem.bc, active=problem.active,
        mask_bc=problem.mask_bc).solve(
            t_end=0.01, dt=0.01, initial_steady=steady,
            tol_step=1e-6, rebalance=True)

    assert transient.initial_state_reused
    assert transient.k0 == steady.k_eff


def test_single_reduction_pcg_preserves_distributed_transient_history():
    from ndgpu.benchmarks import HPMR_KINETICS, build_hpmr2d

    initial = build_hpmr2d(
        refine=1, drum_angle_deg=120.0, absorber="polar")
    perturbed = build_hpmr2d(
        refine=1, drum_angle_deg=118.0, absorber="polar")
    context = DistributedContext.serial("cpu")
    common = dict(
        bc=initial.bc, active=initial.active, mask_bc=initial.mask_bc,
        mix_material=initial.mix_material, mix_weight=initial.mix_weight)
    steady = DistributedTriDiffusionEigenSolver(
        initial.grid, initial.materials, initial.material_map,
        context=context, **common).solve(tol_k=1e-9, tol_source=1e-8)

    def problem_at(time):
        problem = initial if time == 0.0 else perturbed
        return (problem.materials, problem.material_map,
                problem.mix_material, problem.mix_weight)

    solver = DistributedTriTransientSolver(
        initial.grid, problem_at, HPMR_KINETICS, context=context,
        bc=initial.bc, active=initial.active, mask_bc=initial.mask_bc)
    solve_kwargs = dict(
        t_end=0.02, dt=0.01, initial_steady=steady,
        tol_step=1e-7, rebalance=True)
    reference = solver.solve(**solve_kwargs)
    optimized = solver.solve(
        **solve_kwargs, linsolve_kwargs={"single_reduction": True})

    np.testing.assert_allclose(
        optimized.power, reference.power, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(
        optimized.flux, reference.flux, rtol=2e-8, atol=2e-10)
    assert optimized.step_iterations == reference.step_iterations
    assert optimized.initial_state_reused


@pytest.mark.parametrize(
    "solve_kwargs, message",
    [
        ({"step_solver": "monolithic"}, "fixed-point"),
        ({"adaptive_bdf": {}}, "adaptive BDF"),
    ],
)
def test_distributed_tri_transient_rejects_unsupported_modes(
        solve_kwargs, message):
    from ndgpu.benchmarks import HPMR_KINETICS, build_hpmr2d

    problem = build_hpmr2d(refine=1, absorber="polar")
    solver = DistributedTriTransientSolver(
        problem.grid,
        lambda time: (problem.materials, problem.material_map),
        HPMR_KINETICS,
        context=DistributedContext.serial("cpu"),
        bc=problem.bc, active=problem.active, mask_bc=problem.mask_bc)

    with pytest.raises(NotImplementedError, match=message):
        solver.solve(t_end=0.01, dt=0.01, **solve_kwargs)
