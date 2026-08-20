"""Correctness gates for the dependency-safe distributed foundation."""

import numpy as np
import pytest

from ndgpu.distributed import (CartesianSlabPartition, DistributedContext,
                               SerialReductions, TriRowPartition)
from ndgpu.linalg import PCGWorkspace, pcg


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
    assert isinstance(context.reductions, SerialReductions)


def test_cpu_mpi_context_reduces_and_exchanges_backend_arrays():
    context, communicator = _mirrored_context()
    values = np.arange(5.0)
    assert context.reductions.dot(values, values) == 60.0
    assert context.reductions.sum(values) == 20.0
    assert context.reductions.max(values) == 8.0
    np.testing.assert_array_equal(
        context.sendrecv(values, destination=1, source=1, tag=7), values)
    assert communicator.allreduce_calls == 3


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
