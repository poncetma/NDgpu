"""MPI execution primitives for spatially distributed diffusion solves.

The serial package never imports :mod:`mpi4py`.  Callers opt into MPI by
passing an existing communicator to :meth:`DistributedContext.from_mpi`.
One process owns one spatial partition and, on GPU, one visible CUDA device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
import socket
import time
from typing import Sequence

import numpy as np

from . import kernels
from .backend import asnumpy, get_backend, synchronize


_COMMUNICATION_MODES = {
    "auto", "serial", "cpu-mpi", "cuda-aware", "host-staged"}


def _balanced_range(length: int, rank: int, size: int) -> tuple[int, int]:
    if length < 1:
        raise ValueError("partitioned dimensions must be non-empty")
    if size < 1 or not 0 <= rank < size:
        raise ValueError("rank and size are inconsistent")
    if size > length:
        raise ValueError(
            f"cannot divide {length} entries among {size} non-empty ranks")
    width, extra = divmod(length, size)
    start = rank * width + min(rank, extra)
    return start, start + width + (rank < extra)


@dataclass(frozen=True)
class SpatialPartition:
    """Immutable metadata for one rank's contiguous spatial slab."""

    global_shape: tuple[int, ...]
    axis: int
    rank: int
    size: int
    start: int
    stop: int

    def __post_init__(self):
        shape = tuple(int(value) for value in self.global_shape)
        if not shape or any(value < 1 for value in shape):
            raise ValueError("global_shape must contain positive dimensions")
        if not 0 <= self.axis < len(shape):
            raise ValueError("partition axis is outside global_shape")
        expected = _balanced_range(shape[self.axis], self.rank, self.size)
        if (self.start, self.stop) != expected:
            raise ValueError(
                f"owned range {(self.start, self.stop)} does not match "
                f"balanced range {expected}")
        object.__setattr__(self, "global_shape", shape)

    @property
    def local_shape(self) -> tuple[int, ...]:
        shape = list(self.global_shape)
        shape[self.axis] = self.stop - self.start
        return tuple(shape)

    @property
    def global_cell_count(self) -> int:
        return math.prod(self.global_shape)

    @property
    def local_cell_count(self) -> int:
        return math.prod(self.local_shape)

    @property
    def lower_rank(self) -> int | None:
        return self.rank - 1 if self.rank > 0 else None

    @property
    def upper_rank(self) -> int | None:
        return self.rank + 1 if self.rank + 1 < self.size else None

    @property
    def owned_slice(self) -> tuple[slice, ...]:
        result = [slice(None)] * len(self.global_shape)
        result[self.axis] = slice(self.start, self.stop)
        return tuple(result)

    @property
    def counts(self) -> tuple[int, ...]:
        cross_section = self.global_cell_count // self.global_shape[self.axis]
        return tuple(
            (_balanced_range(self.global_shape[self.axis], rank, self.size)[1]
             - _balanced_range(self.global_shape[self.axis], rank, self.size)[0])
            * cross_section
            for rank in range(self.size)
        )

    @property
    def displacements(self) -> tuple[int, ...]:
        counts = self.counts
        return tuple(sum(counts[:rank]) for rank in range(self.size))

    def global_index(self, local_index: Sequence[int]) -> tuple[int, ...]:
        index = tuple(int(value) for value in local_index)
        if len(index) != len(self.local_shape):
            raise ValueError("local index dimensionality does not match partition")
        if any(value < 0 or value >= extent
               for value, extent in zip(index, self.local_shape)):
            raise IndexError("local index is outside the owned partition")
        result = list(index)
        result[self.axis] += self.start
        return tuple(result)


class CartesianSlabPartition(SpatialPartition):
    """Balanced one-axis decomposition of a Cartesian backing array."""

    @classmethod
    def create(cls, global_shape, rank: int, size: int, axis: int | None = None):
        shape = tuple(int(value) for value in global_shape)
        if axis is None:
            axis = max(range(len(shape)), key=shape.__getitem__)
        start, stop = _balanced_range(shape[axis], rank, size)
        return cls(shape, axis, rank, size, start, stop)


class TriRowPartition(SpatialPartition):
    """Row-slab decomposition for triangular 2-D or extruded 3-D arrays."""

    @classmethod
    def create(cls, global_shape, rank: int, size: int):
        shape = tuple(int(value) for value in global_shape)
        if len(shape) not in (3, 4) or shape[2] != 2:
            raise ValueError(
                "triangular shape must be (rows, columns, 2[, axial])")
        start, stop = _balanced_range(shape[0], rank, size)
        return cls(shape, 0, rank, size, start, stop)


class ExtrudedAxialPartition(SpatialPartition):
    """Axial-layer decomposition of ``(radial_cell, axial_layer)`` arrays."""

    @classmethod
    def create(cls, global_shape, rank: int, size: int):
        shape = tuple(int(value) for value in global_shape)
        if len(shape) != 2:
            raise ValueError(
                "extruded mesh shape must be (radial_cells, axial_layers)")
        start, stop = _balanced_range(shape[1], rank, size)
        return cls(shape, 1, rank, size, start, stop)


class SerialReductions:
    """Backend-local reductions implementing the distributed reduction API."""

    distributed = False

    def __init__(self, xp):
        self.xp = xp

    def dot(self, left, right):
        return kernels.dot(self.xp, left, right)

    def sum(self, value):
        return self.xp.sum(value)

    def max(self, value):
        return self.xp.max(value)

    def sum_many(self, values):
        """Reduce several independent arrays into one backend-native vector."""
        values = tuple(values)
        if not values:
            return self.xp.empty(0, dtype=float)
        return self.xp.stack([self.xp.sum(value) for value in values])

    def dot_many(self, pairs):
        """Compute several dot products without retaining product arrays."""
        pairs = tuple(pairs)
        if not pairs:
            return self.xp.empty(0, dtype=float)
        # Match the power iteration's historical multiply-then-sum order.
        # PCG uses the separately injected fused ``dot`` method; changing the
        # Anderson/source-error order here perturbs size-one GPU histories.
        return self.xp.stack([
            self.xp.sum(left * right) for left, right in pairs])

    def project(self, basis, value, *, include_norm=False):
        """Project onto a stacked basis, optionally appending ``value`` norm2."""
        rows = basis.reshape(basis.shape[0], -1)
        flat = value.reshape(-1)
        projected = rows @ flat
        if include_norm:
            projected = self.xp.concatenate(
                (projected, kernels.dot(self.xp, value, value).reshape(1)))
        return projected


class DistributedReductions(SerialReductions):
    """Local array reductions followed by communicator-wide collectives."""

    distributed = True

    def __init__(self, context: "DistributedContext"):
        super().__init__(context.xp)
        self.context = context

    def dot(self, left, right):
        return self.context.allreduce_sum(super().dot(left, right))

    def sum(self, value):
        return self.context.allreduce_sum(super().sum(value))

    def max(self, value):
        return self.context.allreduce_max(super().max(value))

    def sum_many(self, values):
        return self.context.allreduce_sum(super().sum_many(values))

    def dot_many(self, pairs):
        return self.context.allreduce_sum(super().dot_many(pairs))

    def project(self, basis, value, *, include_norm=False):
        return self.context.allreduce_sum(
            super().project(basis, value, include_norm=include_norm))


@dataclass
class DistributedContext:
    """MPI communicator, rank placement, and backend-aware communication."""

    communicator: object | None
    xp: object
    rank: int
    size: int
    local_rank: int
    communication_mode: str
    hostname: str = field(default_factory=socket.gethostname)
    device_id: int | None = None
    device_identity: str | None = None
    mpi_library_version: str | None = None
    batched_halos: bool = False
    _mpi: object | None = field(default=None, repr=False)
    _reductions: object | None = field(default=None, init=False, repr=False)
    _communication_stats: dict = field(
        default_factory=lambda: {
            "allreduce_calls": 0,
            "allreduce_bytes": 0,
            "allreduce_seconds": 0.0,
            "sendrecv_calls": 0,
            "sendrecv_bytes": 0,
            "sendrecv_seconds": 0.0,
        }, init=False, repr=False)

    def __post_init__(self):
        if self.communication_mode not in _COMMUNICATION_MODES - {"auto"}:
            raise ValueError(f"unknown communication mode {self.communication_mode!r}")
        if self.size < 1 or not 0 <= self.rank < self.size:
            raise ValueError("rank and size are inconsistent")
        if self.size > 1 and self.communicator is None:
            raise ValueError("a communicator is required when size > 1")
        if self.communication_mode == "cuda-aware" and self.xp is np:
            raise ValueError("cuda-aware communication requires a GPU backend")

    @classmethod
    def serial(cls, device: str = "cpu") -> "DistributedContext":
        xp = get_backend(device)
        device_id = None
        identity = None
        if xp is not np:
            device_id = int(xp.cuda.runtime.getDevice())
            identity = _cuda_device_identity(xp, device_id)
        return cls(None, xp, 0, 1, 0, "serial",
                   device_id=device_id, device_identity=identity)

    @classmethod
    def from_mpi(cls, communicator, *, device: str = "gpu",
                 communication: str = "auto", local_rank: int | None = None,
                 allow_shared_device: bool = False,
                 batched_halos: bool = False) -> "DistributedContext":
        """Construct from an mpi4py communicator and select this rank's GPU.

        ``communication='auto'`` deliberately chooses explicit host staging on
        GPU. Direct device pointers are used only when ``'cuda-aware'`` is
        requested and subsequently exercised by the environment probe.
        """
        if communication not in _COMMUNICATION_MODES:
            raise ValueError(f"unknown communication mode {communication!r}")
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError(
                "distributed execution requires mpi4py; install ndgpu[mpi] "
                "against the site's MPI implementation") from exc

        rank = int(communicator.Get_rank())
        size = int(communicator.Get_size())
        hostname = socket.gethostname()
        if local_rank is None:
            local_rank = _discover_local_rank(communicator, MPI)

        xp = get_backend(device)
        device_id = None
        identity = None
        if xp is not np:
            count = int(xp.cuda.runtime.getDeviceCount())
            if count < 1:
                raise RuntimeError("no CUDA devices are visible to this MPI rank")
            # Slurm commonly exposes one distinct GPU as device zero per task.
            device_id = 0 if count == 1 else int(local_rank) % count
            xp.cuda.Device(device_id).use()
            identity = _cuda_device_identity(xp, device_id)
            placements = communicator.allgather((hostname, identity))
            duplicates = len(placements) != len(set(placements))
            if duplicates and not allow_shared_device:
                raise RuntimeError(
                    "multiple MPI ranks selected the same GPU; request one GPU "
                    "per rank or pass allow_shared_device=True only for tests; "
                    f"placements={placements}")

        if communication == "auto":
            communication = "host-staged" if xp is not np else "cpu-mpi"
        if communication == "serial" and size > 1:
            raise ValueError("serial communication is invalid for multiple ranks")
        if communication == "cpu-mpi" and xp is not np:
            raise ValueError("cpu-mpi communication requires the NumPy backend")
        if communication in {"cuda-aware", "host-staged"} and xp is np:
            if communication == "cuda-aware":
                raise ValueError("cuda-aware communication requires a GPU backend")
            communication = "cpu-mpi"

        return cls(
            communicator, xp, rank, size, int(local_rank), communication,
            hostname=hostname, device_id=device_id, device_identity=identity,
            mpi_library_version=MPI.Get_library_version().strip(),
            batched_halos=bool(batched_halos), _mpi=MPI)

    @property
    def reductions(self):
        if self._reductions is None:
            reduction_type = DistributedReductions if self.size > 1 else SerialReductions
            self._reductions = reduction_type(self if self.size > 1 else self.xp)
        return self._reductions

    @property
    def is_distributed(self) -> bool:
        return self.size > 1

    def describe(self) -> dict:
        return {
            "rank": self.rank,
            "size": self.size,
            "local_rank": self.local_rank,
            "hostname": self.hostname,
            "communication_mode": self.communication_mode,
            "device_id": self.device_id,
            "device_identity": self.device_identity,
            "mpi_library_version": self.mpi_library_version,
            "batched_halos": self.batched_halos,
        }

    def reset_communication_stats(self):
        for name in self._communication_stats:
            self._communication_stats[name] = (
                0.0 if name.endswith("_seconds") else 0)

    def communication_stats(self) -> dict:
        result = dict(self._communication_stats)
        result["communication_seconds"] = (
            result["allreduce_seconds"] + result["sendrecv_seconds"])
        return result

    def _record_communication(self, kind: str, nbytes: int, seconds: float):
        self._communication_stats[f"{kind}_calls"] += 1
        self._communication_stats[f"{kind}_bytes"] += int(nbytes)
        self._communication_stats[f"{kind}_seconds"] += float(seconds)

    def allreduce_sum(self, value):
        return self._allreduce(value, "SUM")

    def allreduce_max(self, value):
        return self._allreduce(value, "MAX")

    def _allreduce(self, value, operation: str):
        if self.size == 1:
            return value
        local = self.xp.asarray(value)
        if not local.flags.c_contiguous:
            local = self.xp.ascontiguousarray(local)
        mpi_operation = getattr(self._mpi, operation) if self._mpi is not None else None
        started = time.perf_counter()
        if self.communication_mode == "host-staged":
            send = np.array(asnumpy(local), copy=True, order="C")
            receive = np.empty_like(send)
            self.communicator.Allreduce(send, receive, op=mpi_operation)
            result = self.xp.asarray(receive)
        else:
            result = self.xp.empty_like(local)
            synchronize(self.xp)
            self.communicator.Allreduce(local, result, op=mpi_operation)
            synchronize(self.xp)
        self._record_communication(
            "allreduce", local.nbytes, time.perf_counter() - started)
        return result

    def sendrecv(self, value, *, destination: int, source: int, tag: int = 0):
        """Exchange one contiguous array with explicit neighboring ranks."""
        if self.size == 1:
            return self.xp.array(value, copy=True)
        send = self.xp.ascontiguousarray(value)
        started = time.perf_counter()
        if self.communication_mode == "host-staged":
            host_send = np.ascontiguousarray(asnumpy(send))
            host_receive = np.empty_like(host_send)
            self.communicator.Sendrecv(
                host_send, dest=destination, sendtag=tag,
                recvbuf=host_receive, source=source, recvtag=tag)
            result = self.xp.asarray(host_receive)
        else:
            result = self.xp.empty_like(send)
            synchronize(self.xp)
            self.communicator.Sendrecv(
                send, dest=destination, sendtag=tag,
                recvbuf=result, source=source, recvtag=tag)
            synchronize(self.xp)
        self._record_communication(
            "sendrecv", send.nbytes, time.perf_counter() - started)
        return result

    def exchange_halos(self, value, partition: SpatialPartition, *, tag: int = 0):
        """Return lower and upper neighbor planes for an owned slab.

        Every rank performs the same two directional ``Sendrecv`` calls. A
        physical boundary uses ``MPI.PROC_NULL`` and is returned as ``None``;
        this remains safe for one-cell slabs where both owned planes coincide.
        Leading component dimensions are retained, allowing one exchange of a
        stacked multigroup field.
        """
        if partition.rank != self.rank or partition.size != self.size:
            raise ValueError("partition rank/size does not match the context")
        spatial_ndim = len(partition.local_shape)
        if tuple(value.shape[-spatial_ndim:]) != partition.local_shape:
            raise ValueError(
                f"local value spatial shape {value.shape[-spatial_ndim:]} != "
                f"{partition.local_shape}")
        if self.size == 1:
            return None, None

        if self.batched_halos:
            return self._exchange_halos_batched(value, partition, tag=tag)

        lower = partition.lower_rank
        upper = partition.upper_rank
        proc_null = self._mpi.PROC_NULL if self._mpi is not None else -1

        value_axis = value.ndim - spatial_ndim + partition.axis

        def plane(index):
            sl = [slice(None)] * value.ndim
            sl[value_axis] = index
            return value[tuple(sl)]

        lower_halo = self.sendrecv(
            plane(-1), destination=upper if upper is not None else proc_null,
            source=lower if lower is not None else proc_null, tag=tag)
        upper_halo = self.sendrecv(
            plane(0), destination=lower if lower is not None else proc_null,
            source=upper if upper is not None else proc_null, tag=tag + 1)
        return (None if lower is None else lower_halo,
                None if upper is None else upper_halo)

    def _exchange_halos_batched(
            self, value, partition: SpatialPartition, *, tag: int = 0):
        """Exchange both slab faces with one nonblocking MPI completion."""
        state = self._begin_halo_exchange(value, partition, tag=tag)
        return self._finish_halo_exchange(state)

    def exchange_halos_while(
            self, value, partition: SpatialPartition, work, *, tag: int = 0):
        """Exchange halos while executing independent owned-domain work."""
        if not self.batched_halos or self.size == 1:
            halos = self.exchange_halos(value, partition, tag=tag)
            return halos, work()
        if partition.rank != self.rank or partition.size != self.size:
            raise ValueError("partition rank/size does not match the context")
        spatial_ndim = len(partition.local_shape)
        if tuple(value.shape[-spatial_ndim:]) != partition.local_shape:
            raise ValueError(
                f"local value spatial shape {value.shape[-spatial_ndim:]} != "
                f"{partition.local_shape}")
        state = self._begin_halo_exchange(value, partition, tag=tag)
        result = work()
        return self._finish_halo_exchange(state), result

    def _begin_halo_exchange(
            self, value, partition: SpatialPartition, *, tag: int):
        lower = partition.lower_rank
        upper = partition.upper_rank
        spatial_ndim = len(partition.local_shape)
        value_axis = value.ndim - spatial_ndim + partition.axis

        def plane(index):
            sl = [slice(None)] * value.ndim
            sl[value_axis] = index
            return self.xp.ascontiguousarray(value[tuple(sl)])

        sends = []
        lower_receive = upper_receive = None
        requests = []
        started = time.perf_counter()

        if self.communication_mode == "host-staged":
            lower_send = (None if lower is None else
                          np.ascontiguousarray(asnumpy(plane(0))))
            upper_send = (None if upper is None else
                          np.ascontiguousarray(asnumpy(plane(-1))))
            lower_buffer = (None if lower_send is None else
                            np.empty_like(lower_send))
            upper_buffer = (None if upper_send is None else
                            np.empty_like(upper_send))
        else:
            lower_send = None if lower is None else plane(0)
            upper_send = None if upper is None else plane(-1)
            lower_buffer = (None if lower_send is None else
                            self.xp.empty_like(lower_send))
            upper_buffer = (None if upper_send is None else
                            self.xp.empty_like(upper_send))
            if self.communication_mode == "cuda-aware":
                synchronize(self.xp)

        if lower is not None:
            requests.append(self.communicator.Irecv(
                lower_buffer, source=lower, tag=tag))
            requests.append(self.communicator.Isend(
                lower_send, dest=lower, tag=tag + 1))
            sends.append(lower_send)
        if upper is not None:
            requests.append(self.communicator.Irecv(
                upper_buffer, source=upper, tag=tag + 1))
            requests.append(self.communicator.Isend(
                upper_send, dest=upper, tag=tag))
            sends.append(upper_send)
        post_seconds = time.perf_counter() - started
        return (requests, sends, lower_buffer, upper_buffer, post_seconds)

    def _finish_halo_exchange(self, state):
        requests, sends, lower_buffer, upper_buffer, seconds = state
        started = time.perf_counter()
        self._mpi.Request.Waitall(requests)
        if self.communication_mode == "host-staged":
            lower_receive = upper_receive = None
            if lower_buffer is not None:
                lower_receive = self.xp.asarray(lower_buffer)
            if upper_buffer is not None:
                upper_receive = self.xp.asarray(upper_buffer)
        else:
            lower_receive, upper_receive = lower_buffer, upper_buffer
            if self.communication_mode == "cuda-aware":
                synchronize(self.xp)
        self._record_communication(
            "sendrecv", sum(send.nbytes for send in sends),
            seconds + time.perf_counter() - started)
        return lower_receive, upper_receive

    def gather_spatial(self, value, partition: SpatialPartition, *, root: int = 0):
        """Explicitly gather rank-local spatial slabs into one host array."""
        if not 0 <= root < self.size:
            raise ValueError(f"root {root} is outside communicator size {self.size}")
        spatial_ndim = len(partition.local_shape)
        if tuple(value.shape[-spatial_ndim:]) != partition.local_shape:
            raise ValueError(
                f"local value spatial shape {value.shape[-spatial_ndim:]} != "
                f"{partition.local_shape}")
        if partition.rank != self.rank or partition.size != self.size:
            raise ValueError("partition rank/size does not match the context")
        if self.size == 1:
            return asnumpy(value) if self.rank == root else None

        local = np.ascontiguousarray(asnumpy(value))
        pieces = self.communicator.gather(
            (partition.start, partition.stop, local), root=root)
        if self.rank != root:
            return None

        prefix_shape = local.shape[:-spatial_ndim]
        gathered = np.empty(prefix_shape + partition.global_shape,
                            dtype=local.dtype)
        for start, stop, piece in pieces:
            sl = [slice(None)] * gathered.ndim
            sl[len(prefix_shape) + partition.axis] = slice(start, stop)
            gathered[tuple(sl)] = piece
        return gathered


@dataclass
class DistributedResult:
    """Rank-local eigenvalue result with explicit global-field gathering."""

    k_eff: float
    local_flux: object
    partition: SpatialPartition
    context: DistributedContext
    converged: bool
    outer_iterations: int
    inner_iterations: int
    solve_seconds: float
    device: str
    k_history: list = field(default_factory=list)
    source_error_history: list = field(default_factory=list)

    @classmethod
    def from_local_result(cls, result, partition, context):
        return cls(
            k_eff=result.k_eff,
            local_flux=result.flux,
            partition=partition,
            context=context,
            converged=result.converged,
            outer_iterations=result.outer_iterations,
            inner_iterations=result.inner_iterations,
            solve_seconds=result.solve_seconds,
            device=result.device,
            k_history=result.k_history,
            source_error_history=result.source_error_history,
        )

    @property
    def rank(self) -> int:
        return self.context.rank

    @property
    def size(self) -> int:
        return self.context.size

    @property
    def local_flux_numpy(self) -> np.ndarray:
        return asnumpy(self.local_flux)

    @property
    def flux(self):
        """Rank-local compatibility alias for solver-internal handoffs."""
        return self.local_flux

    def gather_flux(self, root: int = 0):
        """Collect the global flux explicitly; collective in multi-rank use."""
        if not 0 <= root < self.size:
            raise ValueError(f"root {root} is outside communicator size {self.size}")
        if self.size == 1:
            return self.local_flux if self.rank == root else None
        return self.context.gather_spatial(
            self.local_flux, self.partition, root=root)

    def __repr__(self):
        status = "converged" if self.converged else "NOT CONVERGED"
        return (
            f"DistributedResult(k_eff={self.k_eff:.6f}, {status}, "
            f"rank {self.rank}/{self.size}, {self.outer_iterations} outers / "
            f"{self.inner_iterations} inners, {self.solve_seconds:.2f} s on "
            f"{self.device})"
        )


class DistributedTransientResult:
    """Rank-local transient result with explicit final-state gathering."""

    def __init__(self, local_result, partition, context):
        self.local_result = local_result
        self.partition = partition
        self.context = context

    @property
    def local_flux(self):
        return self.local_result.flux

    @property
    def local_precursors(self):
        return self.local_result.precursors

    @property
    def rank(self):
        return self.context.rank

    @property
    def size(self):
        return self.context.size

    def gather_flux(self, root=0):
        return self.context.gather_spatial(
            self.local_flux, self.partition, root=root)

    def gather_precursors(self, root=0):
        return self.context.gather_spatial(
            self.local_precursors, self.partition, root=root)

    def __getattr__(self, name):
        return getattr(self.local_result, name)

    def __repr__(self):
        return (f"DistributedTransientResult(rank {self.rank}/{self.size}, "
                f"{self.local_result!r})")


def _discover_local_rank(communicator, mpi) -> int:
    for name in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
                 "MV2_COMM_WORLD_LOCAL_RANK"):
        if name in os.environ:
            return int(os.environ[name])
    local = communicator.Split_type(mpi.COMM_TYPE_SHARED)
    try:
        return int(local.Get_rank())
    finally:
        local.Free()


def _cuda_device_identity(xp, device_id: int) -> str:
    runtime = xp.cuda.runtime
    try:
        uuid = runtime.deviceGetUuid(device_id)
        if isinstance(uuid, bytes):
            return uuid.decode(errors="replace")
        return str(uuid)
    except (AttributeError, RuntimeError):
        try:
            bus_id = runtime.deviceGetPCIBusId(device_id)
            return bus_id.decode() if isinstance(bus_id, bytes) else str(bus_id)
        except AttributeError:
            props = runtime.getDeviceProperties(device_id)
            name = props["name"]
            return name.decode() if isinstance(name, bytes) else str(name)
