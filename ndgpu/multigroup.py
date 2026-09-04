"""Matrix-free monolithic multigroup systems for implicit transient steps.

The production diffusion transient currently converges a fixed point over the
end-of-step fission source and performs Gauss--Seidel group solves inside each
map evaluation.  After analytic precursor elimination the very same step is a
single linear block system.  This module represents that system without
assembling a sparse ``(groups*cells)^2`` matrix and provides the current group
sweep as a flexible right preconditioner.

The classes are intentionally standalone while the method is evaluated.  They
do not alter :class:`ndgpu.transient.TransientSolver` or its validated default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from . import kernels
from .backend import asnumpy
from .linalg import PCGWorkspace, fixed_pcg, neumann_preconditioner, pcg


def _shape_of(operator):
    shape = getattr(operator, "shape", None)
    if shape is None:
        diag = getattr(operator, "inv_diag", None)
        if diag is None:
            raise TypeError("group operators must expose shape or inv_diag")
        shape = diag.shape
    return tuple(shape)


class MultigroupStepOperator:
    r"""Implicit diffusion step as one matrix-free block operator.

    For group ``g`` the row is

    .. math::

       L_g\phi_g - W\left[\sum_{h\ne g}\Sigma_{s,h\to g}\phi_h
       + {w_g\over k_0}\sum_h\nu\Sigma_{f,h}\phi_h\right],

    where ``L_g`` is the existing loss-plus-time group operator and ``W`` is
    its optional cylindrical finite-volume RHS weight.  ``w_g`` already
    contains prompt emission plus the analytically eliminated end-of-step
    precursor contribution.  Thus delayed and old-time sources belong only in
    the fixed right-hand side and the operator remains linear.

    Arrays have shape ``(groups, *cell_shape)``.  All coefficients stay on the
    backend selected by the supplied group operators.
    """

    def __init__(self, operators, sigma_s, nu_sigma_f, emission_weights,
                 k_eff, rhs_weight=None, group_batch=None):
        self.operators = tuple(operators)
        self.groups = len(self.operators)
        if self.groups == 0:
            raise ValueError("at least one energy group is required")
        self.xp = self.operators[0].xp
        self.cell_shape = _shape_of(self.operators[0])
        if any(op.xp is not self.xp for op in self.operators):
            raise ValueError("all group operators must use the same backend")
        if any(_shape_of(op) != self.cell_shape for op in self.operators):
            raise ValueError("all group operators must have the same shape")
        if len(sigma_s) != self.groups or any(
                len(row) != self.groups for row in sigma_s):
            raise ValueError("sigma_s must be a square group-by-group table")
        if len(nu_sigma_f) != self.groups:
            raise ValueError("nu_sigma_f must contain one field per group")
        if len(emission_weights) != self.groups:
            raise ValueError("emission_weights must contain one field per group")
        if not np.isfinite(k_eff) or k_eff <= 0.0:
            raise ValueError("k_eff must be finite and positive")
        self.sigma_s = tuple(tuple(row) for row in sigma_s)
        self.nu_sigma_f = tuple(nu_sigma_f)
        self.emission_weights = tuple(emission_weights)
        self.k_eff = float(k_eff)
        self.rhs_weight = rhs_weight
        self._group_batch = None
        if (group_batch is not None
                and kernels.use_fused(self.xp, "groups")):
            W, S = group_batch.get("W"), group_batch.get("S")
            expected_w = (self.groups,) + self.cell_shape
            expected_s = (self.groups, self.groups) + self.cell_shape
            if W is None or tuple(W.shape) != expected_w:
                raise ValueError("group_batch W has incompatible shape")
            if S is None or tuple(S.shape) != expected_s:
                raise ValueError("group_batch S has incompatible shape")
            self._group_batch = {"W": W, "S": S}
            self._fission_buffer = self.xp.empty(self.cell_shape,
                                                  dtype=W.dtype)
            self._scatter_buffer = self.xp.empty_like(self._fission_buffer)
            self._emission_rows = tuple(
                w if rhs_weight is None else rhs_weight * w
                for w in self.emission_weights)

        # A useful fallback diagonal for generic Krylov interfaces.  It includes
        # the same-group fission derivative; the GS preconditioner below is the
        # intended path and does not rely on this approximation.
        diagonal = []
        for g, op in enumerate(self.operators):
            diag = 1.0 / op.inv_diag
            fdiag = self.emission_weights[g] * self.nu_sigma_f[g] / self.k_eff
            if rhs_weight is not None:
                fdiag = rhs_weight * fdiag
            diagonal.append(diag - fdiag)
        self.inv_diag = 1.0 / self.xp.stack(diagonal)

    @property
    def shape(self):
        return (self.groups,) + self.cell_shape

    def _check_state(self, value):
        if tuple(value.shape) != self.shape:
            raise ValueError(f"multigroup state shape {value.shape} != {self.shape}")

    def fission_source(self, value):
        self._check_state(value)
        if self._group_batch is not None:
            source = self._fission_buffer
            source.fill(0)
            kernels.group_accumulate(
                self.xp, source, self._group_batch["W"], value)
            source *= 1.0 / self.k_eff
            return source
        source = self.nu_sigma_f[0] * value[0]
        for g in range(1, self.groups):
            source = source + self.nu_sigma_f[g] * value[g]
        return source / self.k_eff

    def _apply_group_loss(self, value, out):
        """Apply all within-group operators, batching compatible halos."""
        first = self.operators[0]
        context = getattr(first, "context", None)
        partition = getattr(first, "partition", None)
        batchable = (
            context is not None and partition is not None
            and all(getattr(op, "context", None) is context
                    and getattr(op, "partition", None) == partition
                    and hasattr(op, "apply_local")
                    and hasattr(op, "finish_halo_apply")
                    for op in self.operators))
        if not batchable:
            for g, op in enumerate(self.operators):
                op.apply(value[g], out=out[g])
            return out

        def apply_owned():
            for g, op in enumerate(self.operators):
                op.apply_local(value[g], out=out[g])
            return out

        (lower, upper), out = context.exchange_halos_while(
            value, partition, apply_owned,
            tag=int(getattr(first, "communication_tag", 0)) + 2)
        for g, op in enumerate(self.operators):
            op.finish_halo_apply(
                value[g], out[g],
                lower_phi=None if lower is None else lower[g],
                upper_phi=None if upper is None else upper[g])
        return out

    def apply(self, value, out=None):
        """Return the coupled block product, optionally writing into ``out``."""
        self._check_state(value)
        if out is None:
            out = self.xp.empty_like(value)
        else:
            self._check_state(out)
        fission = self.fission_source(value)
        self._apply_group_loss(value, out)
        if self._group_batch is not None:
            for g in range(self.groups):
                kernels.product_accumulate(
                    self.xp, out[g], self._emission_rows[g], fission, -1.0)
                if self.rhs_weight is None:
                    kernels.group_accumulate(
                        self.xp, out[g], self._group_batch["S"][g], value,
                        alpha=-1.0)
                else:
                    scatter = self._scatter_buffer
                    scatter.fill(0)
                    kernels.group_accumulate(
                        self.xp, scatter, self._group_batch["S"][g], value)
                    kernels.product_accumulate(
                        self.xp, out[g], self.rhs_weight, scatter, -1.0)
            return out
        for g in range(self.groups):
            coupled = self.emission_weights[g] * fission
            for gf in range(self.groups):
                scatter = self.sigma_s[gf][g]
                if gf != g and scatter is not None:
                    coupled = coupled + scatter * value[gf]
            if self.rhs_weight is not None:
                coupled = self.rhs_weight * coupled
            out[g] -= coupled
        return out

    def fixed_rhs(self, inverse_velocity_dt, carried_flux, delayed_source):
        """Build ``W * (theta*phi_carried + delayed)`` for one step."""
        if not (len(inverse_velocity_dt) == len(carried_flux)
                == len(delayed_source) == self.groups):
            raise ValueError("fixed-source inputs need one entry per group")
        rows = []
        for g in range(self.groups):
            row = (inverse_velocity_dt[g] * carried_flux[g]
                   + delayed_source[g])
            if self.rhs_weight is not None:
                row = self.rhs_weight * row
            rows.append(row)
        return self.xp.stack(rows)

    def assemble(self):
        """Assemble the exact host sparse block matrix for coarse setup.

        This is intentionally not used by the fine solve.  The conservative
        two-level prototype needs it once per changed transient operator to
        form ``P.T @ A @ P`` without probing the matrix-free block.
        """
        import scipy.sparse as sp

        n = int(np.prod(self.cell_shape))
        zero = sp.csr_matrix((n, n))
        rows = []
        rhs_weight = (None if self.rhs_weight is None else
                      np.asarray(asnumpy(self.rhs_weight)).reshape(-1))
        nsf = [np.asarray(asnumpy(v)).reshape(-1)
               for v in self.nu_sigma_f]
        emission = [np.asarray(asnumpy(v)).reshape(-1)
                    for v in self.emission_weights]
        for g in range(self.groups):
            row = []
            for source in range(self.groups):
                value = (self.operators[g].assemble()
                         if source == g else zero)
                coupling = emission[g] * nsf[source] / self.k_eff
                scatter = self.sigma_s[source][g]
                if source != g and scatter is not None:
                    coupling = coupling + np.asarray(
                        asnumpy(scatter)).reshape(-1)
                if rhs_weight is not None:
                    coupling = rhs_weight * coupling
                row.append(value - sp.diags(coupling))
            rows.append(row)
        return sp.bmat(rows, format="csr")


class SpatialCoarseSpace:
    """Piecewise-constant conservative spatial aggregation.

    ``P`` injects one value per aggregate into every member fine cell and
    ``R=P.T`` sums fine balance equations. Optional labels prevent an aggregate
    from crossing material/active-region boundaries inside a geometric bin.
    Energy groups are retained exactly; only space is coarsened.
    """

    def __init__(self, shape, factors=2, labels=None):
        import scipy.sparse as sp

        self.shape = tuple(int(v) for v in shape)
        ndim = len(self.shape)
        if np.ndim(factors) == 0:
            factors = (int(factors),) * ndim
        self.factors = tuple(int(v) for v in factors)
        if len(self.factors) != ndim or any(v < 1 for v in self.factors):
            raise ValueError("coarse factors must be positive and match shape")
        if labels is None:
            labels = np.zeros(self.shape, dtype=np.int64)
        labels = np.asarray(labels)
        if labels.shape != self.shape:
            raise ValueError("coarse labels must match the fine cell shape")

        indices = np.indices(self.shape)
        coarse_shape = tuple(
            (size + factor - 1) // factor
            for size, factor in zip(self.shape, self.factors))
        geometric = np.ravel_multi_index(
            tuple(indices[axis] // self.factors[axis]
                  for axis in range(ndim)), coarse_shape).reshape(-1)
        pairs = np.column_stack((geometric, labels.reshape(-1)))
        _, mapping = np.unique(pairs, axis=0, return_inverse=True)
        self.mapping = mapping.astype(np.int64, copy=False)
        self.fine_cells = int(mapping.size)
        self.coarse_cells = int(mapping.max()) + 1
        self.aggregate_sizes = np.bincount(
            self.mapping, minlength=self.coarse_cells)
        self.P = sp.csr_matrix(
            (np.ones(self.fine_cells),
             (np.arange(self.fine_cells), self.mapping)),
            shape=(self.fine_cells, self.coarse_cells))

    def full_operators(self, groups):
        """Return group-major block-diagonal prolongation and restriction."""
        import scipy.sparse as sp

        prolongation = sp.kron(
            sp.eye(int(groups), format="csr"), self.P, format="csr")
        return prolongation, prolongation.T.tocsr()

    def restrict(self, fine, groups):
        value = np.asarray(fine).reshape(int(groups), self.fine_cells)
        return np.asarray((self.P.T @ value.T).T).reshape(-1)

    def prolong(self, coarse, groups):
        value = np.asarray(coarse).reshape(int(groups), self.coarse_cells)
        return value[:, self.mapping].reshape((int(groups),) + self.shape)


@dataclass
class SpatialCoarseStats:
    applications: int = 0
    coarse_solves: int = 0
    fine_residual_applies: int = 0
    setup_seconds: float = 0.0


class BlockJacobiPreconditioner:
    """Allocation-free elementwise inverse of the monolithic block diagonal."""

    ndgpu_out = True

    def __init__(self, inverse_diagonal):
        self.inverse_diagonal = inverse_diagonal
        self.xp = kernels.module_of(inverse_diagonal)
        self.applications = 0

    def __call__(self, residual, out=None):
        if out is None:
            out = self.inverse_diagonal * residual
        else:
            self.xp.multiply(self.inverse_diagonal, residual, out=out)
        self.applications += 1
        return out


class GalerkinCMFDPreconditioner:
    r"""Conservative full-energy two-level correction for a block step.

    The coarse matrix is the exact ``A_H = P^T A_h P`` balance operator. The
    default multiplicative form smooths with ``base``, restricts the remaining
    residual, solves the sparse coarse system exactly, and prolongs the
    correction. ``mode='coarse-first'`` projects the coarse error before
    smoothing, while ``mode='additive'`` applies both corrections to the
    original residual and avoids the extra fine operator application. They are
    included as explicit cost/robustness experiments, not silently selected.

    This first prototype is host-only. It establishes algebra, convergence and
    HP-MR work reduction before GPU restriction/prolongation and coarse solves
    are implemented.
    """

    ndgpu_out = True

    def __init__(self, block: MultigroupStepOperator, base,
                 coarse_space: SpatialCoarseSpace, *, mode="multiplicative"):
        if kernels.is_cupy(block.xp):
            raise ValueError("Galerkin CMFD prototype currently requires CPU")
        if coarse_space.shape != block.cell_shape:
            raise ValueError("coarse space and block shapes do not match")
        if mode not in ("multiplicative", "coarse-first", "additive"):
            raise ValueError("CMFD mode must be multiplicative, coarse-first, "
                             "or additive")
        import scipy.sparse.linalg as spla

        started = time.perf_counter()
        self.block = block
        self.base = base
        self.space = coarse_space
        self.groups = block.groups
        self.mode = mode
        self.prolongation, self.restriction = coarse_space.full_operators(
            self.groups)
        self.fine_matrix = block.assemble()
        self.coarse_matrix = (
            self.restriction @ self.fine_matrix @ self.prolongation).tocsr()
        self._factor = spla.splu(self.coarse_matrix.tocsc())
        self._applied = np.empty(block.shape, dtype=block.inv_diag.dtype)
        self._remaining = np.empty_like(self._applied)
        self._smooth = np.empty_like(self._applied)
        self.stats = SpatialCoarseStats(
            setup_seconds=time.perf_counter() - started)

    @property
    def coarse_cells(self):
        return self.space.coarse_cells

    @property
    def coarse_unknowns(self):
        return self.groups * self.space.coarse_cells

    def reset_stats(self):
        setup = self.stats.setup_seconds
        self.stats = SpatialCoarseStats(setup_seconds=setup)

    def __call__(self, residual, out=None):
        self.block._check_state(residual)
        if self.mode == "coarse-first":
            restricted = self.space.restrict(residual, self.groups)
            coarse_error = self._factor.solve(restricted)
            coarse = self.space.prolong(coarse_error, self.groups)
            if out is None:
                corrected = coarse
            else:
                np.copyto(out, coarse)
                corrected = out
            self.block.apply(corrected, out=self._applied)
            np.subtract(residual, self._applied, out=self._remaining)
            if getattr(self.base, "ndgpu_out", False):
                smooth = self.base(self._remaining, out=self._smooth)
            else:
                np.copyto(self._smooth, self.base(self._remaining))
                smooth = self._smooth
            corrected += smooth
            self.stats.fine_residual_applies += 1
            self.stats.applications += 1
            self.stats.coarse_solves += 1
            return corrected

        if out is None:
            corrected = self.base(residual)
        elif getattr(self.base, "ndgpu_out", False):
            corrected = self.base(residual, out=out)
        else:
            np.copyto(out, self.base(residual))
            corrected = out

        coarse_source = residual
        if self.mode == "multiplicative":
            self.block.apply(corrected, out=self._applied)
            np.subtract(residual, self._applied, out=self._remaining)
            coarse_source = self._remaining
            self.stats.fine_residual_applies += 1
        restricted = self.space.restrict(coarse_source, self.groups)
        coarse_error = self._factor.solve(restricted)
        corrected += self.space.prolong(coarse_error, self.groups)
        self.stats.applications += 1
        self.stats.coarse_solves += 1
        return corrected


@dataclass
class GroupSweepStats:
    applications: int = 0
    group_solves: int = 0
    inner_iterations: int = 0
    inner_iterations_by_group: list = field(default_factory=list)


class EnergyGroupGaussSeidelPreconditioner:
    """Inexact energy-group sweep for flexible right preconditioning.

    Each application approximately solves the loss-plus-scatter block system.
    Lower-index groups use their newly computed values and higher-index groups
    use values from the preceding subsweep.  Optional lagged fission makes this
    the same source/group hierarchy as a production fixed-point evaluation;
    leaving it off lets outer FGMRES resolve fission globally and is usually
    safer near the critical pole.

    For distributed operators each group solve uses the rank-local principal
    block, including interface contributions on the diagonal but omitting
    off-rank unknowns.  This is a non-overlapping domain block-Jacobi (additive
    Schwarz) right preconditioner: its inner PCGs require no communication,
    while outer FGMRES applies the globally coupled operator and repairs the
    subdomain-interface error.

    The group PCG tolerance is deliberately inexact.  FGMRES stores each
    resulting correction separately, so changing inner iteration counts do not
    violate its Krylov recurrence.
    """

    ndgpu_out = True

    def __init__(self, block: MultigroupStepOperator, *, scatter_sweeps=1,
                 include_fission=False, inner_rtol=1e-2, inner_atol=0.0,
                 inner_maxiter=200, precond_degree=0,
                 ordering="forward", anderson_depth=0, relaxation=1.0,
                 inner_fixed_iterations=0, inner_fixed_relaxations=0):
        self.block = block
        self.xp = block.xp
        self.scatter_sweeps = int(scatter_sweeps)
        self.include_fission = bool(include_fission)
        inner_values = np.asarray(inner_rtol, dtype=float)
        if inner_values.ndim == 0:
            inner_values = np.full(block.groups, float(inner_values))
        if inner_values.shape != (block.groups,):
            raise ValueError("inner_rtol must be scalar or one value per group")
        self.inner_rtols = tuple(float(value) for value in inner_values)
        self.inner_rtol = (self.inner_rtols[0]
                           if len(set(self.inner_rtols)) == 1
                           else self.inner_rtols)
        self.inner_atol = float(inner_atol)
        self.inner_maxiter = int(inner_maxiter)
        self.ordering = str(ordering)
        self.anderson_depth = int(anderson_depth)
        self.relaxation = float(relaxation)
        self.inner_fixed_iterations = int(inner_fixed_iterations)
        self.inner_fixed_relaxations = int(inner_fixed_relaxations)
        if self.scatter_sweeps < 1:
            raise ValueError("scatter_sweeps must be positive")
        if any(value < 0.0 for value in self.inner_rtols) or self.inner_atol < 0.0:
            raise ValueError("inner tolerances must be non-negative")
        if self.inner_maxiter < 1:
            raise ValueError("inner_maxiter must be positive")
        if self.inner_fixed_iterations < 0:
            raise ValueError("fixed inner iterations must be non-negative")
        if self.inner_fixed_relaxations < 0:
            raise ValueError("fixed inner relaxations must be non-negative")
        if self.inner_fixed_iterations and self.inner_fixed_relaxations:
            raise ValueError("choose fixed PCG or fixed relaxation, not both")
        if self.ordering not in ("forward", "reverse", "alternating"):
            raise ValueError(
                "energy ordering must be forward, reverse, or alternating")
        if self.anderson_depth not in (0, 1, 2):
            raise ValueError("energy-sweep Anderson depth must be zero, one, or two")
        if (not np.isfinite(self.relaxation)
                or not 0.0 < self.relaxation < 2.0):
            raise ValueError("energy-sweep relaxation must lie between 0 and 2")
        self.group_applies = tuple(
            getattr(op, "preconditioner_apply", op.apply)
            for op in block.operators)
        self.group_preconditioners = tuple(neumann_preconditioner(
            apply, op.inv_diag, int(precond_degree))
            for apply, op in zip(self.group_applies, block.operators))
        templates = [self.xp.zeros(block.cell_shape,
                                   dtype=block.operators[0].inv_diag.dtype)
                     for _ in range(block.groups)]
        self.workspaces = tuple(PCGWorkspace.like(
            template, operator_out=True) for template in templates)
        self.stats = GroupSweepStats(
            inner_iterations_by_group=[0] * block.groups)
        if self.anderson_depth or self.relaxation != 1.0:
            self._aa_before = self.xp.empty(block.shape,
                                            dtype=templates[0].dtype)
            self._aa_delta = self.xp.empty_like(self._aa_before)
        if self.anderson_depth:
            self._aa_previous_f = self.xp.empty_like(self._aa_before)
            self._aa_previous_g = self.xp.empty_like(self._aa_before)
            self._aa_current_g = self.xp.empty_like(self._aa_before)
            self._aa_f = self.xp.empty_like(self._aa_before)
            self._aa_tiny = np.finfo(templates[0].dtype).tiny
            if self.anderson_depth == 2:
                self._aa_older_f = self.xp.empty_like(self._aa_before)
                self._aa_older_g = self.xp.empty_like(self._aa_before)
                self._aa_delta2 = self.xp.empty_like(self._aa_before)

    def reset_stats(self):
        self.stats = GroupSweepStats(
            inner_iterations_by_group=[0] * self.block.groups)

    def __call__(self, residual, out=None):
        self.block._check_state(residual)
        z = self.xp.zeros_like(residual) if out is None else out
        self.block._check_state(z)
        z.fill(0)
        for sweep_index in range(self.scatter_sweeps):
            if self.anderson_depth or self.relaxation != 1.0:
                self.xp.copyto(self._aa_before, z)
            fission = (self.block.fission_source(z)
                       if self.include_fission else None)
            reverse = (self.ordering == "reverse" or
                       (self.ordering == "alternating"
                        and sweep_index % 2 == 1))
            group_order = (range(self.block.groups - 1, -1, -1)
                           if reverse else range(self.block.groups))
            for g in group_order:
                op = self.block.operators[g]
                apply_inner = self.group_applies[g]
                q = residual[g].copy()
                if self.block._group_batch is not None:
                    if self.block.rhs_weight is None:
                        kernels.group_accumulate(
                            self.xp, q, self.block._group_batch["S"][g], z)
                        coupled = None
                    else:
                        coupled = self.block._scatter_buffer
                        coupled.fill(0)
                        kernels.group_accumulate(
                            self.xp, coupled,
                            self.block._group_batch["S"][g], z)
                else:
                    coupled = None
                    for gf in range(self.block.groups):
                        scatter = self.block.sigma_s[gf][g]
                        if gf != g and scatter is not None:
                            term = scatter * z[gf]
                            coupled = term if coupled is None else coupled + term
                if fission is not None:
                    term = self.block.emission_weights[g] * fission
                    coupled = term if coupled is None else coupled + term
                if coupled is not None:
                    if self.block.rhs_weight is not None:
                        coupled = self.block.rhs_weight * coupled
                    q += coupled
                if self.inner_fixed_relaxations:
                    work = self.workspaces[g]
                    for _ in range(self.inner_fixed_relaxations):
                        apply_inner(z[g], out=work.ap)
                        self.xp.subtract(q, work.ap, out=work.r)
                        self.group_preconditioners[g](
                            work.r, out=work.z, scratch=work.ap)
                        z[g] += work.z
                    solved = z[g]
                    iterations = self.inner_fixed_relaxations
                elif self.inner_fixed_iterations:
                    solved, iterations = fixed_pcg(
                        apply_inner, q, z[g], op.inv_diag, self.xp,
                        iterations=self.inner_fixed_iterations,
                        precond=self.group_preconditioners[g],
                        workspace=self.workspaces[g])
                else:
                    solved, iterations = pcg(
                        apply_inner, q, z[g], op.inv_diag, self.xp,
                        rtol=self.inner_rtols[g], atol=self.inner_atol,
                        maxiter=self.inner_maxiter,
                        precond=self.group_preconditioners[g],
                        workspace=self.workspaces[g], raise_on_fail=False)
                z[g] = solved
                self.stats.group_solves += 1
                self.stats.inner_iterations += iterations
                self.stats.inner_iterations_by_group[g] += iterations
            if self.relaxation != 1.0:
                self.xp.subtract(z, self._aa_before, out=self._aa_delta)
                self._aa_delta *= self.relaxation
                self.xp.add(self._aa_before, self._aa_delta, out=z)
            if self.anderson_depth:
                # Walker--Ni depth-one Anderson update for the energy-sweep
                # fixed point.  All full-state work is persistent; on GPU the
                # two dot products are the only added synchronizations.
                self.xp.subtract(z, self._aa_before, out=self._aa_f)
                self.xp.copyto(self._aa_current_g, z)
                if sweep_index and (self.anderson_depth == 1
                                    or sweep_index == 1):
                    self.xp.subtract(
                        self._aa_f, self._aa_previous_f, out=self._aa_delta)
                    denominator = self.xp.sum(
                        self._aa_delta * self._aa_delta)
                    numerator = self.xp.sum(self._aa_delta * self._aa_f)
                    gamma = numerator / (denominator + self._aa_tiny)
                    self.xp.subtract(
                        self._aa_current_g, self._aa_previous_g,
                        out=self._aa_delta)
                    self._aa_delta *= gamma
                    self.xp.subtract(
                        self._aa_current_g, self._aa_delta, out=z)
                elif sweep_index >= 2:
                    # Depth two uses the two latest map differences.  The
                    # closed-form 2x2 normal-equation solve keeps every scalar
                    # on device and avoids a tiny generic solver launch.
                    self.xp.subtract(
                        self._aa_previous_f, self._aa_older_f,
                        out=self._aa_delta)
                    self.xp.subtract(
                        self._aa_f, self._aa_previous_f,
                        out=self._aa_delta2)
                    a = self.xp.sum(self._aa_delta * self._aa_delta)
                    b = self.xp.sum(self._aa_delta * self._aa_delta2)
                    c = self.xp.sum(self._aa_delta2 * self._aa_delta2)
                    u = self.xp.sum(self._aa_delta * self._aa_f)
                    v = self.xp.sum(self._aa_delta2 * self._aa_f)
                    determinant = a * c - b * b
                    regularized = determinant + self._aa_tiny
                    gamma0 = (c * u - b * v) / regularized
                    gamma1 = (a * v - b * u) / regularized
                    self.xp.subtract(
                        self._aa_previous_g, self._aa_older_g,
                        out=self._aa_delta)
                    self.xp.subtract(
                        self._aa_current_g, self._aa_previous_g,
                        out=self._aa_delta2)
                    self._aa_delta *= gamma0
                    self._aa_delta2 *= gamma1
                    self.xp.add(self._aa_delta, self._aa_delta2,
                                out=self._aa_delta)
                    self.xp.subtract(
                        self._aa_current_g, self._aa_delta, out=z)
                if self.anderson_depth == 2 and sweep_index:
                    self.xp.copyto(self._aa_older_f, self._aa_previous_f)
                    self.xp.copyto(self._aa_older_g, self._aa_previous_g)
                self.xp.copyto(self._aa_previous_f, self._aa_f)
                self.xp.copyto(self._aa_previous_g, self._aa_current_g)
        self.stats.applications += 1
        # Each call owns `z`; FGMRES may retain it as a basis vector directly.
        return z


class AdjointCoarseCorrectionPreconditioner:
    r"""Rank-one amplitude correction wrapped around a base preconditioner.

    Let ``p`` be a forward coarse mode, ``q`` its adjoint importance, and
    ``M^-1`` the energy-group sweep.  The adapted-deflation update

    .. math::

       \widetilde M^{-1}r = M^{-1}r +
       (p-M^{-1}Ap){q^T r\over q^T A p}

    makes the preconditioned operator exact on ``p`` while retaining the base
    action on the complementary space.  ``q`` supplies the physically relevant
    neutron-population coordinate instead of an unweighted Euclidean average.

    Setup costs one block apply and one base-preconditioner application. Every
    subsequent call adds only one adjoint dot product and one vector update; it
    does *not* duplicate the outer FGMRES block apply. The base may still be
    inexact/variable because the enclosing solver is flexible GMRES, although
    exactness on ``p`` is relative to the setup application of the base.
    """

    def __init__(self, block: MultigroupStepOperator, base, forward_mode,
                 adjoint_mode):
        self.block = block
        self.base = base
        self.xp = block.xp
        forward = self.xp.asarray(forward_mode)
        adjoint = self.xp.asarray(adjoint_mode)
        block._check_state(forward)
        block._check_state(adjoint)
        if not bool(self.xp.all(self.xp.isfinite(forward))):
            raise ValueError("forward coarse mode must be finite")
        if not bool(self.xp.all(self.xp.isfinite(adjoint))):
            raise ValueError("adjoint coarse mode must be finite")
        applied = block.apply(forward)
        denominator = float(self.xp.sum(adjoint * applied))
        scale = float(self.xp.sqrt(self.xp.sum(adjoint * adjoint))
                          * self.xp.sqrt(self.xp.sum(applied * applied)))
        if not np.isfinite(denominator) or abs(denominator) <= 1e-14 * max(scale, 1e-300):
            raise ValueError("adjoint coarse denominator q^T A p is singular")
        base_on_mode = base(applied)
        self.forward_mode = forward.copy()
        self.adjoint_mode = adjoint.copy()
        self.denominator = denominator
        self.direction = self.forward_mode - base_on_mode
        self.applications = 0

    def __call__(self, residual):
        self.block._check_state(residual)
        corrected = self.base(residual)
        coefficient = float(self.xp.sum(self.adjoint_mode * residual))
        corrected += (coefficient / self.denominator) * self.direction
        self.applications += 1
        return corrected


@dataclass
class EnergyCoarseStats:
    applications: int = 0
    setup_base_applications: int = 0
    setup_seconds: float = 0.0


class EnergyModeCoarseCorrectionPreconditioner:
    r"""Resolve group-to-group amplitude error in a small coarse system.

    The trial space contains one mode per selected energy group.  Mode ``g``
    is the latest forward flux shape in that group and is zero in all other
    groups; the test space is constructed analogously from the adjoint flux.
    For trial matrix ``P``, test matrix ``Q`` and base preconditioner ``B``,
    the adapted-deflation update is

    .. math::

       \widetilde B r = B r + (P-BAP)(Q^TAP)^{-1}Q^T r.

    It is therefore exact on the complete group-amplitude space even when
    ``B`` is not.  Setup costs one block and one base application per retained
    group.  An application performs one batched group projection, an at-most
    ``G x G`` dense solve, and one basis accumulation, with no extra fine block
    application.  Only ``P-BAP`` is stored as full-state basis vectors; the
    group-local trial and test modes use the original forward/adjoint arrays.
    """

    ndgpu_out = True

    def __init__(self, block: MultigroupStepOperator, base, forward_mode,
                 adjoint_mode, *, groups=None):
        started = time.perf_counter()
        self.block = block
        self.base = base
        self.xp = block.xp
        forward = self.xp.asarray(forward_mode)
        adjoint = self.xp.asarray(adjoint_mode)
        block._check_state(forward)
        block._check_state(adjoint)
        if not bool(self.xp.all(self.xp.isfinite(forward))):
            raise ValueError("forward energy modes must be finite")
        if not bool(self.xp.all(self.xp.isfinite(adjoint))):
            raise ValueError("adjoint energy modes must be finite")
        if groups is None:
            groups = range(block.groups)
        selected = tuple(int(g) for g in groups)
        if (not selected or len(set(selected)) != len(selected)
                or any(g < 0 or g >= block.groups for g in selected)):
            raise ValueError("energy groups must be unique valid indices")
        self.groups = selected

        # Independent normalization leaves the two subspaces and the adapted
        # deflation update unchanged, while keeping the coarse matrix well
        # scaled across fast and thermal groups.
        self.forward_modes = self.xp.empty_like(forward)
        self.test_modes = self.xp.empty_like(adjoint)
        self.forward_modes.fill(0)
        self.test_modes.fill(0)
        for g in selected:
            pnorm = float(self.xp.sqrt(self.xp.sum(forward[g] * forward[g])))
            qnorm = float(self.xp.sqrt(self.xp.sum(adjoint[g] * adjoint[g])))
            if (not np.isfinite(pnorm) or not np.isfinite(qnorm)
                    or pnorm <= 0.0 or qnorm <= 0.0):
                raise ValueError(f"energy group {g} has a zero mode")
            self.forward_modes[g] = forward[g] / pnorm
            self.test_modes[g] = adjoint[g] / qnorm

        count = len(selected)
        self.directions = self.xp.empty(
            (count,) + block.shape, dtype=forward.dtype)
        coarse = self.xp.empty((count, count), dtype=forward.dtype)
        trial = self.xp.zeros(block.shape, dtype=forward.dtype)
        for j, group in enumerate(selected):
            trial.fill(0)
            trial[group] = self.forward_modes[group]
            applied = block.apply(trial)
            for i, test_group in enumerate(selected):
                coarse[i, j] = self.xp.sum(
                    self.test_modes[test_group] * applied[test_group])
            base_applied = base(applied)
            self.xp.subtract(trial, base_applied, out=self.directions[j])

        coarse_host = np.asarray(asnumpy(coarse))
        condition = float(np.linalg.cond(coarse_host))
        if not np.isfinite(condition) or condition > 1e13:
            raise ValueError(
                f"energy coarse matrix is singular or ill-conditioned "
                f"(condition {condition:.3e})")
        self.coarse_matrix = coarse
        self.condition = condition
        self.stats = EnergyCoarseStats(
            setup_base_applications=count,
            setup_seconds=time.perf_counter() - started)

    @property
    def coarse_unknowns(self):
        return len(self.groups)

    @property
    def storage_bytes(self):
        return int(self.directions.nbytes + self.forward_modes.nbytes
                   + self.test_modes.nbytes + self.coarse_matrix.nbytes)

    def reset_stats(self):
        setup = self.stats.setup_seconds
        setup_apps = self.stats.setup_base_applications
        self.stats = EnergyCoarseStats(
            setup_base_applications=setup_apps, setup_seconds=setup)

    def __call__(self, residual, out=None):
        self.block._check_state(residual)
        if out is None:
            corrected = self.base(residual)
        elif getattr(self.base, "ndgpu_out", False):
            corrected = self.base(residual, out=out)
        else:
            self.xp.copyto(out, self.base(residual))
            corrected = out
        projected = self.xp.stack([
            self.xp.sum(self.test_modes[g] * residual[g])
            for g in self.groups])
        coefficients = self.xp.linalg.solve(self.coarse_matrix, projected)
        kernels.basis_accumulate(
            self.xp, corrected, self.directions, coefficients,
            self.coarse_unknowns)
        self.stats.applications += 1
        return corrected


class SourceShapeCoarseCorrectionPreconditioner:
    r"""Adapted deflation of regional fission-shape amplitude modes.

    ``spatial_modes[k]`` is a scalar cell field, normally a disjoint material
    region or a low-order spatial bin.  The corresponding multigroup trial
    and test vectors are ``forward_mode * spatial_modes[k]`` and
    ``adjoint_mode * spatial_modes[k]``.  This preserves the physical energy
    spectrum inside each region while allowing its amplitude to move
    independently.  Like :class:`EnergyModeCoarseCorrectionPreconditioner`,
    the online correction needs no fine block application.
    """

    ndgpu_out = True

    def __init__(self, block: MultigroupStepOperator, base, forward_mode,
                 adjoint_mode, spatial_modes):
        started = time.perf_counter()
        self.block = block
        self.base = base
        self.xp = block.xp
        forward = self.xp.asarray(forward_mode)
        adjoint = self.xp.asarray(adjoint_mode)
        modes = self.xp.asarray(spatial_modes)
        block._check_state(forward)
        block._check_state(adjoint)
        if modes.ndim == len(block.cell_shape):
            modes = modes[None, ...]
        if tuple(modes.shape[1:]) != block.cell_shape or modes.shape[0] < 1:
            raise ValueError("spatial modes must have shape (modes, *cells)")
        if not bool(self.xp.all(self.xp.isfinite(modes))):
            raise ValueError("spatial modes must be finite")

        count = int(modes.shape[0])
        self.trial_modes = self.xp.empty(
            (count,) + block.shape, dtype=forward.dtype)
        self.test_modes = self.xp.empty_like(self.trial_modes)
        for j in range(count):
            self.trial_modes[j] = forward * modes[j][None, ...]
            self.test_modes[j] = adjoint * modes[j][None, ...]
            pnorm = float(self.xp.sqrt(
                self.xp.sum(self.trial_modes[j] * self.trial_modes[j])))
            qnorm = float(self.xp.sqrt(
                self.xp.sum(self.test_modes[j] * self.test_modes[j])))
            if (not np.isfinite(pnorm) or not np.isfinite(qnorm)
                    or pnorm <= 0.0 or qnorm <= 0.0):
                raise ValueError(f"spatial mode {j} has zero forward/adjoint weight")
            self.trial_modes[j] *= 1.0 / pnorm
            self.test_modes[j] *= 1.0 / qnorm

        self.directions = self.xp.empty_like(self.trial_modes)
        coarse = self.xp.empty((count, count), dtype=forward.dtype)
        reduction_axes = tuple(range(1, self.test_modes.ndim))
        for j in range(count):
            applied = block.apply(self.trial_modes[j])
            coarse[:, j] = self.xp.sum(
                self.test_modes * applied[None, ...], axis=reduction_axes)
            base_applied = base(applied)
            self.xp.subtract(
                self.trial_modes[j], base_applied, out=self.directions[j])

        coarse_host = np.asarray(asnumpy(coarse))
        condition = float(np.linalg.cond(coarse_host))
        if not np.isfinite(condition) or condition > 1e13:
            raise ValueError(
                f"source-shape coarse matrix is singular or ill-conditioned "
                f"(condition {condition:.3e})")
        self.coarse_matrix = coarse
        self.condition = condition
        self.stats = EnergyCoarseStats(
            setup_base_applications=count,
            setup_seconds=time.perf_counter() - started)

    @property
    def coarse_unknowns(self):
        return int(self.coarse_matrix.shape[0])

    @property
    def storage_bytes(self):
        return int(self.trial_modes.nbytes + self.test_modes.nbytes
                   + self.directions.nbytes + self.coarse_matrix.nbytes)

    def __call__(self, residual, out=None):
        self.block._check_state(residual)
        if out is None:
            corrected = self.base(residual)
        elif getattr(self.base, "ndgpu_out", False):
            corrected = self.base(residual, out=out)
        else:
            self.xp.copyto(out, self.base(residual))
            corrected = out
        reduction_axes = tuple(range(1, self.test_modes.ndim))
        projected = self.xp.sum(
            self.test_modes * residual[None, ...], axis=reduction_axes)
        coefficients = self.xp.linalg.solve(self.coarse_matrix, projected)
        kernels.basis_accumulate(
            self.xp, corrected, self.directions, coefficients,
            self.coarse_unknowns)
        self.stats.applications += 1
        return corrected


__all__ = [
    "MultigroupStepOperator",
    "EnergyGroupGaussSeidelPreconditioner",
    "AdjointCoarseCorrectionPreconditioner",
    "EnergyModeCoarseCorrectionPreconditioner",
    "SourceShapeCoarseCorrectionPreconditioner",
    "BlockJacobiPreconditioner", "SpatialCoarseSpace",
    "GalerkinCMFDPreconditioner",
    "GroupSweepStats", "SpatialCoarseStats", "EnergyCoarseStats",
]
