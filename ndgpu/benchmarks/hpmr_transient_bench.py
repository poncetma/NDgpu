"""Cost model for an **uncoupled** HP-MR diffusion transient, CPU vs GPU.

The harness behind ``examples/hpmr_transient_bench.py`` and
``notebooks/colab_hpmr_transient_gpu.ipynb``. It lives in the package rather
than in the example so the two cannot drift: Colab installs ``dist/ndgpu-src.zip``,
which carries the package and not ``examples/``.

**What a step costs, and which factor a GPU can touch.** A step is a fixed point
over the end-of-step fission source; each sweep does ``subsweeps`` Gauss-Seidel
passes over the G groups, and each pass solves one group system with PCG. So

    ms/step  =  (sweeps/step) x (G x subsweeps) x (CG iters/solve) x (cost/iter)

The first three factors are *algorithm*. They are set by the perturbation, the
tolerances and the accelerator, and they must come out identical on CPU and GPU
-- so every measurement here reports them, and any CPU/GPU comparison should
assert they match. Only the last factor is a property of the bus, and it is what
a speedup number is actually about. Reporting wall time alone hides the
difference between "the GPU ran the same work faster" and "the GPU happened to
converge in fewer sweeps", which is the standard way this measurement goes
wrong.

**The manoeuvre is a uniform absorption scaling, not a drum rotation.** A drum
presents a different absorber area fraction to every mesh, so its worth -- and
hence the iteration counts -- would change with refinement, and the size sweep
would not be comparing like with like. A bulk scaling is shape-preserving and
essentially mesh-independent, so iteration counts across the sweep differ only
through the discretization. It is applied as a step at t=0, which makes every
timed step start far from converged: the expensive regime, and the honest one.
"""

from __future__ import annotations

import time

import numpy as np

from .. import kernels
from ..materials import Material
from ..tri import TriDiffusionEigenSolver, TriGroupOperator
from ..transient import TransientSolver
from .hpmr import HPMR_KINETICS, build_hpmr2d, build_hpmr3d
from .hpmr_thermal import hpmr_endfb8_builtin, hpmr_kinetics_11g

# Uniform Sigma_a multiplier worth about +0.5 $ on the 11-group core at
# refine 4 (measured: +324.9 pcm against beta = 650 pcm). Half a dollar is a
# large but subcritical step -- the power roughly doubles promptly -- so the
# fixed point has real work to do on every step without the run diverging.
SIGMA_A_SCALE = 0.997065

# Gauss-Seidel passes over the energy groups per fixed-point evaluation.
# THE dominant cost knob on this strongly upscattering 11-group set -- worth
# ~7x the CG iterations and ~3x the wall time against the 3 that the solver's
# auto rule used to pick. Measured sweeps/CG for one step, refine 2 and 3:
#
#     subsweeps      3          4          5         6         8
#     refine 2   248/35689  144/23421   43/7661   27/5285   25/5536
#     refine 3   247/52864  144/34609   30/8068   26/7583   25/8254
#
# Optimum 6, plateau 5-8, identical at both sizes, and 6 is the most ACCURATE
# too: subsweeps 3/6/12 all converge to the same power (1.3990072/1.3990070/
# 1.3990067 at tol_step 1e-9) and the production-tolerance errors are
# 2.5e-4/7.1e-5/1.8e-4. The 4->5 transition is a cliff rather than a gradient,
# and the mechanism is NOT the scattering iteration on its own: with
# rebalance=False the subsweep count barely matters (1347 -> 1080 sweeps from 3
# to 6). Subsweeps supply the spectrally CONSISTENT flux the rebalance needs
# for its balance assumption to hold; the two together are worth ~50x.
#
# _UPSCATTER_SUBSWEEPS in transient.py now defaults to 6 as well (raised after
# the same check passed on C5G7-TD), so this constant is belt-and-braces --
# kept because it is where the HP-MR-specific evidence lives.
SUBSWEEPS = 6


def parse_size(spec: str) -> tuple[int, int]:
    """``'2d:4' -> (4, 0)``; ``'3d:4x20' -> (4, 20)``."""
    kind, _, rest = spec.partition(":")
    if kind == "2d":
        return int(rest), 0
    if kind == "3d":
        r, _, nz = rest.partition("x")
        return int(r), int(nz or 20)
    raise ValueError(f"size must look like 2d:4 or 3d:4x20, got {spec!r}")


def scale_absorption(materials, factor):
    """Uniform bulk perturbation: every material's Sigma_a times ``factor``."""
    return [Material(name=m.name, diffusion=m.diffusion,
                     sigma_a=m.sigma_a * factor, nu_sigma_f=m.nu_sigma_f,
                     sigma_s=m.sigma_s, chi=m.chi if m.is_fissile else None,
                     kappa_fission=m.kappa_fission)
            for m in materials]


def build_case(refine: int, nz: int = 0, groups: str = "11"):
    """The problem and its perturbed twin, for the given size and library."""
    three_d = nz > 0
    mats0 = hpmr_endfb8_builtin(three_d=three_d) if groups == "11" else None
    build = ((lambda m: build_hpmr3d(refine=refine, nz=nz, drum_angle_deg=150.0,
                                     absorber="polar", materials=m))
             if three_d else
             (lambda m: build_hpmr2d(refine=refine, drum_angle_deg=150.0,
                                     absorber="polar", materials=m)))
    p = build(mats0)
    kin = hpmr_kinetics_11g() if groups == "11" else HPMR_KINETICS
    return p, kin


def case_dof(refine: int, nz: int = 0, groups: str = "11") -> int:
    """Unknowns (active cells x groups) for a size, without solving anything.

    For deciding whether a leg is worth launching -- a CPU run at a few hundred
    outer sweeps per step is hours, and that call has to be made from the size,
    not from a trial solve.
    """
    p, kin = build_case(refine, nz, groups)
    return int(np.count_nonzero(p.active)) * int(kin.velocities.shape[-1])


def transient_bench(refine: int = 4, nz: int = 0, *, groups: str = "11",
                    steps: int = 6, dt: float = 0.02, tol_step: float = 1e-6,
                    anderson_depth: int = 1, rebalance: bool = True,
                    max_sweeps: int = 4000, subsweeps: int | None = SUBSWEEPS,
                    sigma_a_scale: float = SIGMA_A_SCALE,
                    device: str = "auto", dtype=np.float64,
                    precond_degree: int = 0, check_every: int = 1,
                    precond_dtype=None,
                    graph_block: int = 0,
                    reuse_krylov_workspaces: bool = True,
                    step_solver: str = "fixed-point",
                    multigroup_kwargs: dict | None = None,
                    time_scheme: str = "backward-euler",
                    warmup: bool | None = None, verbose: bool = False) -> dict:
    """Time ``steps`` steps of the uncoupled transient; return counters.

    Returned keys pair cost with the work that produced it: ``ms_step``,
    ``us_per_cg`` and ``ns_per_cg_dof`` alongside ``cg_per_step``, ``dof`` and
    the final ``power`` -- so a run can be checked for having done the same work
    before its time is compared with another's.
    """
    p, kin = build_case(refine, nz, groups)
    mats1 = scale_absorption(p.materials, sigma_a_scale)
    dtype = np.dtype(dtype)

    # Step insertion at t=0. The 0.5*dt threshold puts the switch strictly
    # inside the first step, so no step straddles it.
    problem_at = (lambda t: ((mats1 if t >= 0.5 * dt else p.materials),
                             p.material_map, p.mix_material, p.mix_weight))

    solver = TransientSolver(
        p.grid, problem_at, kin, bc=p.bc, active=p.active, mask_bc=p.mask_bc,
        mix_material=p.mix_material, mix_weight=p.mix_weight,
        group_operator=TriGroupOperator, eig_solver=TriDiffusionEigenSolver,
        precond_degree=precond_degree, precond_dtype=precond_dtype,
        device=device, dtype=dtype)

    kwargs = dict(dt=dt, tol_step=tol_step, anderson_depth=anderson_depth,
                  rebalance=rebalance, max_sweeps=max_sweeps,
                  scatter_subsweeps=subsweeps,
                  reuse_krylov_workspaces=reuse_krylov_workspaces,
                  step_solver=step_solver, time_scheme=time_scheme)
    if multigroup_kwargs is not None:
        kwargs["multigroup_kwargs"] = dict(multigroup_kwargs)
    if step_solver == "fixed-point" and (check_every > 1 or graph_block):
        # Passed only when exercised, so the default leg calls the solver
        # exactly as every other script in the repo does.
        kwargs["linsolve_kwargs"] = dict(
            check_every=check_every, graph_block=graph_block)

    if warmup is None:
        # Default on GPU, off on CPU. A warm-up step here costs as much as a
        # measured one, and NumPy has nothing to warm: no kernel compilation, no
        # memory pool. Paying it on CPU would double every baseline for nothing.
        warmup = kernels.is_cupy(solver.xp)
    if warmup:
        # On GPU the first touch of each kernel compiles it, and the first
        # operator build allocates the pools. Two steps reaches every kernel the
        # loop uses, including the rebuild triggered by the insertion.
        solver.solve(t_end=2 * dt, **kwargs)

    t0 = time.perf_counter()
    res = solver.solve(t_end=steps * dt, verbose=verbose, **kwargs)
    wall = time.perf_counter() - t0

    n_cells = int(np.count_nonzero(p.active))
    G = kin.velocities.shape[-1]
    dof = n_cells * G
    n_steps = len(res.times) - 1
    inner = max(res.total_inner_iterations, 1)
    sweeps = sum(res.step_iterations) / max(len(res.step_iterations), 1)
    return dict(refine=refine, nz=nz, cells=n_cells, groups=G, dof=dof,
                steps=n_steps, wall=wall, ms_step=1e3 * wall / n_steps,
                inner=res.total_inner_iterations, cg_per_step=inner / n_steps,
                sweeps_per_step=sweeps,
                # CG iterations per group per sweep, summed over the Gauss-Seidel
                # subsweeps (n_sub is 3 when the library has upscatter, as the
                # 11-group set does, so divide by 3 for the per-solve count).
                cg_per_group_sweep=inner / max(sweeps * n_steps * G, 1),
                us_per_cg=1e6 * wall / inner,
                ns_per_cg_dof=1e9 * wall / inner / dof,
                k0=res.k0, power=float(res.power[-1]),
                steady_s=res.steady.solve_seconds, device=res.device,
                dtype=str(dtype), precond_degree=precond_degree,
                precond_dtype=(None if precond_dtype is None
                               else str(np.dtype(precond_dtype))),
                mixed_precision_fallbacks=res.mixed_precision_fallbacks,
                cuda_graph_captures=res.cuda_graph_captures,
                cuda_graph_replays=res.cuda_graph_replays,
                cuda_graph_errors=res.cuda_graph_errors,
                check_every=check_every,
                graph_block=int(graph_block),
                step_solver=str(step_solver),
                time_scheme=res.time_scheme,
                time_orders=list(res.time_orders),
                reuse_krylov_workspaces=bool(reuse_krylov_workspaces))


HEADER = (f"{'case':>12}  {'cells':>8}  {'dof':>9}  {'ms/step':>9}  "
          f"{'sweeps':>7}  {'cg/step':>8}  {'cg/g/sw':>8}  {'us/cg':>7}  "
          f"{'ns/cg/dof':>9}  {'P(end)':>9}")


def format_row(r: dict, label: str | None = None) -> str:
    label = label or (f"2d:{r['refine']}" if not r["nz"]
                      else f"3d:{r['refine']}x{r['nz']}")
    return (f"{label:>12}  {r['cells']:>8,}  {r['dof']:>9,}  "
            f"{r['ms_step']:>9.1f}  {r['sweeps_per_step']:>7.0f}  "
            f"{r['cg_per_step']:>8.0f}  {r['cg_per_group_sweep']:>8.1f}  "
            f"{r['us_per_cg']:>7.1f}  {r['ns_per_cg_dof']:>9.2f}  "
            f"{r['power']:>9.5f}")
