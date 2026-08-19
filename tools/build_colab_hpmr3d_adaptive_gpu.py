"""Build the 3-D HP-MR adaptive-BDF GPU validation notebook."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def cell(kind, source):
    result = {
        "cell_type": kind,
        "metadata": {},
        "source": dedent(source).lstrip("\n").splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell("markdown", r"""
    # NDgpu adaptive BDF on a 3-D, 11-group HP-MR core

    This is an executable GPU acceptance experiment for the latest transient work. It does not embed or assume a speedup. Instead it measures the selected Colab GPU and checks that faster runs solve the same problem.

    The notebook answers four questions:

    1. Does adaptive BDF's bookkeeping remain on device? A microbenchmark compares the current one-transfer full-state error norm with the previous per-field synchronization pattern and A/B tests fused history extrapolation.
    2. Is the new monolithic path using its GPU implementation? The run must report energy-group batching and a persistent FGMRES workspace, and a short A/B gate measures the depth-one energy-sweep Anderson accelerator.
    3. Is adaptive BDF accurate on a real 3-D HP-MR model? Both adaptive methods are compared with a finer fixed-step backward-Euler history, including power, thermal exchange temperatures, and final flux shape.
    4. Does BDF beat accuracy-matched adaptive backward Euler here? Accepted/rejected steps, BDF order occupancy, Krylov work, march time, and phase timings are all reported. A low-order result is a physical finding, not a failed GPU benchmark.

    The default `QUICK=True` case is already 3-D and uses the 11-group ENDF/B-VIII-derived library (about 145k active flux unknowns at refinement 2 × 10 axial layers). Set `QUICK=False` for the production-oriented refinement 4 × 20-layer gate. Startup and transient march time are kept separate, and all benchmark legs reuse exactly the same converged hot equilibrium.

    > In Colab choose **Runtime → Change runtime type → T4 GPU**. From the repository, first run `python tools/build_src_zip.py`, then upload `dist/ndgpu-src.zip` in the next cell.
    """),
    cell("code", r"""
    # Colab install. A local Jupyter run uses the current checkout.
    try:
        from google.colab import files
        uploaded = files.upload()
        archive = next(iter(uploaded))
        get_ipython().run_line_magic(
            "pip", f"install -q --force-reinstall --no-deps {archive}")
        try:
            import cupy
        except ImportError:
            get_ipython().run_line_magic("pip", "install -q cupy-cuda12x")
        get_ipython().system("nvidia-smi")
    except ImportError:
        pass
    """),
    cell("code", r"""
    import inspect
    import json
    import platform
    import time
    from collections import Counter

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import ndgpu
    import cupy as cp

    from ndgpu import BDF, kernels
    from ndgpu.benchmarks.hpmr import build_hpmr3d
    from ndgpu.benchmarks.hpmr_thermal import (
        build_hpmr_coupling, hpmr_endfb8_builtin)
    from ndgpu.coupling import CoupledSolver, coupled_transient
    from ndgpu.linalg import FGMRESWorkspace, fgmres

    assert cp.cuda.runtime.getDeviceCount() > 0, (
        "No CUDA GPU is visible. Enable a GPU runtime and rerun from the top.")
    required = {"initial_coupled_steady", "transient_kwargs", "profile"}
    missing = required - set(inspect.signature(coupled_transient).parameters)
    assert not missing, (
        f"The uploaded archive predates this notebook; missing {sorted(missing)}")
    assert "workspace" in inspect.signature(fgmres).parameters
    assert "adaptive" in kernels._GROUPS

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    GPU_NAME = props["name"].decode()
    print("ndgpu:", inspect.getfile(ndgpu))
    print("GPU  :", GPU_NAME)
    print(f"VRAM : {free_b/2**30:.2f} GiB free / {total_b/2**30:.2f} GiB total")
    print("CuPy :", cp.__version__)
    """),
    cell("markdown", r"""
    ## 1. Adaptive-state GPU microbenchmarks

    An 11-group, six-precursor adaptive state contains 17 fields. Previously, numerator and denominator norms converted every field reduction to a Python float: 34 blocking device synchronizations for one error estimate, and up to three estimates during automatic order selection. The current routine streams one field at a time, retains only device scalars, and transfers the final numerator/denominator pair once.

    The second A/B isolates the fused in-place AXPY used by nonuniform BDF prediction. CUDA event timing excludes Python scheduling after the event and synchronizes only at the end of each sample.
    """),
    cell("code", r"""
    def cuda_ms(call, warmup=3, repeats=20):
        for _ in range(warmup):
            call()
        cp.cuda.get_current_stream().synchronize()
        samples = []
        for _ in range(repeats):
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record(); call(); stop.record(); stop.synchronize()
            samples.append(cp.cuda.get_elapsed_time(start, stop))
        return float(np.median(samples)), samples


    rng = cp.random.RandomState(7)
    state = [rng.standard_normal((128, 128, 8), dtype=cp.float64)
             for _ in range(17)]
    predicted = [value * (1.0 + 2e-5) for value in state]
    RTOL_MICRO, ATOL_MICRO = 1e-3, 1e-10

    def legacy_error_norm():
        residual = [c - p for c, p in zip(state, predicted)]
        scale = [ATOL_MICRO + RTOL_MICRO * abs(c) for c in state]
        denominator = sum(float((value * value).sum()) for value in scale)
        numerator = sum(float((value * value).sum()) for value in residual)
        return np.sqrt(numerator / denominator)

    current_value = BDF.error_norm(
        state, predicted, rtol=RTOL_MICRO, atol=ATOL_MICRO)
    legacy_value = legacy_error_norm()
    assert np.isclose(current_value, legacy_value, rtol=2e-13, atol=0.0)
    current_ms, current_samples = cuda_ms(
        lambda: BDF.error_norm(
            state, predicted, rtol=RTOL_MICRO, atol=ATOL_MICRO))
    legacy_ms, legacy_samples = cuda_ms(legacy_error_norm)

    scheme = BDF(5)
    scheme.start([value.copy() for value in state[:11]])
    for step in range(1, 6):
        scheme.prepare_step(step, 0.01)
        scheme.push([(1.0 + 1e-4 * step) * value for value in state[:11]])

    def predictor_ms(fused):
        previous = kernels.set_fused_group("adaptive", fused)
        try:
            return cuda_ms(lambda: scheme.predict(0.013), repeats=30)[0]
        finally:
            kernels.set_fused_group("adaptive", previous)

    predictor_generic_ms = predictor_ms(False)
    predictor_fused_ms = predictor_ms(True)
    microbenchmark = {
        "error_norm_legacy_ms": legacy_ms,
        "error_norm_current_ms": current_ms,
        "error_norm_speedup": legacy_ms / current_ms,
        "error_norm_host_syncs_legacy": 34,
        "error_norm_host_syncs_current": 1,
        "predictor_generic_ms": predictor_generic_ms,
        "predictor_fused_ms": predictor_fused_ms,
        "predictor_speedup": predictor_generic_ms / predictor_fused_ms,
    }
    print(pd.Series(microbenchmark).to_string())
    del state, predicted, scheme
    cp.get_default_memory_pool().free_all_blocks()
    """),
    cell("markdown", r"""
    ## 2. Build one motion-resolving 3-D HP-MR trajectory

    The control callback interpolates pre-rasterized polar absorber volume fractions. Every method sees the identical continuous-in-time geometry. Interior control knots are supplied as BDF history restarts because the piecewise-linear trajectory changes slope there; thermal exchanges occur at the same endpoints.

    `REFERENCE_STEPS` controls the fixed backward-Euler reference. For a publication run, repeat with 2× and 4× as many steps and show that the reported candidate errors have stabilized.
    """),
    cell("code", r"""
    QUICK = True
    REFINE = 2 if QUICK else 4
    NZ = 10 if QUICK else 20          # build_hpmr3d requires a multiple of 10
    DURATION = 0.08 if QUICK else 0.40
    CONTROL_INTERVALS = 4 if QUICK else 8
    REFERENCE_STEPS = 32 if QUICK else 128
    INITIAL_STEPS = 8
    ADAPTIVE_RTOL = 1e-3
    DRUM_FROM, DRUM_TO = 90.0, 94.0
    POLAR_SAMPLES = 10
    FGMRES_RESTART = 10               # bounded 3-D basis memory
    POWER_ERROR_LIMIT = 2e-3
    FLUX_ERROR_LIMIT = 2e-3
    TEMPERATURE_ERROR_LIMIT_K = 2e-2

    materials = hpmr_endfb8_builtin(three_d=True)
    problem = build_hpmr3d(
        refine=REFINE, nz=NZ, drum_angle_deg=DRUM_FROM,
        absorber="polar", materials=materials, samples=POLAR_SAMPLES)

    frames = []
    for angle in np.linspace(DRUM_FROM, DRUM_TO, CONTROL_INTERVALS + 1):
        frame = build_hpmr3d(
            refine=REFINE, nz=NZ, drum_angle_deg=float(angle),
            absorber="polar", materials=materials, samples=POLAR_SAMPLES)
        frames.append((frame.mix_material, frame.mix_weight))
    assert all(np.array_equal(frames[0][0], frame[0]) for frame in frames[1:])
    assert any(np.any(frame[1] != frames[0][1]) for frame in frames[1:]), (
        "Drum motion is unresolved; increase refinement or polar samples.")

    def problem_at(t):
        position = CONTROL_INTERVALS * np.clip(float(t) / DURATION, 0.0, 1.0)
        lo = min(int(np.floor(position)), CONTROL_INTERVALS)
        hi = min(lo + 1, CONTROL_INTERVALS)
        fraction = position - lo
        weight = (1.0 - fraction) * frames[lo][1] + fraction * frames[hi][1]
        return (problem.materials, problem.material_map, frames[0][0], weight)

    ctx = build_hpmr_coupling(problem, device="gpu")
    active_cells = int(np.count_nonzero(problem.active))
    full_cells = int(np.prod(problem.grid.shape))
    groups = problem.materials[0].n_groups
    dof = active_cells * groups
    basis_estimate = (2 * FGMRES_RESTART + 4) * groups * full_cells * 8
    scatter_estimate = groups * groups * full_cells * 8
    print(f"grid={problem.grid.shape}; active cells={active_cells:,}; flux DOF={dof:,}")
    print(f"FGMRES storage estimate={basis_estimate/2**20:.1f} MiB; "
          f"dense group batch={scatter_estimate/2**20:.1f} MiB")
    print(f"trajectory: {DRUM_FROM:g}° -> {DRUM_TO:g}° in {DURATION:g} s")
    """),
    cell("markdown", r"""
    ## 3. Compute startup once, then warm the transient kernels

    The coupled hot equilibrium is a physical initial condition, not part of the time integrator comparison. It is computed once and passed to every leg through `initial_coupled_steady`. `result.seconds` and the CUDA-event phase profile cover the transient, while `hot.seconds` reports startup separately.
    """),
    cell("code", r"""
    hot = CoupledSolver(ctx).solve(tol=1e-8, anderson_depth=5)
    assert hot.converged
    print(hot)

    knots = DURATION * np.arange(1, CONTROL_INTERVALS) / CONTROL_INTERVALS
    common_multigroup = {
        # The T4 quick gate rejected tolerance-based energy Anderson but found
        # this reduction-free path 1.47x faster than plain group PCG.
        "scatter_sweeps": 4,
        "energy_anderson": 1,
        "inner_fixed_relaxations": 1,
        "rtol": 1e-9,
        "inner_rtol": 0.1,
        "precond_degree": 1,
        "restart": FGMRES_RESTART,
        "maxiter": 300,
    }

    def run_leg(label, scheme, *, adaptive, reference_steps=REFERENCE_STEPS,
                duration=DURATION, energy_anderson=None,
                multigroup_overrides=None):
        multigroup = dict(common_multigroup)
        if energy_anderson is not None:
            multigroup["energy_anderson"] = int(energy_anderson)
        if multigroup_overrides:
            multigroup.update(multigroup_overrides)
        transient = {
            "time_scheme": scheme,
            "step_solver": "monolithic",
            "tol_step": 1e-8,
            "multigroup_kwargs": multigroup,
        }
        if adaptive:
            transient["bdf_restart_times"] = knots[knots < duration]
            transient["adaptive_bdf"] = {
                "rtol": ADAPTIVE_RTOL,
                "min_dt": DURATION / REFERENCE_STEPS / 64,
                "max_dt": DURATION / CONTROL_INTERVALS,
                "automatic_order": scheme != "bdf1",
                "rejection_strategy": "error",
                "reject_max_factor": 0.5,
            }
        cp.cuda.get_current_stream().synchronize()
        wall0 = time.perf_counter()
        result = coupled_transient(
            ctx, t_end=duration,
            dt=(DURATION / INITIAL_STEPS if adaptive
                else duration / reference_steps),
            dt_thermal=min(DURATION / CONTROL_INTERVALS, duration),
            problem_at=problem_at, initial_coupled_steady=hot,
            transient_kwargs=transient, profile=True)
        wall = time.perf_counter() - wall0
        print(f"{label:20s}: march={result.seconds:9.3f} s  wall={wall:9.3f} s  "
              f"steps={result.steps:4d}+{result.rejected_steps:<3d}  "
              f"inner={result.counters['neutron_inner_iterations']:,}")
        return result, wall

    # Compile the monolithic, batch, Arnoldi, adaptive AXPY, and coupling kernels
    # before any timed comparison. This result is intentionally discarded.
    warm, _ = run_leg(
        "warm-up", "bdf1", adaptive=False, reference_steps=1,
        duration=DURATION / CONTROL_INTERVALS)
    assert warm.counters["group_batch_active"] == 1
    del warm
    cp.get_default_memory_pool().free_all_blocks()

    # Warm the separate tolerance-based PCG path before its short A/B gate.
    warm_tolerance, _ = run_leg(
        "tolerance warm-up", "bdf1", adaptive=False, reference_steps=1,
        duration=DURATION / CONTROL_INTERVALS,
        multigroup_overrides={"scatter_sweeps": 3,
                              "energy_anderson": 0,
                              "inner_fixed_relaxations": 0})
    del warm_tolerance
    cp.get_default_memory_pool().free_all_blocks()

    GPU_GATE_REPEATS = 3
    gate_configs = [
        ("plain", "plain tolerance",
         {"scatter_sweeps": 3, "energy_anderson": 0,
          "inner_fixed_relaxations": 0}),
        ("anderson", "energy Anderson",
         {"scatter_sweeps": 3, "energy_anderson": 1,
          "inner_fixed_relaxations": 0}),
        ("fixed", "fixed polynomial",
         {"scatter_sweeps": 4, "energy_anderson": 1,
          "inner_fixed_relaxations": 1}),
    ]
    gate_samples = {key: [] for key, _, _ in gate_configs}
    gate_work = {}
    for repeat in range(GPU_GATE_REPEATS):
        ordered = gate_configs if repeat % 2 == 0 else gate_configs[::-1]
        for key, label, overrides in ordered:
            result, _ = run_leg(
                f"{label} {repeat + 1}", "bdf1", adaptive=False,
                reference_steps=1, duration=DURATION / CONTROL_INTERVALS,
                multigroup_overrides=overrides)
            gate_samples[key].append(result.seconds)
            work = (result.counters["neutron_total_outer_iterations"],
                    result.counters["neutron_inner_iterations"])
            if key in gate_work:
                assert gate_work[key] == work
            gate_work[key] = work
            del result
            cp.get_default_memory_pool().free_all_blocks()

    gate_medians = {
        key: float(np.median(samples))
        for key, samples in gate_samples.items()
    }
    energy_sweep_gpu_gate = {
        "repeats": GPU_GATE_REPEATS,
        "baseline_samples": gate_samples["plain"],
        "anderson_samples": gate_samples["anderson"],
        "baseline_seconds": gate_medians["plain"],
        "anderson_seconds": gate_medians["anderson"],
        "speedup": gate_medians["plain"] / gate_medians["anderson"],
        "baseline_outer": gate_work["plain"][0],
        "anderson_outer": gate_work["anderson"][0],
        "baseline_inner": gate_work["plain"][1],
        "anderson_inner": gate_work["anderson"][1],
    }
    print("Energy-sweep GPU A/B:", energy_sweep_gpu_gate)
    reduction_free_gpu_gate = {
        "repeats": GPU_GATE_REPEATS,
        "fixed_polynomial_samples": gate_samples["fixed"],
        "plain_tolerance_seconds": gate_medians["plain"],
        "tolerance_seconds": gate_medians["anderson"],
        "fixed_polynomial_seconds": gate_medians["fixed"],
        "speedup_vs_plain": gate_medians["plain"] / gate_medians["fixed"],
        "speedup_vs_anderson": (gate_medians["anderson"]
                                / gate_medians["fixed"]),
        "tolerance_outer": gate_work["anderson"][0],
        "fixed_polynomial_outer": gate_work["fixed"][0],
        "tolerance_inner": gate_work["anderson"][1],
        "fixed_polynomial_inner": gate_work["fixed"][1],
    }
    print("Reduction-free GPU A/B:", reduction_free_gpu_gate)
    """),
    cell("markdown", r"""
    ## 4. Fine BE reference, adaptive BE, and adaptive BDF5

    The relevant method comparison is adaptive BDF5 versus adaptive BDF1 at the same local tolerance. The fine fixed-step BE leg is an independent accuracy reference, not the denominator for claiming an algorithmic speedup. This distinction is important on HP-MR: mild drum ramps often keep the controller at order one or two, unlike LRA's sharp prompt peaks.
    """),
    cell("code", r"""
    reference, reference_wall = run_leg(
        "fine fixed BE", "bdf1", adaptive=False)
    adaptive_be, adaptive_be_wall = run_leg(
        "adaptive BE", "bdf1", adaptive=True)
    adaptive_bdf, adaptive_bdf_wall = run_leg(
        "adaptive BDF5", "bdf5", adaptive=True)

    assert reference.counters["coupled_steady_reuses"] == 1
    assert adaptive_bdf.counters["group_batch_active"] == 1
    assert adaptive_bdf.counters["fgmres_workspace_bytes"] > 0
    assert max(adaptive_bdf.local_errors, default=0.0) <= 1.0 + 5e-12
    assert np.all(np.isfinite(adaptive_bdf.power))
    """),
    cell("code", r"""
    thermal_times = np.linspace(0.0, DURATION, CONTROL_INTERVALS + 1)

    def values_at(result, values, targets):
        indices = []
        for target in targets:
            j = int(np.argmin(np.abs(result.times - target)))
            assert np.isclose(result.times[j], target, rtol=1e-12,
                              atol=1e-14 * max(1.0, DURATION))
            indices.append(j)
        return np.asarray(values)[indices]

    def normalized_flux(result):
        values = cp.asnumpy(result.flux)[:, np.asarray(problem.active, bool)].ravel()
        return values / np.linalg.norm(values)

    reference_flux = normalized_flux(reference)

    def summarize(name, result, wall):
        power_on_ref = np.interp(reference.times, result.times, result.power)
        power_error = float(np.max(
            np.abs(power_on_ref - reference.power)
            / np.maximum(np.abs(reference.power), 1e-14)))
        mean_error = float(np.max(np.abs(
            values_at(result, result.mean_temperature, thermal_times)
            - values_at(reference, reference.mean_temperature, thermal_times))))
        peak_error = float(np.max(np.abs(
            values_at(result, result.peak_temperature, thermal_times)
            - values_at(reference, reference.peak_temperature, thermal_times))))
        flux_error = float(np.linalg.norm(normalized_flux(result) - reference_flux))
        return {
            "method": name,
            "accepted": result.steps,
            "rejected": result.rejected_steps,
            "max_order": max(result.time_orders, default=1),
            "order_counts": dict(Counter(result.time_orders)),
            "march_seconds": result.seconds,
            "end_to_end_seconds": wall,
            "fine_be_speedup": reference.seconds / result.seconds,
            "inner_iterations": result.counters["neutron_inner_iterations"],
            "outer_iterations_including_rejects": result.counters[
                "neutron_total_outer_iterations"],
            "max_power_relative_error": power_error,
            "max_mean_temperature_error_k": mean_error,
            "max_peak_temperature_error_k": peak_error,
            "final_flux_shape_l2_error": flux_error,
            "group_batch_active": bool(result.counters["group_batch_active"]),
            "fgmres_workspace_mib": result.counters[
                "fgmres_workspace_bytes"] / 2**20,
        }

    rows = [
        summarize("fine fixed BE", reference, reference_wall),
        summarize("adaptive BE", adaptive_be, adaptive_be_wall),
        summarize("adaptive BDF5", adaptive_bdf, adaptive_bdf_wall),
    ]
    table = pd.DataFrame(rows)
    display(table.drop(columns="order_counts"))
    print("\nOrder occupancy")
    for row in rows:
        print(f"  {row['method']:20s}: {row['order_counts']}")
    matched_speedup = adaptive_be.seconds / adaptive_bdf.seconds
    print(f"\nAdaptive BDF5 speedup versus accuracy-matched adaptive BE: "
          f"{matched_speedup:.3f}x")

    for row in rows[1:]:
        assert row["max_power_relative_error"] < POWER_ERROR_LIMIT
        assert row["final_flux_shape_l2_error"] < FLUX_ERROR_LIMIT
        assert row["max_mean_temperature_error_k"] < TEMPERATURE_ERROR_LIMIT_K
        assert row["max_peak_temperature_error_k"] < TEMPERATURE_ERROR_LIMIT_K
    print("PASS: both adaptive methods satisfy the configured 3-D reference gates.")
    """),
    cell("code", r"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for label, result in [
        ("fine fixed BE", reference), ("adaptive BE", adaptive_be),
        ("adaptive BDF5", adaptive_bdf)]:
        axes[0, 0].plot(result.times, result.power, marker=".", label=label)
    axes[0, 0].set(xlabel="time [s]", ylabel="P/P0", title="Power history")
    axes[0, 0].legend()

    axes[0, 1].step(adaptive_bdf.times[1:], adaptive_bdf.step_widths,
                    where="post")
    axes[0, 1].set(xlabel="accepted time [s]", ylabel="step width [s]",
                   title="Adaptive BDF step sizes")

    axes[1, 0].step(adaptive_bdf.times[1:], adaptive_bdf.time_orders,
                    where="post")
    axes[1, 0].set(xlabel="accepted time [s]", ylabel="BDF order",
                   title="Accepted order")
    axes[1, 0].set_yticks(range(1, 6))

    axes[1, 1].semilogy(adaptive_bdf.times[1:], adaptive_bdf.local_errors,
                        marker=".")
    axes[1, 1].axhline(1.0, color="k", linestyle="--", linewidth=1)
    axes[1, 1].set(xlabel="accepted time [s]", ylabel="normalized defect",
                   title="Accepted local-error estimates")
    fig.savefig("hpmr3d_adaptive_bdf_gpu.png", dpi=160)
    plt.show()
    """),
    cell("markdown", r"""
    ## 5. Phase attribution and machine-readable artifact

    Whole-run speed without work counters is ambiguous. The cells below show whether a difference came from time integration, Krylov work, geometry/operator rebuilds, or transfers. `telemetry_transfers` should scale with thermal windows, not adaptive neutron steps.
    """),
    cell("code", r"""
    phases = pd.DataFrame({
        "adaptive BE": adaptive_be.phase_seconds,
        "adaptive BDF5": adaptive_bdf.phase_seconds,
    }).fillna(0.0)
    display(phases.sort_values("adaptive BDF5", ascending=False))

    important = [
        "neutronics_steps", "neutron_rejected_steps",
        "neutron_total_outer_iterations", "neutron_inner_iterations",
        "thermal_steps", "telemetry_transfers", "operator_rebuilds",
        "group_batch_active", "fgmres_workspace_bytes",
    ]
    counters = pd.DataFrame({
        "adaptive BE": {k: adaptive_be.counters.get(k, 0) for k in important},
        "adaptive BDF5": {k: adaptive_bdf.counters.get(k, 0) for k in important},
    })
    display(counters)

    payload = {
        "environment": {
            "gpu": GPU_NAME,
            "cupy": cp.__version__,
            "python": platform.python_version(),
            "ndgpu_path": inspect.getfile(ndgpu),
        },
        "configuration": {
            "quick": QUICK, "refine": REFINE, "nz": NZ,
            "active_cells": active_cells, "groups": groups, "dof": dof,
            "duration_s": DURATION, "control_intervals": CONTROL_INTERVALS,
            "reference_steps": REFERENCE_STEPS,
            "adaptive_initial_steps": INITIAL_STEPS,
            "adaptive_rtol": ADAPTIVE_RTOL,
            "fgmres_restart": FGMRES_RESTART,
            "energy_anderson": common_multigroup["energy_anderson"],
            "scatter_sweeps": common_multigroup["scatter_sweeps"],
            "inner_rtol": common_multigroup["inner_rtol"],
            "inner_fixed_relaxations": common_multigroup[
                "inner_fixed_relaxations"],
            "drum_degrees": [DRUM_FROM, DRUM_TO],
            "coupled_steady_seconds": hot.seconds,
        },
        "gpu_microbenchmark": microbenchmark,
        "energy_sweep_gpu_gate": energy_sweep_gpu_gate,
        "reduction_free_gpu_gate": reduction_free_gpu_gate,
        "matched_adaptive_speedup": matched_speedup,
        "results": rows,
        "phases": {
            "adaptive_be": adaptive_be.phase_seconds,
            "adaptive_bdf5": adaptive_bdf.phase_seconds,
        },
        "counters": {
            "adaptive_be": adaptive_be.counters,
            "adaptive_bdf5": adaptive_bdf.counters,
        },
    }
    with open("hpmr3d_adaptive_bdf_gpu.json", "w") as stream:
        json.dump(payload, stream, indent=2)
    table.to_csv("hpmr3d_adaptive_bdf_gpu.csv", index=False)
    print("Wrote hpmr3d_adaptive_bdf_gpu.json/.csv/.png")

    try:
        from google.colab import files
        # Uncomment to download automatically.
        # files.download("hpmr3d_adaptive_bdf_gpu.json")
        # files.download("hpmr3d_adaptive_bdf_gpu.csv")
        # files.download("hpmr3d_adaptive_bdf_gpu.png")
    except ImportError:
        pass
    """),
    cell("markdown", r"""
    ## Interpreting the result

    - A speedup above 1 in `matched_adaptive_speedup` is the defensible adaptive-BDF versus backward-Euler comparison. `fine_be_speedup` measures savings relative to an accuracy reference and must not be presented as a same-tolerance method advantage.
    - If BDF5 is mostly order 1–2, a result near or below 1× is expected: it performs candidate-order error estimates without exploiting sustained high order. This is the mechanism previously observed on the mild 2-D HP-MR ramp and is different from LRA's high-order prompt transient.
    - If error-norm microbenchmark speedup is small but its values agree, retain the one-transfer implementation for scaling: automatic order selection can call it three times per accepted step, and synchronization cost grows much more visibly when the neutron solve itself becomes faster.
    - `group_batch_active=False` means free-VRAM protection selected the sparse fallback. Reduce `FGMRES_RESTART`, close other GPU sessions, or use a larger GPU before interpreting wall time.
    - The persistent workspace is deliberately bounded by `FGMRES_RESTART`. Raise it only if outer iteration histories show frequent restarts; the reported MiB figure makes the cost explicit.
    - `energy_sweep_gpu_gate` isolates the depth-one energy-sweep Anderson update. The three samples are interleaved with the other configurations and the reported seconds are medians; outer and inner counters must remain deterministic.
    - `reduction_free_gpu_gate` compares tolerance-based group PCG with four fixed polynomial sweeps. The fixed path is slower on CPU but eliminates every inner convergence synchronization; interpret its median together with the full-run phase and work counters.
    - Before quoting an accuracy or performance number, rerun with `REFERENCE_STEPS` doubled, then repeat the timed legs at least three times and report medians. Colab GPU type and load vary between sessions.
    """),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "colab_hpmr3d_adaptive_bdf_gpu.ipynb",
                  "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).resolve().parents[1] / "notebooks" / \
    "colab_hpmr3d_adaptive_bdf_gpu.ipynb"
target.write_text(json.dumps(notebook, indent=1) + "\n")
print(target)
