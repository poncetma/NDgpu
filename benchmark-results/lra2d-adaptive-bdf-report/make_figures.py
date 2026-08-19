"""Generate the figures used by the adaptive-BDF LRA report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def load(name):
    return json.loads((DATA / name).read_text())


summary = json.loads((ROOT / "benchmark-summary.json").read_text())
tight = load("bdf5-rtol-1e-6.json")
practical = load("bdf5-rtol-1e-3.json")
backward = load("backward-euler-rtol-1e-4.json")
cell_tight = load("bdf5-cell-refine4-rtol-1e-6.json")
cell_matched = load("bdf5-cell-refine4-rtol-1e-3.json")
cell_backward = load("backward-euler-cell-refine4-rtol-1e-4.json")
cell_spatial = [load(f"bdf5-cell-refine{refine}-rtol-1e-5.json")
                for refine in (1, 2, 4, 8, 15)]
cell_fine = cell_spatial[-1]


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


# LRA geometry, reconstructed from the public benchmark problem.
from ndgpu.benchmarks import build_lra2d  # noqa: E402

problem = build_lra2d(refine=1)
regions = problem.material_map[:, :, 0].copy()
# The dedicated transient-control slot is displayed as R rather than region 6.
display = regions.copy()
display[problem.control_mask[:, :, 0]] = 5
fig, ax = plt.subplots(figsize=(5.4, 4.5))
cmap = ListedColormap(["#4477aa", "#66ccee", "#228833", "#ccbb44",
                       "#bbbbbb", "#ee6677"])
ax.imshow(display.T, origin="lower", cmap=cmap, vmin=0, vmax=5,
          extent=(0, 165, 0, 165), interpolation="nearest")
for edge in np.arange(0, 166, 15):
    ax.axhline(edge, color="white", lw=0.45, alpha=0.8)
    ax.axvline(edge, color="white", lw=0.45, alpha=0.8)
for i in range(11):
    for j in range(11):
        value = "R" if problem.control_mask[i, j, 0] else str(regions[i, j] + 1)
        ax.text((i + 0.5) * 15, (j + 0.5) * 15, value,
                ha="center", va="center", fontsize=6,
                color="white" if regions[i, j] in (0, 2) else "black")
ax.set(xlabel="x (cm)", ylabel="y (cm)",
       title="LRA quarter-core material map (R: withdrawing control region)")
ax.set_aspect("equal")
save(fig, "lra_geometry")


# Complete power histories and the original ANL point references.
fig, ax = plt.subplots(figsize=(7.0, 4.1))
for payload, label, color, lw in (
    (tight, r"automatic BDF5, $10^{-6}$", "#0077bb", 1.8),
    (practical, r"automatic BDF5, $10^{-3}$", "#ee7733", 1.5),
    (backward, r"backward Euler, $10^{-4}$", "#009988", 1.2),
    (cell_fine, r"cell-local FV, 1 cm, $10^{-5}$", "#aa4499", 1.5),
):
    history = payload["adaptive_history"]
    times = np.asarray(history["times_s"])
    # The solver's relative power is converted using P0=1e-6 W/cm3.
    # Reconstruct physical power by interpolating the stored aggregate anchors
    # is unnecessary: TransientResult power is not emitted in history JSON.
    # Instead use accepted temperature-independent metric interpolation only
    # for the visual curve by re-scaling the normalized local error? No: the
    # benchmark JSON intentionally lacks the full power vector, so require it.
    powers = np.asarray(history["power_w_cm3"])
    ax.plot(times, powers, label=label, color=color, lw=lw)
ref = summary["anl_reference"]
ax.scatter([ref["first_peak_time_s"], 2.0, 3.0],
           [ref["first_peak_power_w_cm3"],
            ref["second_peak_power_w_cm3"], ref["power_at_3s_w_cm3"]],
           marker="D", s=35, color="black", zorder=5,
           label="ANL-7416/2 reference points")
ax.axvline(2.0, ls="--", lw=0.8, color="0.35", label="rod withdrawal ends")
ax.set(xlim=(0, 3), xlabel="time (s)", ylabel=r"core-average power (W cm$^{-3}$)",
       title="LRA transient: temporal solutions and original benchmark points")
ax.legend(loc="upper right")
save(fig, "power_history")


# Adaptive controller behavior for the practical BDF case.
h = practical["adaptive_history"]
times = np.asarray(h["times_s"])[1:]
widths = np.asarray(h["widths_s"])
orders = np.asarray(h["orders"])
errors = np.asarray(h["accepted_errors"])
fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.1), sharex=True)
axes[0].semilogy(times, widths, color="#0077bb", lw=1.1)
axes[0].set_ylabel("accepted h (s)")
axes[1].step(times, orders, where="post", color="#ee7733", lw=1.1)
axes[1].set(ylabel="BDF order q", yticks=range(1, 6), ylim=(0.7, 5.3))
axes[2].semilogy(times, np.maximum(errors, 1e-12), color="#009988", lw=1.0)
axes[2].axhline(1.0, color="black", ls="--", lw=0.8)
axes[2].set(xlabel="time (s)", ylabel="accepted defect")
for ax in axes:
    ax.axvline(2.0, ls=":", lw=0.9, color="0.25")
    ax.axvline(practical["ndgpu"]["first_peak_time_s"], ls=":", lw=0.9,
               color="#cc3311")
axes[0].set_title("Automatic BDF response near the prompt peak and rod-stop event")
save(fig, "controller_history")


# Tolerance convergence: peak error versus portable work.
ladder = summary["tolerance_ladder"]
reference = ladder[-1]
rtols = np.asarray([row["rtol"] for row in ladder])
peak_errors = 100 * np.abs(np.asarray([
    row["first_peak_power_w_cm3"] / reference["first_peak_power_w_cm3"] - 1
    for row in ladder]))
inner = np.asarray([row["inner"] for row in ladder])
fig, ax = plt.subplots(figsize=(5.7, 4.1))
ax.plot(inner[:-1], peak_errors[:-1], "o-", color="#0077bb")
for row, x, y in zip(ladder[:-1], inner[:-1], peak_errors[:-1]):
    ax.annotate(f"RTOL={row['rtol']:.0e}", (x, y),
                xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.axvline(inner[-1], color="0.35", ls="--", lw=1,
           label=r"RTOL=$10^{-6}$ temporal reference work")
ax.set_yscale("log")
ax.set(xlabel="total inner iterations", ylabel=r"$|P_1-P_{1,10^{-6}}|/P_{1,10^{-6}}$ (\%)",
       title="Temporal convergence versus solver work")
ax.legend(loc="lower left")
save(fig, "tolerance_tradeoff")


# Accuracy against ANL, not against FEMCORE.
keys = ["rods_in", "rods_out", "first_peak_time_s",
        "first_peak_power_w_cm3", "second_peak_power_w_cm3",
        "power_at_3s_w_cm3", "mean_temperature_at_3s_k",
        "peak_temperature_at_3s_k"]
labels = [r"$k_{in}$", r"$k_{out}$", r"$t_1$", r"$P_1$", r"$P_2$",
          r"$P(3s)$", r"$\bar T(3s)$", r"$T_{max}(3s)$"]
anl = summary["anl_reference"]

def complete_metrics(payload):
    result = dict(payload["ndgpu"])
    result.update(summary["static_ndgpu"])
    return result

series = [
    ("ndgpu assembly T, tight", complete_metrics(tight), "#0077bb"),
    ("ndgpu cell-local T, tight", complete_metrics(cell_tight), "#ee7733"),
    ("FEMCORE", summary["cherezov_femcore"], "#aa4499"),
]
x = np.arange(len(keys))
width = 0.25
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for offset, (name, values, color) in zip((-width, 0, width), series):
    error = [100 * (values[key] / anl[key] - 1) for key in keys]
    ax.bar(x + offset, error, width, label=name, color=color)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, labels, rotation=25, ha="right")
ax.set(ylabel="relative difference from ANL reference (%)",
       title="Accuracy is assessed against ANL-7416/2 reference data")
ax.legend(ncols=3, loc="upper left")
save(fig, "anl_accuracy")


# Spatial convergence of the benchmark-faithful cell-local feedback model.
cell_widths = np.asarray([15.0 / row["case"]["refine"]
                          for row in cell_spatial])
spatial_metrics = [row["ndgpu"] for row in cell_spatial]
fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.25), sharex=True)
for key, label, color, marker in (
    ("first_peak_time_s", r"$t_1$", "#0077bb", "o"),
    ("first_peak_power_w_cm3", r"$P_1$", "#ee7733", "s"),
    ("second_peak_power_w_cm3", r"$P_2$", "#009988", "^"),
    ("power_at_3s_w_cm3", r"$P(3s)$", "#aa4499", "D"),
):
    error = [100 * (row[key] / anl[key] - 1) for row in spatial_metrics]
    axes[0].plot(cell_widths, error, marker=marker, label=label,
                 color=color, lw=1.1, ms=4)
for key, label, color, marker in (
    ("mean_temperature_at_3s_k", r"$\bar T(3s)$", "#0077bb", "o"),
    ("peak_temperature_at_3s_k", r"assembly $T_{max}(3s)$", "#cc3311", "s"),
):
    error = [100 * (row[key] / anl[key] - 1) for row in spatial_metrics]
    axes[1].plot(cell_widths, error, marker=marker, label=label,
                 color=color, lw=1.1, ms=4)
for key, label, color, marker in (
    ("rods_in", r"$k_{in}$", "#0077bb", "o"),
    ("rods_out", r"$k_{out}$", "#ee7733", "s"),
):
    error = [1e5 * (row[key] - anl[key])
             for row in summary["static_convergence"]]
    axes[2].plot(cell_widths, error, marker=marker, label=label,
                 color=color, lw=1.1, ms=4)
for ax in axes:
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(cell_widths, [f"{value:g}" for value in cell_widths])
    ax.set_xlabel("FV cell width (cm)")
    ax.legend()
axes[0].set(ylabel="difference from ANL (%)", title="Power and timing")
axes[1].set(ylabel="difference from ANL (%)", title="Temperature")
axes[2].set(ylabel="difference from ANL (pcm)", title="Static endpoints")
fig.suptitle("Spatial convergence with cell-local heat and Doppler feedback",
             y=1.02)
save(fig, "spatial_convergence")


# Matched-error normalized cost on the benchmark-faithful cell-local model.
bdf = cell_matched["ndgpu"]
be = cell_backward["ndgpu"]
names = ["accepted steps", "FGMRES applications", "inner iterations", "wall time"]
ratios = [be["steps"] / bdf["steps"],
          be["fgmres_applications"] / bdf["fgmres_applications"],
          be["inner_iterations"] / bdf["inner_iterations"],
          be["solve_seconds"] / bdf["solve_seconds"]]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
bars = ax.bar(names, ratios, color=["#cc3311", "#ee7733", "#009988", "#0077bb"])
ax.bar_label(bars, fmt="%.1fx", padding=3)
ax.set(ylabel="backward Euler / automatic BDF5",
       title="Cost at comparable first-peak temporal error", ylim=(0, max(ratios) * 1.18))
ax.tick_params(axis="x", rotation=18)
save(fig, "matched_cost")


# Published FEMCORE CPU ladder and ndgpu measurements.  The paper does not
# identify its CPU, so this is a raw-time comparison rather than a speedup.
fem = summary["cherezov_performance"]["fem_rows"]
fig, ax = plt.subplots(figsize=(5.9, 4.0))
ax.loglog([row["dof"] for row in fem], [row["seconds"] for row in fem],
          "o-", color="#aa4499", label="FEMCORE (published)")
ndgpu_dof = [44**2, 165**2]
ndgpu_seconds = [cell_spatial[2]["ndgpu"]["solve_seconds"],
                 cell_spatial[-1]["ndgpu"]["solve_seconds"]]
ax.loglog(ndgpu_dof, ndgpu_seconds, "s", color="#0077bb", ms=6,
          label="ndgpu (i5-1145G7)")
for x_value, y_value, label in zip(
        ndgpu_dof, ndgpu_seconds, ("3.75 cm", "1 cm")):
    ax.annotate(label, (x_value, y_value), xytext=(5, -11),
                textcoords="offset points", fontsize=8)
ax.set(xlabel="scalar spatial degrees of freedom / FV cells",
       ylabel="reported transient CPU time (s)",
       title="Raw CPU time on different, non-normalized hosts")
ax.legend()
save(fig, "cherezov_cpu")
