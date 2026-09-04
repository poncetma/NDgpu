#!/usr/bin/env bash

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/NDgpu"
stage_root="/data/scratch/shared/poncet_m/ndgpu-multigpu-phase7-performance"
ranks="${1:-2}"
precond_degree="${2:-1}"
refine="${3:-8}"
local_levels="${4:-3}"
steps="${5:-5}"
nz="${6:-10}"
scatter_subsweeps="${7:-0}"
step_solver="${8:-fixed-point}"
partition="${9:-gh-hourly}"
communication="${10:-cuda-aware}"
multigroup_scatter_sweeps="${11:-3}"
multigroup_energy_anderson="${12:-0}"
multigroup_fixed_relaxations="${13:-0}"
multigroup_fixed_iterations="${14:-0}"
multigroup_inner_rtol="${15:-1e-3}"
drum_motion="${16:-step}"
if [[ "${ranks}" != "1" && "${ranks}" != "2" && "${ranks}" != "4" ]]; then
    echo "usage: $0 [1|2|4] [preconditioner-degree] [refine] [local-levels] [steps] [nz] [scatter-subsweeps] [fixed-point|monolithic] [gh-hourly|a100-hourly] [cuda-aware|host-staged] [energy-sweeps] [anderson-depth] [fixed-relaxations] [fixed-pcg-iterations] [inner-rtol] [step|linear-ramp]" >&2
    exit 2
fi
if ! [[ "${precond_degree}" =~ ^[0-9]+$ \
        && "${refine}" =~ ^[1-9][0-9]*$ \
        && "${local_levels}" =~ ^[0-9]+$ \
        && "${steps}" =~ ^[1-9][0-9]*$ \
        && "${nz}" =~ ^[1-9][0-9]*$ \
        && "${scatter_subsweeps}" =~ ^[0-9]+$ \
        && "${multigroup_scatter_sweeps}" =~ ^[1-9][0-9]*$ \
        && "${multigroup_energy_anderson}" =~ ^[0-2]$ \
        && "${multigroup_fixed_relaxations}" =~ ^[0-9]+$ \
        && "${multigroup_fixed_iterations}" =~ ^[0-9]+$ \
        && "${multigroup_inner_rtol}" =~ ^[0-9.eE+-]+$ ]]; then
    echo "degrees/levels must be non-negative; refine, steps, and nz must be positive" >&2
    exit 2
fi
if (( multigroup_fixed_relaxations && multigroup_fixed_iterations )); then
    echo "choose fixed inner relaxations or fixed PCG iterations, not both" >&2
    exit 2
fi
if (( ranks > nz )); then
    echo "rank count cannot exceed nz" >&2
    exit 2
fi
if [[ "${step_solver}" != "fixed-point" && "${step_solver}" != "monolithic" ]]; then
    echo "step solver must be fixed-point or monolithic" >&2
    exit 2
fi
if [[ "${partition}" != "gh-hourly" && "${partition}" != "a100-hourly" ]]; then
    echo "partition must be gh-hourly or a100-hourly" >&2
    exit 2
fi
if [[ "${communication}" != "cuda-aware" && "${communication}" != "host-staged" ]]; then
    echo "communication must be cuda-aware or host-staged" >&2
    exit 2
fi
if [[ "${drum_motion}" != "step" && "${drum_motion}" != "linear-ramp" ]]; then
    echo "drum motion must be step or linear-ramp" >&2
    exit 2
fi
time_limit="00:55:00"
if (( steps <= 5 )); then
    time_limit="00:04:00"
fi

mkdir -p "${stage_root}/repo" "${stage_root}/logs"
rsync -a --delete --exclude '.git' --exclude '.agents' --exclude '.codex' \
    "${src_root}/" "${stage_root}/repo/"
export NDGPU_REPO="${stage_root}/repo"
sbatch --clusters=gmerlin7 --partition="${partition}" --ntasks="${ranks}" \
    --time="${time_limit}" \
    --export="ALL,NDGPU_PRECOND_DEGREE=${precond_degree},NDGPU_REFINE=${refine},NDGPU_LOCAL_LEVELS=${local_levels},NDGPU_STEPS=${steps},NDGPU_NZ=${nz},NDGPU_SCATTER_SUBSWEEPS=${scatter_subsweeps},NDGPU_STEP_SOLVER=${step_solver},NDGPU_MPI_COMMUNICATION=${communication},NDGPU_MULTIGROUP_SCATTER_SWEEPS=${multigroup_scatter_sweeps},NDGPU_MULTIGROUP_ENERGY_ANDERSON=${multigroup_energy_anderson},NDGPU_MULTIGROUP_FIXED_RELAXATIONS=${multigroup_fixed_relaxations},NDGPU_MULTIGROUP_FIXED_ITERATIONS=${multigroup_fixed_iterations},NDGPU_MULTIGROUP_INNER_RTOL=${multigroup_inner_rtol},NDGPU_DRUM_MOTION=${drum_motion}" \
    "${stage_root}/repo/slurm/run_ndgpu_phase7_gh_extruded_performance.sh"
