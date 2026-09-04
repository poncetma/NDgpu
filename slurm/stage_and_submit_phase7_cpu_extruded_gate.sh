#!/usr/bin/env bash

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/NDgpu"
stage_root="/data/scratch/shared/poncet_m/ndgpu-multigpu-phase7-cpu"
step_solver="${1:-fixed-point}"
if [[ "${step_solver}" != "fixed-point" && "${step_solver}" != "monolithic" ]]; then
    echo "usage: $0 [fixed-point|monolithic]" >&2
    exit 2
fi

mkdir -p "${stage_root}/repo" "${stage_root}/logs"
rsync -a --delete --exclude '.git' --exclude '.agents' --exclude '.codex' \
    "${src_root}/" "${stage_root}/repo/"
export NDGPU_REPO="${stage_root}/repo"
sbatch --clusters=merlin7 --export="ALL,NDGPU_STEP_SOLVER=${step_solver}" \
    "${stage_root}/repo/slurm/run_ndgpu_phase7_cpu_extruded_gate.sh"
