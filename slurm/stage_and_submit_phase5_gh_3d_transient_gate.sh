#!/usr/bin/env bash

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/NDgpu"
stage_root="/data/scratch/shared/poncet_m/ndgpu-multigpu-phase5"

mkdir -p "${stage_root}/repo" "${stage_root}/logs"
rsync -a --delete --exclude '.git' --exclude '.agents' --exclude '.codex' \
    "${src_root}/" "${stage_root}/repo/"
export NDGPU_REPO="${stage_root}/repo"
if [[ "${1:-}" == "--skip-eigen" ]]; then
    sbatch --clusters=gmerlin7 --export=ALL,NDGPU_SKIP_EIGEN_GATE=1 \
        "${stage_root}/repo/slurm/run_ndgpu_phase5_gh_3d_transient_gate.sh"
elif [[ "${1:-}" == "--eigen-only" ]]; then
    sbatch --clusters=gmerlin7 --export=ALL,NDGPU_SKIP_TRANSIENT_GATE=1 \
        "${stage_root}/repo/slurm/run_ndgpu_phase5_gh_3d_transient_gate.sh"
else
    sbatch --clusters=gmerlin7 \
        "${stage_root}/repo/slurm/run_ndgpu_phase5_gh_3d_transient_gate.sh"
fi
