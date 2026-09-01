#!/usr/bin/env bash

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/NDgpu"
stage_root="/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6"

mkdir -p "${stage_root}/repo" "${stage_root}/logs"
rsync -a --delete --exclude '.git' --exclude '.agents' --exclude '.codex' \
    "${src_root}/" "${stage_root}/repo/"
export NDGPU_REPO="${stage_root}/repo"

sbatch --clusters=gmerlin7 --ntasks=2 \
    --export=ALL,NDGPU_MPI_COMMUNICATION=cuda-aware,NDGPU_SINGLE_REDUCTION=1 \
    "${stage_root}/repo/slurm/run_ndgpu_phase6_gh_transient_performance.sh"
