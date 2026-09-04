#!/usr/bin/env bash

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/NDgpu"
stage_root="/data/scratch/shared/poncet_m/ndgpu-hpmr-drum-refinement"

mkdir -p "${stage_root}/repo" "${stage_root}/logs"
rsync -a --delete --exclude '.git' --exclude '.agents' --exclude '.codex' \
    "${src_root}/" "${stage_root}/repo/"

for mode in local global; do
    sbatch --clusters=gmerlin7 --ntasks=1 \
        --export="ALL,NDGPU_REPO=${stage_root}/repo,NDGPU_LOGDIR=${stage_root}/logs,NDGPU_REFINEMENT_MODE=${mode}" \
        "${stage_root}/repo/slurm/run_ndgpu_gh_drum_refinement.sh"
done
