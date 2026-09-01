#!/usr/bin/env bash

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/NDgpu"
stage_root="/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6-batched"

mkdir -p "${stage_root}/repo" "${stage_root}/logs"
rsync -a --delete --exclude '.git' --exclude '.agents' --exclude '.codex' \
    "${src_root}/" "${stage_root}/repo/"
export NDGPU_REPO="${stage_root}/repo"
sbatch --clusters=gmerlin7 \
    --chdir="${stage_root}/logs" \
    --output="${stage_root}/logs/batched-halo-%j.log" \
    --error="${stage_root}/logs/batched-halo-%j.log" \
    --export=ALL,NDGPU_MPI_COMMUNICATION=cuda-aware,NDGPU_BATCHED_HALOS=1,NDGPU_PROBE_ELEMENTS=4096,NDGPU_PROBE_ITERATIONS=200 \
    "${stage_root}/repo/slurm/run_ndgpu_phase6_gh_communication_probe.sh"
