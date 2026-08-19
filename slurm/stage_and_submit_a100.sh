#!/usr/bin/env bash
# Stage an independent NDgpu snapshot and submit the full speed benchmark to
# an x86_64 A100 node. Command-line sbatch options override the GH defaults in
# the shared GPU runner without mutating queued job snapshots.

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/NDgpu"
stage_root="${NDGPU_A100_STAGE:-/data/scratch/shared/poncet_m/ndgpu-a100-stage}"
logdir="${NDGPU_A100_LOG_DIR:-/data/scratch/shared/poncet_m/ndgpu-a100-logs}"

mkdir -p "$stage_root" "$logdir"
rsync -a --delete \
    --exclude '.git' \
    --exclude '.agents' \
    --exclude '.codex' \
    --exclude 'results' \
    "$src_root/" "$stage_root/repo/"

export NDGPU_REPO="$stage_root/repo"
export NDGPU_PYTHON_BIN="${NDGPU_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}"
export NDGPU_DEBUG_DIR="$logdir"

sbatch \
    --cluster=gmerlin7 \
    --partition=a100-hourly \
    --job-name=ndgpu-a100 \
    --chdir="$logdir" \
    --output="$logdir/ndgpu-a100-%j.out" \
    --error="$logdir/ndgpu-a100-%j.err" \
    "$stage_root/repo/slurm/run_ndgpu_gh.sh"
