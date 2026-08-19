#!/usr/bin/env bash
# Stage an independent NDgpu snapshot in shared scratch and submit the complete
# GPU regression suite. The separate stage avoids mutating queued benchmarks.

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/NDgpu"
stage_root="${NDGPU_REGRESSION_STAGE:-/data/scratch/shared/poncet_m/ndgpu-regression-stage}"
logdir="${NDGPU_REGRESSION_LOG_DIR:-/data/scratch/shared/poncet_m/ndgpu-regression-logs}"

mkdir -p "$stage_root" "$logdir"
rsync -a --delete \
    --exclude '.git' \
    --exclude '.agents' \
    --exclude '.codex' \
    --exclude 'results' \
    "$src_root/" "$stage_root/repo/"

export NDGPU_REPO="$stage_root/repo"
export NDGPU_PYTHON_BIN="${NDGPU_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}"
export NDGPU_TEST_DEPS="${NDGPU_TEST_DEPS:-/data/scratch/shared/poncet_m/ndgpu-test-deps/pytest8}"
export NDGPU_REGRESSION_LOG_DIR="$logdir"
sbatch "$stage_root/repo/slurm/run_ndgpu_gpu_regression.sh"
