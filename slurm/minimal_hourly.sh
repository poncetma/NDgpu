#!/usr/bin/env bash
#
# Ultra-minimal Merlin7 CPU batch job for debugging batch execution.
#
# This script is intentionally tiny: it only prints a few lines, writes a
# timestamped log in a shared scratch directory, and sleeps briefly. If this
# job does not leave output behind, the issue is with the batch environment or
# filesystem path rather than NDgpu.

#SBATCH --job-name=slurm-mini
#SBATCH --cluster=merlin7
#SBATCH --partition=hourly
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-slurm-debug
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/slurm-mini-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/slurm-mini-%j.err

set -euo pipefail

base_dir="${NDGPU_DEBUG_DIR:-/data/scratch/shared/poncet_m/ndgpu-slurm-debug}"
mkdir -p "$base_dir"

job_id="${SLURM_JOB_ID:-manual}"
logfile="$base_dir/slurm-mini-${job_id}.log"
touch "$logfile"
exec >>"$logfile" 2>&1

echo "=== minimal job start ==="
echo "time: $(date -Is)"
echo "host: $(hostname)"
echo "job:  ${SLURM_JOB_ID:-manual}"
echo "part: ${SLURM_JOB_PARTITION:-unknown}"
echo "node: ${SLURM_NODELIST:-unknown}"
sleep 10
echo "=== minimal job end ==="
