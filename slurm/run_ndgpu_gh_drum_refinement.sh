#!/usr/bin/env bash

#SBATCH --job-name=ndgpu-hpmr-drum-refine
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-hpmr-drum-refinement/logs
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-hpmr-drum-refinement/logs/refinement-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-hpmr-drum-refinement/logs/refinement-%j.log

set -euo pipefail

repo="${NDGPU_REPO:-/data/scratch/shared/poncet_m/ndgpu-hpmr-drum-refinement/repo}"
logdir="${NDGPU_LOGDIR:-/data/scratch/shared/poncet_m/ndgpu-hpmr-drum-refinement/logs}"
python_bin="${NDGPU_PYTHON_BIN:-/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1/bin/python}"
mode="${NDGPU_REFINEMENT_MODE:-local}"
samples="${NDGPU_SAMPLES:-0}"

unset PMODULES_ENV
module purge
module use Spack unstable
module load gcc/12.3 cuda/12.6.0-wak5 python/3.13.5-xbg5
export PYTHONPATH="${repo}"
export OMP_NUM_THREADS=1

printf 'HPMR drum refinement: mode=%s samples=%s job=%s host=%s\n' \
    "${mode}" "${samples}" "${SLURM_JOB_ID}" "$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
"${python_bin}" -u -c \
    'import cupy; print(f"CuPy {cupy.__version__}, devices={cupy.cuda.runtime.getDeviceCount()}")'

args=(--mode "${mode}" --samples "${samples}" --device gpu
      --output "${logdir}/${mode}-${SLURM_JOB_ID}.json")
if [[ "${mode}" == "local" ]]; then
    args+=(--refine "${NDGPU_REFINE:-4}"
           --local-levels "${NDGPU_LOCAL_LEVELS:-0,1,2,3}")
else
    args+=(--refines "${NDGPU_REFINES:-4,6,8,12,16}")
fi

"${python_bin}" -u "${repo}/examples/hpmr_drum_refinement.py" "${args[@]}"
