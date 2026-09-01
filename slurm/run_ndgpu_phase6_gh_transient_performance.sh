#!/usr/bin/env bash

#SBATCH --job-name=ndgpu-p6-gh-transient
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6/logs
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6/logs/transient-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6/logs/transient-%j.log

set -euo pipefail

repo="${NDGPU_REPO:-/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6/repo}"
python_bin="${NDGPU_PYTHON_BIN:-/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1/bin/python}"
mpi_root="${NDGPU_MPI_ROOT:-/afs/psi.ch/sys/spack/develop/opt/spack/unstable/linux-sles15-aarch64/gcc-14.2.0/openmpi-5.0.7-iw2cnaeburexauadnbenfkytesx62sq2}"
communication="${NDGPU_MPI_COMMUNICATION:-cuda-aware}"
refine="${NDGPU_REFINE:-8}"
nz="${NDGPU_NZ:-10}"
steps="${NDGPU_STEPS:-5}"
dt="${NDGPU_DT:-0.01}"
check_every="${NDGPU_CHECK_EVERY:-1}"
single_reduction="${NDGPU_SINGLE_REDUCTION:-0}"
batched_halos="${NDGPU_BATCHED_HALOS:-0}"

unset PMODULES_ENV
module purge
module use Spack unstable
module load gcc/12.3 cuda/12.6.0-wak5 python/3.13.5-xbg5
if [[ ! -x "${python_bin}" || ! -f "${mpi_root}/lib/libmpi.so" ]]; then
    echo "GH Python or OpenMPI environment is incomplete" >&2
    exit 2
fi
export PATH="${mpi_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${mpi_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SLURM_MPI_TYPE=pmix
export PYTHONPATH="${repo}"

printf 'NDgpu Phase 6 GH transient performance: ranks=%s mode=%s r=%s nz=%s steps=%s check_every=%s single_reduction=%s batched_halos=%s host=%s\n' \
    "${SLURM_NTASKS}" "${communication}" "${refine}" "${nz}" \
    "${steps}" "${check_every}" "${single_reduction}" "${batched_halos}" \
    "$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
single_reduction_arg=()
if [[ "${single_reduction}" == "1" ]]; then
    single_reduction_arg+=(--single-reduction)
fi
batched_halos_arg=()
if [[ "${batched_halos}" == "1" ]]; then
    batched_halos_arg+=(--batched-halos)
fi
srun --mpi=pmix --ntasks="${SLURM_NTASKS}" --kill-on-bad-exit=1 \
    --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1 \
    "${python_bin}" "${repo}/examples/distributed_tri_hpmr_transient_performance.py" \
    --device gpu --communication "${communication}" \
    --refine "${refine}" --nz "${nz}" --groups 11 \
    --steps "${steps}" --dt "${dt}" --tol-step 1e-8 \
    --check-every "${check_every}" "${single_reduction_arg[@]}" \
    "${batched_halos_arg[@]}"
