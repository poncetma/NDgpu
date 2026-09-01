#!/usr/bin/env bash

#SBATCH --job-name=ndgpu-p6-gh-comm
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6/logs
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6/logs/comm-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6/logs/comm-%j.log

set -euo pipefail

repo="${NDGPU_REPO:-/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6/repo}"
python_bin="${NDGPU_PYTHON_BIN:-/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1/bin/python}"
mpi_root="${NDGPU_MPI_ROOT:-/afs/psi.ch/sys/spack/develop/opt/spack/unstable/linux-sles15-aarch64/gcc-14.2.0/openmpi-5.0.7-iw2cnaeburexauadnbenfkytesx62sq2}"
communication="${NDGPU_MPI_COMMUNICATION:-host-staged}"
elements="${NDGPU_PROBE_ELEMENTS:-1048576}"
iterations="${NDGPU_PROBE_ITERATIONS:-100}"
allreduce_iterations="${NDGPU_ALLREDUCE_ITERATIONS:-2000}"

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

printf 'NDgpu Phase 6 GH communication probe: mode=%s host=%s\n' \
    "${communication}" "$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
srun --mpi=pmix --ntasks=2 --kill-on-bad-exit=1 --cpu-bind=cores \
    --gpus-per-task=1 --gpu-bind=single:1 \
    "${python_bin}" "${repo}/examples/mpi_environment_probe.py" \
    --device gpu --communication "${communication}" \
    --elements "${elements}" --iterations "${iterations}" \
    --allreduce-iterations "${allreduce_iterations}"
