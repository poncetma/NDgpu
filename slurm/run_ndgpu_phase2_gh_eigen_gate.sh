#!/usr/bin/env bash

#SBATCH --job-name=ndgpu-p2-eigen-gh
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase2/logs
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase2/logs/gh-eigen-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase2/logs/gh-eigen-%j.log

set -euo pipefail

repo="${NDGPU_REPO:-/data/scratch/shared/poncet_m/ndgpu-multigpu-phase2/repo}"
python_bin="${NDGPU_PYTHON_BIN:-/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1/bin/python}"
mpi_root="${NDGPU_MPI_ROOT:-/afs/psi.ch/sys/spack/develop/opt/spack/unstable/linux-sles15-aarch64/gcc-14.2.0/openmpi-5.0.7-iw2cnaeburexauadnbenfkytesx62sq2}"

unset PMODULES_ENV
module purge
module use Spack unstable
module load gcc/12.3 cuda/12.6.0-wak5 python/3.13.5-xbg5
module list 2>&1

if [[ ! -x "${python_bin}" ]]; then
    printf 'GH Python environment is missing: %s\n' "${python_bin}" >&2
    exit 2
fi
if [[ ! -f "${mpi_root}/lib/libmpi.so" ]]; then
    printf 'OpenMPI library is missing: %s\n' "${mpi_root}/lib/libmpi.so" >&2
    exit 2
fi
export PATH="${mpi_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${mpi_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export SLURM_MPI_TYPE=pmix
export PYTHONPATH="${repo}"

printf 'NDgpu Phase 2 multi-GH eigen gate: host=%s tasks=%s repo=%s\n' \
    "$(hostname)" "${SLURM_NTASKS}" "${repo}"
"${python_bin}" -c \
    'import cupy, mpi4py, numpy; print(f"cupy={cupy.__version__} mpi4py={mpi4py.__version__} numpy={numpy.__version__}")'
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

srun --mpi=pmix --ntasks=2 --kill-on-bad-exit=1 --cpu-bind=cores \
    --gpus-per-task=1 --gpu-bind=single:1 \
    "${python_bin}" "${repo}/examples/distributed_cartesian_eigen_gate.py" \
    --device gpu --communication host-staged --shape 128,96,64 --axis 0
