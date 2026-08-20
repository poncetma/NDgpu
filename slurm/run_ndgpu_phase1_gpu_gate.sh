#!/usr/bin/env bash

#SBATCH --job-name=ndgpu-phase1-gh
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase1/logs
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase1/logs/gh-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase1/logs/gh-%j.log

set -euo pipefail

repo="${NDGPU_REPO:-/data/scratch/shared/poncet_m/ndgpu-multigpu-phase1/repo}"
python_bin="${NDGPU_PYTHON_BIN:-/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1/bin/python}"
wheelhouse="${NDGPU_WHEELHOUSE:-/data/scratch/shared/poncet_m/ndgpu-gh-wheelhouse}"
mpi_root="${NDGPU_MPI_ROOT:-/afs/psi.ch/sys/spack/develop/opt/spack/unstable/linux-sles15-aarch64/gcc-14.2.0/openmpi-5.0.7-iw2cnaeburexauadnbenfkytesx62sq2}"

unset PMODULES_ENV
module purge
module use Spack unstable
module load gcc/12.3 cuda/12.6.0-wak5 python/3.13.5-xbg5
module list 2>&1

# The generated OpenMPI module currently names stale dependency modules, but
# the installation and its dependency RPATHs are intact. Use that site prefix
# directly until PSI regenerates the aarch64 module hierarchy.
if [[ ! -f "${mpi_root}/lib/libmpi.so" ]]; then
    printf 'OpenMPI library is missing: %s\n' "${mpi_root}/lib/libmpi.so" >&2
    exit 2
fi
export PATH="${mpi_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${mpi_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPICC="${mpi_root}/bin/mpicc"
export SLURM_MPI_TYPE=pmix

"${python_bin}" -m pip install \
    --no-index --find-links "${wheelhouse}" mpi4py==4.1.2

export PYTHONPATH="${repo}"
printf 'NDgpu Phase 1 GH gate: host=%s python=%s repo=%s\n' \
    "$(hostname)" "${python_bin}" "${repo}"
nvidia-smi --query-gpu=name,uuid --format=csv,noheader

srun --ntasks=1 --kill-on-bad-exit=1 --cpu-bind=cores --gpus-per-task=1 \
    "${python_bin}" "${repo}/examples/distributed_size_one_gate.py" \
    --device gpu --communication host-staged
