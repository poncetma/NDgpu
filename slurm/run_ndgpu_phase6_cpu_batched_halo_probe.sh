#!/usr/bin/env bash

#SBATCH --job-name=ndgpu-p6-cpu-halo
#SBATCH --cluster=merlin7
#SBATCH --partition=hourly
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:02:00
#SBATCH --mem=2G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6-cpu/logs
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6-cpu/logs/batched-halo-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6-cpu/logs/batched-halo-%j.log

set -euo pipefail

repo="${NDGPU_REPO:-/data/scratch/shared/poncet_m/ndgpu-multigpu-phase6-cpu/repo}"
python_bin="${NDGPU_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}"
mpi4py_path="${NDGPU_MPI4PY_PATH:-/data/scratch/shared/poncet_m/ndgpu-mpi4py-x86-py312}"

unset PMODULES_ENV
module purge
module use Spack
module load openmpi/4.1.6-57rc-A100-gpu
export PYTHONPATH="${mpi4py_path}:${repo}"

srun --ntasks=2 --kill-on-bad-exit=1 --cpu-bind=cores \
    "${python_bin}" "${repo}/examples/mpi_environment_probe.py" \
    --device cpu --communication cpu-mpi --batched-halos \
    --elements 4096 --iterations 200 --allreduce-iterations 2000
