#!/usr/bin/env bash

#SBATCH --job-name=ndgpu-p2-eigen-cpu
#SBATCH --cluster=merlin7
#SBATCH --partition=hourly
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase2/logs
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase2/logs/cpu-eigen-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase2/logs/cpu-eigen-%j.log

set -euo pipefail

repo="${NDGPU_REPO:-/data/scratch/shared/poncet_m/ndgpu-multigpu-phase2/repo}"
python_bin="${NDGPU_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}"
mpi4py_path="${NDGPU_MPI4PY_PATH:-/data/scratch/shared/poncet_m/ndgpu-mpi4py-x86-py312}"

unset PMODULES_ENV
module purge
module use Spack
module load openmpi/4.1.6-57rc-A100-gpu
module list 2>&1

export PYTHONPATH="${mpi4py_path}:${repo}"
printf 'NDgpu Phase 2 CPU eigen gate: host=%s tasks=%s repo=%s\n' \
    "$(hostname)" "${SLURM_NTASKS}" "${repo}"

srun --ntasks="${SLURM_NTASKS}" --kill-on-bad-exit=1 --cpu-bind=cores \
    "${python_bin}" "${repo}/examples/distributed_cartesian_eigen_gate.py" \
    --device cpu --communication cpu-mpi --shape 17,13,9 --axis 0
