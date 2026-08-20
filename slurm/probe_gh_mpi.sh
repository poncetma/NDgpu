#!/usr/bin/env bash

#SBATCH --job-name=ndgpu-gh-mpi-probe
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase1/logs
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase1/logs/gh-mpi-probe-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase1/logs/gh-mpi-probe-%j.log

set -u

printf 'GH MPI probe: host=%s arch=%s\n' "$(hostname)" "$(uname -m)"
unset PMODULES_ENV
module purge
module use Spack
printf '\nStable OpenMPI modules:\n'
module -t avail openmpi 2>&1
printf '\nStable MPICH modules:\n'
module -t avail mpich 2>&1
module use Spack unstable
printf '\nUnstable OpenMPI modules:\n'
module -t avail openmpi 2>&1
printf '\nUnstable MPICH modules:\n'
module -t avail mpich 2>&1
