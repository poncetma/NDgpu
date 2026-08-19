#!/usr/bin/env bash
# Inventory the native aarch64 Python/CUDA environment on a GH200 node before
# creating NDgpu's persistent scratch environment.

#SBATCH --job-name=ndgpu-gh-probe
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-slurm-debug
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/ndgpu-gh-probe-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/ndgpu-gh-probe-%j.log

set -o pipefail

printf '=== GH200 environment probe: %s ===\n' "$(date -Is)"
printf 'host=%s arch=%s kernel=%s\n' "$(hostname)" "$(uname -m)" "$(uname -r)"
printf 'slurm_job=%s cuda_visible_devices=%s\n' \
    "${SLURM_JOB_ID:-unknown}" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

unset PMODULES_ENV
module purge
module use Spack unstable

printf '\n=== Candidate modules ===\n'
module -t avail python py-numpy py-scipy py-pip py-virtualenv cuda 2>&1 || true
printf '\n=== CuPy module search ===\n'
module spider py-cupy 2>&1 || true
module spider cupy 2>&1 || true

printf '\n=== Load documented GH200 Python/CUDA stack ===\n'
module load gcc/12.3 cuda/12.6.0-wak5 python/3.13.5-xbg5 py-numpy/2.3.2-yoqr
module list 2>&1

printf '\n=== Python capabilities ===\n'
command -v python
python -VV
python -c 'import platform, sys; print(platform.machine()); print(sys.executable); print(sys.path)'
python -c 'import numpy; print("numpy", numpy.__version__, numpy.__file__)'
python -c 'import importlib.util as i; print("scipy", i.find_spec("scipy")); print("cupy", i.find_spec("cupy")); print("pip", i.find_spec("pip")); print("venv", i.find_spec("venv"))'
python -m pip --version || true
python -m ensurepip --version || true

printf '=== GH200 environment probe finished: %s ===\n' "$(date -Is)"
