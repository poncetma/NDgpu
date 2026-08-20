#!/usr/bin/env bash
# Build and validate NDgpu's native aarch64 Python environment on GH200 from
# wheels prefetched into shared scratch. This job performs no network access.

#SBATCH --job-name=ndgpu-gh-env
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-interactive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-slurm-debug
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/ndgpu-gh-env-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/ndgpu-gh-env-%j.log

set -euo pipefail

wheelhouse="/data/scratch/shared/poncet_m/ndgpu-gh-wheelhouse"
env_root="/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1"
pip_wheel="$wheelhouse/pip-25.3-py3-none-any.whl"

printf '=== GH200 environment setup started: %s ===\n' "$(date -Is)"
printf 'host=%s arch=%s env=%s\n' "$(hostname)" "$(uname -m)" "$env_root"
if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "error: GH200 environment setup requires an aarch64 node" >&2
    exit 1
fi
if [[ ! -f "$pip_wheel" ]]; then
    echo "error: missing offline pip wheel: $pip_wheel" >&2
    exit 1
fi

unset PMODULES_ENV
module purge
module use Spack unstable
module load gcc/12.3 cuda/12.6.0-wak5 python/3.13.5-xbg5
module list 2>&1

if [[ ! -x "$env_root/bin/python" ]]; then
    python -m venv --without-pip "$env_root"
fi

export PYTHONPATH="$pip_wheel"
"$env_root/bin/python" -m pip install \
    --no-index \
    --find-links "$wheelhouse" \
    --upgrade \
    pip==25.3 \
    numpy==2.3.2 \
    scipy==1.16.3 \
    cupy-cuda12x==13.6.0 \
    mpi4py==4.1.2 \
    pytest==8.4.2
unset PYTHONPATH

"$env_root/bin/python" - <<'PY'
import platform

import cupy
import mpi4py
import numpy
import pytest
import scipy
from cupy.cuda import runtime

x = cupy.arange(1_000_000, dtype=cupy.float64)
total = float(cupy.sum(x).get())
cupy.cuda.Stream.null.synchronize()
expected = 999_999 * 1_000_000 / 2
if total != expected:
    raise RuntimeError(f"CuPy reduction mismatch: {total} != {expected}")

props = runtime.getDeviceProperties(0)
name = props["name"]
if isinstance(name, bytes):
    name = name.decode()
print(f"arch={platform.machine()} device={name} cuda_devices={runtime.getDeviceCount()}")
print(f"numpy={numpy.__version__} scipy={scipy.__version__} "
      f"cupy={cupy.__version__} mpi4py={mpi4py.__version__} "
      f"pytest={pytest.__version__}")
print(f"cupy_sum={total}")
PY

printf '=== GH200 environment setup finished: %s ===\n' "$(date -Is)"
