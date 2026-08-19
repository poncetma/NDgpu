#!/usr/bin/env bash
#
# Submit a single-node NDgpu run to a Merlin7 GPU partition.
#
# Usage:
#   sbatch slurm/run_ndgpu_gh.sh
#
# Optional overrides are passed as environment variables:
#   NDGPU_EXAMPLE   Python entrypoint to run
#   NDGPU_ARGS      Arguments for the example
#   NDGPU_DEVICE    Device string passed to NDgpu (default: gpu)
#
# Default workload:
#   examples/speed_benchmark.py 32 64 96 128

#SBATCH --job-name=ndgpu-gh
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --mem=46G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-slurm-debug
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/ndgpu-gh-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/ndgpu-gh-%j.err

set -euo pipefail

if [[ "$(uname -m)" == "aarch64" ]]; then
    unset PMODULES_ENV
    module purge
    module use Spack unstable
    module load gcc/12.3 cuda/12.6.0-wak5 python/3.13.5-xbg5
    default_python="/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1/bin/python"
else
    default_python="/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python"
fi
python_bin="${NDGPU_PYTHON_BIN:-$default_python}"
if [[ ! -x "$python_bin" ]]; then
    if [[ -n "${MODULESHOME:-}" ]]; then
        module load "${NDGPU_PYTHON_MODULE:-Python/3.11.11}"
    fi
    python_bin="${PYTHON_BIN:-}"
fi
if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
    python_bin="$(command -v python3.11 || command -v python3.12 || command -v python3 || true)"
fi

repo="${NDGPU_REPO:-}"
if [[ -z "$repo" ]]; then
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo="$(cd "$here/.." && pwd)"
fi

logdir="${NDGPU_DEBUG_DIR:-/data/scratch/shared/poncet_m/ndgpu-slurm-debug}"
mkdir -p "$logdir"
logfile="$logdir/ndgpu-gh-${SLURM_JOB_ID:-manual}.log"
touch "$logfile"
exec >>"$logfile" 2>&1
cd "$logdir"
printf '=== NDgpu GPU speed benchmark started at %s on %s ===\n' "$(date -Is)" "$(hostname)"

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

example="${NDGPU_EXAMPLE:-$repo/examples/speed_benchmark.py}"
args="${NDGPU_ARGS:-32 64 96 128}"

printf 'job=%s host=%s cwd=%s\n' "${SLURM_JOB_ID:-manual}" "$(hostname)" "$PWD"
printf 'example=%s\nargs=%s\n' "$example" "$args"

if [[ -z "$python_bin" ]]; then
    echo "error: no usable python interpreter found on PATH" >&2
    exit 127
fi

if ! "$python_bin" - <<'PY'
import cupy
import numpy
import scipy
from cupy.cuda import runtime

from ndgpu import TransientSolver
from ndgpu.multigroup import MultigroupStepOperator

device_count = runtime.getDeviceCount()
if device_count < 1:
    raise RuntimeError("CuPy cannot see an allocated CUDA device")

props = runtime.getDeviceProperties(0)
name = props["name"]
if isinstance(name, bytes):
    name = name.decode()
print(f"python preflight: numpy={numpy.__version__} scipy={scipy.__version__} "
      f"cupy={cupy.__version__} cuda_devices={device_count} device0={name}")
print(f"transient preflight: {TransientSolver.__name__}, "
      f"{MultigroupStepOperator.__name__}")
PY
then
    cat >&2 <<EOF
error: the selected Python failed the NumPy/SciPy/CuPy/CUDA/NDgpu preflight
selected interpreter: $python_bin
Set NDGPU_PYTHON_BIN to a CuPy-enabled environment, for example one with
cuda-aware cupy installed alongside numpy and scipy.
EOF
    exit 1
fi

"$python_bin" -u "$example" $args
