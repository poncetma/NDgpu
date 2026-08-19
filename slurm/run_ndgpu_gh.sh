#!/usr/bin/env bash
#
# Submit a single-node NDgpu run to Merlin7's Grace-Hopper cluster.
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
#   examples/speed_benchmark.py 32
#
# That is a short CPU/GPU sanity check that verifies the solver can start and
# complete a small, correct run on the selected node.

#SBATCH --job-name=ndgpu-gh
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --mem=46G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-slurm-debug
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/ndgpu-gh-%j.out
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-slurm-debug/ndgpu-gh-%j.err

set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" && -n "${MODULESHOME:-}" ]]; then
    # Merlin's system Python is too old and usually lacks NumPy/SciPy.
    module load "${NDGPU_PYTHON_MODULE:-Python/3.11.11}"
fi

python_bin="${PYTHON_BIN:-${NDGPU_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}}"
if [[ ! -x "$python_bin" ]]; then
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
printf '=== NDgpu GH smoke test started at %s on %s ===\n' "$(date -Is)" "$(hostname)"

export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

example="${NDGPU_EXAMPLE:-$repo/examples/speed_benchmark.py}"
args="${NDGPU_ARGS:-32}"

printf 'job=%s host=%s cwd=%s\n' "${SLURM_JOB_ID:-manual}" "$(hostname)" "$PWD"
printf 'example=%s\nargs=%s\n' "$example" "$args"

if [[ -z "$python_bin" ]]; then
    echo "error: no usable python interpreter found on PATH" >&2
    exit 127
fi

if ! "$python_bin" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("cupy") is not None else 1)
PY
then
    cat >&2 <<EOF
error: the selected Python does not provide CuPy
selected interpreter: $python_bin
Set NDGPU_PYTHON_BIN to a CuPy-enabled environment, for example one with
cuda-aware cupy installed alongside numpy and scipy.
EOF
    exit 1
fi

"$python_bin" -u "$example" $args
