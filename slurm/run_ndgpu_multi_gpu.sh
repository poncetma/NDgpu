#!/usr/bin/env bash
# Launch the MPI communication probe inside an existing multi-GPU allocation.
# Site modules/environment creation remain outside this script so mpi4py is
# always linked against the MPI implementation used by srun.

set -euo pipefail

: "${SLURM_JOB_ID:?run this script inside an sbatch or salloc allocation}"
: "${SLURM_NTASKS:?request one Slurm task per GPU}"

if (( SLURM_NTASKS < 2 )); then
    printf 'NDgpu multi-GPU probe requires at least two Slurm tasks\n' >&2
    exit 2
fi

repo_dir="${NDGPU_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${NDGPU_PYTHON_BIN:-$(command -v python)}"
device="${NDGPU_DISTRIBUTED_DEVICE:-gpu}"
communication="${NDGPU_MPI_COMMUNICATION:-auto}"
probe="${repo_dir}/examples/mpi_environment_probe.py"

if [[ ! -x "${python_bin}" ]]; then
    printf 'Python interpreter is not executable: %s\n' "${python_bin}" >&2
    exit 2
fi
if [[ ! -f "${probe}" ]]; then
    printf 'MPI probe is missing: %s\n' "${probe}" >&2
    exit 2
fi

"${python_bin}" -c 'import mpi4py, numpy'
if [[ "${device}" == "gpu" ]]; then
    "${python_bin}" -c 'import cupy; assert cupy.cuda.runtime.getDeviceCount() >= 1'
fi

printf 'NDgpu MPI launch: job=%s nodes=%s tasks=%s device=%s communication=%s\n' \
    "${SLURM_JOB_ID}" "${SLURM_JOB_NUM_NODES:-unknown}" "${SLURM_NTASKS}" \
    "${device}" "${communication}"

srun_args=(
    --ntasks="${SLURM_NTASKS}"
    --kill-on-bad-exit=1
    --cpu-bind=cores
)
if [[ "${device}" == "gpu" ]]; then
    srun_args+=(--gpus-per-task=1)
fi

exec srun "${srun_args[@]}" \
    "${python_bin}" "${probe}" \
    --device "${device}" \
    --communication "${communication}" \
    "${@}"
