#!/usr/bin/env bash

#SBATCH --job-name=ndgpu-p7-extr-perf
#SBATCH --cluster=gmerlin7
#SBATCH --partition=gh-hourly
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:55:00
#SBATCH --mem=160G
#SBATCH --chdir=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase7-performance/logs
#SBATCH --output=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase7-performance/logs/extruded-%j.log
#SBATCH --error=/data/scratch/shared/poncet_m/ndgpu-multigpu-phase7-performance/logs/extruded-%j.log

set -euo pipefail

repo="${NDGPU_REPO:-/data/scratch/shared/poncet_m/ndgpu-multigpu-phase7-performance/repo}"
refine="${NDGPU_REFINE:-8}"
local_levels="${NDGPU_LOCAL_LEVELS:-3}"
nz="${NDGPU_NZ:-10}"
steps="${NDGPU_STEPS:-5}"
dt="${NDGPU_DT:-0.01}"
check_every="${NDGPU_CHECK_EVERY:-1}"
precond_degree="${NDGPU_PRECOND_DEGREE:-1}"
scatter_subsweeps="${NDGPU_SCATTER_SUBSWEEPS:-0}"
step_solver="${NDGPU_STEP_SOLVER:-fixed-point}"
communication="${NDGPU_MPI_COMMUNICATION:-cuda-aware}"
multigroup_scatter_sweeps="${NDGPU_MULTIGROUP_SCATTER_SWEEPS:-3}"
multigroup_energy_anderson="${NDGPU_MULTIGROUP_ENERGY_ANDERSON:-0}"
multigroup_fixed_relaxations="${NDGPU_MULTIGROUP_FIXED_RELAXATIONS:-0}"
multigroup_fixed_iterations="${NDGPU_MULTIGROUP_FIXED_ITERATIONS:-0}"
multigroup_inner_rtol="${NDGPU_MULTIGROUP_INNER_RTOL:-1e-3}"
drum_motion="${NDGPU_DRUM_MOTION:-step}"

unset PMODULES_ENV
module purge
if [[ "$(uname -m)" == "aarch64" ]]; then
    module use Spack unstable
    module load gcc/12.3 cuda/12.6.0-wak5 python/3.13.5-xbg5
    python_bin="${NDGPU_PYTHON_BIN:-/data/scratch/shared/poncet_m/ndgpu-gh-py313-v1/bin/python}"
    mpi_root="${NDGPU_MPI_ROOT:-/afs/psi.ch/sys/spack/develop/opt/spack/unstable/linux-sles15-aarch64/gcc-14.2.0/openmpi-5.0.7-iw2cnaeburexauadnbenfkytesx62sq2}"
    if [[ ! -x "${python_bin}" || ! -f "${mpi_root}/lib/libmpi.so" ]]; then
        echo "GH Python or OpenMPI environment is incomplete" >&2
        exit 2
    fi
    export PATH="${mpi_root}/bin:${PATH}"
    export LD_LIBRARY_PATH="${mpi_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    mpi4py_prefix=""
else
    module use Spack
    module load openmpi/4.1.6-57rc-A100-gpu
    python_bin="${NDGPU_PYTHON_BIN:-/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python}"
    mpi4py_prefix="${NDGPU_MPI4PY_PATH:-/data/scratch/shared/poncet_m/ndgpu-mpi4py-x86-py312}:"
    if [[ ! -x "${python_bin}" ]]; then
        echo "A100 Python environment is incomplete" >&2
        exit 2
    fi
fi
export SLURM_MPI_TYPE=pmix
export PYTHONPATH="${mpi4py_prefix}${repo}"
export OMP_NUM_THREADS=1

printf 'NDgpu Phase 7 local-mesh performance: ranks=%s r=%s+%s nz=%s steps=%s precond=%s scatter=%s method=%s communication=%s host=%s\n' \
    "${SLURM_NTASKS}" "${refine}" "${local_levels}" "${nz}" "${steps}" \
    "${precond_degree}" "${scatter_subsweeps}" "${step_solver}" \
    "${communication}" "$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
srun --mpi=pmix --ntasks="${SLURM_NTASKS}" --kill-on-bad-exit=1 \
    --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1 \
    "${python_bin}" \
    "${repo}/examples/distributed_extruded_hpmr_performance.py" \
    --device gpu --communication "${communication}" --groups 11 \
    --refine "${refine}" --local-levels "${local_levels}" --nz "${nz}" \
    --steps "${steps}" --dt "${dt}" --tol-step 1e-7 \
    --drum-motion "${drum_motion}" \
    --precond-degree "${precond_degree}" \
    --scatter-subsweeps "${scatter_subsweeps}" \
    --check-every "${check_every}" --step-solver "${step_solver}" \
    --multigroup-scatter-sweeps "${multigroup_scatter_sweeps}" \
    --multigroup-energy-anderson "${multigroup_energy_anderson}" \
    --multigroup-inner-fixed-relaxations "${multigroup_fixed_relaxations}" \
    --multigroup-inner-fixed-iterations "${multigroup_fixed_iterations}" \
    --multigroup-inner-rtol "${multigroup_inner_rtol}" \
    --single-reduction --batched-halos
