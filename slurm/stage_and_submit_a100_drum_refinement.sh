#!/usr/bin/env bash

set -euo pipefail

src_root="/afs/psi.ch/project/stars/workspace/RND/SB-RND-ACT-011-24/WP2/PM41/NDgpu"
stage_root="/data/scratch/shared/poncet_m/ndgpu-hpmr-drum-refinement-a100"
logdir="${stage_root}/logs"

mkdir -p "${stage_root}/repo" "${logdir}"
rsync -a --delete --exclude '.git' --exclude '.agents' --exclude '.codex' \
    "${src_root}/" "${stage_root}/repo/"

export NDGPU_REPO="${stage_root}/repo"
export NDGPU_PYTHON_BIN="/opt/psi/conda-envs/x86_64/ra-standard_py312/bin/python"
export NDGPU_DEBUG_DIR="${logdir}"
export NDGPU_EXAMPLE="${stage_root}/repo/examples/hpmr_drum_refinement.py"

selection="${1:-both}"
profile="${2:-baseline}"
case "${selection}" in
    both) modes=(local global) ;;
    local|global) modes=("${selection}") ;;
    *) echo "selection must be both, local, or global" >&2; exit 2 ;;
esac

for mode in "${modes[@]}"; do
    if [[ "${mode}" == "local" ]]; then
        base_refine=4
        case "${profile}" in
            baseline) levels="0,1,2,3"; angles="90,90.5,95"; samples=24 ;;
            fine) levels="3,4"; angles="90,90.5,95"; samples=48 ;;
            corrected) levels="0,1,2,3,4"; angles="90,90.5,95"; samples=48 ;;
            curve) levels="4"; angles="85,87,89,90,90.25,90.5,91,93,95"; samples=48 ;;
            exact) levels="0,1,2,3"; angles="90,90.5,95"; samples=0 ;;
            exact-curve) levels="3"; angles="85,87,89,90,90.25,90.5,91,93,95"; samples=0 ;;
            exact-fine-curve) levels="4"; angles="85,87,89,90,90.25,90.5,91,93,95"; samples=0 ;;
            exact-balanced-curve) base_refine=8; levels="3"; angles="85,87,89,90,90.25,90.5,91,93,95"; samples=0 ;;
            exact-balanced16-curve) base_refine=16; levels="2"; angles="85,87,89,90,90.25,90.5,91,93,95"; samples=0 ;;
            *) echo "unknown profile: ${profile}" >&2; exit 2 ;;
        esac
        export NDGPU_ARGS="--mode local --refine ${base_refine} --local-levels ${levels} --angles ${angles} --samples ${samples} --device gpu --output ${logdir}/local-${profile}-result.json"
    else
        case "${profile}" in
            baseline) refines="4,6,8,12,16"; angles="90,90.5,95"; samples=24 ;;
            fine) refines="16,24,32"; angles="90,90.5,95"; samples=48 ;;
            corrected) refines="16,24,32"; angles="90,90.5,95"; samples=48 ;;
            curve) refines="32"; angles="85,87,89,90,90.25,90.5,91,93,95"; samples=48 ;;
            exact) refines="16,24,32"; angles="90,90.5,95"; samples=0 ;;
            exact-curve) refines="32"; angles="85,87,89,90,90.25,90.5,91,93,95"; samples=0 ;;
            exact-fine-curve) refines="64"; angles="85,87,89,90,90.25,90.5,91,93,95"; samples=0 ;;
            *) echo "unknown profile: ${profile}" >&2; exit 2 ;;
        esac
        export NDGPU_ARGS="--mode global --refines ${refines} --angles ${angles} --samples ${samples} --device gpu --output ${logdir}/global-${profile}-result.json"
    fi
    sbatch --clusters=gmerlin7 --partition=a100-hourly --ntasks=1 \
        --job-name="ndgpu-drum-${mode}-${profile}" --chdir="${logdir}" \
        --output="${logdir}/${mode}-${profile}-%j.out" \
        --error="${logdir}/${mode}-${profile}-%j.err" \
        "${stage_root}/repo/slurm/run_ndgpu_gh.sh"
done
