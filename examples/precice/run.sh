#!/usr/bin/env bash
# Launch both HP-MR coupling participants and wait for them.
#
#   bash examples/precice/run.sh [extra args passed to BOTH participants]
#
# e.g.  bash examples/precice/run.sh --refine 6 --groups 11
#
# Needs pyprecice. The system libprecice on this machine is an Ubuntu 24.04
# build on a 22.04 box and cannot load (it wants GLIBC_2.38); use the
# conda-forge environment instead:
#
#   conda create -n ndgpu-precice --override-channels -c conda-forge \
#       python=3.13 pyprecice numpy scipy pytest
#   conda run -n ndgpu-precice python -m pip install -e . --no-deps
#   conda run -n ndgpu-precice bash examples/precice/run.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
workdir="${NDGPU_PRECICE_WORKDIR:-$here/run}"

mkdir -p "$workdir"
cd "$workdir"
rm -rf precice-run precice-profiling

# Both participants import common.py from the examples dir, and ndgpu from the
# repo; run from a scratch dir so preCICE's own files land there.
export PYTHONPATH="$here:$repo${PYTHONPATH:+:$PYTHONPATH}"

echo "workdir: $workdir"
python "$here/neutronics.py" --csv neutronics.csv "$@" &
neutronics=$!
python "$here/thermal.py" --csv thermal.csv "$@" &
thermal=$!

status=0
wait $neutronics || status=$?
wait $thermal || status=$?
exit $status
