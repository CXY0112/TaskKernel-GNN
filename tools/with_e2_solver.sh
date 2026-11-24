#!/usr/bin/env bash
# Reuse the verified WSL solver environment; activation is local to this process.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/taskkernel-e2/venv_carputils/bin/activate
export PATH="/usr/local/lib/opencarp/lib/petsc/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLBACKEND=Agg
cd "$script_dir/.."
exec "$@"
