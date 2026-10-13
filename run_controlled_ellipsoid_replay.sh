#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_DYNAMIC=FALSE PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
args=()
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
exec "${PYTHON_BIN:-python}" "$repo_root/controlled_floor_replay.py" "${args[@]}" "$@"
