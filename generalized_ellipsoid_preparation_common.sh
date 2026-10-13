#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
stage="$1"
shift
args=()
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
if [[ "$stage" != train ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
fi
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python}" "$repo_root/nag/scripts/generalized_ellipsoid_preparation.py" "$stage" "${args[@]}" "$@"
