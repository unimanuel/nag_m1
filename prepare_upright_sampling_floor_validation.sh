#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
args=()
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python}" "$repo_root/nag/scripts/upright_sampling_floor_validation.py" prepare "${args[@]}" "$@"
