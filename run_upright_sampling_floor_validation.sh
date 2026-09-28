#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
args=()
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python}" "$repo_root/nag/scripts/upright_sampling_floor_validation.py" train "${args[@]}" "$@"
