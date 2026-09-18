#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
args=()
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
exec "${PYTHON_BIN:-python}" "$repo_root/nag/scripts/sampling_floor_validation.py" train "${args[@]}" "$@"
