#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
args=()
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
export PYTHONDONTWRITEBYTECODE=1
exec "${NAG_PYTHON:-/home/mang/anaconda3/envs/nag1/bin/python}" \
  reports/paper_depth_floor_ablations.py train "${args[@]}" "$@"
