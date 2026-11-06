#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$repo_root/run_paper_depth_floor.sh" \
  --intervals W254_0_37 W254_37_81 W254_86_137 W254_139_195 "$@"
