#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$repo_root/run_final_ellipsoid_benchmarks.sh" \
  --intervals W324_0_42 W324_42_96 W324_96_161 W324_161_197 \
  --training-root "$repo_root/runs/autowork_final_ellipsoid_W324_4workers" "$@"
