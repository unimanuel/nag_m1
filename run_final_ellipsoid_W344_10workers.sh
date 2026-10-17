#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$repo_root/run_final_ellipsoid_benchmarks.sh" \
  --intervals W344_0_51 W344_52_95 W344_95_135 W344_135_197 \
  --training-root "$repo_root/runs/autowork_final_ellipsoid_W344_10workers" "$@"
