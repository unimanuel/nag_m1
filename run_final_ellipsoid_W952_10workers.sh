#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$repo_root/run_final_ellipsoid_benchmarks.sh" \
  --intervals W952_0_63 W952_64_140 W952_141_198 \
  --training-root "$repo_root/runs/autowork_final_ellipsoid_W952_10workers" "$@"
