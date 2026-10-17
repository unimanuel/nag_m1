#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# The other three W975 intervals remain blocked by missing depth TIFFs.
exec bash "$repo_root/run_final_ellipsoid_benchmarks.sh" \
  --intervals W975_175_195 \
  --training-root "$repo_root/runs/autowork_final_ellipsoid_W975_ready_4workers" "$@"
