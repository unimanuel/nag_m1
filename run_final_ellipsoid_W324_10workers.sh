#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Keep the historical cohort identity to reuse completed ten-worker jobs.
# Scheduling skips all repeats/layouts of 96-161 without rewriting its manifest.
exec bash "$repo_root/run_final_ellipsoid_benchmarks.sh" \
  --intervals W324_0_42 W324_42_96 W324_96_161 W324_161_197 \
  --skip-intervals W324_96_161 \
  --training-root "$repo_root/runs/autowork_final_ellipsoid_W324_10workers" "$@"
