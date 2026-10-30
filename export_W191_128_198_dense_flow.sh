#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
exec "${NAG_PYTHON:-/home/mang/anaconda3/envs/nag1/bin/python}" \
  reports/dense_background_flow.py \
  --run "$script_dir/runs/2026-09-25_ego_poles_repeatability_part2of4/W191_128_198_regular_same_seed_seed42_repeat01/attempt_001" \
  --output /mnt/d/CGProjectData/waymo/diagnostics/autowork_W191_128_198_dense_flow_2026-10-30 \
  --resume "$@"
