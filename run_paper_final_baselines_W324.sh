#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Resource/runtime exclusion applies equally to plane and both sphere layouts.
exec bash "$repo_root/run_paper_final_baselines.sh" \
  --intervals W324_0_42 W324_42_96 W324_161_197 \
  --output-root "$repo_root/runs/autowork_paper_final_baselines_W324_without_96_161" "$@"
