#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$repo_root/run_paper_pca_baselines.sh" \
  --intervals W191_0_40 W191_40_128 W191_128_198 "$@"
