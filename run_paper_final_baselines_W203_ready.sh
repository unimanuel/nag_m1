#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Keep 60-107 out until its original depth TIFFs have been restored and audited.
exec bash "$repo_root/run_paper_final_baselines.sh" --intervals W203_3_58 "$@"
