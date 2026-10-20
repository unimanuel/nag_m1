#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Other W975 intervals are intentionally deferred until depth restoration.
exec bash "$repo_root/run_paper_final_baselines.sh" --intervals W975_175_195 "$@"
