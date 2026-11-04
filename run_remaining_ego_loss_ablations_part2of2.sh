#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$repo_root/run_remaining_ego_loss_ablations.sh" \
  --intervals W254_0_37 W254_139_195 "$@"
