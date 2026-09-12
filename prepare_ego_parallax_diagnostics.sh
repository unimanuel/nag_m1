#!/usr/bin/env bash
# Shared LiDAR observation preparation; see docs/ego_parallax_diagnostics.md.
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_ROOT/ego_parallax_diagnostics_common.sh" prepare "$@"
