#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$repo_root/run_ego_repeatability.sh" --batch part2of4 --texture-orientations legacy world_y "$@"
