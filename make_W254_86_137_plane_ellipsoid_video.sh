#!/usr/bin/env bash
# First-repeat, regular-mapping comparison. No training/GPU use.
# Usage: bash make_W254_86_137_plane_ellipsoid_video.sh [extra CLI options]
# Preview without writing: DRY_RUN=1 bash make_W254_86_137_plane_ellipsoid_video.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEFT_RUN="${LEFT_RUN:-$REPO_ROOT/runs/autowork_paper_plane_baselines_part2of2/autowork_W254_86_137_plane_same_seed_seed42_repeat01/attempt_001}"
RIGHT_RUN="${RIGHT_RUN:-$REPO_ROOT/runs/2026-09-25_ego_poles_repeatability_part4of4/W254_86_137_regular_same_seed_seed42_repeat01/attempt_001}"
OUTPUT="${OUTPUT:-/mnt/d/CGProjectData/waymo/diagnostics/autowork_presentation_comparisons/W254_86_137_plane_vs_ellipsoid_repeat01.mp4}"
extra=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then extra+=(--dry-run); fi
exec "$PYTHON_BIN" "$REPO_ROOT/nag/scripts/make_result_comparison_video.py" \
    --left-run "$LEFT_RUN" --right-run "$RIGHT_RUN" \
    --sequence-label 'W254 86–137' --output "$OUTPUT" "${extra[@]}" "$@"
