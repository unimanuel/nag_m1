#!/usr/bin/env bash
# Export per-pixel RGB-MAE diagnostics for every completed v4 ellipsoid run.
#
# This covers the raw v4 and Moderate-Y v4 runs, in both regular and
# Visible-UV modes, under runs/2026-09-02_depth_proxy_v4.
#
# Usage from the repository root:
#   bash export_depth_proxy_v4_pixel_errors.sh
#
# Optional environment variables:
#   DISPLAY_CAP=0.25  Fixed normalized RGB-MAE cap for all heatmaps.
#   FORCE=1           Recreate an existing final/pixel_error directory.
#   DRY_RUN=1         Resolve and print jobs without writing files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${REPO_ROOT}/runs/2026-09-02_depth_proxy_v4"
EXPORTER="${REPO_ROOT}/nag/scripts/export_pixel_error_maps.py"
DATA_ROOT="${REPO_ROOT}/data/datasets/waymo"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DISPLAY_CAP="${DISPLAY_CAP:-0.25}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$EXPORTER" ]] || { echo "Missing exporter: $EXPORTER" >&2; exit 1; }
[[ -d "$RUN_ROOT" ]] || { echo "Missing run root: $RUN_ROOT" >&2; exit 1; }

"$PYTHON_BIN" - "$DISPLAY_CAP" <<'PY'
import sys
cap = float(sys.argv[1])
if not 0.0 < cap <= 1.0:
    raise SystemExit("DISPLAY_CAP must be in (0, 1].")
PY

frame_start_for() {
    local label="$1"
    case "$label" in
        W191_0_40|W254_0_37) echo 0 ;;
        W191_40_128) echo 40 ;;
        W191_128_198) echo 128 ;;
        W254_37_81) echo 37 ;;
        W254_86_137) echo 86 ;;
        W254_139_195) echo 139 ;;
        *) return 1 ;;
    esac
}

segment_for() {
    case "$1" in
        W191_*) echo 1918764220984209654 ;;
        W254_*) echo 2547899409721197155 ;;
        *) return 1 ;;
    esac
}

jobs=0
while IFS= read -r -d '' run_dir; do
    name="$(basename "$run_dir")"
    sequence_label="$(printf '%s' "$name" | grep -oE 'W(191|254)_[0-9]+_[0-9]+' | head -n1 || true)"
    if [[ -z "$sequence_label" ]]; then
        echo "[skip] Could not identify sequence in: $name" >&2
        continue
    fi
    if [[ "$name" == *"_moderate_y_visuv" ]]; then
        mode="Moderate-Y Visible-UV"
    elif [[ "$name" == *"_moderate_y" ]]; then
        mode="Moderate-Y regular"
    elif [[ "$name" == *"_visuv" ]]; then
        mode="Raw v4 Visible-UV"
    else
        mode="Raw v4 regular"
    fi
    frame_start="$(frame_start_for "$sequence_label")"
    segment_id="$(segment_for "$sequence_label")"
    ground_truth_dir="$DATA_ROOT/segment-${segment_id}/images_undistorted"
    [[ -d "$run_dir/final/complete" ]] || { echo "[skip] No complete renders: $run_dir"; continue; }
    [[ -d "$ground_truth_dir" ]] || { echo "Missing ground truth: $ground_truth_dir" >&2; exit 1; }
    if [[ "$FORCE" != 1 && -f "$run_dir/final/pixel_error/summary.json" ]]; then
        echo "[skip] $mode | $sequence_label | pixel_error already exists"
        continue
    fi
    echo "[export] $mode | $sequence_label | $name"
    jobs=$((jobs + 1))
    if [[ "$DRY_RUN" == 1 ]]; then
        continue
    fi
    "$PYTHON_BIN" "$EXPORTER" \
        --run-dir "$run_dir" \
        --ground-truth-dir "$ground_truth_dir" \
        --frame-start "$frame_start" \
        --frame-step 1 \
        --display-cap "$DISPLAY_CAP"
done < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "Completed $jobs v4 pixel-error export(s); DISPLAY_CAP=$DISPLAY_CAP."
