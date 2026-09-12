#!/usr/bin/env bash
# Export individual pixel-MAE maps and signed spatial comparisons for the
# constrained reprojection/depth ellipsoid experiment on W254 37-81 and 86-137.
#
# The spatial comparison reads saved final renders directly; it is independent
# of the individual final/pixel_error folders.  This wrapper makes both kinds
# of inspection artifact in one reproducible command:
#   * per-run normalized RGB-MAE heatmaps/panels for Sphere, raw v4,
#     Moderate-Y, and reprojection-depth v1, in regular and Visible-UV modes;
#   * signed MAE maps and region statistics for every ellipsoid versus the
#     matching-layout Sphere, plus v1 directly versus Moderate-Y and raw v4.
#
# Usage from the repository root:
#   DRY_RUN=1 bash export_W254_reprojection_depth_spatial_comparison.sh
#   bash export_W254_reprojection_depth_spatial_comparison.sh
#
# Optional environment variables:
#   PYTHON_BIN=python3             Python environment with NumPy, Pillow, PyYAML.
#   PIXEL_DISPLAY_CAP=0.25         Fixed [0,1] heatmap saturation for all runs.
#   SPATIAL_DISPLAY_CAP=0.10       Fixed [0,1] saturation for signed differences.
#   SPATIAL_OUTPUT_ROOT=/mnt/d/... Spatial diagnostics output root.
#   FORCE=1                        Recreate existing pixel and spatial artifacts.
#   PIXEL_ERRORS=0                 Skip individual per-run pixel-error exports.
#   SPATIAL_COMPARISON=0           Skip signed spatial comparisons/statistics.
#   DRY_RUN=1                      Resolve and validate inputs without writing files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORTER="$REPO_ROOT/nag/scripts/export_pixel_error_maps.py"
DATA_ROOT="$REPO_ROOT/data/datasets/waymo"
SPHERE_ROOT="$REPO_ROOT/runs/2026-08-19_ellipsoid_non_center"
V4_ROOT="$REPO_ROOT/runs/2026-09-02_depth_proxy_v4"
REPROJECTION_ROOT_NAME="2026-09-12_reprojection_depth_v1"
REPROJECTION_ROOT="$REPO_ROOT/runs/$REPROJECTION_ROOT_NAME"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIXEL_DISPLAY_CAP="${PIXEL_DISPLAY_CAP:-0.25}"
SPATIAL_DISPLAY_CAP="${SPATIAL_DISPLAY_CAP:-0.10}"
SPATIAL_OUTPUT_ROOT="${SPATIAL_OUTPUT_ROOT:-/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1}"
FORCE="${FORCE:-0}"
PIXEL_ERRORS="${PIXEL_ERRORS:-1}"
SPATIAL_COMPARISON="${SPATIAL_COMPARISON:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$EXPORTER" ]] || { echo "Missing pixel-error exporter: $EXPORTER" >&2; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "Missing Waymo dataset root: $DATA_ROOT" >&2; exit 1; }
for root in "$SPHERE_ROOT" "$V4_ROOT" "$REPROJECTION_ROOT"; do
    [[ -d "$root" ]] || { echo "Missing run root: $root" >&2; exit 1; }
done

"$PYTHON_BIN" - "$PIXEL_DISPLAY_CAP" "$SPATIAL_DISPLAY_CAP" <<'PY'
import sys
for name, value in zip(('PIXEL_DISPLAY_CAP', 'SPATIAL_DISPLAY_CAP'), sys.argv[1:]):
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise SystemExit(f'{name} must be in (0, 1].')
PY

resolve_run() {
    local root="$1"
    local sequence_label="$2"
    local suffix="$3"
    local candidate name normalized
    local -a matches=()

    while IFS= read -r -d '' candidate; do
        name="$(basename "$candidate")"
        normalized="${name//$'\r'/}"
        if [[ "$normalized" == *"${sequence_label}_large-undistorted${suffix}" ]]; then
            matches+=("$candidate")
        fi
    done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "Expected one run in $root ending in ${sequence_label}${suffix}; found ${#matches[@]}." >&2
        printf '  %q\n' "${matches[@]}" >&2
        exit 1
    fi
    printf '%s' "${matches[0]}"
}

export_one() {
    local geometry="$1"
    local layout="$2"
    local root="$3"
    local sequence_label="$4"
    local segment_id="$5"
    local frame_start="$6"
    local suffix="$7"
    local run_dir ground_truth_dir

    run_dir="$(resolve_run "$root" "$sequence_label" "$suffix")"
    ground_truth_dir="$DATA_ROOT/segment-${segment_id}/images_undistorted"
    [[ -d "$ground_truth_dir" ]] || { echo "Missing ground truth: $ground_truth_dir" >&2; exit 1; }
    [[ -d "$run_dir/final/complete" ]] || { echo "Missing final renders: $run_dir/final/complete" >&2; exit 1; }

    if [[ "$FORCE" != "1" && -f "$run_dir/final/pixel_error/summary.json" ]]; then
        echo "[skip] $sequence_label | $geometry | $layout | final/pixel_error already exists"
        return
    fi

    echo "[pixel] $sequence_label | $geometry | $layout"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  run: %q\n  truth: %q\n' "$run_dir" "$ground_truth_dir"
        return
    fi
    "$PYTHON_BIN" "$EXPORTER" \
        --run-dir "$run_dir" \
        --ground-truth-dir "$ground_truth_dir" \
        --frame-start "$frame_start" \
        --frame-step 1 \
        --display-cap "$PIXEL_DISPLAY_CAP"
}

# Label, Waymo segment, first source frame.  The saved render count determines
# the final frame; no source images outside each trained interval are read.
SEQUENCES=(
    "W254_37_81 2547899409721197155 37"
    "W254_86_137 2547899409721197155 86"
)

if [[ "$PIXEL_ERRORS" == "1" ]]; then
    for record in "${SEQUENCES[@]}"; do
        read -r label segment_id frame_start <<<"$record"
        export_one "Sphere" "regular" "$SPHERE_ROOT" "$label" "$segment_id" "$frame_start" \
            "_radius_visible_plane_centers"
        export_one "Sphere" "Visible-UV" "$SPHERE_ROOT" "$label" "$segment_id" "$frame_start" \
            "_sph_visuv_radius_visible_plane_centers"
        export_one "Raw v4" "regular" "$V4_ROOT" "$label" "$segment_id" "$frame_start" \
            "_ellipsoid_ray_depth_proxy_yup_v4"
        export_one "Raw v4" "Visible-UV" "$V4_ROOT" "$label" "$segment_id" "$frame_start" \
            "_ellipsoid_ray_depth_proxy_yup_v4_visuv"
        export_one "Moderate-Y" "regular" "$V4_ROOT" "$label" "$segment_id" "$frame_start" \
            "_ellipsoid_ray_depth_proxy_yup_v4_moderate_y"
        export_one "Moderate-Y" "Visible-UV" "$V4_ROOT" "$label" "$segment_id" "$frame_start" \
            "_ellipsoid_ray_depth_proxy_yup_v4_moderate_y_visuv"
        export_one "Reprojection-depth v1" "regular" "$REPROJECTION_ROOT" "$label" "$segment_id" "$frame_start" \
            "_ellipsoid_reprojection_depth_v1"
        export_one "Reprojection-depth v1" "Visible-UV" "$REPROJECTION_ROOT" "$label" "$segment_id" "$frame_start" \
            "_ellipsoid_reprojection_depth_v1_visuv"
    done
fi

if [[ "$SPATIAL_COMPARISON" == "1" ]]; then
    echo "[spatial] W254 37-81 and 86-137; signed maps use blue=the first-named geometry improves, red=it worsens"
    SEQUENCES="W254_37_81 W254_86_137" \
    PARALLAX_EXTRA_RUN_ROOTS="$REPROJECTION_ROOT_NAME" \
    PARALLAX_FAMILIES="sphere raw_v4 moderate_y reprojection_v1" \
    PARALLAX_COHORTS="benchmark" \
    OUTPUT_ROOT="$SPATIAL_OUTPUT_ROOT" \
    DISPLAY_CAP="$SPATIAL_DISPLAY_CAP" \
    FORCE="$FORCE" \
    DRY_RUN="$DRY_RUN" \
    PYTHON_BIN="$PYTHON_BIN" \
        bash "$REPO_ROOT/export_ego_spatial_error_comparison.sh"

    if [[ "$DRY_RUN" != "1" ]]; then
        "$PYTHON_BIN" - "$SPATIAL_OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

for label in ('W254_37_81', 'W254_86_137'):
    summary_path = Path(sys.argv[1]) / label / 'spatial.json'
    summary = json.loads(summary_path.read_text())
    print(f"[spatial output] {label}: {summary['directory']}")
    for candidate, reference in summary['comparisons']:
        if candidate.startswith('reprojection_v1'):
            print(f'  {candidate} vs {reference}')
PY
    fi
fi

echo "Completed W254 reprojection-depth pixel/spatial comparison."
