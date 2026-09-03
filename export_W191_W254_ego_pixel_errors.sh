#!/usr/bin/env bash
# Export per-pixel RGB-MAE diagnostics for the completed ego-background runs.
#
# Covers seven frame intervals in each of:
#   - camera-centred, sphere-clipped AR3 ellipsoid, regular and Visible-UV;
#   - v4 ray-depth-proxy Moderate-Y ellipsoid, regular and Visible-UV;
#   - Sphere Visible-UV.
#
# The script resolves run folders by stable experiment suffixes, rather than
# timestamps. This tolerates the carriage return in some 2026-09-02 names.
#
# Usage from the repository root:
#   bash export_W191_W254_ego_pixel_errors.sh
#
# Optional environment variables:
#   DISPLAY_CAP=0.25  Fixed normalized RGB-MAE cap for every heatmap.
#   FORCE=1           Recreate an existing final/pixel_error directory.
#   DRY_RUN=1         Resolve and print all 35 jobs without writing files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORTER="${REPO_ROOT}/nag/scripts/export_pixel_error_maps.py"
DATA_ROOT="${REPO_ROOT}/data/datasets/waymo"
AR3_ROOT="${REPO_ROOT}/runs/2026-08-21_missing_ego"
SPHERE_W254_ROOT="${REPO_ROOT}/runs/2026-08-19_ellipsoid_non_center"
MODERATE_Y_ROOT="${REPO_ROOT}/runs/2026-09-02_depth_proxy_v4"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DISPLAY_CAP="${DISPLAY_CAP:-0.25}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$EXPORTER" ]]; then
    echo "Missing pixel-error exporter: $EXPORTER" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Missing Waymo dataset root: $DATA_ROOT" >&2
    exit 1
fi

if ! "$PYTHON_BIN" - "$DISPLAY_CAP" <<'PY'
import sys
cap = float(sys.argv[1])
if not 0.0 < cap <= 1.0:
    raise SystemExit("DISPLAY_CAP must be in (0, 1].")
PY
then
    exit 1
fi

resolve_run() {
    local root="$1"
    local sequence_label="$2"
    local suffix="$3"
    local -a matches=()
    local candidate

    while IFS= read -r -d '' candidate; do
        matches+=("$candidate")
    done < <(find "$root" -mindepth 1 -maxdepth 1 -type d \
        -name "*${sequence_label}*${suffix}" -print0)

    if [[ "${#matches[@]}" -ne 1 ]]; then
        echo "Expected exactly one run under $root for ${sequence_label}${suffix}; found ${#matches[@]}." >&2
        printf '  %q\n' "${matches[@]}" >&2
        exit 1
    fi
    printf '%s' "${matches[0]}"
}

export_one() {
    local mode="$1"
    local run_root="$2"
    local sequence_label="$3"
    local segment_id="$4"
    local frame_start="$5"
    local suffix="$6"
    local run_dir ground_truth_dir

    run_dir="$(resolve_run "$run_root" "$sequence_label" "$suffix")"
    ground_truth_dir="$DATA_ROOT/segment-${segment_id}/images_undistorted"

    if [[ ! -d "$ground_truth_dir" ]]; then
        echo "Missing undistorted ground-truth images: $ground_truth_dir" >&2
        exit 1
    fi
    if [[ ! -d "$run_dir/final/complete" ]]; then
        echo "Missing final complete renders: $run_dir/final/complete" >&2
        exit 1
    fi
    if [[ "$FORCE" != "1" && -f "$run_dir/final/pixel_error/summary.json" ]]; then
        echo "[skip] $mode | $sequence_label | already has final/pixel_error"
        return
    fi

    echo "[export] $mode | $sequence_label"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  run: %q\n  images: %q\n' "$run_dir" "$ground_truth_dir"
        return
    fi

    "$PYTHON_BIN" "$EXPORTER" \
        --run-dir "$run_dir" \
        --ground-truth-dir "$ground_truth_dir" \
        --frame-start "$frame_start" \
        --frame-step 1 \
        --display-cap "$DISPLAY_CAP"
}

# Arguments: short label, run-name label, Waymo segment ID, first source frame.
SEQUENCES=(
    "W191_0_40 W191_0_40 1918764220984209654 0"
    "W191_40_128 W191_40_128 1918764220984209654 40"
    "W191_128_198 W191_128_198 1918764220984209654 128"
    "W254_0_37 W254_0_37 2547899409721197155 0"
    "W254_37_81 W254_37_81 2547899409721197155 37"
    "W254_86_137 W254_86_137 2547899409721197155 86"
    "W254_139_195 W254_139_195 2547899409721197155 139"
)

for record in "${SEQUENCES[@]}"; do
    read -r label sequence_label segment_id frame_start <<<"$record"

    # W254 86-137 predates the batch and has a historical AR3 run suffix.
    if [[ "$label" == "W254_86_137" ]]; then
        ar3_regular_suffix="_ellipsoid_sphere_clipped_current"
        ar3_uv_suffix="_ellipsoid_sphere_clipped_current_visuv"
        ar3_root="$REPO_ROOT/runs/2026-08-20_ellipsoid_center_clipped"
    else
        ar3_regular_suffix="_ellipsoid_centered_ar3_sphere_clipped"
        ar3_uv_suffix="_ellipsoid_centered_ar3_sphere_clipped_visuv"
        ar3_root="$AR3_ROOT"
    fi

    export_one "AR3 regular" "$ar3_root" "$sequence_label" "$segment_id" "$frame_start" "$ar3_regular_suffix"
    export_one "AR3 Visible-UV" "$ar3_root" "$sequence_label" "$segment_id" "$frame_start" "$ar3_uv_suffix"
    export_one "Moderate-Y regular" "$MODERATE_Y_ROOT" "$sequence_label" "$segment_id" "$frame_start" "_ellipsoid_ray_depth_proxy_yup_v4_moderate_y"
    export_one "Moderate-Y Visible-UV" "$MODERATE_Y_ROOT" "$sequence_label" "$segment_id" "$frame_start" "_ellipsoid_ray_depth_proxy_yup_v4_moderate_y_visuv"

    if [[ "$label" == W191_* ]]; then
        sphere_root="$AR3_ROOT"
    else
        sphere_root="$SPHERE_W254_ROOT"
    fi
    export_one "Sphere Visible-UV" "$sphere_root" "$sequence_label" "$segment_id" "$frame_start" "_sph_visuv_radius_visible_plane_centers"
done

echo "Completed per-pixel diagnostics. All heatmaps use DISPLAY_CAP=$DISPLAY_CAP."
