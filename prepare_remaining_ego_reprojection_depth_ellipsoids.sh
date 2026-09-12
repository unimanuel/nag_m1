#!/usr/bin/env bash
# Fit constrained reprojection/depth-v1 ellipsoids for the five ego intervals
# not used in the initial W254 37-81 / 86-137 controlled trial.
#
# This is geometry preparation only.  It reuses each interval's existing raw
# v4 road-aligned orientation and the identical objective, sampling budget,
# safety limits, and deterministic fitting settings used in the initial trial.
# It does not create training runs.
#
# Usage from the repository root after activating nag1:
#   DRY_RUN=1 bash prepare_remaining_ego_reprojection_depth_ellipsoids.sh
#   bash prepare_remaining_ego_reprojection_depth_ellipsoids.sh
#
# Set FORCE=1 only to replace an existing generated v1 geometry folder.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
dry_run="${DRY_RUN:-0}"
force="${FORCE:-0}"
diagnostic_root="${OUTPUT_ROOT:-/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1}"
ellipsoid_root="${ELLIPSOID_ROOT:-/mnt/d/CGProjectData/waymo/ellipsoids}"

# Label, Waymo segment, padded first frame, padded last frame.  These are all
# held out from the initial two-interval v1 fitting/training experiment.
records=(
  "W191_0_40 1918764220984209654 000 040"
  "W191_40_128 1918764220984209654 040 128"
  "W191_128_198 1918764220984209654 128 198"
  "W254_0_37 2547899409721197155 000 037"
  "W254_139_195 2547899409721197155 139 195"
)
labels=()
for record in "${records[@]}"; do
  read -r label _ <<< "$record"
  labels+=("$label")
done

[[ -x "$python_bin" || "$python_bin" == python || "$python_bin" == python3 ]] || {
  echo "PYTHON_BIN is not executable: $python_bin" >&2; exit 1;
}
for required in "$diagnostic_root" "$ellipsoid_root"; do
  [[ -e "$required" ]] || { echo "Missing required directory: $required" >&2; exit 1; }
done

# Prepared observations are shared with the earlier diagnostics.  In dry-run
# mode the helper validates provenance and all inputs without writing a cache.
if [[ "$dry_run" == 1 ]]; then
  SEQUENCES="${labels[*]}" DRY_RUN=1 PYTHON_BIN="$python_bin" bash prepare_ego_parallax_diagnostics.sh
else
  SEQUENCES="${labels[*]}" PYTHON_BIN="$python_bin" bash prepare_ego_parallax_diagnostics.sh
fi

for record in "${records[@]}"; do
  read -r label segment first last <<< "$record"
  prepared="$diagnostic_root/$label/prepared.json"
  source="$ellipsoid_root/segment-$segment/frames_${first}_${last}_ray_depth_proxy_yup_v4/ellipsoid.json"
  output="$ellipsoid_root/segment-$segment/frames_${first}_${last}_reprojection_depth_v1"
  for required in "$prepared" "$source"; do
    [[ -f "$required" ]] || { echo "$label: missing required file: $required" >&2; exit 1; }
  done

  command=("$python_bin" nag/scripts/fit_reprojection_ellipsoid.py
    --label "$label"
    --prepared "$prepared"
    --source-ellipsoid "$source"
    --output-dir "$output"
    --max-pairs 24000
    --max-depth-rays 14000
    --max-camera-normalized-radius 0.65
    --minimum-camera-surface-distance 6.0
    --minimum-axis-radius 8.0
    --max-axis-ratio 10.0
    --reprojection-weight 1.0
    --depth-weight 0.15
    --max-iterations 200)
  [[ "$force" == 1 ]] && command+=(--overwrite)
  [[ "$dry_run" == 1 ]] && command+=(--dry-run)

  printf '\n[%s]\n' "$label"
  printf '%q ' "${command[@]}"
  printf '\n'
  "${command[@]}"
done

printf '\nFinished held-out reprojection/depth geometry preparation (DRY_RUN=%s).\n' "$dry_run"
printf 'Inspect every ellipsoid.ply and reprojection_fit_diagnostics.json before training.\n'
