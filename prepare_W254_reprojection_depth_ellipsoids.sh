#!/usr/bin/env bash
# Build the first constrained cross-frame reprojection/depth ellipsoids.
#
# This is a geometry-only stage: it does not start NAG training.  It uses the
# full cached static-background pairs already produced by parallax diagnostics,
# rather than raw Waymo camera pixel labels.  The two intervals were selected
# because W254 37-81 favors raw v4 while W254 86-137 exposes its failure mode.
#
# Usage from the repository root after activating nag1:
#   DRY_RUN=1 bash prepare_W254_reprojection_depth_ellipsoids.sh
#   bash prepare_W254_reprojection_depth_ellipsoids.sh
#
# Set FORCE=1 only to deliberately replace these generated candidate folders.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
dry_run="${DRY_RUN:-0}"
force="${FORCE:-0}"
segment="2547899409721197155"
diagnostic_root="${OUTPUT_ROOT:-/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1}"
ellipsoid_root="${ELLIPSOID_ROOT:-/mnt/d/CGProjectData/waymo/ellipsoids}"
labels=("W254_37_81" "W254_86_137")
intervals=("037 081" "086 137")

[[ -x "$python_bin" || "$python_bin" == python || "$python_bin" == python3 ]] || {
  echo "PYTHON_BIN is not executable: $python_bin" >&2; exit 1;
}
for required in "$diagnostic_root" "$ellipsoid_root/segment-$segment"; do
  [[ -e "$required" ]] || { echo "Missing required directory: $required" >&2; exit 1; }
done

# This reuses the completed prepared caches when their provenance is current;
# in DRY_RUN mode it validates the same files without writing anything.
if [[ "$dry_run" == 1 ]]; then
  SEQUENCES="${labels[*]}" DRY_RUN=1 PYTHON_BIN="$python_bin" bash prepare_ego_parallax_diagnostics.sh
else
  SEQUENCES="${labels[*]}" PYTHON_BIN="$python_bin" bash prepare_ego_parallax_diagnostics.sh
fi

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  read -r first last <<< "${intervals[$index]}"
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

printf '\nFinished constrained reprojection/depth geometry preparation (DRY_RUN=%s).\n' "$dry_run"
printf 'Inspect each output folder for ellipsoid.ply and reprojection_fit_diagnostics.json before training.\n'
