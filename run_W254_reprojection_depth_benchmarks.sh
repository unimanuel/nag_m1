#!/usr/bin/env bash
# Train the first constrained cross-frame reprojection/depth ellipsoid trial.
#
# This runs only W254 37-81 and W254 86-137, each in regular and Visible-UV
# modes.  The fitted geometry and legacy texture orientation are identical
# within each regular/Visible-UV pair; only the texture allocation differs.
#
# Prerequisite:
#   bash prepare_W254_reprojection_depth_ellipsoids.sh
#
# Usage:
#   DRY_RUN=1 bash run_W254_reprojection_depth_benchmarks.sh
#   bash run_W254_reprojection_depth_benchmarks.sh
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
dry_run="${DRY_RUN:-0}"
segment="2547899409721197155"
ellipsoid_root="${ELLIPSOID_ROOT:-/mnt/d/CGProjectData/waymo/ellipsoids}"
model_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"
intervals=("37 81" "86 137")

for interval in "${intervals[@]}"; do
  read -r first last <<< "$interval"
  printf -v first_padded '%03d' "$((10#$first))"
  printf -v last_padded '%03d' "$((10#$last))"
  regular_config="config/waymo/large-ego/segment-${segment}_${first}_${last}_large-undistorted.yaml"
  visuv_config="${regular_config%.yaml}_sph_visuv.yaml"
  ellipsoid="$ellipsoid_root/segment-${segment}/frames_${first_padded}_${last_padded}_reprojection_depth_v1/ellipsoid.json"
  for required in "$regular_config" "$visuv_config" "$ellipsoid"; do
    [[ -f "$required" ]] || {
      echo "Missing required configuration or prepared geometry: $required" >&2
      echo "Run bash prepare_W254_reprojection_depth_ellipsoids.sh first." >&2
      exit 1
    }
  done
  "$python_bin" - "$ellipsoid" "$segment" "$first" "$last" <<'PY'
import json, math, sys
path, segment, first, last = sys.argv[1:]
with open(path, encoding='utf-8') as f:
    doc = json.load(f)
fit = doc['fit']
assert doc['format_version'] == 4
assert doc['sequence'] == 'segment-' + segment
assert fit['mode'] == 'reprojection_depth_proxy'
assert (doc['frame_slice']['start'], doc['frame_slice']['stop'], doc['frame_slice']['step']) == (int(first), int(last) + 1, 1)
assert all(math.isfinite(value) and value > 0 for value in fit['radii'])
assert fit['conservative_minimum_camera_surface_distance'] >= fit['target_minimum_camera_surface_distance'] - 1e-6
print(f'W254 {first}-{last}: center={fit["center"]}, radii={fit["radii"]}, clearance={fit["conservative_minimum_camera_surface_distance"]:.3f}m')
PY

  for mode in regular visuv; do
    config="$regular_config"
    suffix=""
    [[ "$mode" == visuv ]] && { config="$visuv_config"; suffix="_visuv"; }
    name="W254_${first}_${last}_large-undistorted_ellipsoid_reprojection_depth_v1${suffix}"
    command=("$python_bin" nag/scripts/run_nag.py
      --config-path "$config"
      --background-plane-type "$model_type"
      --background-ellipsoid-path "$ellipsoid"
      --background-ellipsoid-scale-margin 0
      --background-ellipsoid-texture-orientation legacy
      --name "$name")
    printf '\n[%s]\n' "$name"
    if [[ "$dry_run" == 1 ]]; then
      printf '%q ' "${command[@]}"
      printf '\n'
    else
      "${command[@]}"
    fi
  done
done

printf '\nFinished four constrained reprojection/depth benchmark jobs (DRY_RUN=%s).\n' "$dry_run"
