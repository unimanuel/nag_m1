#!/usr/bin/env bash
# Train only the three intervals where the ORIGINAL reprojection/depth-v1 fit
# inflated the lateral axis to 157--171 m.  This is the focused six-job test
# of the already generated, scale-limited new_radius_scale_v1 PLY geometries:
#
#   W191 0--40, W254 0--37, W254 139--195
#   x regular and Visible-UV texture layouts.
#
# The four intervals whose scale-limited PLYs are unchanged are intentionally
# excluded.  Existing original-v1 result folders are never reused or
# overwritten: every result gets the distinct new_radius_scale_v1 name.
#
# Prerequisite:
#   bash prepare_all_ego_reprojection_depth_new_radius_scale_ellipsoids.sh
#
# Usage from the repository root after activating nag1:
#   DRY_RUN=1 bash run_pathological_ego_new_radius_scale_v1_benchmarks.sh
#   bash run_pathological_ego_new_radius_scale_v1_benchmarks.sh
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
dry_run="${DRY_RUN:-0}"
ellipsoid_root="${ELLIPSOID_ROOT:-/mnt/d/CGProjectData/waymo/ellipsoids}"
model_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"

# Dataset label, Waymo segment, first and final inclusive frame.  These are
# exactly the fits whose source-scale prior reduced an extreme lateral radius.
records=(
  "W191 1918764220984209654 0 40"
  "W254 2547899409721197155 0 37"
  "W254 2547899409721197155 139 195"
)

for record in "${records[@]}"; do
  read -r dataset segment first last <<< "$record"
  printf -v first_padded '%03d' "$((10#$first))"
  printf -v last_padded '%03d' "$((10#$last))"
  regular_config="config/waymo/large-ego/segment-${segment}_${first}_${last}_large-undistorted.yaml"
  visuv_config="${regular_config%.yaml}_sph_visuv.yaml"
  ellipsoid="$ellipsoid_root/segment-${segment}/frames_${first_padded}_${last_padded}_reprojection_depth_new_radius_scale_v1/ellipsoid.json"
  for required in "$regular_config" "$visuv_config" "$ellipsoid"; do
    [[ -f "$required" ]] || {
      echo "Missing required configuration or scale-limited geometry: $required" >&2
      echo "Run bash prepare_all_ego_reprojection_depth_new_radius_scale_ellipsoids.sh first." >&2
      exit 1
    }
  done

  "$python_bin" - "$ellipsoid" "$segment" "$first" "$last" <<'PY'
import json
import math
import sys

path, segment, first, last = sys.argv[1:]
with open(path, encoding='utf-8') as file:
    document = json.load(file)
fit = document['fit']
assert document['format_version'] == 4
assert document['sequence'] == 'segment-' + segment
assert fit['mode'] == 'reprojection_depth_proxy'
assert fit['regularizer_version'] == 'new_radius_scale'
assert fit['radius_excess_weight'] > 0.0
assert math.isfinite(fit['radius_excess_reference_max_axis_radius_m'])
assert (document['frame_slice']['start'], document['frame_slice']['stop'], document['frame_slice']['step']) == (int(first), int(last) + 1, 1)
assert all(math.isfinite(value) and value > 0 for value in fit['radii'])
assert fit['conservative_minimum_camera_surface_distance'] >= fit['target_minimum_camera_surface_distance'] - 1e-6
label = document.get('label', f"segment-{segment} frames {first}-{last}")
print(
    f"{label}: regularizer={fit['regularizer_version']}, "
    f"reference={fit['radius_excess_reference_max_axis_radius_m']:.3f}m, "
    f"center={fit['center']}, radii={fit['radii']}"
)
PY

  for mode in regular visuv; do
    config="$regular_config"
    suffix=""
    [[ "$mode" == visuv ]] && { config="$visuv_config"; suffix="_visuv"; }
    name="${dataset}_${first}_${last}_large-undistorted_ellipsoid_reprojection_depth_new_radius_scale_v1${suffix}"
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

printf '\nFinished six pathological-interval scale-limited benchmark jobs (DRY_RUN=%s).\n' "$dry_run"
