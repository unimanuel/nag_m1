#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

segment_id="2547899409721197155"
config_dir="config/waymo/large-ego"
ellipsoid_root="/mnt/d/CGProjectData/waymo/ellipsoids/segment-$segment_id"
ellipsoid_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"
intervals=(0_37 37_81 86_137 139_195)
run_number=0
total_runs=$((5 * ${#intervals[@]}))

# Check every required configuration and fitted ellipsoid before starting the
# first long training run.
for interval in "${intervals[@]}"; do
  start="${interval%%_*}"
  end="${interval##*_}"
  base_config="$config_dir/segment-${segment_id}_${interval}_large-undistorted.yaml"
  visuv_config="$config_dir/segment-${segment_id}_${interval}_large-undistorted_sph_visuv.yaml"
  plane_config="$config_dir/segment-${segment_id}_${interval}_large-undistorted_bgplane.yaml"
  ellipsoid_path="$(printf '%s/frames_%03d_%03d/ellipsoid.json' \
    "$ellipsoid_root" "$((10#$start))" "$((10#$end))")"

  for required_path in "$base_config" "$visuv_config" "$plane_config" "$ellipsoid_path"; do
    if [[ ! -f "$required_path" ]]; then
      echo "Missing required input: $required_path" >&2
      echo "Run ./prepare_W254_ellipsoids.sh first if an ellipsoid is missing." >&2
      exit 1
    fi
  done
done

run_training() {
  local description="$1"
  shift
  run_number=$((run_number + 1))
  echo
  echo "[$run_number/$total_runs] $description"
  python nag/scripts/run_nag.py "$@"
}

for interval in "${intervals[@]}"; do
  start="${interval%%_*}"
  end="${interval##*_}"
  base_config="$config_dir/segment-${segment_id}_${interval}_large-undistorted.yaml"
  visuv_config="$config_dir/segment-${segment_id}_${interval}_large-undistorted_sph_visuv.yaml"
  plane_config="$config_dir/segment-${segment_id}_${interval}_large-undistorted_bgplane.yaml"
  ellipsoid_path="$(printf '%s/frames_%03d_%03d/ellipsoid.json' \
    "$ellipsoid_root" "$((10#$start))" "$((10#$end))")"
  run_prefix="W254_${interval}_large-undistorted"

  run_training "W254 frames ${start}-${end}: LiDAR ellipsoid" \
    --config-path "$base_config" \
    --background-plane-type "$ellipsoid_type" \
    --background-ellipsoid-path "$ellipsoid_path" \
    --background-ellipsoid-scale-margin 0.02 \
    --name "${run_prefix}_ellipsoid"

  run_training "W254 frames ${start}-${end}: visible-UV LiDAR ellipsoid" \
    --config-path "$visuv_config" \
    --background-plane-type "$ellipsoid_type" \
    --background-ellipsoid-path "$ellipsoid_path" \
    --background-ellipsoid-scale-margin 0.02 \
    --name "${run_prefix}_ellipsoid_visuv"

  run_training "W254 frames ${start}-${end}: regular sphere" \
    --config-path "$base_config" \
    --background-sphere-radius-from-visible-plane-centers \
    --name "${run_prefix}_radius_visible_plane_centers"

  run_training "W254 frames ${start}-${end}: visible-UV sphere" \
    --config-path "$visuv_config" \
    --background-sphere-radius-from-visible-plane-centers \
    --name "${run_prefix}_sph_visuv_radius_visible_plane_centers"

  run_training "W254 frames ${start}-${end}: planar background" \
    --config-path "$plane_config" \
    --name "${run_prefix}_bgplane"
done

echo
echo "Completed all $total_runs W254 background-comparison runs."
