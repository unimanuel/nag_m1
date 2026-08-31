#!/usr/bin/env bash
set -euo pipefail

# Controlled physical-vertical expansion of the W254 86-137 ray-depth proxy.
#
# The source PLY uses world X as forward, world Y as vertical, and world Z as
# left/right.  Its nearly axis-aligned rotation maps local radii to those same
# axes, so the second local radius is the physical vertical radius here.
#
# Source radii:       [104.8339564,  2.5801558, 12.6104140] m
# This experiment:    [104.8339564, 12.6104140, 12.6104140] m
#
# It therefore increases only vertical Y from 2.58 m to 12.61 m, matching the
# original left/right radius and avoiding the much larger 29.7054 m test.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

segment_id="2547899409721197155"
config_dir="config/waymo/large-ego"
base_config="$config_dir/segment-${segment_id}_86_137_large-undistorted.yaml"
visuv_config="$config_dir/segment-${segment_id}_86_137_large-undistorted_sph_visuv.yaml"
ellipsoid_path="/mnt/d/CGProjectData/waymo/ellipsoids/segment-${segment_id}/frames_086_137_ray_depth_proxy_bgmask/ellipsoid.json"
ellipsoid_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"

# Local [X-forward, Y-vertical, Z-left/right] radii for this source geometry.
vertical_y_radii=("104.8339564120489" "12.610414004830998" "12.610414004830998")

for required_path in "$base_config" "$visuv_config" "$ellipsoid_path"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    echo "Run ./prepare_W254_86_137_ray_depth_ellipsoid.sh first." >&2
    exit 1
  fi
done

echo "[1/2] W254 86-137 ray-depth proxy with physical Y/vertical radius 12.6104 m"
python nag/scripts/run_nag.py \
  --config-path "$base_config" \
  --background-plane-type "$ellipsoid_type" \
  --background-ellipsoid-path "$ellipsoid_path" \
  --background-ellipsoid-scale-margin 0 \
  --background-ellipsoid-radii-override "${vertical_y_radii[@]}" \
  --name W254_86_137_large-undistorted_ellipsoid_ray_depth_proxy_y_vertical_moderate

echo
echo "[2/2] W254 86-137 ray-depth proxy with physical Y/vertical radius 12.6104 m, Visible-UV"
python nag/scripts/run_nag.py \
  --config-path "$visuv_config" \
  --background-plane-type "$ellipsoid_type" \
  --background-ellipsoid-path "$ellipsoid_path" \
  --background-ellipsoid-scale-margin 0 \
  --background-ellipsoid-radii-override "${vertical_y_radii[@]}" \
  --name W254_86_137_large-undistorted_ellipsoid_ray_depth_proxy_y_vertical_moderate_visuv

echo
echo "Completed both moderate physical-Y/vertical ray-depth-proxy ablation runs."
