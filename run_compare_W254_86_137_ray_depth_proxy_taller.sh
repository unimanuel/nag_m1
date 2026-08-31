#!/usr/bin/env bash
set -euo pipefail

# Controlled Z/lateral-only ablation of the W254 86-137 ray-depth proxy.
#
# In these NAG-world PLYs the source rotation maps the three local radii to
# approximately [world X forward, world Y vertical, world Z left/right].
# This script leaves local Y unchanged and enlarges local Z, so it is a
# left/right-width ablation, not the vertical test.
# This experiment preserves the fitted center, orientation, forward radius,
# vertical radius, point/mask selection, and zero runtime scale margin.  Only
# the local Z/left-right radius is increased to 29.7053833 m.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

segment_id="2547899409721197155"
config_dir="config/waymo/large-ego"
base_config="$config_dir/segment-${segment_id}_86_137_large-undistorted.yaml"
visuv_config="$config_dir/segment-${segment_id}_86_137_large-undistorted_sph_visuv.yaml"
ellipsoid_path="/mnt/d/CGProjectData/waymo/ellipsoids/segment-${segment_id}/frames_086_137_ray_depth_proxy_bgmask/ellipsoid.json"
ellipsoid_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"

# NAG interprets these in the source ellipsoid's local road-aligned frame.
taller_radii=("104.8339564120489" "2.5801557803285866" "29.70538330078125")

for required_path in "$base_config" "$visuv_config" "$ellipsoid_path"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    echo "Run ./prepare_W254_86_137_ray_depth_ellipsoid.sh first." >&2
    exit 1
  fi
done

echo "[1/2] W254 86-137 ray-depth proxy with local Z/left-right radius 29.7054 m"
python nag/scripts/run_nag.py \
  --config-path "$base_config" \
  --background-plane-type "$ellipsoid_type" \
  --background-ellipsoid-path "$ellipsoid_path" \
  --background-ellipsoid-scale-margin 0 \
  --background-ellipsoid-radii-override "${taller_radii[@]}" \
  --name W254_86_137_large-undistorted_ellipsoid_ray_depth_proxy_taller

echo
echo "[2/2] W254 86-137 ray-depth proxy with local Z/left-right radius 29.7054 m, Visible-UV"
python nag/scripts/run_nag.py \
  --config-path "$visuv_config" \
  --background-plane-type "$ellipsoid_type" \
  --background-ellipsoid-path "$ellipsoid_path" \
  --background-ellipsoid-scale-margin 0 \
  --background-ellipsoid-radii-override "${taller_radii[@]}" \
  --name W254_86_137_large-undistorted_ellipsoid_ray_depth_proxy_taller_visuv

echo
echo "Completed both local-Z/left-right ray-depth-proxy ablation runs."
