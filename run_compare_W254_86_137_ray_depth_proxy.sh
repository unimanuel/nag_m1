#!/usr/bin/env bash
set -euo pipefail

# Run only after prepare_W254_86_137_ray_depth_ellipsoid.sh passes the
# geometry-diagnostic gate.  These are the regular and Visible-UV comparisons.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

segment_id="2547899409721197155"
config_dir="config/waymo/large-ego"
base_config="$config_dir/segment-${segment_id}_86_137_large-undistorted.yaml"
visuv_config="$config_dir/segment-${segment_id}_86_137_large-undistorted_sph_visuv.yaml"
ellipsoid_path="/mnt/d/CGProjectData/waymo/ellipsoids/segment-${segment_id}/frames_086_137_ray_depth_proxy_bgmask/ellipsoid.json"
ellipsoid_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"

for required_path in "$base_config" "$visuv_config" "$ellipsoid_path"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    echo "Run ./prepare_W254_86_137_ray_depth_ellipsoid.sh first." >&2
    exit 1
  fi
done

echo "[1/2] W254 86-137 ray-depth ellipsoid, regular texture"
python nag/scripts/run_nag.py \
  --config-path "$base_config" \
  --background-plane-type "$ellipsoid_type" \
  --background-ellipsoid-path "$ellipsoid_path" \
  --background-ellipsoid-scale-margin 0 \
  --name W254_86_137_large-undistorted_ellipsoid_ray_depth_proxy

echo
echo "[2/2] W254 86-137 ray-depth ellipsoid, Visible-UV texture"
python nag/scripts/run_nag.py \
  --config-path "$visuv_config" \
  --background-plane-type "$ellipsoid_type" \
  --background-ellipsoid-path "$ellipsoid_path" \
  --background-ellipsoid-scale-margin 0 \
  --name W254_86_137_large-undistorted_ellipsoid_ray_depth_proxy_visuv
