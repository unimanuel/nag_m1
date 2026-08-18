#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

ellipsoid_path="/mnt/d/CGProjectData/waymo/ellipsoids/segment-2547899409721197155/frames_086_137/ellipsoid.json"
if [[ ! -f "$ellipsoid_path" ]]; then
  echo "Missing ellipsoid: $ellipsoid_path" >&2
  echo "Run ./prepare_W254_86_137_ellipsoid.sh first." >&2
  exit 1
fi

echo
echo "[1/4] W254 frames 86-137: LiDAR ellipsoid"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-2547899409721197155_86_137_large-undistorted.yaml \
  --background-plane-type nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D \
  --background-ellipsoid-path "$ellipsoid_path" \
  --background-ellipsoid-scale-margin 0.02 \
  --name W254_86_137_large-undistorted_ellipsoid

echo
echo "[2/4] W254 frames 86-137: regular sphere with visible-plane-center radius"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-2547899409721197155_86_137_large-undistorted.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W254_86_137_large-undistorted_radius_visible_plane_centers

echo
echo "[3/4] W254 frames 86-137: visible-UV sphere with the same radius rule"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-2547899409721197155_86_137_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W254_86_137_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "[4/4] W254 frames 86-137: planar background"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-2547899409721197155_86_137_large-undistorted_bgplane.yaml \
  --name W254_86_137_large-undistorted_bgplane
