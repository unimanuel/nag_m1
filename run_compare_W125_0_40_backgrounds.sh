#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

echo
echo "[1/3] W125 frames 0-40: regular sphere with visible-plane-center radius"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-12511696717465549299_0_40_large-undistorted.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W125_0_40_large-undistorted_radius_visible_plane_centers

echo
echo "[2/3] W125 frames 0-40: visible-UV sphere with the same radius rule"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-12511696717465549299_0_40_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W125_0_40_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "[3/3] W125 frames 0-40: planar background"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-12511696717465549299_0_40_large-undistorted_bgplane.yaml \
  --name W125_0_40_large-undistorted_bgplane
