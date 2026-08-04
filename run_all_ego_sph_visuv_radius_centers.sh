#!/usr/bin/env bash
set -euo pipefail

echo
echo "Running visible-UV config: config/waymo/large-ego/segment-1918764220984209654_0_40_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-1918764220984209654_0_40_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W191_0_40_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/large-ego/segment-1918764220984209654_40_128_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-1918764220984209654_40_128_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W191_40_128_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/large-ego/segment-1918764220984209654_128_198_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-1918764220984209654_128_198_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W191_128_198_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/large-ego/segment-2547899409721197155_0_37_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-2547899409721197155_0_37_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W254_0_37_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/large-ego/segment-2547899409721197155_37_81_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-2547899409721197155_37_81_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W254_37_81_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/large-ego/segment-2547899409721197155_86_137_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-2547899409721197155_86_137_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W254_86_137_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/large-ego/segment-2547899409721197155_139_195_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/large-ego/segment-2547899409721197155_139_195_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W254_139_195_large-undistorted_sph_visuv_radius_visible_plane_centers
