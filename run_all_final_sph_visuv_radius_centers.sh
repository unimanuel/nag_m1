#!/usr/bin/env bash
set -euo pipefail

echo
echo "Running visible-UV config: config/waymo/final/segment-2036908808378190283_3_58_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-2036908808378190283_3_58_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W203_3_58_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/final/segment-3247914894323111613_0_42_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-3247914894323111613_0_42_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W324_0_42_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/final/segment-3247914894323111613_42_96_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-3247914894323111613_42_96_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W324_42_96_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/final/segment-3247914894323111613_96_161_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-3247914894323111613_96_161_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W324_96_161_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running visible-UV config: config/waymo/final/segment-3247914894323111613_161_197_large-undistorted_sph_visuv.yaml"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-3247914894323111613_161_197_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W324_161_197_large-undistorted_sph_visuv_radius_visible_plane_centers
