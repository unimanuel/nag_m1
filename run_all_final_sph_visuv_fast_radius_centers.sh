#!/usr/bin/env bash
set -euo pipefail

# Static-camera visible-UV runs, ordered from the smallest estimated workload
# to the larger known-fast slices. This matches the six sequences in the
# regular- and double-resolution baseline runner scripts.

echo
echo "Running visible-UV config: W975 175-195"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-9758342966297863572_175_195_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W975_175_195_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running fast visible-UV config: W125 124-149"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-12511696717465549299_124_149_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W125_124_149_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running fast visible-UV config: W141 174-197"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-14133920963894906769_174_197_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W141_174_197_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running fast visible-UV config: W203 3-58"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-2036908808378190283_3_58_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W203_3_58_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running fast visible-UV config: W344 95-135"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-3441838785578020259_95_135_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W344_95_135_large-undistorted_sph_visuv_radius_visible_plane_centers

echo
echo "Running fast visible-UV config: W324 161-197"
python nag/scripts/run_nag.py \
  --config-path config/waymo/final/segment-3247914894323111613_161_197_large-undistorted_sph_visuv.yaml \
  --background-sphere-radius-from-visible-plane-centers \
  --name W324_161_197_large-undistorted_sph_visuv_radius_visible_plane_centers
