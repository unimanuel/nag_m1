#!/usr/bin/env bash
set -euo pipefail

# Fit the first geometry-driven background proxy: a free-centre, road-aligned
# ellipsoid whose camera-ray intersections are fitted to static FRONT LiDAR
# depths.  It reuses the already extracted W254 point clouds and NAG masks.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
processing_root="$repo_root/libs/nag-waymo-processing"
nagwp="$processing_root/.venv/bin/nagwp"
pointcloud_dir="/mnt/d/CGProjectData/waymo/pointclouds"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
segment_id="2547899409721197155"
config_path="$repo_root/config/waymo/large-ego/segment-${segment_id}_86_137_large-undistorted.yaml"
output_suffix="ray_depth_proxy_bgmask"
output_dir="$ellipsoid_dir/segment-${segment_id}/frames_086_137_${output_suffix}"

if [[ ! -x "$nagwp" ]]; then
  echo "Missing processing CLI: $nagwp" >&2
  exit 1
fi
for required_path in \
  "$config_path" \
  "$pointcloud_dir/segment-$segment_id/137.npz" \
  "$repo_root/data/datasets/waymo/segment-$segment_id/masks/img_86_ov_0.png"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 1
  fi
done

"$nagwp" ellipsoid \
  --config-dir "" \
  --config-paths "$config_path" \
  --point-cloud-dir "$pointcloud_dir" \
  --nag-data-dir "$repo_root/data/datasets/waymo" \
  --target-dir "$ellipsoid_dir" \
  --segment-ids "$segment_id" \
  --correct-pose-mismatches \
  --fit-mode ray_depth_proxy \
  --ray-depth-max-samples 24000 \
  --ray-depth-max-camera-normalized-radius 0.8 \
  --ray-depth-minimum-camera-surface-distance 2.0 \
  --output-suffix "$output_suffix"

echo
echo "Prepared ray-depth proxy ellipsoid: $output_dir"
echo "Inspect: $output_dir/ray_depth_diagnostics.json"
echo "Inspect: $output_dir/ray_depth_error_by_frame.csv"
