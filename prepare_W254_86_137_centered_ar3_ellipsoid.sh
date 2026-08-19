#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
processing_root="$repo_root/libs/nag-waymo-processing"
nagwp="$processing_root/.venv/bin/nagwp"
pointcloud_dir="/mnt/d/CGProjectData/waymo/pointclouds"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
segment_id="2547899409721197155"
config_path="$repo_root/config/waymo/large-ego/segment-${segment_id}_86_137_large-undistorted.yaml"
output_suffix="camera_centered_ar3_q995"
output_path="$ellipsoid_dir/segment-$segment_id/frames_086_137_$output_suffix/ellipsoid.json"

if [[ ! -x "$nagwp" ]]; then
  echo "Missing processing CLI: $nagwp" >&2
  exit 1
fi
if [[ ! -f "$config_path" ]]; then
  echo "Missing W254 configuration: $config_path" >&2
  exit 1
fi
if [[ ! -f "$pointcloud_dir/segment-$segment_id/137.npz" ]]; then
  echo "Missing extracted W254 point clouds through frame 137." >&2
  echo "Run ./prepare_W254_ellipsoids.sh first." >&2
  exit 1
fi

mkdir -p "$ellipsoid_dir"

"$nagwp" ellipsoid \
  --config-dir "" \
  --config-paths "$config_path" \
  --point-cloud-dir "$pointcloud_dir" \
  --nag-data-dir "$repo_root/data/datasets/waymo" \
  --target-dir "$ellipsoid_dir" \
  --segment-ids "$segment_id" \
  --correct-pose-mismatches \
  --fit-mode camera_centered_robust \
  --robust-quantile 0.995 \
  --max-axis-ratio 3 \
  --max-camera-normalized-radius 0.4 \
  --minimum-camera-surface-distance 15 \
  --minimum-axis-radius 15 \
  --output-suffix "$output_suffix"

echo "Prepared: $output_path"
