#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
processing_root="$repo_root/libs/nag-waymo-processing"
nagwp="$processing_root/.venv/bin/nagwp"
raw_dir="/mnt/d/CGProjectData/waymo/raw"
pointcloud_dir="/mnt/d/CGProjectData/waymo/pointclouds"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
raw_file="$raw_dir/segment-2547899409721197155_1380_000_1400_000_with_camera_labels.tfrecord"
config_path="$repo_root/config/waymo/large-ego/segment-2547899409721197155_86_137_large-undistorted.yaml"

if [[ ! -x "$nagwp" ]]; then
  echo "Missing processing CLI: $nagwp" >&2
  echo "Create the nag-waymo-processing virtual environment before running this script." >&2
  exit 1
fi

mkdir -p "$raw_dir" "$pointcloud_dir" "$ellipsoid_dir"

if [[ -f "$raw_file" ]]; then
  echo "Reusing downloaded TFRecord: $raw_file"
else
  echo "[1/3] Downloading Waymo scene ID 438 (segment-2547899409721197155)"
  "$nagwp" download \
    --scene-ids 438 \
    --target-dir "$raw_dir"
fi

echo "[2/3] Extracting registered point clouds through frame 137"
"$nagwp" pointcloud \
  --scene-ids 438 \
  --source-dir "$raw_dir" \
  --target-dir "$pointcloud_dir" \
  --max-frames 138

echo "[3/3] Fitting the FRONT-visible frames 86-137 ellipsoid"
"$nagwp" ellipsoid \
  --config-dir "" \
  --config-paths "$config_path" \
  --point-cloud-dir "$pointcloud_dir" \
  --nag-data-dir "$repo_root/data/datasets/waymo" \
  --target-dir "$ellipsoid_dir" \
  --segment-ids 2547899409721197155

echo "Prepared: $ellipsoid_dir/segment-2547899409721197155/frames_086_137/ellipsoid.json"
