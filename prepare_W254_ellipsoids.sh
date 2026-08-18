#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
processing_root="$repo_root/libs/nag-waymo-processing"
nagwp="$processing_root/.venv/bin/nagwp"
raw_dir="/mnt/d/CGProjectData/waymo/raw"
pointcloud_dir="/mnt/d/CGProjectData/waymo/pointclouds"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
segment_id="2547899409721197155"
scene_id="438"
raw_file="$raw_dir/segment-${segment_id}_1380_000_1400_000_with_camera_labels.tfrecord"
config_dir="$repo_root/config/waymo/large-ego"
intervals=(0_37 37_81 86_137 139_195)
config_paths=()

if [[ ! -x "$nagwp" ]]; then
  echo "Missing processing CLI: $nagwp" >&2
  echo "Create the nag-waymo-processing virtual environment before running this script." >&2
  exit 1
fi

for interval in "${intervals[@]}"; do
  config_path="$config_dir/segment-${segment_id}_${interval}_large-undistorted.yaml"
  if [[ ! -f "$config_path" ]]; then
    echo "Missing W254 configuration: $config_path" >&2
    exit 1
  fi
  config_paths+=("$config_path")
done

mkdir -p "$raw_dir" "$pointcloud_dir" "$ellipsoid_dir"

if [[ -f "$raw_file" ]]; then
  echo "Reusing downloaded TFRecord: $raw_file"
else
  echo "[1/3] Downloading Waymo scene ID $scene_id (segment-$segment_id)"
  "$nagwp" download \
    --scene-ids "$scene_id" \
    --target-dir "$raw_dir"
fi

echo "[2/3] Extracting registered point clouds through frame 195"
"$nagwp" pointcloud \
  --scene-ids "$scene_id" \
  --source-dir "$raw_dir" \
  --target-dir "$pointcloud_dir" \
  --max-frames 196

echo "[3/3] Fitting FRONT-visible ellipsoids for all four W254 intervals"
"$nagwp" ellipsoid \
  --config-dir "" \
  --config-paths "${config_paths[@]}" \
  --point-cloud-dir "$pointcloud_dir" \
  --nag-data-dir "$repo_root/data/datasets/waymo" \
  --target-dir "$ellipsoid_dir" \
  --segment-ids "$segment_id"

echo
echo "Prepared W254 ellipsoids:"
for interval in "${intervals[@]}"; do
  start="${interval%%_*}"
  end="${interval##*_}"
  printf '  %s/segment-%s/frames_%03d_%03d/ellipsoid.json\n' \
    "$ellipsoid_dir" "$segment_id" "$((10#$start))" "$((10#$end))"
done
