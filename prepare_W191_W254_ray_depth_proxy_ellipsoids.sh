#!/usr/bin/env bash
set -euo pipefail

# Fit one corrected-Y-up ray-depth-proxy ellipsoid for every large-ego interval
# in W191 and W254. Each completed fit writes an inspectable folder containing
# ellipsoid.ply, background_points.ply, front_points.ply, and camera_centers.ply.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
processing_root="$repo_root/libs/nag-waymo-processing"
nagwp="$processing_root/.venv/bin/nagwp"
config_dir="$repo_root/config/waymo/large-ego"
pointcloud_dir="/mnt/d/CGProjectData/waymo/pointclouds"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
data_dir="$repo_root/data/datasets/waymo"
output_suffix="ray_depth_proxy_yup_v4"
segment_ids=("1918764220984209654" "2547899409721197155")
config_paths=("$config_dir"/segment-*_large-undistorted.yaml)

if [[ ! -x "$nagwp" ]]; then
  echo "Missing processing CLI: $nagwp" >&2
  exit 1
fi
if [[ ${#config_paths[@]} -ne 7 ]]; then
  echo "Expected seven base large-ego YAML configurations; found ${#config_paths[@]}." >&2
  exit 1
fi
for required_path in "$pointcloud_dir" "$ellipsoid_dir" "$data_dir" "${config_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    exit 1
  fi
done

echo "Fitting corrected Y-up ray-depth proxy ellipsoids for W191 and W254."
echo "Output suffix: $output_suffix"
"$nagwp" ellipsoid \
  --config-dir "" \
  --config-paths "${config_paths[@]}" \
  --point-cloud-dir "$pointcloud_dir" \
  --nag-data-dir "$data_dir" \
  --target-dir "$ellipsoid_dir" \
  --segment-ids "${segment_ids[@]}" \
  --correct-pose-mismatches \
  --fit-mode ray_depth_proxy \
  --ray-depth-max-samples 24000 \
  --ray-depth-max-camera-normalized-radius 0.8 \
  --ray-depth-minimum-camera-surface-distance 2.0 \
  --output-suffix "$output_suffix"

echo
echo "Depth-proxy PLY folders are under:"
echo "  $ellipsoid_dir/segment-*/frames_*_${output_suffix}/"
