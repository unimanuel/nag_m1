#!/usr/bin/env bash
set -euo pipefail

# Recreate the original (pre-camera-centering) W191 ellipsoids.  This is the
# same default PCA-enclosing fit used for the existing W254 folders without a
# suffix: no robust quantile, AR3, camera-centering, runtime sphere cap, or
# radius override is applied here.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
processing_root="$repo_root/libs/nag-waymo-processing"
nagwp="$processing_root/.venv/bin/nagwp"
pointcloud_dir="/mnt/d/CGProjectData/waymo/pointclouds"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
segment_id="1918764220984209654"
config_dir="$repo_root/config/waymo/large-ego"
intervals=("0_40" "40_128" "128_198")
config_paths=()

if [[ ! -x "$nagwp" ]]; then
  echo "Missing processing CLI: $nagwp" >&2
  echo "Create the nag-waymo-processing virtual environment before running this script." >&2
  exit 1
fi

for interval in "${intervals[@]}"; do
  config_path="$config_dir/segment-${segment_id}_${interval}_large-undistorted.yaml"
  if [[ ! -f "$config_path" ]]; then
    echo "Missing W191 configuration: $config_path" >&2
    exit 1
  fi
  config_paths+=("$config_path")
done

# The final W191 interval reaches frame 198.  The point-cloud preparation from
# the existing W191 AR3 workflow already produced these files, so no Waymo
# download or point-cloud extraction is needed here.
if [[ ! -f "$pointcloud_dir/segment-$segment_id/198.npz" ]]; then
  echo "Missing W191 point clouds through frame 198: $pointcloud_dir/segment-$segment_id/198.npz" >&2
  echo "Run the W191 point-cloud preparation step before fitting ellipsoids." >&2
  exit 1
fi

mkdir -p "$ellipsoid_dir"

echo "Fitting original point-centred PCA enclosing ellipsoids for W191"
"$nagwp" ellipsoid \
  --config-dir "" \
  --config-paths "${config_paths[@]}" \
  --point-cloud-dir "$pointcloud_dir" \
  --nag-data-dir "$repo_root/data/datasets/waymo" \
  --target-dir "$ellipsoid_dir" \
  --segment-ids "$segment_id" \
  --correct-pose-mismatches \
  --fit-mode pca_enclosing

echo
echo "Prepared original W191 ellipsoids:"
for interval in "${intervals[@]}"; do
  start="${interval%%_*}"
  end="${interval##*_}"
  printf '  %s/segment-%s/frames_%03d_%03d/ellipsoid.json\n' \
    "$ellipsoid_dir" "$segment_id" "$((10#$start))" "$((10#$end))"
done
