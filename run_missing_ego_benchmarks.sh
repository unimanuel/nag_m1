#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

processing_root="$repo_root/libs/nag-waymo-processing"
nagwp="$processing_root/.venv/bin/nagwp"
raw_dir="/mnt/d/CGProjectData/waymo/raw"
pointcloud_dir="/mnt/d/CGProjectData/waymo/pointclouds"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
config_dir="$repo_root/config/waymo/large-ego"
runs_dir="$repo_root/runs"
ellipsoid_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"
output_suffix="camera_centered_ar3_q995"

if [[ ! -x "$nagwp" ]]; then
  echo "Missing processing CLI: $nagwp" >&2
  echo "Create the nag-waymo-processing virtual environment before running this script." >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1; then
  echo "python is not available. Activate the nag1 Conda environment first." >&2
  exit 1
fi

mkdir -p "$raw_dir" "$pointcloud_dir" "$ellipsoid_dir"

prepare_segment() {
  local label="$1"
  local segment_id="$2"
  local scene_id="$3"
  local raw_filename="$4"
  local max_frames="$5"
  shift 5
  local intervals=("$@")
  local raw_path="$raw_dir/$raw_filename"
  local segment_pointcloud_dir="$pointcloud_dir/segment-$segment_id"
  local config_paths=()
  local interval config_path frame_index
  local pointclouds_complete=true

  echo
  echo "Preparing $label (segment-$segment_id)"
  if [[ -f "$raw_path" ]]; then
    echo "Reusing downloaded TFRecord: $raw_path"
  else
    echo "Downloading Waymo scene ID $scene_id"
    "$nagwp" download \
      --scene-ids "$scene_id" \
      --target-dir "$raw_dir"
  fi

  for ((frame_index = 0; frame_index < max_frames; frame_index++)); do
    if [[ ! -f "$segment_pointcloud_dir/$frame_index.npz" ]]; then
      pointclouds_complete=false
      break
    fi
  done
  if [[ "$pointclouds_complete" == true ]]; then
    echo "Reusing all $max_frames extracted point-cloud frames."
  else
    echo "Extracting registered point clouds through frame $((max_frames - 1))"
    "$nagwp" pointcloud \
      --scene-ids "$scene_id" \
      --source-dir "$raw_dir" \
      --target-dir "$pointcloud_dir" \
      --max-frames "$max_frames"
  fi

  for interval in "${intervals[@]}"; do
    config_path="$config_dir/segment-${segment_id}_${interval}_large-undistorted.yaml"
    if [[ ! -f "$config_path" ]]; then
      echo "Missing ego configuration: $config_path" >&2
      exit 1
    fi
    config_paths+=("$config_path")
  done

  echo "Fitting or reusing camera-centered robust AR3 ellipsoids"
  "$nagwp" ellipsoid \
    --config-dir "" \
    --config-paths "${config_paths[@]}" \
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
}

completed_run_directories=()
while IFS= read -r -d '' exit_path; do
  if grep -q '^Exit code: 0\.' "$exit_path"; then
    completed_run_directories+=("$(basename "$(dirname "$exit_path")")")
  fi
done < <(
  find "$runs_dir" -mindepth 3 -maxdepth 3 -type f -name exit_000.txt -print0 2>/dev/null
)

has_completed_run() {
  local candidate completed_directory
  for candidate in "$@"; do
    [[ -n "$candidate" ]] || continue
    for completed_directory in "${completed_run_directories[@]}"; do
      if [[ "$completed_directory" == *_"$candidate" ]]; then
        return 0
      fi
    done
  done
  return 1
}

run_if_missing() {
  local run_name="$1"
  local existing_alias="$2"
  shift 2
  if has_completed_run "$run_name" "$existing_alias"; then
    echo "Skipping completed benchmark: $run_name"
    return
  fi

  echo
  echo "Running missing benchmark: $run_name"
  "$@"
  completed_run_directories+=("completed_now_$run_name")
}

prepare_segment \
  "W191" \
  "1918764220984209654" \
  "402" \
  "segment-1918764220984209654_5680_000_5700_000_with_camera_labels.tfrecord" \
  "199" \
  "0_40" "40_128" "128_198"

prepare_segment \
  "W254" \
  "2547899409721197155" \
  "438" \
  "segment-2547899409721197155_1380_000_1400_000_with_camera_labels.tfrecord" \
  "196" \
  "0_37" "37_81" "86_137" "139_195"

# The radii below are the final visible-plane-center sphere radii recorded by
# the already-completed regular sphere benchmarks for these exact intervals.
labels=(
  "W191" "W191" "W191"
  "W254" "W254" "W254" "W254"
)
segment_ids=(
  "1918764220984209654" "1918764220984209654" "1918764220984209654"
  "2547899409721197155" "2547899409721197155" "2547899409721197155" "2547899409721197155"
)
intervals=(
  "0_40" "40_128" "128_198"
  "0_37" "37_81" "86_137" "139_195"
)
sphere_radii=(
  "52.4692" "28.6621" "18.2372"
  "30.3448" "44.4949" "58.6208" "31.9631"
)

total_intervals="${#intervals[@]}"
for ((index = 0; index < total_intervals; index++)); do
  label="${labels[$index]}"
  segment_id="${segment_ids[$index]}"
  interval="${intervals[$index]}"
  sphere_radius="${sphere_radii[$index]}"
  base_config="$config_dir/segment-${segment_id}_${interval}_large-undistorted.yaml"
  visuv_config="$config_dir/segment-${segment_id}_${interval}_large-undistorted_sph_visuv.yaml"
  start="${interval%%_*}"
  last="${interval##*_}"
  ellipsoid_path="$ellipsoid_dir/segment-$segment_id/frames_$(printf '%03d' "$((10#$start))")_$(printf '%03d' "$((10#$last))")_${output_suffix}/ellipsoid.json"
  sphere_name="${label}_${interval}_large-undistorted_radius_visible_plane_centers"
  sphere_visuv_name="${label}_${interval}_large-undistorted_sph_visuv_radius_visible_plane_centers"
  ellipsoid_name="${label}_${interval}_large-undistorted_ellipsoid_centered_ar3_sphere_clipped"
  ellipsoid_visuv_name="${ellipsoid_name}_visuv"
  ellipsoid_existing_alias=""
  ellipsoid_visuv_existing_alias=""

  for required_path in "$base_config" "$visuv_config" "$ellipsoid_path"; do
    if [[ ! -f "$required_path" ]]; then
      echo "Missing required benchmark input: $required_path" >&2
      exit 1
    fi
  done

  # The original W254 86-137 controlled runs predate the standardized names.
  if [[ "$label" == "W254" && "$interval" == "86_137" ]]; then
    ellipsoid_existing_alias="W254_86_137_large-undistorted_ellipsoid_sphere_clipped_current"
    ellipsoid_visuv_existing_alias="W254_86_137_large-undistorted_ellipsoid_sphere_clipped_current_visuv"
  fi

  run_if_missing "$sphere_name" "" \
    python nag/scripts/run_nag.py \
      --config-path "$base_config" \
      --background-sphere-radius-from-visible-plane-centers \
      --name "$sphere_name"

  run_if_missing "$sphere_visuv_name" "" \
    python nag/scripts/run_nag.py \
      --config-path "$visuv_config" \
      --background-sphere-radius-from-visible-plane-centers \
      --name "$sphere_visuv_name"

  run_if_missing "$ellipsoid_name" "$ellipsoid_existing_alias" \
    python nag/scripts/run_nag.py \
      --config-path "$base_config" \
      --background-plane-type "$ellipsoid_type" \
      --background-ellipsoid-path "$ellipsoid_path" \
      --background-ellipsoid-scale-margin 0.02 \
      --background-ellipsoid-max-radius "$sphere_radius" \
      --name "$ellipsoid_name"

  run_if_missing "$ellipsoid_visuv_name" "$ellipsoid_visuv_existing_alias" \
    python nag/scripts/run_nag.py \
      --config-path "$visuv_config" \
      --background-plane-type "$ellipsoid_type" \
      --background-ellipsoid-path "$ellipsoid_path" \
      --background-ellipsoid-scale-margin 0.02 \
      --background-ellipsoid-max-radius "$sphere_radius" \
      --name "$ellipsoid_visuv_name"
done

echo
echo "All missing ego benchmarks are complete."
