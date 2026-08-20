#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

segment_id="2547899409721197155"
config_dir="config/waymo/large-ego"
base_config="$config_dir/segment-${segment_id}_86_137_large-undistorted.yaml"
visuv_config="$config_dir/segment-${segment_id}_86_137_large-undistorted_sph_visuv.yaml"
ellipsoid_path="/mnt/d/CGProjectData/waymo/ellipsoids/segment-$segment_id/frames_086_137_camera_centered_ar3_q995/ellipsoid.json"
ellipsoid_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"

# All values below are final rendered radii in meters. The reference sphere
# radius is the 58.6208 m visible-plane-center radius reported by the matching
# W254 86-137 sphere run. The current values are the robust AR3 ellipsoid after
# its former 2% runtime margin was applied.
sphere_radius="58.6208"
current_radii=("89.11614990234375" "29.70538330078125" "29.70538330078125")
sphere_equivalent_radii=("$sphere_radius" "$sphere_radius" "$sphere_radius")
halfway_radii=("73.86847495117188" "44.163091650390625" "44.163091650390625")
sphere_clipped_radii=("$sphere_radius" "${current_radii[1]}" "${current_radii[2]}")

for required_path in "$base_config" "$visuv_config" "$ellipsoid_path"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    echo "Run ./prepare_W254_86_137_centered_ar3_ellipsoid.sh first." >&2
    exit 1
  fi
done

run_number=0
total_runs=6

run_geometry() {
  local geometry_name="$1"
  local description="$2"
  shift 2
  local radii=("$@")

  run_number=$((run_number + 1))
  echo
  echo "[$run_number/$total_runs] $description"
  python nag/scripts/run_nag.py \
    --config-path "$base_config" \
    --background-plane-type "$ellipsoid_type" \
    --background-ellipsoid-path "$ellipsoid_path" \
    --background-ellipsoid-scale-margin 0 \
    --background-ellipsoid-radii-override "${radii[@]}" \
    --name "W254_86_137_large-undistorted_ellipsoid_${geometry_name}"

  run_number=$((run_number + 1))
  echo
  echo "[$run_number/$total_runs] $description with visible-UV densification"
  python nag/scripts/run_nag.py \
    --config-path "$visuv_config" \
    --background-plane-type "$ellipsoid_type" \
    --background-ellipsoid-path "$ellipsoid_path" \
    --background-ellipsoid-scale-margin 0 \
    --background-ellipsoid-radii-override "${radii[@]}" \
    --name "W254_86_137_large-undistorted_ellipsoid_${geometry_name}_visuv"
}

run_geometry \
  "sphere_equivalent" \
  "Sphere-equivalent ellipsoid: [${sphere_equivalent_radii[*]}] m" \
  "${sphere_equivalent_radii[@]}"

run_geometry \
  "halfway" \
  "Halfway sphere/current ellipsoid: [${halfway_radii[*]}] m" \
  "${halfway_radii[@]}"

run_geometry \
  "sphere_clipped_current" \
  "Current AR3 ellipsoid clipped per axis to sphere radius: [${sphere_clipped_radii[*]}] m" \
  "${sphere_clipped_radii[@]}"

echo
echo "Completed all $total_runs controlled W254 86-137 ellipsoid runs."
