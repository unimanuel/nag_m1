#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

segment_id="2547899409721197155"
config_dir="config/waymo/large-ego"
base_config="$config_dir/segment-${segment_id}_86_137_large-undistorted.yaml"
visuv_config="$config_dir/segment-${segment_id}_86_137_large-undistorted_sph_visuv.yaml"
ellipsoid_path="/mnt/d/CGProjectData/waymo/ellipsoids/segment-$segment_id/frames_086_137_camera_centered_enclosing_all_points/ellipsoid.json"
ellipsoid_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"

# The existing comparison runs provide two corners of this cross-section test:
#   new all-point clipped: [58.6208, 49.0124,  7.0083]
#   sphere-clipped AR3:    [58.6208, 29.7054, 29.7054]
# These two missing corners independently test the wider second axis and the
# thinner third axis while preserving the same center, orientation, and first axis.
wide_thick_radii=("58.6208" "49.012413024902344" "29.70538330078125")
narrow_thin_radii=("58.6208" "29.70538330078125" "7.00830078125")

for required_path in "$base_config" "$visuv_config" "$ellipsoid_path"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required input: $required_path" >&2
    echo "Run ./prepare_W254_86_137_camera_centered_enclosing_ellipsoid.sh first." >&2
    exit 1
  fi
done

run_number=0
total_runs=4

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
  "cross_section_wide_thick" \
  "Wide-and-thick cross-section: [${wide_thick_radii[*]}] m" \
  "${wide_thick_radii[@]}"

run_geometry \
  "cross_section_narrow_thin" \
  "Narrow-and-thin cross-section: [${narrow_thin_radii[*]}] m" \
  "${narrow_thin_radii[@]}"

echo
echo "Completed all $total_runs W254 86-137 cross-section ablation runs."
