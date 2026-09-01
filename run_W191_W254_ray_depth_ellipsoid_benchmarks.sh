#!/usr/bin/env bash
set -euo pipefail

# Run both corrected ray-depth ellipsoid geometries for every W191 and W254
# large-ego interval, with both regular and Visible-UV background texture
# layouts.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

config_dir="config/waymo/large-ego"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
ellipsoid_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"
source_suffix="ray_depth_proxy_yup_v4"
moderate_suffix="y_vertical_lateral_matched"
config_paths=("$config_dir"/segment-*_large-undistorted.yaml)

if [[ ${#config_paths[@]} -ne 7 ]]; then
  echo "Expected seven base large-ego YAML configurations; found ${#config_paths[@]}." >&2
  exit 1
fi

for config_path in "${config_paths[@]}"; do
  config_base="$(basename "$config_path" .yaml)"
  if [[ ! "$config_base" =~ ^segment-([0-9]+)_([0-9]+)_([0-9]+)_large-undistorted$ ]]; then
    echo "Cannot derive segment and frame interval from: $config_path" >&2
    exit 1
  fi
  segment_id="${BASH_REMATCH[1]}"
  first="${BASH_REMATCH[2]}"
  last="${BASH_REMATCH[3]}"
  printf -v first_padded '%03d' "$((10#$first))"
  printf -v last_padded '%03d' "$((10#$last))"
  run_base="$(awk -F ': ' '/^  name: / {print $2; exit}' "$config_path")"
  if [[ -z "$run_base" ]]; then
    echo "Could not read NAGConfig.name from: $config_path" >&2
    exit 1
  fi

  depth_json="$ellipsoid_dir/segment-$segment_id/frames_${first_padded}_${last_padded}_${source_suffix}/ellipsoid.json"
  moderate_json="$ellipsoid_dir/segment-$segment_id/frames_${first_padded}_${last_padded}_${source_suffix}_${moderate_suffix}/ellipsoid.json"
  visuv_config="${config_path%.yaml}_sph_visuv.yaml"
  for required_path in "$depth_json" "$moderate_json" "$visuv_config"; do
    if [[ ! -f "$required_path" ]]; then
      echo "Missing required geometry or Visible-UV configuration: $required_path" >&2
      echo "Run the preparation and moderate-Y PLY scripts first." >&2
      exit 1
    fi
  done

  echo
  echo "[$run_base] corrected depth-proxy ellipsoid, regular texture"
  python nag/scripts/run_nag.py \
    --config-path "$config_path" \
    --background-plane-type "$ellipsoid_type" \
    --background-ellipsoid-path "$depth_json" \
    --background-ellipsoid-scale-margin 0 \
    --name "${run_base}_ellipsoid_ray_depth_proxy_yup_v4"

  echo
  echo "[$run_base] corrected depth-proxy ellipsoid, Visible-UV texture"
  python nag/scripts/run_nag.py \
    --config-path "$visuv_config" \
    --background-plane-type "$ellipsoid_type" \
    --background-ellipsoid-path "$depth_json" \
    --background-ellipsoid-scale-margin 0 \
    --name "${run_base}_ellipsoid_ray_depth_proxy_yup_v4_visuv"

  echo
  echo "[$run_base] moderate-Y depth-proxy ellipsoid, regular texture"
  python nag/scripts/run_nag.py \
    --config-path "$config_path" \
    --background-plane-type "$ellipsoid_type" \
    --background-ellipsoid-path "$moderate_json" \
    --background-ellipsoid-scale-margin 0 \
    --name "${run_base}_ellipsoid_ray_depth_proxy_yup_v4_moderate_y"

  echo
  echo "[$run_base] moderate-Y depth-proxy ellipsoid, Visible-UV texture"
  python nag/scripts/run_nag.py \
    --config-path "$visuv_config" \
    --background-plane-type "$ellipsoid_type" \
    --background-ellipsoid-path "$moderate_json" \
    --background-ellipsoid-scale-margin 0 \
    --name "${run_base}_ellipsoid_ray_depth_proxy_yup_v4_moderate_y_visuv"
done

echo
echo "Completed 28 ray-depth ellipsoid runs: regular and Visible-UV for both geometries."
