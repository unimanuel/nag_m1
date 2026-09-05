#!/usr/bin/env bash
# Reuse the existing v4 Moderate-Y geometry. Only its texture coordinates change.
# Default: two regular-layout runs, W254 37-81 and 86-137.
#   DRY_RUN=1 bash run_W254_fixed_geometry_texture_orientation.sh
#   bash run_W254_fixed_geometry_texture_orientation.sh
# Optional full ablation (six runs):
#   TEXTURE_MODES="legacy world_y world_y_auto_seam" bash run_W254_fixed_geometry_texture_orientation.sh
# PYTHON_BIN defaults to the current environment's python (activate nag1).
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
dry_run="${DRY_RUN:-0}"
read -r -a modes <<< "${TEXTURE_MODES:-world_y_auto_seam}"
segment="2547899409721197155"
geometry_root="${ELLIPSOID_ROOT:-/mnt/d/CGProjectData/waymo/ellipsoids}"
model_type="nag.model.view_dependent_background_image_ellipsoid_scene_node_3d.ViewDependentBackgroundImageEllipsoidSceneNode3D"
intervals=("37 81" "86 137")
configs=()
geometries=()

for mode in "${modes[@]}"; do
    case "$mode" in
        legacy|world_y|world_y_auto_seam) ;;
        *) echo "Unknown texture mode: $mode" >&2; exit 1 ;;
    esac
done
[[ "${#modes[@]}" -gt 0 ]] || { echo "No texture modes selected." >&2; exit 1; }

# Validate BOTH intervals before starting any training. Never regenerate or
# modify the JSON/PLY geometry as part of this texture-only experiment.
for interval in "${intervals[@]}"; do
    read -r first last <<< "$interval"
    config_path="config/waymo/large-ego/segment-${segment}_${first}_${last}_large-undistorted.yaml"
    printf -v frame_dir 'frames_%03d_%03d_ray_depth_proxy_yup_v4_y_vertical_lateral_matched' "$first" "$last"
    geometry="$geometry_root/segment-$segment/$frame_dir/ellipsoid.json"
    [[ -f "$config_path" && -f "$geometry" ]] || {
        echo "Missing existing config or Moderate-Y JSON: $config_path | $geometry" >&2; exit 1;
    }
    "$python_bin" - "$geometry" "$segment" "$first" "$last" <<'PY'
import json, math, sys
path, segment, first, last = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    geometry = json.load(f)
assert geometry["sequence"] == "segment-" + segment, "Segment mismatch"
assert geometry["format_version"] == 4, "Expected the existing v4 geometry"
frames = geometry["frame_slice"]
assert (frames["start"], frames["stop"], frames["step"]) == (int(first), int(last)+1, 1), "Interval mismatch"
radii = geometry["fit"]["radii"]
assert all(math.isfinite(r) and r > 0 for r in radii)
print(f"W254 {first}-{last}: reuse center={geometry['fit']['center']}, radii={radii}")
PY
    configs+=("$config_path")
    geometries+=("$geometry")
done

for index in "${!intervals[@]}"; do
    read -r first last <<< "${intervals[$index]}"
    for mode in "${modes[@]}"; do
        name="W254_${first}_${last}_large-undistorted_ellipsoid_v4_moderate_y_texture_${mode}_regular"
        command=("$python_bin" nag/scripts/run_nag.py
            --config-path "${configs[$index]}"
            --background-plane-type "$model_type"
            --background-ellipsoid-path "${geometries[$index]}"
            --background-ellipsoid-scale-margin 0
            --background-ellipsoid-texture-orientation "$mode"
            --name "$name")
        printf '\n[%s]\n' "$name"
        if [[ "$dry_run" == 1 ]]; then
            printf '%q ' "${command[@]}"
            printf '\n'
        else
            "${command[@]}"
        fi
    done
done
printf '\nFinished %d texture-orientation jobs (DRY_RUN=%s).\n' "$((2 * ${#modes[@]}))" "$dry_run"
