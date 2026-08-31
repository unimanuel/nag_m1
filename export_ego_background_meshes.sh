#!/usr/bin/env bash
set -euo pipefail

# Export red sphere and blue sphere-clipped AR3 ellipsoid PLY meshes for every
# large-ego interval used in the benchmark.  These constants are the final
# visible-plane-center sphere radii recorded by the matching sphere runs.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_python="$repo_root/libs/nag-waymo-processing/.venv/bin/python"
python_bin="${PYTHON_BIN:-$default_python}"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
exporter="$repo_root/nag/scripts/export_background_geometry_meshes.py"
output_suffix="camera_centered_ar3_q995"

if [[ ! -x "$python_bin" ]] && ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  echo "Run the Waymo-processing setup first, or set PYTHON_BIN to a Python executable with NumPy." >&2
  exit 1
fi
if [[ ! -f "$exporter" ]]; then
  echo "Missing mesh exporter: $exporter" >&2
  exit 1
fi

segment_ids=(
  "1918764220984209654" "1918764220984209654" "1918764220984209654"
  "2547899409721197155" "2547899409721197155" "2547899409721197155" "2547899409721197155"
)
intervals=("0_40" "40_128" "128_198" "0_37" "37_81" "86_137" "139_195")
sphere_radii=("52.4692" "28.6621" "18.2372" "30.3448" "44.4949" "58.6208" "31.9631")

for ((index = 0; index < ${#intervals[@]}; index++)); do
  segment_id="${segment_ids[$index]}"
  interval="${intervals[$index]}"
  sphere_radius="${sphere_radii[$index]}"
  start="${interval%%_*}"
  last="${interval##*_}"
  ellipsoid_path="$ellipsoid_dir/segment-$segment_id/frames_$(printf '%03d' "$((10#$start))")_$(printf '%03d' "$((10#$last))")_${output_suffix}/ellipsoid.json"
  # Keep comparison outputs next to the fitted ellipsoid inspection files,
  # but separate from the fit's own front_points/camera_centers/ellipsoid PLYs.
  output_dir="$ellipsoid_dir/segment-$segment_id/frames_$(printf '%03d' "$((10#$start))")_$(printf '%03d' "$((10#$last))")_sphere_ar3clipped"

  if [[ ! -f "$ellipsoid_path" ]]; then
    echo "Missing AR3 ellipsoid for segment-$segment_id frames $interval: $ellipsoid_path" >&2
    echo "Run the ellipsoid-preparation script before exporting its mesh." >&2
    exit 1
  fi

  echo
  echo "[$((index + 1))/${#intervals[@]}] segment-$segment_id, frames $interval"
  "$python_bin" "$exporter" \
    --ellipsoid-json "$ellipsoid_path" \
    --sphere-radius "$sphere_radius" \
    --output-dir "$output_dir" \
    --scale-margin 0.02 \
    --max-radius "$sphere_radius"
done

echo
echo "Final sphere radii used (m):"
for ((index = 0; index < ${#intervals[@]}; index++)); do
  printf '  segment-%s frames %s: %s m\n' \
    "${segment_ids[$index]}" "${intervals[$index]}" "${sphere_radii[$index]}"
done
echo
echo "Done. Open each sphere_red_and_ellipsoid_ar3_clipped_blue.ply in a PLY mesh viewer."
echo "Output folders: $ellipsoid_dir/segment-*/frames_*_sphere_ar3clipped"
