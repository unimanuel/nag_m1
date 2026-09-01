#!/usr/bin/env bash
set -euo pipefail

# Derive PLY folders with a moderate physical-Y radius from the corrected
# ray-depth fits. For each fit, physical vertical is raised only when it is
# smaller than the fit's physical left/right radius.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
ellipsoid_dir="/mnt/d/CGProjectData/waymo/ellipsoids"
source_suffix="ray_depth_proxy_yup_v4"
variant_suffix="y_vertical_lateral_matched"
deriver="$repo_root/nag/scripts/derive_ellipsoid_axis_variant.py"
segment_ids=("1918764220984209654" "2547899409721197155")

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$deriver" ]]; then
  echo "Missing geometry derivation utility: $deriver" >&2
  exit 1
fi

created=0
reused=0
for segment_id in "${segment_ids[@]}"; do
  source_paths=("$ellipsoid_dir/segment-$segment_id"/frames_*_"$source_suffix"/ellipsoid.json)
  if [[ ${#source_paths[@]} -eq 0 || ! -f "${source_paths[0]}" ]]; then
    echo "No corrected ray-depth-proxy outputs found for segment-$segment_id." >&2
    echo "Run ./prepare_W191_W254_ray_depth_proxy_ellipsoids.sh first." >&2
    exit 1
  fi
  for source_path in "${source_paths[@]}"; do
    source_dir="${source_path%/ellipsoid.json}"
    output_dir="${source_dir}_${variant_suffix}"
    if [[ -f "$output_dir/ellipsoid.ply" && -f "$output_dir/ellipsoid.json" ]]; then
      echo "Reusing existing moderate-Y PLY: $output_dir"
      reused=$((reused + 1))
      continue
    fi
    echo "Creating moderate-Y PLY: $output_dir"
    "$python_bin" "$deriver" \
      --source-json "$source_path" \
      --output-dir "$output_dir" \
      --match-vertical-to-lateral \
      --world-up-axis 1 \
      --label "Moderate physical-Y vertical radius matched to the left/right radius"
    created=$((created + 1))
  done
done

echo
echo "Created $created moderate-Y folders; reused $reused existing folders."
echo "Outputs are under: $ellipsoid_dir/segment-*/frames_*_${source_suffix}_${variant_suffix}/"
