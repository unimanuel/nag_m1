#!/usr/bin/env bash
# Train the three missing W191 regular-sphere ego baselines.
#
# These mirror the completed W191 Sphere Visible-UV runs, using the same
# visible-plane-center radius calculation and all non-UV configuration values.
# The only intended difference is that the regular YAML does not allocate the
# sphere texture into the visible-UV tile.
#
# Usage from the repository root after activating nag1:
#   bash run_W191_regular_sphere_benchmarks.sh
#   DRY_RUN=1 bash run_W191_regular_sphere_benchmarks.sh
#
# After completion, add the date directory printed at the end to
# PARALLAX_EXTRA_RUN_ROOTS when running the parallax diagnostics.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
CONFIG_ROOT="$REPO_ROOT/config/waymo/large-ego"
SEGMENT_ID="1918764220984209654"

intervals=("0_40" "40_128" "128_198")

for interval in "${intervals[@]}"; do
    config="$CONFIG_ROOT/segment-${SEGMENT_ID}_${interval}_large-undistorted.yaml"
    visuv_config="${config%.yaml}_sph_visuv.yaml"
    [[ -f "$config" && -f "$visuv_config" ]] || {
        echo "Missing W191 regular or Visible-UV configuration: $config | $visuv_config" >&2
        exit 1
    }
    "$PYTHON_BIN" - "$config" "$visuv_config" "$interval" <<'PY'
import sys
import yaml

regular_path, uv_path, interval = sys.argv[1:]
with open(regular_path, encoding="utf-8") as f:
    regular = yaml.safe_load(f)["NAGConfig"]
with open(uv_path, encoding="utf-8") as f:
    visuv = yaml.safe_load(f)["NAGConfig"]
for key in ("data_path", "frame_indices_filter", "background_plane_type", "background_relative_scale_margin"):
    if regular.get(key) != visuv.get(key):
        raise SystemExit(f"{interval}: regular and Visible-UV configs differ in {key!r}")
if regular.get("background_plane_kwargs", {}).get("sphere_visible_uv_texture", {}).get("enabled"):
    raise SystemExit(f"{interval}: regular config unexpectedly enables Visible-UV")
if not visuv.get("background_plane_kwargs", {}).get("sphere_visible_uv_texture", {}).get("enabled"):
    raise SystemExit(f"{interval}: Visible-UV config does not enable Visible-UV")
print(f"W191 {interval}: regular sphere config verified against its completed UV counterpart")
PY
done

for interval in "${intervals[@]}"; do
    config="$CONFIG_ROOT/segment-${SEGMENT_ID}_${interval}_large-undistorted.yaml"
    name="W191_${interval}_large-undistorted_radius_visible_plane_centers"
    command=("$PYTHON_BIN" "$REPO_ROOT/nag/scripts/run_nag.py"
        --config-path "$config"
        --background-sphere-radius-from-visible-plane-centers
        --name "$name")
    printf '\n[%s]\n' "$name"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
done

printf '\nCompleted W191 regular-sphere jobs (DRY_RUN=%s). Expected run folder: runs/%s\n' "$DRY_RUN" "$RUN_DATE"
printf 'For parallax diagnostics after completion: PARALLAX_EXTRA_RUN_ROOTS=%q ...\n' "$RUN_DATE"
