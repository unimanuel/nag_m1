#!/usr/bin/env bash
# Fit the NEW radius-scale-regularized reprojection/depth ellipsoid for every
# ego interval.  This is a paired experiment against the existing ORIGINAL
# reprojection/depth-v1 geometries: it uses the same cached observations,
# source raw-v4 orientation, seed, sample counts, depth anchor, safety terms,
# and optimiser.  The only optimisation difference is the soft source-scale
# radius-excess regularizer below.
#
# It does not launch NAG training.  It never overwrites the original v1
# ellipsoid folders; new geometry goes in *_reprojection_depth_new_radius_scale_v1.
#
# Usage from the repository root after activating nag1:
#   DRY_RUN=1 bash prepare_all_ego_reprojection_depth_new_radius_scale_ellipsoids.sh
#   bash prepare_all_ego_reprojection_depth_new_radius_scale_ellipsoids.sh
#
# Set FORCE=1 only to replace an existing NEW geometry folder.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python}"
dry_run="${DRY_RUN:-0}"
force="${FORCE:-0}"
diagnostic_root="${OUTPUT_ROOT:-/mnt/d/CGProjectData/waymo/diagnostics/parallax_v1}"
ellipsoid_root="${ELLIPSOID_ROOT:-/mnt/d/CGProjectData/waymo/ellipsoids}"

# This is a soft prior, not a cap.  It penalizes log(r/reference) only when an
# axis r exceeds the largest radius of the safety-projected raw-v4 initializer.
# A 20.0 weight was calibrated on W254 0-37: it eliminates the 171 m lateral
# degeneracy while leaving already-within-scale original fits unpenalized.
radius_excess_weight="${RADIUS_EXCESS_WEIGHT:-20.0}"
radius_excess_huber_log="${RADIUS_EXCESS_HUBER_LOG:-0.20}"

# Label, Waymo segment, padded first and last frame.  The seven entries are
# the complete W191 + W254 ego benchmark set, including the two original
# controlled intervals that were already trained under the original objective.
records=(
  "W191_0_40 1918764220984209654 000 040"
  "W191_40_128 1918764220984209654 040 128"
  "W191_128_198 1918764220984209654 128 198"
  "W254_0_37 2547899409721197155 000 037"
  "W254_37_81 2547899409721197155 037 081"
  "W254_86_137 2547899409721197155 086 137"
  "W254_139_195 2547899409721197155 139 195"
)
labels=()
for record in "${records[@]}"; do
  read -r label _ <<< "$record"
  labels+=("$label")
done

[[ -x "$python_bin" || "$python_bin" == python || "$python_bin" == python3 ]] || {
  echo "PYTHON_BIN is not executable: $python_bin" >&2; exit 1;
}
for required in "$diagnostic_root" "$ellipsoid_root"; do
  [[ -e "$required" ]] || { echo "Missing required directory: $required" >&2; exit 1; }
done

# This validates/refreshes the shared static, visible cross-frame observations.
# It does not change a completed original geometry or training result.
if [[ "$dry_run" == 1 ]]; then
  SEQUENCES="${labels[*]}" DRY_RUN=1 PYTHON_BIN="$python_bin" bash prepare_ego_parallax_diagnostics.sh
else
  SEQUENCES="${labels[*]}" PYTHON_BIN="$python_bin" bash prepare_ego_parallax_diagnostics.sh
fi

for record in "${records[@]}"; do
  read -r label segment first last <<< "$record"
  prepared="$diagnostic_root/$label/prepared.json"
  source="$ellipsoid_root/segment-$segment/frames_${first}_${last}_ray_depth_proxy_yup_v4/ellipsoid.json"
  output="$ellipsoid_root/segment-$segment/frames_${first}_${last}_reprojection_depth_new_radius_scale_v1"
  for required in "$prepared" "$source"; do
    [[ -f "$required" ]] || { echo "$label: missing required file: $required" >&2; exit 1; }
  done

  command=("$python_bin" nag/scripts/fit_reprojection_ellipsoid.py
    --label "$label"
    --prepared "$prepared"
    --source-ellipsoid "$source"
    --output-dir "$output"
    --max-pairs 24000
    --max-depth-rays 14000
    --max-camera-normalized-radius 0.65
    --minimum-camera-surface-distance 6.0
    --minimum-axis-radius 8.0
    --max-axis-ratio 10.0
    --reprojection-weight 1.0
    --depth-weight 0.15
    --radius-excess-weight "$radius_excess_weight"
    --radius-excess-huber-log "$radius_excess_huber_log"
    --max-iterations 200)
  [[ "$force" == 1 ]] && command+=(--overwrite)
  [[ "$dry_run" == 1 ]] && command+=(--dry-run)

  printf '\n[%s: new radius-scale regularizer]\n' "$label"
  printf '%q ' "${command[@]}"
  printf '\n'
  "${command[@]}"
done

printf '\nFinished seven NEW radius-scale geometry fits (DRY_RUN=%s).\n' "$dry_run"
printf 'Review each ellipsoid.ply and reprojection_fit_diagnostics.json before training.\n'
