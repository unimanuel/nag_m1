#!/usr/bin/env bash
# Independent restored cohorts: never expand an existing scene manifest in place.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 2 )); then
  echo 'Usage: missing_final_sequences_common.sh {prepare|baseline|ellipsoid} {all|W203|W975} [native CLI options]' >&2
  exit 2
fi
stage="$1"
scene="$2"
shift 2
case "$scene" in
  W203) intervals=(W203_60_107) ;;
  W975) intervals=(W975_0_68 W975_69_99 W975_99_162) ;;
  all) intervals=(W203_60_107 W975_0_68 W975_69_99 W975_99_162) ;;
  *) echo "Unknown restored-depth scene: $scene" >&2; exit 2 ;;
esac
geometry_root=/mnt/d/CGProjectData/waymo/ellipsoids/autowork_restored_final_legacy_v1
if [[ "$stage" == prepare ]]; then
  exec bash "$repo_root/prepare_final_ellipsoids.sh" \
    --config-dir "$repo_root/config/waymo/autowork_paper-final" \
    --intervals "${intervals[@]}" --output-root "$geometry_root" "$@"
fi
if [[ "$stage" != baseline && "$stage" != ellipsoid ]] || [[ "$scene" == all ]]; then
  echo 'Choose prepare/all, or baseline/ellipsoid with one scene (W203 or W975).' >&2
  exit 2
fi
if [[ "$stage" == baseline ]]; then
  family=baselines
  output_flag=--output-root
else
  family=ellipsoids
  output_flag=--training-root
fi
output_root="$repo_root/runs/autowork_restored_${family}_$scene"
legacy_root="$repo_root/runs/autowork_paper_final_baselines_$scene/autowork_restored_$family"
explicit_output=0
for option in "$@"; do
  if [[ "$option" == "$output_flag" || "$option" == "$output_flag="* ]]; then
    explicit_output=1
  fi
done
if [[ ( -e "$legacy_root" || -L "$legacy_root" ) && "$explicit_output" == 0 ]]; then
  echo "Legacy nested restored output exists: $legacy_root" >&2
  echo "Use $output_flag explicitly to continue that cohort; no migration or duplicate queue is created." >&2
  exit 2
fi
if [[ "$stage" == baseline ]]; then
  exec bash "$repo_root/run_paper_final_baselines.sh" \
    --intervals "${intervals[@]}" \
    --output-root "$output_root" "$@"
fi
exec bash "$repo_root/run_final_ellipsoid_benchmarks.sh" \
  --config-dir "$repo_root/config/waymo/autowork_paper-final" \
  --intervals "${intervals[@]}" --output-root "$geometry_root" \
  --training-root "$output_root" "$@"
