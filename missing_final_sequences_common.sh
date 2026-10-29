#!/usr/bin/env bash
# New child cohorts only: never expand an existing scene manifest in place.
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
scene_root="$repo_root/runs/autowork_paper_final_baselines_$scene"
if [[ ! -f "$scene_root/manifest.json" ]]; then
  echo "Missing renamed scene manifest: $scene_root/manifest.json" >&2
  echo 'Restore the existing scene folder; this launcher never recreates or migrates it.' >&2
  exit 2
fi
if [[ "$stage" == baseline ]]; then
  exec bash "$repo_root/run_paper_final_baselines.sh" \
    --intervals "${intervals[@]}" \
    --output-root "$scene_root/autowork_restored_baselines" "$@"
fi
exec bash "$repo_root/run_final_ellipsoid_benchmarks.sh" \
  --config-dir "$repo_root/config/waymo/autowork_paper-final" \
  --intervals "${intervals[@]}" --output-root "$geometry_root" \
  --training-root "$scene_root/autowork_restored_ellipsoids" "$@"
