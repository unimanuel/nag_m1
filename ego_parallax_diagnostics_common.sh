#!/usr/bin/env bash
# Shared entry point. Arguments after the stage are forwarded verbatim to Python.
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage="$1"
shift
args=("$stage")
if [[ -n "${SEQUENCES:-}" ]]; then
    read -r -a labels <<< "$SEQUENCES"
    args+=(--sequences "${labels[@]}")
fi
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
[[ "${FORCE:-0}" == 1 ]] && args+=(--force)
[[ -n "${OUTPUT_ROOT:-}" ]] && args+=(--output "$OUTPUT_ROOT")
[[ -n "${RAYS_PER_FRAME:-}" ]] && args+=(--rays-per-frame "$RAYS_PER_FRAME")
[[ -n "${TARGETS_PER_FRAME:-}" ]] && args+=(--targets-per-frame "$TARGETS_PER_FRAME")
[[ -n "${FRAME_LIMIT:-}" ]] && args+=(--frame-limit "$FRAME_LIMIT")
[[ -n "${DISPLAY_CAP:-}" ]] && args+=(--display-cap "$DISPLAY_CAP")
[[ -n "${BOUNDARY_WIDTH:-}" ]] && args+=(--boundary-width "$BOUNDARY_WIDTH")
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python3}" "$SCRIPT_ROOT/nag/scripts/ego_parallax_diagnostics.py" "${args[@]}" "$@"
