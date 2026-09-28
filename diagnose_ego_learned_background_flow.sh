#!/usr/bin/env bash
# Final-checkpoint rendering + learned background flow; never trains or edits runs.
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=()
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
[[ -n "${OUTPUT_ROOT:-}" ]] && args+=(--output "$OUTPUT_ROOT")
[[ -n "${INVENTORY_PATH:-}" ]] && args+=(--inventory "$INVENTORY_PATH")
if [[ -n "${SEQUENCES:-}" ]]; then
    read -r -a labels <<< "$SEQUENCES"
    args+=(--sequences "${labels[@]}")
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python3}" "$SCRIPT_ROOT/nag/scripts/ego_learned_background_flow.py" "${args[@]}" "$@"
