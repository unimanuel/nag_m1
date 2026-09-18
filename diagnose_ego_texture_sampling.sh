#!/usr/bin/env bash
# Uses the existing full-interval parallax caches; no rendering or training.
set -euo pipefail
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
args=()
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
[[ -n "${OUTPUT_ROOT:-}" ]] && args+=(--output "$OUTPUT_ROOT")
[[ -n "${PREPARED_ROOT:-}" ]] && args+=(--prepared-root "$PREPARED_ROOT")
if [[ -n "${SEQUENCES:-}" ]]; then
    read -r -a labels <<< "$SEQUENCES"
    args+=(--sequences "${labels[@]}")
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON_BIN:-python3}" "$SCRIPT_ROOT/nag/scripts/texture_sampling_diagnostics.py" "${args[@]}" "$@"
