#!/usr/bin/env bash
# Merge existing seeds 0-2 with extension seeds 3-4, then summarize five seeds.
# The merged directories contain metrics/audit tables only, not model checkpoints.
set -euo pipefail

DATASET=${DATASET:-cifar-100}
ROOT=${ROOT:-outputs/${DATASET}/10-task}
MERGED_ROOT=${MERGED_ROOT:-${ROOT}/forgetting-audit-five-seed}
INCLUDE_VALUE=${INCLUDE_VALUE:-1}
INCLUDE_KEY_VALUE=${INCLUDE_KEY_VALUE:-1}

merge_pair() {
  local first=$1
  local extension=$2
  local output=$3
  python -u utils/merge_forgetting_audit_runs.py \
    "${first}" "${extension}" --output_dir "${output}"
}

BASELINE_OUT="${MERGED_ROOT}/forgetting-audit-router"
KEY_OUT="${MERGED_ROOT}/forgetting-audit-freeze-key"
merge_pair "${ROOT}/forgetting-audit-router" \
  "${ROOT}/forgetting-audit-router-seeds34" "${BASELINE_OUT}"
merge_pair "${ROOT}/forgetting-audit-freeze-key" \
  "${ROOT}/forgetting-audit-freeze-key-seeds34" "${KEY_OUT}"

RUNS=("${BASELINE_OUT}" "${KEY_OUT}")
if [[ "${INCLUDE_VALUE}" == "1" ]]; then
  VALUE_OUT="${MERGED_ROOT}/forgetting-audit-freeze-value"
  merge_pair "${ROOT}/forgetting-audit-freeze-value" \
    "${ROOT}/forgetting-audit-freeze-value-seeds34" "${VALUE_OUT}"
  RUNS+=("${VALUE_OUT}")
fi
if [[ "${INCLUDE_KEY_VALUE}" == "1" ]]; then
  KEY_VALUE_OUT="${MERGED_ROOT}/forgetting-audit-freeze-key-value"
  merge_pair "${ROOT}/forgetting-audit-freeze-key-value" \
    "${ROOT}/forgetting-audit-freeze-key-value-seeds34" "${KEY_VALUE_OUT}"
  RUNS+=("${KEY_VALUE_OUT}")
fi

python -u utils/summarize_forgetting_audit.py \
  "${RUNS[@]}" \
  --baseline "${BASELINE_OUT}" \
  --output_dir "${MERGED_ROOT}/summary" \
  --plasticity_warn_threshold "${PLASTICITY_WARN_THRESHOLD:-2.0}" \
  --retention_gain_threshold "${RETENTION_GAIN_THRESHOLD:-1.0}" \
  --enable_significance_tests
