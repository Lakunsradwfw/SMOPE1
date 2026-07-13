#!/usr/bin/env bash
# Strict seed-paired aggregation for the baseline and four freeze interventions.
set -euo pipefail

ROOT=${ROOT:-outputs/cifar-100/10-task}
BASELINE=${BASELINE:-${ROOT}/forgetting-audit-router}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/forgetting-audit-summary}

python -u utils/summarize_forgetting_audit.py \
  "${BASELINE}" \
  "${ROOT}/forgetting-audit-freeze-prompt" \
  "${ROOT}/forgetting-audit-freeze-key" \
  "${ROOT}/forgetting-audit-freeze-value" \
  "${ROOT}/forgetting-audit-freeze-classifier" \
  --baseline "${BASELINE}" \
  --output_dir "${OUTPUT_DIR}" \
  --plasticity_warn_threshold "${PLASTICITY_WARN_THRESHOLD:-2.0}" \
  --retention_gain_threshold "${RETENTION_GAIN_THRESHOLD:-1.0}"

