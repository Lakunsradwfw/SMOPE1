#!/usr/bin/env bash
# Small matched baseline/freeze cross-check before spending the full budget.
set -euo pipefail

GPUID=${GPUID:-0}
ROOT=${ROOT:-outputs/cifar-100/crosscheck/forgetting-audit-v2}
COMMON=(MAX_TASK=3 REPEAT=2 CRCT_EPOCHS=2 AUDIT_MAX_SAMPLES=128 AUDIT_CHECKPOINTS="3" SEEDS="0 1" GPUID="${GPUID}" CONFIG=configs/cifar-100_prompt_smope_audit_smoke.yaml)

env "${COMMON[@]}" OUTDIR="${ROOT}/forgetting-audit-router" \
  bash experiments/cifar-100_forgetting_audit_router.sh
MANIFEST="${ROOT}/forgetting-audit-router/forgetting_audit/audit_sample_manifest.json"
env "${COMMON[@]}" AUDIT_SAMPLE_MANIFEST="${MANIFEST}" OUTDIR="${ROOT}/forgetting-audit-freeze-key" \
  bash experiments/cifar-100_forgetting_audit_freeze_key.sh
env "${COMMON[@]}" AUDIT_SAMPLE_MANIFEST="${MANIFEST}" OUTDIR="${ROOT}/forgetting-audit-freeze-value" \
  bash experiments/cifar-100_forgetting_audit_freeze_value.sh
env "${COMMON[@]}" AUDIT_SAMPLE_MANIFEST="${MANIFEST}" OUTDIR="${ROOT}/forgetting-audit-freeze-classifier" \
  bash experiments/cifar-100_forgetting_audit_freeze_classifier.sh
