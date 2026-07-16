#!/usr/bin/env bash
# Extend freeze-key with seeds 3 and 4, paired to the seeds34 router manifest.
set -euo pipefail

DATASET=${DATASET:-cifar-100}
ROOT=${ROOT:-outputs/${DATASET}/10-task}
BASELINE_SEEDS34=${BASELINE_SEEDS34:-${ROOT}/forgetting-audit-router-seeds34}
MANIFEST=${AUDIT_SAMPLE_MANIFEST:-${BASELINE_SEEDS34}/forgetting_audit/audit_sample_manifest.json}
OUTDIR=${OUTDIR:-${ROOT}/forgetting-audit-freeze-key-seeds34}

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing seeds 3/4 baseline manifest: ${MANIFEST}" >&2
  echo "Run experiments/cifar-100_forgetting_audit_router_seeds34.sh first." >&2
  exit 2
fi

REPEAT=2 SEEDS="3 4" OUTDIR="${OUTDIR}" AUDIT_SAMPLE_MANIFEST="${MANIFEST}" \
GPUID=${GPUID:-1} OVERWRITE=${OVERWRITE:-1} \
  bash experiments/cifar-100_forgetting_audit_freeze_key.sh
