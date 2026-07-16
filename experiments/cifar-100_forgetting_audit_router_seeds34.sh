#!/usr/bin/env bash
# Extend the matched router baseline with only seeds 3 and 4.
set -euo pipefail

DATASET=${DATASET:-cifar-100}
ROOT=${ROOT:-outputs/${DATASET}/10-task}
OUTDIR=${OUTDIR:-${ROOT}/forgetting-audit-router-seeds34}

REPEAT=2 SEEDS="3 4" OUTDIR="${OUTDIR}" \
AUDIT_SAVE_FULL_CHECKPOINTS=${AUDIT_SAVE_FULL_CHECKPOINTS:-0} \
GPUID=${GPUID:-0} OVERWRITE=${OVERWRITE:-1} \
  bash experiments/cifar-100_forgetting_audit_router.sh
