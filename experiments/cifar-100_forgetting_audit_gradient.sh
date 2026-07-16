#!/usr/bin/env bash
# Natural-training baseline with expensive task-boundary direction diagnostics.
set -euo pipefail

OUTDIR=${OUTDIR:-outputs/cifar-100/10-task/forgetting-audit-gradient}
AUDIT_SAVE_FULL_CHECKPOINTS=${AUDIT_SAVE_FULL_CHECKPOINTS:-0}
AUDIT_GRADIENT_DIRECTION=1 AUDIT_SAVE_FULL_CHECKPOINTS="${AUDIT_SAVE_FULL_CHECKPOINTS}" \
OUTDIR="${OUTDIR}" bash experiments/cifar-100_forgetting_audit_router.sh

