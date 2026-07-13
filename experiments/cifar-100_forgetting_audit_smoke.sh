#!/usr/bin/env bash
# Two-task implementation smoke test.  This does not support a paper conclusion.
set -euo pipefail

GPUID=${GPUID:-0}
OUTDIR=${OUTDIR:-outputs/cifar-100/smoke/forgetting-audit-v2}

if [[ ! -f pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.bin ]]; then
  echo "Missing pretrained/vit_base_patch16_224_augreg2_in21k_ft_in1k.bin; follow README.md before running the smoke test." >&2
  exit 2
fi

MAX_TASK=2 REPEAT=1 CRCT_EPOCHS=1 AUDIT_MAX_SAMPLES=${AUDIT_MAX_SAMPLES:-128} \
AUDIT_CHECKPOINTS="2" AUDIT_SAVE_FULL_CHECKPOINTS=1 AUDIT_EXPERT_USAGE=1 \
CONFIG=configs/cifar-100_prompt_smope_audit_smoke.yaml \
GPUID="${GPUID}" OUTDIR="${OUTDIR}" \
bash experiments/cifar-100_forgetting_audit_router.sh
