#!/usr/bin/env bash
# Baseline + router drift + Historical Router Replay + component drift.
# Example: GPUID=0 bash experiments/cifar-100_forgetting_audit_router.sh

set -euo pipefail

DATASET=${DATASET:-cifar-100}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task/forgetting-audit-router}
GPUID=${GPUID:-0}
REPEAT=${REPEAT:-3}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-10}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
AUDIT_MAX_SAMPLES=${AUDIT_MAX_SAMPLES:-0}
SEEDS=${SEEDS:-"0 1 2"}
SAVE_ROUTER_LOGITS=${SAVE_ROUTER_LOGITS:-0}

read -r -a SEED_ARRAY <<< "${SEEDS}"
LOGIT_ARGS=()
if [[ "${SAVE_ROUTER_LOGITS}" == "1" ]]; then
  LOGIT_ARGS+=(--audit_save_logits)
fi

mkdir -p "${OUTDIR}"
python -u run.py \
  --config configs/cifar-100_prompt_smope.yaml \
  --gpuid "${GPUID}" --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
  --learner_type prompt --learner_name OnePrompt \
  --prompt_param 50 5 1e-5 1e-5 0.4 --seeds "${SEED_ARRAY[@]}" \
  --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
  --audit_router --audit_checkpoints 5 10 \
  --audit_router_max_samples "${AUDIT_MAX_SAMPLES}" \
  "${LOGIT_ARGS[@]}" \
  --log_dir "${OUTDIR}"
