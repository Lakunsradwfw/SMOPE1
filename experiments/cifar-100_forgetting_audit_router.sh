#!/usr/bin/env bash
# Baseline + router drift + Historical Router Replay + component drift.
# Example: GPUID=0 bash experiments/cifar-100_forgetting_audit_router.sh

set -euo pipefail

DATASET=${DATASET:-cifar-100}
CONFIG=${CONFIG:-configs/cifar-100_prompt_smope.yaml}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task/forgetting-audit-router}
GPUID=${GPUID:-0}
REPEAT=${REPEAT:-3}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-10}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
AUDIT_MAX_SAMPLES=${AUDIT_MAX_SAMPLES:-0}
AUDIT_CHECKPOINTS=${AUDIT_CHECKPOINTS:-"5 10"}
SEEDS=${SEEDS:-"0 1 2"}
SAVE_ROUTER_LOGITS=${SAVE_ROUTER_LOGITS:-0}
AUDIT_EXPERT_USAGE=${AUDIT_EXPERT_USAGE:-1}
AUDIT_GRADIENT_DIRECTION=${AUDIT_GRADIENT_DIRECTION:-0}
AUDIT_SAVE_FULL_CHECKPOINTS=${AUDIT_SAVE_FULL_CHECKPOINTS:-1}
AUDIT_CLEANUP_CLASS_CHECKPOINTS=${AUDIT_CLEANUP_CLASS_CHECKPOINTS:-1}
AUDIT_SAMPLE_MANIFEST=${AUDIT_SAMPLE_MANIFEST:-}
AUDIT_MAIN_EPOCHS=${AUDIT_MAIN_EPOCHS:-0}

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a CHECKPOINT_ARRAY <<< "${AUDIT_CHECKPOINTS}"
LOGIT_ARGS=()
if [[ "${SAVE_ROUTER_LOGITS}" == "1" ]]; then
  LOGIT_ARGS+=(--audit_save_logits)
fi
AUDIT_ARGS=()
if [[ "${AUDIT_EXPERT_USAGE}" == "1" ]]; then AUDIT_ARGS+=(--audit_expert_usage); fi
if [[ "${AUDIT_GRADIENT_DIRECTION}" == "1" ]]; then AUDIT_ARGS+=(--audit_gradient_direction); fi
if [[ "${AUDIT_SAVE_FULL_CHECKPOINTS}" == "1" ]]; then AUDIT_ARGS+=(--audit_save_full_checkpoints); fi
if [[ "${AUDIT_CLEANUP_CLASS_CHECKPOINTS}" == "1" ]]; then AUDIT_ARGS+=(--audit_cleanup_class_checkpoints); fi
if [[ -n "${AUDIT_SAMPLE_MANIFEST}" ]]; then AUDIT_ARGS+=(--audit_sample_manifest "${AUDIT_SAMPLE_MANIFEST}"); fi

mkdir -p "${OUTDIR}"
python -u run.py \
  --config "${CONFIG}" \
  --gpuid "${GPUID}" --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
  --learner_type prompt --learner_name OnePrompt \
  --prompt_param 50 5 1e-5 1e-5 0.4 --seeds "${SEED_ARRAY[@]}" \
  --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
  --audit_main_epochs "${AUDIT_MAIN_EPOCHS}" \
  --audit_router --audit_checkpoints "${CHECKPOINT_ARRAY[@]}" \
  --audit_router_replay_modes identity identity_prompt_logits \
  --audit_router_max_samples "${AUDIT_MAX_SAMPLES}" \
  "${LOGIT_ARGS[@]}" \
  "${AUDIT_ARGS[@]}" \
  --log_dir "${OUTDIR}"
