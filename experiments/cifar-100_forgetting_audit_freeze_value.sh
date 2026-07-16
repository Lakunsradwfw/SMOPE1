#!/usr/bin/env bash
# Intervention: freeze prompt values (e_pv) from Task 2 onward.
# Example: GPUID=3 bash experiments/cifar-100_forgetting_audit_freeze_value.sh

set -euo pipefail
DATASET=${DATASET:-cifar-100}
CONFIG=${CONFIG:-configs/cifar-100_prompt_smope.yaml}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task/forgetting-audit-freeze-value}
GPUID=${GPUID:-3}
REPEAT=${REPEAT:-3}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-10}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
AUDIT_MAX_SAMPLES=${AUDIT_MAX_SAMPLES:-0}
AUDIT_CHECKPOINTS=${AUDIT_CHECKPOINTS:-"5 10"}
SAVE_ROUTER_LOGITS=${SAVE_ROUTER_LOGITS:-0}
AUDIT_SAMPLE_MANIFEST=${AUDIT_SAMPLE_MANIFEST:-}
AUDIT_CLEANUP_CLASS_CHECKPOINTS=${AUDIT_CLEANUP_CLASS_CHECKPOINTS:-1}
SEEDS=${SEEDS:-"0 1 2"}
read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a CHECKPOINT_ARRAY <<< "${AUDIT_CHECKPOINTS}"
AUDIT_ARGS=(--audit_expert_usage)
if [[ "${SAVE_ROUTER_LOGITS}" == "1" ]]; then AUDIT_ARGS+=(--audit_save_full_router_logits); fi
if [[ "${AUDIT_CLEANUP_CLASS_CHECKPOINTS}" == "1" ]]; then AUDIT_ARGS+=(--audit_cleanup_class_checkpoints); fi
if [[ -n "${AUDIT_SAMPLE_MANIFEST}" ]]; then AUDIT_ARGS+=(--audit_sample_manifest "${AUDIT_SAMPLE_MANIFEST}"); fi

mkdir -p "${OUTDIR}"
python -u run.py --config "${CONFIG}" \
  --gpuid "${GPUID}" --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
  --learner_type prompt --learner_name OnePrompt \
  --prompt_param 50 5 1e-5 1e-5 0.4 --seeds "${SEED_ARRAY[@]}" \
  --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
  --audit_freeze_component value --audit_freeze_from_task 2 \
  --audit_router --audit_checkpoints "${CHECKPOINT_ARRAY[@]}" \
  --audit_router_replay_modes identity identity_prompt_logits \
  --audit_router_max_samples "${AUDIT_MAX_SAMPLES}" "${AUDIT_ARGS[@]}" \
  --log_dir "${OUTDIR}"
