#!/usr/bin/env bash
# Intervention: freeze both prompt keys (e_pk) and values (e_pv) from Task 2 onward.
# Example: GPUID=4 bash experiments/cifar-100_forgetting_audit_freeze_key_value.sh

set -euo pipefail

DATASET=${DATASET:-cifar-100}
CONFIG=${CONFIG:-configs/cifar-100_prompt_smope.yaml}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task/forgetting-audit-freeze-key-value}
GPUID=${GPUID:-4}
REPEAT=${REPEAT:-3}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-10}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
AUDIT_MAX_SAMPLES=${AUDIT_MAX_SAMPLES:-0}
AUDIT_CHECKPOINTS=${AUDIT_CHECKPOINTS:-"5 10"}
SAVE_ROUTER_LOGITS=${SAVE_ROUTER_LOGITS:-0}
SEEDS=${SEEDS:-"0 1 2"}
BASELINE_OUTDIR=${BASELINE_OUTDIR:-outputs/${DATASET}/10-task/forgetting-audit-router}
AUDIT_SAMPLE_MANIFEST=${AUDIT_SAMPLE_MANIFEST:-${BASELINE_OUTDIR}/forgetting_audit/audit_sample_manifest.json}
AUDIT_CLEANUP_CLASS_CHECKPOINTS=${AUDIT_CLEANUP_CLASS_CHECKPOINTS:-1}

if [[ ! -f "${AUDIT_SAMPLE_MANIFEST}" ]]; then
  echo "Missing matched baseline manifest: ${AUDIT_SAMPLE_MANIFEST}" >&2
  echo "Run the paired router baseline first or set AUDIT_SAMPLE_MANIFEST explicitly." >&2
  exit 2
fi

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a CHECKPOINT_ARRAY <<< "${AUDIT_CHECKPOINTS}"
AUDIT_ARGS=(--audit_expert_usage --audit_sample_manifest "${AUDIT_SAMPLE_MANIFEST}")
if [[ "${SAVE_ROUTER_LOGITS}" == "1" ]]; then AUDIT_ARGS+=(--audit_save_full_router_logits); fi
if [[ "${AUDIT_CLEANUP_CLASS_CHECKPOINTS}" == "1" ]]; then AUDIT_ARGS+=(--audit_cleanup_class_checkpoints); fi

mkdir -p "${OUTDIR}"
python -u run.py --config "${CONFIG}" \
  --gpuid "${GPUID}" --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
  --learner_type prompt --learner_name OnePrompt \
  --prompt_param 50 5 1e-5 1e-5 0.4 --seeds "${SEED_ARRAY[@]}" \
  --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
  --audit_freeze_component key_value --audit_freeze_from_task 2 \
  --audit_router --audit_checkpoints "${CHECKPOINT_ARRAY[@]}" \
  --audit_router_replay_modes identity identity_prompt_logits \
  --audit_router_max_samples "${AUDIT_MAX_SAMPLES}" "${AUDIT_ARGS[@]}" \
  --log_dir "${OUTDIR}"
