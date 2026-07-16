#!/usr/bin/env bash
# Calibrate non-freeze baselines with fewer main-training epochs, then select
# controls whose seed-level diagonal loss matches freeze-prompt/freeze-value.
set -euo pipefail

DATASET=${DATASET:-cifar-100}
ROOT=${ROOT:-outputs/${DATASET}/10-task}
BASELINE=${BASELINE:-${ROOT}/forgetting-audit-router}
TARGET_PROMPT=${TARGET_PROMPT:-${ROOT}/forgetting-audit-freeze-prompt}
TARGET_VALUE=${TARGET_VALUE:-${ROOT}/forgetting-audit-freeze-value}
CONTROL_ROOT=${CONTROL_ROOT:-${ROOT}/matched-plasticity-controls}
MANIFEST=${AUDIT_SAMPLE_MANIFEST:-${BASELINE}/forgetting_audit/audit_sample_manifest.json}
MAIN_EPOCHS_LIST=${MAIN_EPOCHS_LIST:-"17 18"}
SEEDS=${SEEDS:-"0 1 2"}
REPEAT=${REPEAT:-3}
TOLERANCE=${TOLERANCE:-0.35}
EXTRA_CANDIDATES=${EXTRA_CANDIDATES:-}

for required in "${MANIFEST}" "${TARGET_PROMPT}/args.yaml" "${TARGET_VALUE}/args.yaml"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing matched-plasticity input: ${required}" >&2
    exit 2
  fi
done

read -r -a EPOCH_ARRAY <<< "${MAIN_EPOCHS_LIST}"
CANDIDATES=()
for epochs in "${EPOCH_ARRAY[@]}"; do
  OUTDIR="${CONTROL_ROOT}/baseline-main-epochs-${epochs}"
  CANDIDATES+=("${OUTDIR}")
  AUDIT_MAIN_EPOCHS="${epochs}" \
  AUDIT_SAMPLE_MANIFEST="${MANIFEST}" \
  AUDIT_SAVE_FULL_CHECKPOINTS=0 AUDIT_EXPERT_USAGE=0 \
  AUDIT_GRADIENT_DIRECTION=0 SAVE_ROUTER_LOGITS=0 \
  REPEAT="${REPEAT}" SEEDS="${SEEDS}" OUTDIR="${OUTDIR}" \
  GPUID=${GPUID:-0} OVERWRITE=${OVERWRITE:-1} \
    bash experiments/cifar-100_forgetting_audit_router.sh
done
if [[ -n "${EXTRA_CANDIDATES}" ]]; then
  read -r -a EXTRA_CANDIDATE_ARRAY <<< "${EXTRA_CANDIDATES}"
  CANDIDATES+=("${EXTRA_CANDIDATE_ARRAY[@]}")
fi

python -u utils/select_matched_plasticity_control.py \
  --baseline "${BASELINE}" \
  --targets "${TARGET_PROMPT}" "${TARGET_VALUE}" \
  --candidates "${CANDIDATES[@]}" \
  --output_dir "${CONTROL_ROOT}/selection" \
  --tolerance "${TOLERANCE}"
