#!/usr/bin/env bash
# Matched no-probe control for checking that audit instrumentation is inert.
set -euo pipefail

CONFIG=${CONFIG:-configs/cifar-100_prompt_smope.yaml}
OUTDIR=${OUTDIR:-outputs/cifar-100/10-task/within-expert-interference-control}
GPUID=${GPUID:-0}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-10}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
SEEDS=${SEEDS:-"0 1 2"}

read -r -a SEED_ARRAY <<< "${SEEDS}"
REPEAT=${REPEAT:-${#SEED_ARRAY[@]}}
if [[ "${REPEAT}" -ne "${#SEED_ARRAY[@]}" ]]; then
  echo "REPEAT must equal the number of explicit SEEDS." >&2
  exit 2
fi

mkdir -p "${OUTDIR}"
python -u run.py \
  --config "${CONFIG}" \
  --gpuid "${GPUID}" --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
  --learner_type prompt --learner_name OnePrompt \
  --prompt_param 50 5 1e-5 1e-5 0.4 --seeds "${SEED_ARRAY[@]}" \
  --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" \
  --ca_batch_size_ratio 1 \
  --audit_cleanup_class_checkpoints \
  --log_dir "${OUTDIR}"
