#!/usr/bin/env bash
# Intervention: freeze prompt values (e_pv) from Task 2 onward.
# Example: GPUID=3 bash experiments/cifar-100_forgetting_audit_freeze_value.sh

set -euo pipefail
DATASET=${DATASET:-cifar-100}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task/forgetting-audit-freeze-value}
GPUID=${GPUID:-3}
REPEAT=${REPEAT:-3}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-10}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
SEEDS=${SEEDS:-"0 1 2"}
read -r -a SEED_ARRAY <<< "${SEEDS}"

mkdir -p "${OUTDIR}"
python -u run.py --config configs/cifar-100_prompt_smope.yaml \
  --gpuid "${GPUID}" --repeat "${REPEAT}" --overwrite "${OVERWRITE}" \
  --learner_type prompt --learner_name OnePrompt \
  --prompt_param 50 5 1e-5 1e-5 0.4 --seeds "${SEED_ARRAY[@]}" \
  --max_task "${MAX_TASK}" --crct_epochs "${CRCT_EPOCHS}" --ca_batch_size_ratio 1 \
  --audit_freeze_component value --audit_freeze_from_task 2 \
  --log_dir "${OUTDIR}"
