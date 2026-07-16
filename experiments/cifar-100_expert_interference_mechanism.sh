#!/usr/bin/env bash
# Compact within-expert task-pair audit for:
# persistent shared routing -> cross-task conflict above split-half noise ->
# actual update harm -> incremental functional drift -> forgetting.
# It intentionally does not save router snapshots or full audit checkpoints.
set -euo pipefail

DATASET=${DATASET:-cifar-100}
CONFIG=${CONFIG:-configs/cifar-100_prompt_smope.yaml}
OUTDIR=${OUTDIR:-outputs/${DATASET}/10-task/within-expert-interference}
GPUID=${GPUID:-0}
OVERWRITE=${OVERWRITE:-1}
MAX_TASK=${MAX_TASK:-10}
CRCT_EPOCHS=${CRCT_EPOCHS:-50}
SEEDS=${SEEDS:-"0 1 2"}
AUDIT_MECHANISM_MAX_SAMPLES=${AUDIT_MECHANISM_MAX_SAMPLES:-256}
AUDIT_MECHANISM_GRADIENT_EPSILON=${AUDIT_MECHANISM_GRADIENT_EPSILON:-1e-12}

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
  --audit_expert_interference \
  --audit_mechanism_max_samples "${AUDIT_MECHANISM_MAX_SAMPLES}" \
  --audit_mechanism_gradient_epsilon "${AUDIT_MECHANISM_GRADIENT_EPSILON}" \
  --audit_cleanup_class_checkpoints \
  --log_dir "${OUTDIR}"

python utils/analyze_expert_interference.py "${OUTDIR}"
