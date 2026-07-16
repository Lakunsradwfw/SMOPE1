#!/usr/bin/env bash
# Fast implementation crosscheck only; not paper evidence.
set -euo pipefail

GPUID=${GPUID:-0} \
CONFIG=${CONFIG:-configs/cifar-100_prompt_smope_audit_smoke.yaml} \
OUTDIR=${OUTDIR:-outputs/cifar-100/3-task/within-expert-interference-smoke} \
MAX_TASK=${MAX_TASK:-3} \
CRCT_EPOCHS=${CRCT_EPOCHS:-2} \
SEEDS=${SEEDS:-"0"} \
REPEAT=${REPEAT:-1} \
AUDIT_MECHANISM_MAX_SAMPLES=${AUDIT_MECHANISM_MAX_SAMPLES:-32} \
bash experiments/cifar-100_expert_interference_mechanism.sh
