#!/usr/bin/env bash
# Offline final-model component restoration; requires baseline audit checkpoints.
set -euo pipefail

RUN_DIR=${RUN_DIR:-outputs/cifar-100/10-task/forgetting-audit-router}
GPUID=${GPUID:-0}
FINAL_CHECKPOINT=${FINAL_CHECKPOINT:-10}
USAGE_COVERAGE=${USAGE_COVERAGE:-0.90}

python -u utils/evaluate_component_restoration.py \
  --run_dir "${RUN_DIR}" \
  --final_checkpoint "${FINAL_CHECKPOINT}" \
  --components final router_identity router_identity_logits key value classifier full_historical \
  --combinations key+value router_identity+value router_identity+classifier key+value+classifier \
  --restore_scope full_pool used_experts high_frequency_experts \
  --usage_coverage "${USAGE_COVERAGE}" \
  --gpuid "${GPUID}" \
  --output "${RUN_DIR}/forgetting_audit/component_restoration.jsonl"

