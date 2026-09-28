#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/qwen/run.sh {zero_shot|lora|smope} [preflight|smoke|full] [extra arguments]'
  exit 2
fi
METHOD="$1"; shift
MODE="${1:-smoke}"; if [[ $# -gt 0 ]]; then shift; fi
case "$METHOD" in zero_shot|lora|smope) ;; *) echo "Unknown method: $METHOD"; exit 2;; esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for SEED in ${SEEDS:-0}; do
  OUT="${OUTPUT_ROOT:-outputs/coin-qwen3.5-9b}/${METHOD}/${MODE}/seed-${SEED}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" -m qwen_smope.train \
    --method "$METHOD" --mode "$MODE" --seed "$SEED" \
    --model-path "${MODEL_PATH:-pretrained/Qwen3.5-9B}" \
    --coin-root "${COIN_ROOT:-datas/CoIN}" --output "$OUT" "$@"
done
