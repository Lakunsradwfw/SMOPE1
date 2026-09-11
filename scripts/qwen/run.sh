#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/qwen/run.sh DATASET [smoke|full|preflight] [smope|head_only] [extra arguments]'
  exit 2
fi
DATASET="$1"; shift
MODE="${1:-smoke}"; if [[ $# -gt 0 ]]; then shift; fi
METHOD="${1:-smope}"; if [[ $# -gt 0 ]]; then shift; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
for SEED in ${SEEDS:-0}; do
  OUT="${OUTPUT_ROOT:-outputs/qwen}/${DATASET}/${METHOD}/${MODE}/seed-${SEED}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-2}" -m qwen_smope.train \
    --dataset "$DATASET" --method "$METHOD" --mode "$MODE" --seed "$SEED" \
    --model-path "${MODEL_PATH:-pretrained/Qwen3.5-9B-Base}" --data-root "${DATA_ROOT:-data}" \
    --output "$OUT" "$@"
done
