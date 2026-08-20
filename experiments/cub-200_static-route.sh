#!/usr/bin/env bash
set -euo pipefail

GPUID=${GPUID:-0}
REPEAT=${REPEAT:-5}
OVERWRITE=${OVERWRITE:-1}
OUTDIR=${OUTDIR:-outputs/efficiency/cub-200/10-task/static-route}

mkdir -p "$OUTDIR"
python -u run.py --config configs/cub-200_prompt_smope.yaml \
    --gpuid "$GPUID" --repeat "$REPEAT" --overwrite "$OVERWRITE" \
    --learner_type prompt --learner_name OnePrompt \
    --prompt_param 50 5 1e-5 5e-5 0.4 --seeds 0 1 2 3 4 \
    --crct_epochs 50 --ca_batch_size_ratio 1 --pretrained_weight sup21k \
    --smope_mode static_route --profile_flops \
    --log_dir "$OUTDIR"
