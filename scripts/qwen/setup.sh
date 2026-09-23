#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-qwen.txt
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__); print("CUDA", torch.cuda.is_available(), "GPUs", torch.cuda.device_count())'
python -m unittest discover -s tests_qwen -v
