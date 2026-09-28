#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-full}"; if [[ $# -gt 0 ]]; then shift; fi
ROOT="${OUTPUT_ROOT:-outputs/coin-qwen3.5-9b}"
for METHOD in zero_shot lora smope; do
  OUTPUT_ROOT="$ROOT" bash "$(dirname "${BASH_SOURCE[0]}")/run.sh" "$METHOD" "$MODE" "$@"
done
