#!/usr/bin/env bash
# Run all 5 CIFAR-100 forgetting audit experiments on 2 GPUs.
#   GPU 0 (2 serial): router → freeze_classifier
#   GPU 1 (3 serial): freeze_prompt → freeze_key → freeze_value
# Both GPUs start in parallel; exits non-zero if any script fails.
#
# Usage:   bash experiments/cifar-100_forgetting_audit_all.sh
# Override: DATASET=... OUTDIR_PREFIX=... bash experiments/...

set -euo pipefail

DATASET="${DATASET:-cifar-100}"
OUTDIR_PREFIX="${OUTDIR_PREFIX:-outputs/${DATASET}/10-task}"

# ── GPU 0 queue ──────────────────────────────────────────────
run_gpu0() {
  echo "━━━ [GPU 0] router baseline ━━━"
  GPUID=0 OUTDIR="${OUTDIR_PREFIX}/forgetting-audit-router" \
    bash experiments/cifar-100_forgetting_audit_router.sh
  echo "━━━ [GPU 0] router done ✓ ━━━"

  echo "━━━ [GPU 0] freeze classifier ━━━"
  GPUID=0 OUTDIR="${OUTDIR_PREFIX}/forgetting-audit-freeze-classifier" \
    bash experiments/cifar-100_forgetting_audit_freeze_classifier.sh
  echo "━━━ [GPU 0] freeze classifier done ✓ ━━━"
}

# ── GPU 1 queue ──────────────────────────────────────────────
run_gpu1() {
  echo "━━━ [GPU 1] freeze prompt ━━━"
  GPUID=1 OUTDIR="${OUTDIR_PREFIX}/forgetting-audit-freeze-prompt" \
    bash experiments/cifar-100_forgetting_audit_freeze_prompt.sh
  echo "━━━ [GPU 1] freeze prompt done ✓ ━━━"

  echo "━━━ [GPU 1] freeze key ━━━"
  GPUID=1 OUTDIR="${OUTDIR_PREFIX}/forgetting-audit-freeze-key" \
    bash experiments/cifar-100_forgetting_audit_freeze_key.sh
  echo "━━━ [GPU 1] freeze key done ✓ ━━━"

  echo "━━━ [GPU 1] freeze value ━━━"
  GPUID=1 OUTDIR="${OUTDIR_PREFIX}/forgetting-audit-freeze-value" \
    bash experiments/cifar-100_forgetting_audit_freeze_value.sh
  echo "━━━ [GPU 1] freeze value done ✓ ━━━"
}

# ── Launch both queues in parallel ───────────────────────────
run_gpu0 & PID0=$!
run_gpu1 & PID1=$!

FAIL=0
wait $PID0 || { echo "❌ GPU 0 queue FAILED!"; FAIL=1; }
wait $PID1 || { echo "❌ GPU 1 queue FAILED!"; FAIL=1; }

if [[ $FAIL -ne 0 ]]; then
  echo "❌ One or both queues failed — check logs above."
  exit 1
fi

echo "═══════════════════════════════════════════"
echo "✓ All 5 experiments finished successfully."
echo "═══════════════════════════════════════════"
