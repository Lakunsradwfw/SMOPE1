#!/usr/bin/env bash
set -euo pipefail

RUNDIR=${RUNDIR:-outputs/cifar-100/10-task/within-expert-interference}
python utils/analyze_expert_interference.py "${RUNDIR}"
