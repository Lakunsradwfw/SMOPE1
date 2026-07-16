#!/usr/bin/env bash
# Safely prune completed forgetting-audit outputs on a server.
#
# Default: preview only. Use --apply to delete files.
# The router baseline's checkpoint.pt files are always retained for component
# restoration. Use --delete-nonbaseline-checkpoints only for non-baseline runs
# (for example, a historical gradient run that unnecessarily saved them).

set -euo pipefail

ROOT="outputs/cifar-100/10-task"
APPLY=0
DELETE_NONBASELINE_CHECKPOINTS=0

usage() {
  cat <<'EOF'
Usage:
  bash experiments/cleanup_forgetting_audit_outputs.sh [options]

Options:
  --root PATH                         Parent directory containing forgetting-audit-* runs.
                                      Default: outputs/cifar-100/10-task
  --apply                             Actually delete files. Default is a dry run.
  --delete-nonbaseline-checkpoints    Also delete checkpoint.pt outside the router baseline.
  -h, --help                          Show this help.

The script only cleans a condition after all three result YAML files exist,
parse successfully, and report the same positive number of completed repeats:
  results-acc/pt.yaml
  results-acc/global.yaml
  results-fr/global.yaml

For each completed repeat it removes class.pth and component_reference/*.pt.
It never deletes forgetting-audit-router/models/.../checkpoint.pt.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || { echo "--root requires a path" >&2; exit 2; }
      ROOT="$2"
      shift 2
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --delete-nonbaseline-checkpoints)
      DELETE_NONBASELINE_CHECKPOINTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -d "$ROOT" ]] || { echo "Output root does not exist: $ROOT" >&2; exit 2; }
ROOT="$(cd "$ROOT" && pwd -P)"

remove_file() {
  local path="$1"
  if [[ "$APPLY" == "1" ]]; then
    rm -f -- "$path"
    echo "deleted  $path"
  else
    echo "would delete  $path"
  fi
}

completed_repeat_count() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except ImportError as error:
    raise SystemExit(f"PyYAML is required for safe cleanup: {error}")

run_dir = Path(sys.argv[1])
required = (
    run_dir / "results-acc" / "pt.yaml",
    run_dir / "results-acc" / "global.yaml",
    run_dir / "results-fr" / "global.yaml",
)

def repeat_count(history):
    current = history
    while isinstance(current, list) and current:
        first = current[0]
        if not isinstance(first, list):
            return len(current)
        current = first
    return 0

counts = []
for path in required:
    if not path.is_file():
        raise SystemExit(f"missing result YAML: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        raise SystemExit(f"cannot parse {path}: {error}")
    count = repeat_count(payload.get("history"))
    if count <= 0:
        raise SystemExit(f"incomplete history in {path}")
    counts.append(count)

if len(set(counts)) != 1:
    raise SystemExit(f"result YAML repeat counts disagree: {counts}")
print(counts[0])
PY
}

conditions=(
  forgetting-audit-router
  forgetting-audit-gradient
  forgetting-audit-freeze-prompt
  forgetting-audit-freeze-key
  forgetting-audit-freeze-value
  forgetting-audit-freeze-classifier
  forgetting-audit-freeze-key-value
)

class_count=0
reference_count=0
checkpoint_count=0

for condition in "${conditions[@]}"; do
  run_dir="$ROOT/$condition"
  [[ -d "$run_dir" ]] || continue

  if ! completed="$(completed_repeat_count "$run_dir")"; then
    echo "skip $condition: result YAML is missing, invalid, or incomplete" >&2
    continue
  fi
  echo "$condition: $completed completed repeat(s)"

  for ((repeat_id = 1; repeat_id <= completed; repeat_id++)); do
    model_dir="$run_dir/models/repeat-$repeat_id"
    if [[ -d "$model_dir" ]]; then
      while IFS= read -r -d '' path; do
        remove_file "$path"
        ((class_count += 1))
      done < <(find "$model_dir" -type f -name class.pth -print0)
    fi

    reference_dir="$run_dir/forgetting_audit/repeat-$repeat_id/component_reference"
    if [[ -d "$reference_dir" ]]; then
      while IFS= read -r -d '' path; do
        remove_file "$path"
        ((reference_count += 1))
      done < <(find "$reference_dir" -type f -name '*.pt' -print0)
    fi

    if [[ "$DELETE_NONBASELINE_CHECKPOINTS" == "1" && "$condition" != "forgetting-audit-router" && -d "$model_dir" ]]; then
      while IFS= read -r -d '' path; do
        remove_file "$path"
        ((checkpoint_count += 1))
      done < <(find "$model_dir" -type f -name checkpoint.pt -print0)
    fi
  done
done

echo "class.pth files: $class_count"
echo "component references: $reference_count"
echo "non-baseline checkpoint.pt files: $checkpoint_count"
echo "router baseline checkpoint.pt files: retained"
if [[ "$APPLY" != "1" ]]; then
  echo "Dry run only. Re-run with --apply after checking the listed paths."
fi
