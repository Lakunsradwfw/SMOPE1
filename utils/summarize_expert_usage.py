"""Offline expert-usage and soft-ownership summaries for router snapshots."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.audit_metrics import SCHEMA_VERSION


def _load_snapshot(path):
    return torch.load(path, map_location="cpu")


def discover_snapshots(run_dir):
    root = Path(run_dir) / "forgetting_audit"
    current = sorted(root.glob("repeat-*/router_current/*.pt"))
    if current:
        return current
    return sorted(root.glob("repeat-*/router_reference/*.pt"))


def summarize_expert_usage(run_dir, condition=None):
    run_dir = Path(run_dir)
    condition = condition or run_dir.name
    usage_rows = []
    grouped_counts = defaultdict(lambda: defaultdict(int))
    grouped_samples = {}

    for path in discover_snapshots(run_dir):
        snapshot = _load_snapshot(path)
        repeat_id = int(snapshot["repeat_id"])
        seed = int(snapshot["seed"])
        checkpoint_task = int(snapshot["checkpoint_task"])
        eval_task = int(snapshot["eval_task"])
        topk = int(snapshot["topk"])
        samples = sum(int(batch["targets"].numel()) for batch in snapshot["batches"])
        for batch in snapshot["batches"]:
            for layer, layer_state in batch["layers"].items():
                indices = layer_state["indices"]
                num_heads = int(indices.size(1))
                num_experts = int(
                    max(
                        snapshot.get("num_experts", 0),
                        int(indices.max()) + 1 if indices.numel() else 0,
                    )
                )
                for head in range(num_heads):
                    values, counts = torch.unique(
                        indices[:, head].reshape(-1), return_counts=True
                    )
                    count_map = dict(zip(values.tolist(), counts.tolist()))
                    # Router snapshots may not explicitly store pool size.  The
                    # maximum observed expert is a lower bound; structure.json
                    # is used below to fill the complete pool when available.
                    group = (repeat_id, seed, checkpoint_task, int(layer), head)
                    grouped_samples[(group, eval_task)] = (samples, topk)
                    for expert in range(num_experts):
                        grouped_counts[group][(eval_task, expert)] += int(
                            count_map.get(expert, 0)
                        )

    structure_path = run_dir / "forgetting_audit" / "expert_structure.json"
    num_experts = None
    if structure_path.exists():
        import json

        with structure_path.open("r", encoding="utf-8") as handle:
            num_experts = int(json.load(handle)["num_experts_per_layer_head"])

    ownership_rows = []
    for group, counts in sorted(grouped_counts.items()):
        repeat_id, seed, checkpoint_task, layer, head = group
        eval_tasks = sorted({eval_task for eval_task, _ in counts})
        pool_size = num_experts or (
            max((expert for _, expert in counts), default=-1) + 1
        )
        for expert in range(pool_size):
            task_counts = [counts.get((task, expert), 0) for task in eval_tasks]
            total_for_expert = sum(task_counts)
            probabilities = [
                count / total_for_expert for count in task_counts if count > 0
            ]
            entropy = -sum(value * math.log(value) for value in probabilities)
            normalized_entropy = (
                entropy / math.log(len(eval_tasks)) if len(eval_tasks) > 1 else 0.0
            )
            for eval_task, usage_count in zip(eval_tasks, task_counts):
                samples, topk = grouped_samples[(group, eval_task)]
                denominator = samples * topk
                task_share = (
                    usage_count / total_for_expert if total_for_expert else math.nan
                )
                base = {
                    "schema_version": SCHEMA_VERSION,
                    "run": run_dir.name,
                    "condition": condition,
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "checkpoint_task": checkpoint_task,
                    "eval_task": eval_task,
                    "layer": layer,
                    "head": head,
                    "expert": expert,
                    "samples": samples,
                    "topk": topk,
                    "usage_count": usage_count,
                    "usage_rate": usage_count / denominator if denominator else math.nan,
                    "task_share": task_share,
                    "expert_task_entropy": entropy,
                    "normalized_entropy": normalized_entropy,
                }
                usage_rows.append(dict(base))
                ownership_rows.append(dict(base))

        for eval_task in eval_tasks:
            observed = sum(
                counts.get((eval_task, expert), 0) for expert in range(pool_size)
            )
            samples, topk = grouped_samples[(group, eval_task)]
            expected = samples * topk
            if observed != expected:
                raise ValueError(
                    "Expert usage conservation failed for "
                    f"seed={seed}, checkpoint={checkpoint_task}, eval={eval_task}, "
                    f"layer={layer}, head={head}: {observed} != {expected}."
                )

    return usage_rows, ownership_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "forgetting_audit"
    usage, ownership = summarize_expert_usage(args.run_dir)
    write_csv(output_dir / "expert_usage_long.csv", usage)
    write_csv(output_dir / "expert_ownership_long.csv", ownership)
    print(f"Wrote {len(usage)} usage rows and {len(ownership)} ownership rows.")


if __name__ == "__main__":
    main()
