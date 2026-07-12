"""Summarize matched SMoPE forgetting-audit runs.

Example:
    python utils/summarize_forgetting_audit.py \
      outputs/cifar-100/10-task/forgetting-audit-*
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def transpose_history(history):
    if not history:
        return []
    return [list(repeat) for repeat in zip(*history)]


def load_router_records(run_dir: Path):
    path = run_dir / "forgetting_audit" / "router_audit.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_run(run_dir: Path):
    acc_path = run_dir / "results-acc" / "global.yaml"
    fr_path = run_dir / "results-fr" / "global.yaml"
    if not acc_path.exists() or not fr_path.exists():
        raise FileNotFoundError(f"Missing result YAML files under {run_dir}")

    acc_repeats = transpose_history(load_yaml(acc_path)["history"])
    fr_repeats = transpose_history(load_yaml(fr_path)["history"])
    args_path = run_dir / "args.yaml"
    run_args = load_yaml(args_path) if args_path.exists() else {}
    router_records = load_router_records(run_dir)
    final_checkpoint = len(acc_repeats[0])

    rows = []
    for repeat_index, (acc, fr) in enumerate(zip(acc_repeats, fr_repeats), start=1):
        final_router = [
            record
            for record in router_records
            if record["repeat_id"] == repeat_index
            and record["checkpoint_task"] == final_checkpoint
            and record["eval_task"] < final_checkpoint
        ]
        rows.append(
            {
                "run": run_dir.name,
                "freeze_component": run_args.get("audit_freeze_component", "none"),
                "repeat": repeat_index,
                "seed": (
                    run_args.get("seeds", [repeat_index - 1])[repeat_index - 1]
                    if len(run_args.get("seeds", [])) >= repeat_index
                    else repeat_index - 1
                ),
                "FAA": float(acc[-1]),
                "CAA": float(mean(acc)),
                "FR": float(fr[-1]),
                "final_router_change_rate": (
                    mean(item["router_change_rate"] for item in final_router)
                    if final_router
                    else ""
                ),
                "final_historical_replay_gain": (
                    mean(item["replay_gain"] for item in final_router)
                    if final_router
                    else ""
                ),
            }
        )
    return rows


def print_group_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["run"], []).append(row)
    for run, run_rows in grouped.items():
        print(f"\n{run} (n={len(run_rows)})")
        for metric in ("FAA", "CAA", "FR"):
            values = [float(row[metric]) for row in run_rows]
            print(f"  {metric}: {mean(values):.4f} +/- {pstdev(values):.4f}")
        for metric in (
            "final_router_change_rate",
            "final_historical_replay_gain",
        ):
            values = [float(row[metric]) for row in run_rows if row[metric] != ""]
            if values:
                print(f"  {metric}: {mean(values):.4f} +/- {pstdev(values):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("forgetting_audit_summary.csv")
    )
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        rows.extend(summarize_run(run_dir))
    if not rows:
        raise SystemExit("No completed audit runs found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print_group_summary(rows)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
