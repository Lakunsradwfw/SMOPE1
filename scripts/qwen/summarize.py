"""Summarize completed CoIN runs without importing training dependencies."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="outputs/coin-qwen3.5-9b")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for summary_path in sorted(root.rglob("summary.json")):
        environment_path = summary_path.parent / "environment.json"
        events_path = summary_path.parent / "events.jsonl"
        if not environment_path.is_file() or not events_path.is_file():
            continue
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line]
        run_args = environment["arguments"]
        row = {"method": run_args["method"], "mode": run_args["mode"], "seed": run_args["seed"],
               "tasks": ",".join(run_args["tasks"]),
               "complete": any(item["type"] == "run_complete" for item in events),
               "peak_allocated_gib": max((memory["allocated_gib"] for item in events
                                           if item["type"] == "stage" for memory in item["memory"]), default=0),
               "path": str(summary_path.parent)}
        row.update({key: value for key, value in summary.items() if isinstance(value, (int, float))})
        rows.append(row)
    if not rows:
        raise SystemExit("No completed-format summary.json found")
    fields = sorted({key for row in rows for key in row})
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        value = row.get("final_average_accuracy", row.get("average_accuracy"))
        if row["complete"] and value is not None:
            groups.setdefault((row["method"], row["mode"], row["tasks"]), []).append(value)
    lines = ["# CoIN Qwen3.5 comparison", ""]
    for key, values in groups.items():
        deviation = stdev(values) if len(values) > 1 else 0.0
        lines.append(f"- {' / '.join(key)}: {mean(values):.3f} ± {deviation:.3f}; n={len(values)}")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "comparison.csv")


if __name__ == "__main__":
    main()
