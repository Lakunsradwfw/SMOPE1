"""Combine all completed task summaries under an output root (stdlib only)."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
from statistics import mean, stdev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="outputs/qwen")
    a = p.parse_args()
    root = Path(a.root)
    rows = []
    for path in sorted(root.rglob("summary.json")):
        env = json.loads((path.parent / "environment.json").read_text(encoding="utf-8"))
        summary = json.loads(path.read_text(encoding="utf-8"))
        events = [json.loads(line) for line in (path.parent / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        args = env["arguments"]
        key = {k: args[k] for k in ("dataset", "method", "mode", "seed", "experts", "topk")}
        comparable = {k: v for k, v in env["signature"]["arguments"].items() if k != "seed"}
        comparable.update(world_size=env["signature"]["world_size"], model_config=env["signature"]["config"],
                          git_commit=env["git_commit"], torch=env["torch"], transformers=env["transformers"])
        key["configuration_id"] = hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()[:12]
        stages = [e for e in events if e["type"] == "stage"]
        key.update({k: v for k, v in summary.items() if not isinstance(v, list)})
        key.update(tasks=len(summary["accuracy_history"]), complete=any(e["type"] == "run_complete" for e in events),
                   elapsed_stage_seconds=sum(e["seconds"] for e in stages),
                   peak_allocated_gib=max((m["allocated_gib"] for e in stages for m in e["memory"]), default=0),
                   path=str(path.parent))
        rows.append(key)
    if not rows:
        raise SystemExit("No summary.json found")
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Never combine different modes, k, expert counts or incomplete runs.
    groups = {}
    for row in rows:
        if row["complete"]:
            key = tuple(row[k] for k in ("dataset", "method", "mode", "experts", "topk", "configuration_id"))
            groups.setdefault(key, []).append(row["final_average_accuracy"])
    lines = ["# Comparison", "", "See comparison.csv for per-run metrics and paths.", "",
             "Timing includes retried stages after interrupted runs; incomplete runs are excluded below.", ""]
    for key, values in groups.items():
        lines.append(f"- {' / '.join(map(str, key))}: accuracy {mean(values):.3f} ± {stdev(values) if len(values) > 1 else 0:.3f}; n={len(values)}")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "comparison.csv")


if __name__ == "__main__":
    main()
