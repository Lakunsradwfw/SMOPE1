"""Small auditable experiment outputs; never write base-model weights."""
import csv
import json
from pathlib import Path

import numpy as np


def metrics(matrix):
    a = np.asarray(matrix, dtype=float)
    t = len(a)
    history = [float(a[i, :i + 1].mean()) for i in range(t)]
    forgetting = float(np.mean([max(a[j:t - 1, j]) - a[t - 1, j]
                               for j in range(t - 1)])) if t > 1 else 0.0
    bwt = float(np.mean([a[t - 1, j] - a[j, j] for j in range(t - 1)])) if t > 1 else 0.0
    return dict(final_average_accuracy=history[-1], average_incremental_accuracy=float(np.mean(history)),
                forgetting=forgetting, backward_transfer=bwt, accuracy_history=history)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def event(directory, value):
    with (Path(directory) / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def summarize(directory, matrix):
    directory = Path(directory)
    # Matrix is lower triangular; unseen-task entries are blank, never fake zero accuracy.
    with (directory / "accuracy_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["after_task"] + [f"test_task_{i + 1}" for i in range(len(matrix))])
        for i, row in enumerate(matrix):
            writer.writerow([i + 1] + row[:i + 1] + [""] * (len(matrix) - i - 1))
    padded = [row + [0.] * (len(matrix) - len(row)) for row in matrix]
    result = metrics(padded)
    write_json(directory / "summary.json", result)
    lines = ["# Qwen SMoPE experiment", "", "Accuracy/forgetting/BWT units: percentage points.", ""]
    lines += [f"- {k}: {v:.4f}" for k, v in result.items() if isinstance(v, float)]
    lines += ["", "See accuracy_matrix.csv, per_class_task_*.json, experts_task_*.json and events.jsonl.",
              "Train accuracy uses current-task classes; test accuracy uses all seen classes.",
              "Prototype correction uses diagonal covariance; no raw-image replay.",
              "Only latest adapter/resume state is retained; no base weights are copied."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
