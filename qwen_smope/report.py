"""Auditable, lightweight experiment outputs; base-model weights are never copied."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def continual_metrics(matrix: list[list[float]]) -> dict[str, Any]:
    if not matrix:
        return {}
    stages = len(matrix)
    history = [float(np.mean(row[:index + 1])) for index, row in enumerate(matrix)]
    final = matrix[-1]
    diagonal = [matrix[index][index] for index in range(stages)]
    bwt = float(sum(final[index] - diagonal[index] for index in range(stages)) / stages)
    forgetting = float(np.mean([
        max(matrix[stage][task] for stage in range(task, stages)) - final[task]
        for task in range(stages - 1)
    ])) if stages > 1 else 0.0
    return {"final_average_accuracy": history[-1],
            "mean_average_accuracy": float(np.mean(history)),
            "new_task_accuracy": float(np.mean(diagonal)),
            "backward_transfer": bwt, "forgetting": forgetting,
            "accuracy_history": history}


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: str | Path, values: list[dict] | dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [values] if isinstance(values, dict) else values
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def event(directory: str | Path, value: dict) -> None:
    append_jsonl(Path(directory) / "events.jsonl", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def write_matrix(directory: str | Path, tasks: list[str], matrix: list[list[float]]) -> dict[str, Any]:
    directory = Path(directory)
    with (directory / "accuracy_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_task", *tasks])
        for index, row in enumerate(matrix):
            writer.writerow([tasks[index], *row[:index + 1], *([""] * (len(tasks) - index - 1))])
    result = continual_metrics(matrix)
    write_json(directory / "summary.json", result)
    return result
