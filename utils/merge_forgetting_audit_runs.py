"""Merge completed multi-repeat forgetting-audit runs into a summary-only run.

Unlike the one-seed shard merger, each input may contain one or more explicit
seeds. Metric histories, top-level audit JSONL, and manifests are concatenated.
Model checkpoints are intentionally not copied because the merged directory is
for five-seed paired statistics, not offline restoration.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml


CHECKED_ARGS = (
    "dataset",
    "max_task",
    "crct_epochs",
    "prompt_param",
    "rand_split",
    "audit_freeze_component",
    "audit_freeze_from_task",
    "audit_freeze_until_task",
    "audit_router_max_samples",
    "audit_checkpoints",
    "audit_main_epochs",
)


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)


def load_run_info(run_dir):
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing {args_path}")
    run_args = load_yaml(args_path)
    seeds = [int(seed) for seed in run_args.get("seeds") or []]
    repeat = int(run_args.get("repeat", 0))
    if repeat <= 0 or repeat != len(seeds):
        raise ValueError(
            f"{run_dir}: repeat={repeat} must equal the number of explicit seeds={seeds}."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{run_dir}: duplicate seeds {seeds}")
    return {"dir": run_dir, "args": run_args, "seeds": seeds, "repeat": repeat}


def validate_runs(runs):
    reference = runs[0]
    all_seeds = []
    for run in runs:
        all_seeds.extend(run["seeds"])
        for key in CHECKED_ARGS:
            default = 0 if key == "audit_main_epochs" else None
            if run["args"].get(key, default) != reference["args"].get(key, default):
                raise ValueError(
                    f"{run['dir']} differs on {key}: {run['args'].get(key, default)!r} "
                    f"!= {reference['args'].get(key, default)!r}"
                )
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError(f"Seeds overlap across input runs: {all_seeds}")
    return all_seeds


def merge_metrics(runs, output_dir):
    metric_sets = [
        {path.relative_to(run["dir"]) for path in run["dir"].glob("results-*/*.yaml")}
        for run in runs
    ]
    if not metric_sets[0] or any(current != metric_sets[0] for current in metric_sets[1:]):
        raise ValueError("Metric YAML file sets are empty or differ across runs.")
    for relative in sorted(metric_sets[0]):
        histories = []
        for run in runs:
            payload = load_yaml(run["dir"] / relative)
            history = np.asarray(payload.get("history"), dtype=float)
            if history.ndim not in (2, 3) or history.shape[-1] != run["repeat"]:
                raise ValueError(
                    f"{run['dir'] / relative}: history repeat axis {history.shape} "
                    f"does not match repeat={run['repeat']}"
                )
            histories.append(history)
        if any(history.shape[:-1] != histories[0].shape[:-1] for history in histories[1:]):
            raise ValueError(f"Metric history shape mismatch for {relative}")
        combined = np.concatenate(histories, axis=-1)
        payload = {"mean": combined.mean(axis=-1).tolist(), "history": combined.tolist()}
        if combined.shape[-1] > 2:
            payload["std"] = combined.std(axis=-1).tolist()
        dump_yaml(output_dir / relative, payload)


def repeat_offsets(runs):
    offsets = []
    current = 0
    for run in runs:
        offsets.append(current)
        current += run["repeat"]
    return offsets


def merge_jsonl(runs, output_dir, offsets):
    name_sets = [
        {path.name for path in (run["dir"] / "forgetting_audit").glob("*.jsonl")}
        for run in runs
    ]
    common_names = set.intersection(*name_sets)
    required_names = {"router_audit.jsonl", "component_drift.jsonl"}
    missing_required = required_names - common_names
    if missing_required:
        raise ValueError(
            f"Required audit JSONL files are not common to all runs: {sorted(missing_required)}"
        )
    omitted_partial = set.union(*name_sets) - common_names
    for name in sorted(common_names):
        rows = []
        for run, offset in zip(runs, offsets):
            with (run["dir"] / "forgetting_audit" / name).open(
                "r", encoding="utf-8"
            ) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "repeat_id" in row:
                        source_repeat = int(row["repeat_id"])
                        if not 1 <= source_repeat <= run["repeat"]:
                            raise ValueError(
                                f"{name}: repeat_id={source_repeat} outside {run['dir']} repeat range"
                            )
                        row["repeat_id"] = offset + source_repeat
                    rows.append(row)
        target = output_dir / "forgetting_audit" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sorted(omitted_partial)


def merge_manifests(runs, output_dir, offsets):
    merged = None
    seen = set()
    for run, offset in zip(runs, offsets):
        path = run["dir"] / "forgetting_audit" / "audit_sample_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if merged is None:
            merged = {key: value for key, value in payload.items() if key != "entries"}
            merged["entries"] = []
        for row in payload.get("entries", []):
            row = dict(row)
            source_repeat = int(row.get("repeat_id", run["seeds"].index(int(row["seed"])) + 1))
            row["repeat_id"] = offset + source_repeat
            key = (int(row["seed"]), int(row["eval_task"]))
            if key in seen:
                raise ValueError(f"Duplicate manifest seed/task entry: {key}")
            seen.add(key)
            merged["entries"].append(row)
    merged["entries"].sort(
        key=lambda row: (int(row["repeat_id"]), int(row["seed"]), int(row["eval_task"]))
    )
    target = output_dir / "forgetting_audit" / "audit_sample_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_common_metadata(runs, output_dir):
    name = "expert_structure.json"
    sources = [run["dir"] / "forgetting_audit" / name for run in runs]
    existing = [path for path in sources if path.exists()]
    if not existing:
        return
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in existing]
    if any(payload != payloads[0] for payload in payloads[1:]):
        raise ValueError("expert_structure.json differs across input runs")
    target = output_dir / "forgetting_audit" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(existing[0], target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    runs = [load_run_info(path.resolve()) for path in args.run_dirs]
    if len(runs) < 2:
        raise ValueError("At least two run directories are required.")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    seeds = validate_runs(runs)
    offsets = repeat_offsets(runs)

    output_dir.mkdir(parents=True)
    merged_args = dict(runs[0]["args"])
    merged_args.update(
        {
            "repeat": len(seeds),
            "seeds": seeds,
            "log_dir": str(output_dir),
            "audit_main_epochs": int(merged_args.get("audit_main_epochs", 0)),
        }
    )
    dump_yaml(output_dir / "args.yaml", merged_args)
    merge_metrics(runs, output_dir)
    omitted_partial_jsonl = merge_jsonl(runs, output_dir, offsets)
    merge_manifests(runs, output_dir, offsets)
    copy_common_metadata(runs, output_dir)
    (output_dir / "merge_sources.json").write_text(
        json.dumps(
            {
                "summary_only": True,
                "seeds": seeds,
                "source_dirs": [str(run["dir"]) for run in runs],
                "omitted_partial_jsonl": omitted_partial_jsonl,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Merged seeds {seeds} into summary-only run {output_dir}")


if __name__ == "__main__":
    main()
