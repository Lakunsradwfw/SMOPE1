"""Merge one-seed forgetting-audit shards into one seed-paired run directory.

Each input directory must be produced with ``REPEAT=1`` and one explicit seed.
The output layout is compatible with ``utils/summarize_forgetting_audit.py``.
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
    "audit_router_max_samples",
    "audit_checkpoints",
)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, default_flow_style=False, sort_keys=False)


def shard_args(run_dir: Path):
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing {args_path}")
    args = load_yaml(args_path)
    seeds = list(args.get("seeds") or [])
    if int(args.get("repeat", 0)) != 1 or len(seeds) != 1:
        raise ValueError(
            f"{run_dir} must have REPEAT=1 and exactly one seed; got "
            f"repeat={args.get('repeat')}, seeds={seeds}."
        )
    return args, int(seeds[0])


def validate_args(shards):
    reference, _ = shards[0]
    for args, seed in shards[1:]:
        for key in CHECKED_ARGS:
            if args.get(key) != reference.get(key):
                raise ValueError(
                    f"Shard seed={seed} differs on {key}: "
                    f"{args.get(key)!r} != {reference.get(key)!r}"
                )


def merge_metrics(input_dirs, output_dir: Path) -> None:
    metric_files = {
        path.relative_to(input_dirs[0])
        for path in input_dirs[0].glob("results-*/*.yaml")
    }
    for run_dir in input_dirs[1:]:
        current = {path.relative_to(run_dir) for path in run_dir.glob("results-*/*.yaml")}
        if current != metric_files:
            raise ValueError(f"Metric files differ in {run_dir}.")

    for relative in sorted(metric_files):
        payloads = [load_yaml(run_dir / relative) for run_dir in input_dirs]
        histories = [np.asarray(payload["history"], dtype=float) for payload in payloads]
        if any(history.ndim not in (2, 3) or history.shape[-1] != 1 for history in histories):
            raise ValueError(f"{relative} must contain one-repeat 2D or 3D histories.")
        if any(history.shape[:-1] != histories[0].shape[:-1] for history in histories[1:]):
            raise ValueError(f"History shape mismatch for {relative}.")
        combined = np.concatenate(histories, axis=-1)
        output = {"mean": combined.mean(axis=-1).tolist(), "history": combined.tolist()}
        if combined.shape[-1] > 2:
            output["std"] = combined.std(axis=-1).tolist()
        dump_yaml(output_dir / relative, output)


def merge_audit_jsonl(input_dirs, output_dir: Path) -> None:
    names = set()
    for run_dir in input_dirs:
        audit_dir = run_dir / "forgetting_audit"
        names.update(path.name for path in audit_dir.glob("*.jsonl"))
    for name in sorted(names):
        rows = []
        for repeat_id, run_dir in enumerate(input_dirs, start=1):
            path = run_dir / "forgetting_audit" / name
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; cannot merge partial audit logs.")
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if "repeat_id" in row:
                            row["repeat_id"] = repeat_id
                        rows.append(row)
        target = output_dir / "forgetting_audit" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_manifests(input_dirs, output_dir: Path) -> None:
    merged = None
    seen = set()
    for repeat_id, run_dir in enumerate(input_dirs, start=1):
        path = run_dir / "forgetting_audit" / "audit_sample_manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if merged is None:
            merged = {key: value for key, value in payload.items() if key != "entries"}
            merged["entries"] = []
        for row in payload.get("entries", []):
            row = dict(row)
            row["repeat_id"] = repeat_id
            key = (int(row["seed"]), int(row["eval_task"]))
            if key in seen:
                raise ValueError(f"Duplicate manifest entry for seed/task={key}.")
            seen.add(key)
            merged["entries"].append(row)
    merged["entries"].sort(key=lambda row: (int(row["repeat_id"]), int(row["seed"]), int(row["eval_task"])))
    target = output_dir / "forgetting_audit" / "audit_sample_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_repeat_artifacts(input_dirs, output_dir: Path) -> None:
    for repeat_id, run_dir in enumerate(input_dirs, start=1):
        for parent in ("forgetting_audit", "models"):
            source = run_dir / parent / "repeat-1"
            if source.exists():
                target = output_dir / parent / f"repeat-{repeat_id}"
                shutil.copytree(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="one-seed shard directories")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    input_dirs = [path.resolve() for path in args.run_dirs]
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if len(set(input_dirs)) != len(input_dirs):
        raise ValueError("Input shard directories must be distinct.")

    shards = [shard_args(path) for path in input_dirs]
    seeds = [seed for _, seed in shards]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique; got {seeds}.")
    validate_args(shards)

    output_dir.mkdir(parents=True)
    merged_args = dict(shards[0][0])
    merged_args.update({"repeat": len(seeds), "seeds": seeds, "log_dir": str(output_dir)})
    dump_yaml(output_dir / "args.yaml", merged_args)
    merge_metrics(input_dirs, output_dir)
    merge_audit_jsonl(input_dirs, output_dir)
    merge_manifests(input_dirs, output_dir)
    copy_repeat_artifacts(input_dirs, output_dir)
    (output_dir / "merge_sources.json").write_text(
        json.dumps({"seeds": seeds, "source_dirs": [str(path) for path in input_dirs]}, indent=2),
        encoding="utf-8",
    )
    print(f"Merged seeds {seeds} into {output_dir}")


if __name__ == "__main__":
    main()
