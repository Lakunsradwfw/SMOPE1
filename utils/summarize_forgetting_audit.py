"""Build matched long-form tables and a conservative forgetting-audit report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.audit_metrics import (
    SCHEMA_VERSION,
    clipped_recovery_ratio,
    interpretation_flags,
    paired_effect_summary,
    route_set_metrics,
    safe_recovery_ratio,
)
from utils.summarize_expert_usage import summarize_expert_usage


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def _history_value(history, checkpoint, repeat):
    return float(history[checkpoint][repeat])


def load_run(run_dir):
    run_dir = Path(run_dir)
    args_path = run_dir / "args.yaml"
    pt_path = run_dir / "results-acc" / "pt.yaml"
    global_path = run_dir / "results-acc" / "global.yaml"
    fr_path = run_dir / "results-fr" / "global.yaml"
    missing = [path for path in (args_path, pt_path, global_path, fr_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing audit inputs: " + ", ".join(map(str, missing)))

    run_args = load_yaml(args_path) or {}
    pt_history = load_yaml(pt_path).get("history", [])
    global_history = load_yaml(global_path).get("history", [])
    fr_history = load_yaml(fr_path).get("history", [])
    if not pt_history or not global_history or not fr_history:
        raise ValueError(f"Incomplete metric history under {run_dir}")
    num_tasks = len(pt_history)
    num_repeats = len(pt_history[0][0])
    seeds_arg = list(run_args.get("seeds") or [])
    if seeds_arg and len(seeds_arg) < num_repeats:
        raise ValueError(
            f"{run_dir}: {num_repeats} repeats but only {len(seeds_arg)} explicit seeds."
        )
    seeds = [int(seeds_arg[index] if seeds_arg else index) for index in range(num_repeats)]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{run_dir}: duplicate seeds cannot be paired safely: {seeds}")

    matrices = {}
    for repeat, seed in enumerate(seeds):
        matrices[seed] = [
            [float(pt_history[eval_task][checkpoint][repeat]) for checkpoint in range(num_tasks)]
            for eval_task in range(num_tasks)
        ]
    condition = str(run_args.get("audit_freeze_component", "none"))
    if condition == "none":
        condition = "baseline"
    return {
        "dir": run_dir,
        "run": run_dir.name,
        "condition": condition,
        "args": run_args,
        "num_tasks": num_tasks,
        "seeds": seeds,
        "matrices": matrices,
        "global_history": global_history,
        "fr_history": fr_history,
        "router_records": load_jsonl(run_dir / "forgetting_audit" / "router_audit.jsonl"),
        "drift_records": load_jsonl(run_dir / "forgetting_audit" / "component_drift.jsonl"),
        "gradient_records": load_jsonl(run_dir / "forgetting_audit" / "gradient_direction.jsonl"),
    }


def select_baseline(runs, baseline_dir=None):
    if baseline_dir is not None:
        target = Path(baseline_dir).resolve()
        matches = [run for run in runs if run["dir"].resolve() == target]
    else:
        matches = [run for run in runs if run["condition"] == "baseline"]
    if len(matches) != 1:
        raise ValueError(
            "Exactly one baseline run is required; pass --baseline when inference is ambiguous."
        )
    return matches[0]


def validate_run_pairing(runs, baseline):
    warnings = []
    critical = ("dataset", "max_task", "crct_epochs", "prompt_param", "rand_split")
    for run in runs:
        if run["num_tasks"] != baseline["num_tasks"]:
            raise ValueError(
                f"Task-count mismatch: {run['run']}={run['num_tasks']} vs baseline={baseline['num_tasks']}"
            )
        if set(run["seeds"]) != set(baseline["seeds"]):
            missing = sorted(set(baseline["seeds"]) - set(run["seeds"]))
            extra = sorted(set(run["seeds"]) - set(baseline["seeds"]))
            raise ValueError(
                f"Seed mismatch for {run['run']}: missing={missing}, extra={extra}"
            )
        for key in critical:
            if run["args"].get(key) != baseline["args"].get(key):
                raise ValueError(
                    f"Configuration mismatch for {key}: {run['run']}={run['args'].get(key)!r}, "
                    f"baseline={baseline['args'].get(key)!r}"
                )
        if run["args"].get("audit_router_max_samples") != baseline["args"].get(
            "audit_router_max_samples"
        ):
            warnings.append(
                f"{run['run']} uses a different audit_router_max_samples; router rows are not paired."
            )
    return warnings


def validate_manifests(runs, baseline):
    warnings = []
    baseline_path = baseline["dir"] / "forgetting_audit" / "audit_sample_manifest.json"
    if not baseline_path.exists():
        return ["Baseline audit sample manifest is missing."]
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline_entries = json.load(handle).get("entries", [])
    baseline_index = {
        (int(row["seed"]), int(row["eval_task"])): row for row in baseline_entries
    }
    for run in runs:
        path = run["dir"] / "forgetting_audit" / "audit_sample_manifest.json"
        if not path.exists():
            warnings.append(f"{run['run']} audit sample manifest is missing.")
            continue
        with path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle).get("entries", [])
        index = {(int(row["seed"]), int(row["eval_task"])): row for row in entries}
        for key, baseline_entry in baseline_index.items():
            if key not in index:
                raise ValueError(f"{run['run']} lacks manifest entry seed/task={key}.")
            for field in ("dataset_indices", "sample_ids", "class_ids"):
                if list(index[key][field]) != list(baseline_entry[field]):
                    raise ValueError(
                        f"Manifest mismatch for {run['run']} seed/task={key}, field={field}."
                    )
    return warnings


def build_accuracy_rows(runs, baseline):
    rows = []
    baseline_by_seed = baseline["matrices"]
    for run in runs:
        for repeat_id, seed in enumerate(run["seeds"], start=1):
            matrix = run["matrices"][seed]
            baseline_matrix = baseline_by_seed[seed]
            final_checkpoint = run["num_tasks"] - 1
            for eval_task in range(run["num_tasks"]):
                at_learning = matrix[eval_task][eval_task]
                final_accuracy = matrix[eval_task][final_checkpoint]
                baseline_at_learning = baseline_matrix[eval_task][eval_task]
                baseline_final = baseline_matrix[eval_task][final_checkpoint]
                for checkpoint in range(eval_task, run["num_tasks"]):
                    accuracy = matrix[eval_task][checkpoint]
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "run": run["run"],
                            "condition": run["condition"],
                            "freeze_component": run["args"].get("audit_freeze_component", "none"),
                            "repeat_id": repeat_id,
                            "seed": seed,
                            "checkpoint_task": checkpoint + 1,
                            "eval_task": eval_task + 1,
                            "accuracy": accuracy,
                            "accuracy_at_learning": at_learning,
                            "final_accuracy": final_accuracy,
                            "diagonal_forgetting_gap": at_learning - accuracy,
                            "max_history_forgetting": max(matrix[eval_task][eval_task : checkpoint + 1]) - accuracy,
                            "baseline_accuracy_at_learning": baseline_at_learning,
                            "plasticity_loss": baseline_at_learning - at_learning,
                            "baseline_final_accuracy": baseline_final,
                            "final_retention_gain": final_accuracy - baseline_final,
                            "is_old_task": eval_task < checkpoint,
                        }
                    )
    return rows


def build_condition_rows(
    runs,
    baseline,
    plasticity_warn_threshold,
    retention_gain_threshold,
    enable_significance_tests=False,
    bootstrap_samples=10000,
    confidence_level=0.95,
):
    rows = []
    for run in runs:
        condition_rows = []
        for repeat_id, seed in enumerate(run["seeds"], start=1):
            matrix = run["matrices"][seed]
            baseline_matrix = baseline["matrices"][seed]
            final_index = run["num_tasks"] - 1
            diagonal = [matrix[index][index] for index in range(run["num_tasks"])]
            baseline_diagonal = [
                baseline_matrix[index][index] for index in range(run["num_tasks"])
            ]
            final_old = [matrix[index][final_index] for index in range(final_index)]
            baseline_final_old = [
                baseline_matrix[index][final_index] for index in range(final_index)
            ]
            repeat_index = run["seeds"].index(seed)
            faa = _history_value(run["global_history"], final_index, repeat_index)
            caa = mean(
                _history_value(run["global_history"], checkpoint, repeat_index)
                for checkpoint in range(run["num_tasks"])
            )
            fr = _history_value(run["fr_history"], final_index, repeat_index)
            baseline_repeat_index = baseline["seeds"].index(seed)
            baseline_faa = _history_value(
                baseline["global_history"], final_index, baseline_repeat_index
            )
            baseline_caa = mean(
                _history_value(
                    baseline["global_history"], checkpoint, baseline_repeat_index
                )
                for checkpoint in range(run["num_tasks"])
            )
            baseline_fr = _history_value(
                baseline["fr_history"], final_index, baseline_repeat_index
            )
            mean_plasticity = mean(
                baseline_diagonal[index] - diagonal[index]
                for index in range(1, run["num_tasks"])
            ) if run["num_tasks"] > 1 else 0.0
            mean_final_old = mean(final_old) if final_old else math.nan
            baseline_mean_final_old = mean(baseline_final_old) if baseline_final_old else math.nan
            row = {
                "schema_version": SCHEMA_VERSION,
                "run": run["run"],
                "condition": run["condition"],
                "repeat_id": repeat_id,
                "seed": seed,
                "matched": True,
                "FAA": faa,
                "CAA": caa,
                "FR": fr,
                "mean_diagonal_accuracy": mean(diagonal),
                "mean_final_old_accuracy": mean_final_old,
                "mean_new_task_plasticity_loss": mean_plasticity,
                "delta_FAA_vs_baseline": faa - baseline_faa,
                "delta_CAA_vs_baseline": caa - baseline_caa,
                "delta_FR_vs_baseline": fr - baseline_fr,
                "delta_final_old_accuracy_vs_baseline": mean_final_old - baseline_mean_final_old,
            }
            row["interpretation_flags"] = ";".join(
                interpretation_flags(
                    delta_faa=row["delta_FAA_vs_baseline"],
                    delta_fr=row["delta_FR_vs_baseline"],
                    delta_final_old_accuracy=row["delta_final_old_accuracy_vs_baseline"],
                    mean_plasticity_loss=mean_plasticity,
                    plasticity_warn_threshold=plasticity_warn_threshold,
                    retention_gain_threshold=retention_gain_threshold,
                )
            )
            rows.append(row)
            condition_rows.append(row)
        metric_deltas = {
            "FAA": [row["delta_FAA_vs_baseline"] for row in condition_rows],
            "CAA": [row["delta_CAA_vs_baseline"] for row in condition_rows],
            "FR": [row["delta_FR_vs_baseline"] for row in condition_rows],
            "final_old_accuracy": [
                row["delta_final_old_accuracy_vs_baseline"] for row in condition_rows
            ],
        }
        effects = paired_effect_summary(metric_deltas["FAA"])
        for row in condition_rows:
            row.update(effects)
        for metric, deltas in metric_deltas.items():
            metric_effects = paired_effect_summary(deltas)
            for row in condition_rows:
                row.update(
                    {f"{metric}_{name}": value for name, value in metric_effects.items()}
                )
            t_p = wilcoxon_p = ci_low = ci_high = math.nan
            if enable_significance_tests and len(deltas) >= 5:
                try:
                    from scipy import stats

                    t_p = float(stats.ttest_1samp(deltas, popmean=0.0).pvalue)
                    if any(float(value) != 0.0 for value in deltas):
                        wilcoxon_p = float(stats.wilcoxon(deltas).pvalue)
                except (ImportError, ValueError, FloatingPointError):
                    pass
                if bootstrap_samples > 0:
                    rng = random.Random(0)
                    bootstrap_means = sorted(
                        mean(rng.choices(deltas, k=len(deltas)))
                        for _ in range(bootstrap_samples)
                    )
                    tail = (1.0 - confidence_level) / 2.0
                    low_index = max(
                        0, min(len(bootstrap_means) - 1, int(tail * len(bootstrap_means)))
                    )
                    high_index = max(
                        0,
                        min(
                            len(bootstrap_means) - 1,
                            int((1.0 - tail) * len(bootstrap_means)) - 1,
                        ),
                    )
                    ci_low = bootstrap_means[low_index]
                    ci_high = bootstrap_means[high_index]
            for row in condition_rows:
                row[f"{metric}_paired_t_test_p"] = t_p
                row[f"{metric}_wilcoxon_p"] = wilcoxon_p
                row[f"{metric}_bootstrap_ci_low"] = ci_low
                row[f"{metric}_bootstrap_ci_high"] = ci_high
    return rows


def _snapshot_path(run, repeat_id, checkpoint_task, eval_task):
    return (
        run["dir"]
        / "forgetting_audit"
        / f"repeat-{repeat_id}"
        / "router_current"
        / f"checkpoint-{checkpoint_task}_eval-task-{eval_task}.pt"
    )


def _load_route_layers(path):
    snapshot = torch.load(path, map_location="cpu")
    layers = defaultdict(list)
    sample_ids = []
    for batch in snapshot["batches"]:
        sample_ids.extend(batch.get("sample_ids", []))
        for layer, state in batch["layers"].items():
            layers[int(layer)].append(state["indices"])
    return sample_ids, {layer: torch.cat(values, dim=0) for layer, values in layers.items()}


def _baseline_route_metrics(run, baseline, repeat_id, seed, checkpoint_task, eval_task):
    run_path = _snapshot_path(run, repeat_id, checkpoint_task, eval_task)
    baseline_repeat = baseline["seeds"].index(seed) + 1
    baseline_path = _snapshot_path(
        baseline, baseline_repeat, checkpoint_task, eval_task
    )
    if not run_path.exists() or not baseline_path.exists():
        return None, {}
    run_ids, run_layers = _load_route_layers(run_path)
    baseline_ids, baseline_layers = _load_route_layers(baseline_path)
    if run_ids and baseline_ids and run_ids != baseline_ids:
        raise ValueError(
            f"Router snapshot sample-ID mismatch: {run_path} vs {baseline_path}"
        )
    if set(run_layers) != set(baseline_layers):
        raise ValueError("Router snapshot layers do not match the baseline.")
    per_layer = {
        layer: route_set_metrics(run_layers[layer], baseline_layers[layer])
        for layer in run_layers
    }
    changed = sum(item["changed"] for item in per_layer.values())
    decisions = sum(item["decisions"] for item in per_layer.values())
    jaccard_sum = sum(item["jaccard_sum"] for item in per_layer.values())
    aggregate = {
        "change_rate": changed / max(decisions, 1),
        "mean_jaccard": jaccard_sum / max(decisions, 1),
    }
    return aggregate, per_layer


def build_router_rows(runs, baseline):
    rows = []
    for run in runs:
        records = [
            record for record in run["router_records"]
            if record.get("event") == "historical_router_replay"
        ]
        for record in records:
            repeat_id = int(record["repeat_id"])
            seed = int(record["seed"])
            checkpoint_task = int(record["checkpoint_task"])
            eval_task = int(record["eval_task"])
            matrix = run["matrices"][seed]
            forgetting_gap = matrix[eval_task - 1][eval_task - 1] - matrix[eval_task - 1][checkpoint_task - 1]
            identity_accuracy = record.get("identity_replay_accuracy", record.get("historical_router_accuracy"))
            identity_logits_accuracy = record.get("identity_prompt_logits_replay_accuracy")
            identity_gain = record.get("identity_replay_gain", record.get("replay_gain"))
            total_gain = record.get("total_router_replay_gain")
            if total_gain is None and identity_logits_accuracy is not None:
                total_gain = float(identity_logits_accuracy) - float(record["current_accuracy"])
            aggregate_baseline, layer_baseline = _baseline_route_metrics(
                run, baseline, repeat_id, seed, checkpoint_task, eval_task
            )

            def append_row(layer, within, baseline_metrics):
                identity_ratio = safe_recovery_ratio(identity_gain, forgetting_gap)
                total_ratio = safe_recovery_ratio(total_gain, forgetting_gap) if total_gain is not None else math.nan
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run": run["run"],
                        "condition": run["condition"],
                        "repeat_id": repeat_id,
                        "seed": seed,
                        "checkpoint_task": checkpoint_task,
                        "eval_task": eval_task,
                        "layer": layer,
                        "samples": int(record["samples"]),
                        "current_accuracy": float(record["current_accuracy"]),
                        "identity_replay_accuracy": identity_accuracy,
                        "identity_prompt_logits_replay_accuracy": identity_logits_accuracy,
                        "identity_replay_gain": identity_gain,
                        "additional_prompt_logit_gain": record.get("additional_prompt_logit_gain"),
                        "total_router_replay_gain": total_gain,
                        "within_condition_router_change_rate": within.get("within_condition_router_change_rate", within.get("router_change_rate")),
                        "within_condition_topk_jaccard": within.get("within_condition_topk_jaccard", within.get("mean_topk_jaccard")),
                        "baseline_route_divergence_rate": baseline_metrics.get("change_rate", math.nan) if baseline_metrics else math.nan,
                        "baseline_route_jaccard": baseline_metrics.get("mean_jaccard", math.nan) if baseline_metrics else math.nan,
                        "forgetting_gap": forgetting_gap,
                        "identity_recovery_ratio": identity_ratio,
                        "identity_recovery_ratio_clipped": clipped_recovery_ratio(identity_ratio),
                        "total_router_recovery_ratio": total_ratio,
                        "total_router_recovery_ratio_clipped": clipped_recovery_ratio(total_ratio),
                    }
                )

            append_row("ALL", record, aggregate_baseline)
            for layer, within in sorted(record.get("per_layer", {}).items(), key=lambda item: int(item[0])):
                append_row(int(layer), within, layer_baseline.get(int(layer)))
    return rows


def build_drift_rows(runs):
    rows = []
    for run in runs:
        for record in run["drift_records"]:
            components = record.get("components")
            if not components:
                components = {
                    component: {"global_pool": record[component]}
                    for component in ("key", "value", "classifier")
                    if component in record
                }
            for component, scopes in components.items():
                for scope in (
                    "global_pool",
                    "old_task_used_experts",
                    "old_task_high_frequency_experts",
                ):
                    if scope not in scopes:
                        continue
                    values = scopes[scope]
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "run": run["run"],
                            "condition": run["condition"],
                            "repeat_id": int(record["repeat_id"]),
                            "seed": int(record["seed"]),
                            "checkpoint_task": int(record["checkpoint_task"]),
                            "reference_task": int(record["reference_task"]),
                            "component": component,
                            "layer": "ALL",
                            "head": "ALL",
                            "expert": "ALL",
                            "scope": scope,
                            "l2": values.get("l2", math.nan),
                            "relative_l2": values.get("relative_l2", math.nan),
                            "usage_weighted_l2": values.get("usage_weighted_l2", math.nan),
                            "old_task_used_only_l2": values.get("old_task_used_only_l2", math.nan),
                        }
                    )
                for expert in scopes.get("per_expert", []):
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "run": run["run"],
                            "condition": run["condition"],
                            "repeat_id": int(record["repeat_id"]),
                            "seed": int(record["seed"]),
                            "checkpoint_task": int(record["checkpoint_task"]),
                            "reference_task": int(record["reference_task"]),
                            "component": component,
                            "layer": expert["layer"],
                            "head": expert["head"],
                            "expert": expert["expert"],
                            "scope": "global_pool",
                            "l2": expert["l2"],
                            "relative_l2": math.nan,
                            "usage_weighted_l2": math.nan,
                            "old_task_used_only_l2": math.nan,
                        }
                    )
    return rows


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NaN"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(path, runs, baseline, condition_rows, router_rows, drift_rows, warnings):
    lines = [
        "# SMoPE Forgetting Audit Report",
        "",
        f"Schema version: `{SCHEMA_VERSION}`",
        "",
        "## 1. Configuration and pairing checks",
        "",
        f"Baseline: `{baseline['run']}`. Conditions: " + ", ".join(f"`{run['condition']}`" for run in runs) + ".",
        "",
    ]
    if warnings:
        lines.extend(["Warnings:", ""] + [f"- {warning}" for warning in warnings] + [""])
    else:
        lines.extend(["All runs have matched seeds, task counts, core hyperparameters, and sample manifests.", ""])

    lines.extend([
        "## 2. Expert structure",
        "",
        "SMoPE uses one globally shared expert pool: architecture-level shared experts are 100%, task-private experts are 0%. Soft ownership is reported separately from access counts.",
        "",
        "## 3. Accuracy matrix and freeze plasticity-retention summary",
        "",
        "| condition | seed | FAA | CAA | FR | final old acc | plasticity loss | flags |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in condition_rows:
        lines.append(
            "| {condition} | {seed} | {FAA} | {CAA} | {FR} | {old} | {plasticity} | {flags} |".format(
                condition=row["condition"], seed=row["seed"], FAA=_fmt(row["FAA"]),
                CAA=_fmt(row["CAA"]), FR=_fmt(row["FR"]), old=_fmt(row["mean_final_old_accuracy"]),
                plasticity=_fmt(row["mean_new_task_plasticity_loss"]), flags=row["interpretation_flags"] or "-",
            )
        )
    lines.extend([
        "",
        "## 4. Router replay",
        "",
        "Identity replay isolates historical expert identity. Identity plus Prompt logits additionally restores historical Prompt-side attention strength. Negative gains are retained.",
        "",
    ])
    final_router = [row for row in router_rows if row["layer"] == "ALL" and row["eval_task"] < row["checkpoint_task"]]
    if final_router:
        lines.extend([
            "| condition | seed | checkpoint | eval task | route change | identity gain | extra logits gain | total gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in final_router:
            lines.append(
                f"| {row['condition']} | {row['seed']} | {row['checkpoint_task']} | {row['eval_task']} | "
                f"{_fmt(row['within_condition_router_change_rate'])} | {_fmt(row['identity_replay_gain'])} | "
                f"{_fmt(row['additional_prompt_logit_gain'])} | {_fmt(row['total_router_replay_gain'])} |"
            )
    else:
        lines.append("No router records were found; router fields remain unavailable.")
    lines.extend([
        "",
        "## 5. Parameter drift and gradient direction",
        "",
        f"Drift rows: {len(drift_rows)}. Gradient-direction records: {sum(len(run['gradient_records']) for run in runs)}.",
        "",
        "L2 drift is correlation evidence only. Positive `first_order_old_loss_change`, restoration gain, and matched freeze retention are needed before a causal contribution claim.",
        "",
        "## 6. Component restoration",
        "",
        "Offline restoration results are written by `utils/evaluate_component_restoration.py` to `component_restoration.jsonl` and should be interpreted together with single-component interaction terms.",
        "",
        "## 7. Allowed conclusions",
        "",
        "A component may be described as an important contributor only when change, harmful local direction, final-state restoration, and matched freeze retention agree without a large plasticity loss.",
        "",
        "## 8. Prohibited conclusions",
        "",
        "- Lower FR alone does not identify the forgetting source.",
        "- Larger relative L2 alone does not establish importance.",
        "- Small router replay gain does not exclude router involvement.",
        "- A large single-component restoration gain does not exclude interactions.",
        "- Fewer than five paired seeds must not be described as statistically significant by default.",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--output", type=Path, help="legacy summary CSV path")
    parser.add_argument("--plasticity_warn_threshold", type=float, default=2.0)
    parser.add_argument("--retention_gain_threshold", type=float, default=1.0)
    parser.add_argument("--enable_significance_tests", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--confidence_level", type=float, default=0.95)
    args = parser.parse_args()

    runs = [load_run(path) for path in args.run_dirs]
    baseline = select_baseline(runs, args.baseline)
    warnings = validate_run_pairing(runs, baseline)
    warnings.extend(validate_manifests(runs, baseline))
    if args.enable_significance_tests and len(baseline["seeds"]) < 5:
        warnings.append("Significance tests were requested but fewer than five paired seeds are available; only descriptive effects were emitted.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.output.parent if args.output else baseline["dir"] / "forgetting_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_rows = build_accuracy_rows(runs, baseline)
    condition_rows = build_condition_rows(
        runs,
        baseline,
        args.plasticity_warn_threshold,
        args.retention_gain_threshold,
        args.enable_significance_tests,
        args.bootstrap_samples,
        args.confidence_level,
    )
    router_rows = build_router_rows(runs, baseline)
    drift_rows = build_drift_rows(runs)
    usage_rows = []
    ownership_rows = []
    for run in runs:
        try:
            usage, ownership = summarize_expert_usage(run["dir"], run["condition"])
            usage_rows.extend(usage)
            ownership_rows.extend(ownership)
        except (FileNotFoundError, ValueError) as error:
            warnings.append(f"Expert usage for {run['run']} was not summarized: {error}")

    write_csv(output_dir / "accuracy_long.csv", accuracy_rows)
    write_csv(output_dir / "router_long.csv", router_rows)
    write_csv(output_dir / "drift_long.csv", drift_rows)
    write_csv(output_dir / "paired_condition_summary.csv", condition_rows)
    write_csv(output_dir / "expert_usage_long.csv", usage_rows)
    write_csv(output_dir / "expert_ownership_long.csv", ownership_rows)
    if args.output:
        write_csv(args.output, condition_rows)
    write_report(
        output_dir / "forgetting_audit_report.md",
        runs,
        baseline,
        condition_rows,
        router_rows,
        drift_rows,
        warnings,
    )
    print(f"Wrote forgetting-audit tables and report to {output_dir}")


if __name__ == "__main__":
    main()
