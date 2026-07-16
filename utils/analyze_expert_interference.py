"""Analyze within-expert cross-task interference with task-pair controls."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


COORDINATE_FIELDS = ("layer", "head", "expert")


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def coordinate(record):
    return tuple(int(record[field]) for field in COORDINATE_FIELDS)


def finite_values(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def finite_mean(values):
    values = finite_values(values)
    return float(np.mean(values)) if values else math.nan


def finite_median(values):
    values = finite_values(values)
    return float(np.median(values)) if values else math.nan


def safe_spearman(x_values, y_values):
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return math.nan, len(pairs)
    x, y = zip(*pairs)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan, len(pairs)
    rho, _ = spearmanr(x, y)
    return float(rho), len(pairs)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_usage_conservation(usage_rows):
    grouped = defaultdict(lambda: {"observed": 0, "expected": None})
    for row in usage_rows:
        key = (
            int(row["repeat_id"]),
            int(row["seed"]),
            int(row["task"]),
            int(row["layer"]),
            int(row["head"]),
        )
        grouped[key]["observed"] += int(row["usage_count"])
        grouped[key]["expected"] = int(row["samples"]) * int(row["topk"])
    failures = [
        (key, values)
        for key, values in grouped.items()
        if values["observed"] != values["expected"]
    ]
    if failures:
        key, values = failures[0]
        raise ValueError(
            "Expert usage conservation failed for "
            f"{key}: {values['observed']} != {values['expected']}"
        )


def validate_probe_manifests(manifest_rows, conflict_rows):
    tasks_by_run = defaultdict(set)
    for row in manifest_rows:
        counts = [int(value) for value in row["class_counts"].values()]
        if not counts or max(counts) - min(counts) > 1:
            raise ValueError(
                "Probe manifest is not class-balanced for "
                f"repeat={row['repeat_id']}, seed={row['seed']}, task={row['task']}"
            )
        tasks_by_run[(int(row["repeat_id"]), int(row["seed"]))].add(
            int(row["task"])
        )
    maximum_task_by_run = defaultdict(int)
    for row in conflict_rows:
        run = (int(row["repeat_id"]), int(row["seed"]))
        maximum_task_by_run[run] = max(
            maximum_task_by_run[run], int(row["new_task"])
        )
    for run, maximum_task in maximum_task_by_run.items():
        expected = set(range(1, maximum_task + 1))
        if tasks_by_run.get(run, set()) != expected:
            raise ValueError(
                f"Probe manifests for repeat/seed {run} are incomplete: "
                f"observed={sorted(tasks_by_run.get(run, set()))}, "
                f"expected={sorted(expected)}"
            )


def validate_task_pair_grid(conflict_rows):
    keys = []
    pair_counts = defaultdict(int)
    maximum_task_by_run = defaultdict(int)
    observed_pairs = defaultdict(set)
    for row in conflict_rows:
        run = (int(row["repeat_id"]), int(row["seed"]))
        old_task = int(row["old_task"])
        new_task = int(row["new_task"])
        key = (run, old_task, new_task, coordinate(row))
        keys.append(key)
        pair_counts[(run, old_task, new_task)] += 1
        observed_pairs[run].add((old_task, new_task))
        maximum_task_by_run[run] = max(maximum_task_by_run[run], new_task)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate expert task-pair conflict keys were found.")
    for run, maximum_task in maximum_task_by_run.items():
        expected_pairs = {
            (old_task, new_task)
            for new_task in range(2, maximum_task + 1)
            for old_task in range(1, new_task)
        }
        if observed_pairs[run] != expected_pairs:
            raise ValueError(
                f"Task-pair grid for repeat/seed {run} is incomplete."
            )
        counts = {
            pair_counts[(run, old_task, new_task)]
            for old_task, new_task in expected_pairs
        }
        if len(counts) != 1:
            raise ValueError(
                f"Expert-unit counts differ across task pairs for repeat/seed {run}: "
                f"{sorted(counts)}"
            )


def build_final_checkpoint_usage_rows(reference_usage_rows, drift_rows):
    """Reconstruct final-model usage for every task without sample-level logs."""
    final_task_by_run = defaultdict(int)
    for row in drift_rows:
        run = (int(row["repeat_id"]), int(row["seed"]))
        final_task_by_run[run] = max(final_task_by_run[run], int(row["new_task"]))
    final_rows = []
    for row in drift_rows:
        run = (int(row["repeat_id"]), int(row["seed"]))
        if int(row["new_task"]) != final_task_by_run[run]:
            continue
        final_rows.append(
            {
                "repeat_id": run[0],
                "seed": run[1],
                "task": int(row["old_task"]),
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "expert": int(row["expert"]),
                "samples": int(row["samples"]),
                "topk": int(row["topk"]),
                "usage_count": int(row["old_current_usage_count"]),
            }
        )
    for row in reference_usage_rows:
        run = (int(row["repeat_id"]), int(row["seed"]))
        if int(row["task"]) == final_task_by_run.get(run, int(row["task"])):
            final_rows.append(dict(row))
    return final_rows


def build_usage_summaries(usage_rows, routing_scope="learning_boundary"):
    """Keep layer/head pools intact, then derive descriptive whole-run rollups."""
    task_pools = defaultdict(list)
    for row in usage_rows:
        key = (
            int(row["repeat_id"]),
            int(row["seed"]),
            int(row["task"]),
            int(row["layer"]),
            int(row["head"]),
        )
        task_pools[key].append(row)

    topk_sets = defaultdict(dict)
    coordinate_totals = defaultdict(
        lambda: {
            "usage_count": 0,
            "selection_slots": 0,
            "active_tasks": 0,
            "tasks": 0,
        }
    )
    expert_units_by_pool = {}
    for key, rows in sorted(task_pools.items()):
        repeat_id, seed, task, layer, head = key
        experts = [int(row["expert"]) for row in rows]
        if len(experts) != len(set(experts)):
            raise ValueError(f"Duplicate expert usage rows for task pool {key}.")
        topk_values = {int(row["topk"]) for row in rows}
        sample_values = {int(row["samples"]) for row in rows}
        if len(topk_values) != 1 or len(sample_values) != 1:
            raise ValueError(f"Inconsistent topk/sample counts inside task pool {key}.")
        topk = next(iter(topk_values))
        samples = next(iter(sample_values))
        ranked = sorted(
            rows,
            key=lambda row: (-int(row["usage_count"]), int(row["expert"])),
        )
        topk_sets[(repeat_id, seed, layer, head)][task] = {
            int(row["expert"]) for row in ranked[:topk]
        }
        expert_units_by_pool[(repeat_id, seed, layer, head)] = len(rows)
        for row in rows:
            coordinate_key = (
                repeat_id,
                seed,
                layer,
                head,
                int(row["expert"]),
            )
            totals = coordinate_totals[coordinate_key]
            count = int(row["usage_count"])
            totals["usage_count"] += count
            totals["selection_slots"] += samples * topk
            totals["active_tasks"] += int(count > 0)
            totals["tasks"] += 1

    global_counts = defaultdict(int)
    for (repeat_id, seed, _, _, _), totals in coordinate_totals.items():
        global_counts[(repeat_id, seed)] += totals["usage_count"]

    coordinate_rows = []
    for key, totals in sorted(coordinate_totals.items()):
        repeat_id, seed, layer, head, expert = key
        within_pool_share = totals["usage_count"] / totals["selection_slots"]
        coordinate_rows.append(
            {
                "routing_scope": routing_scope,
                "repeat_id": repeat_id,
                "seed": seed,
                "layer": layer,
                "head": head,
                "expert": expert,
                "tasks": totals["tasks"],
                "active_tasks": totals["active_tasks"],
                "total_usage_count": totals["usage_count"],
                "total_selection_slots_within_pool": totals["selection_slots"],
                "within_layer_head_selection_share": within_pool_share,
                "global_coordinate_selection_share": (
                    totals["usage_count"] / global_counts[(repeat_id, seed)]
                ),
            }
        )

    coordinate_by_pool = defaultdict(list)
    for row in coordinate_rows:
        coordinate_by_pool[
            (row["repeat_id"], row["seed"], row["layer"], row["head"])
        ].append(row)

    pool_rows = []
    for key, rows in sorted(coordinate_by_pool.items()):
        repeat_id, seed, layer, head = key
        task_sets = [
            topk_sets[key][task] for task in sorted(topk_sets[key])
        ]
        topk = len(task_sets[0])
        shares = sorted(
            (row["within_layer_head_selection_share"] for row in rows),
            reverse=True,
        )
        consecutive_jaccards = []
        for first, second in zip(task_sets, task_sets[1:]):
            union = first | second
            consecutive_jaccards.append(len(first & second) / len(union))
        positive_shares = [share for share in shares if share > 0.0]
        entropy = -sum(share * math.log(share) for share in positive_shares)
        normalized_entropy = entropy / math.log(len(shares)) if len(shares) > 1 else 0.0
        pool_rows.append(
            {
                "routing_scope": routing_scope,
                "repeat_id": repeat_id,
                "seed": seed,
                "layer": layer,
                "head": head,
                "tasks": len(task_sets),
                "experts_in_pool": expert_units_by_pool[key],
                "active_expert_count": len(positive_shares),
                "topk": topk,
                "topk_selection_share": float(sum(shares[:topk])),
                "topk_intersection_size_across_tasks": len(
                    set.intersection(*task_sets)
                ),
                "exact_same_topk_all_tasks": bool(
                    all(task_set == task_sets[0] for task_set in task_sets[1:])
                ),
                "mean_consecutive_topk_jaccard": (
                    float(np.mean(consecutive_jaccards))
                    if consecutive_jaccards
                    else 1.0
                ),
                "normalized_selection_entropy": normalized_entropy,
                "effective_expert_count": math.exp(entropy),
            }
        )

    pools_by_run = defaultdict(list)
    for row in pool_rows:
        pools_by_run[(row["repeat_id"], row["seed"])].append(row)
    overall_rows = []
    for (repeat_id, seed), rows in sorted(pools_by_run.items()):
        overall_rows.append(
            {
                "routing_scope": routing_scope,
                "repeat_id": repeat_id,
                "seed": seed,
                "layer_head_pools": len(rows),
                "coordinate_experts": sum(row["experts_in_pool"] for row in rows),
                "median_active_experts_per_pool": finite_median(
                    row["active_expert_count"] for row in rows
                ),
                "median_topk_selection_share": finite_median(
                    row["topk_selection_share"] for row in rows
                ),
                "minimum_topk_selection_share": min(
                    row["topk_selection_share"] for row in rows
                ),
                "exact_same_topk_pool_fraction": finite_mean(
                    row["exact_same_topk_all_tasks"] for row in rows
                ),
                "topk_share_ge_95pct_pool_fraction": finite_mean(
                    row["topk_selection_share"] >= 0.95 for row in rows
                ),
                "mean_consecutive_topk_jaccard": finite_mean(
                    row["mean_consecutive_topk_jaccard"] for row in rows
                ),
                "median_effective_expert_count": finite_median(
                    row["effective_expert_count"] for row in rows
                ),
            }
        )
    return coordinate_rows, pool_rows, overall_rows


def build_expert_task_pair_rows(run_dir):
    audit_dir = Path(run_dir) / "expert_interference"
    manifest_rows = list(read_jsonl(audit_dir / "probe_manifest.jsonl"))
    usage_rows = list(read_jsonl(audit_dir / "expert_usage.jsonl.gz"))
    validate_usage_conservation(usage_rows)
    conflict_rows = list(
        read_jsonl(audit_dir / "expert_task_pair_conflict.jsonl.gz")
    )
    validate_probe_manifests(manifest_rows, conflict_rows)
    validate_task_pair_grid(conflict_rows)
    drift_rows = list(
        read_jsonl(audit_dir / "expert_functional_drift.jsonl.gz")
    )
    drift_by_key = {
        (
            int(row["repeat_id"]),
            int(row["seed"]),
            int(row["old_task"]),
            int(row["new_task"]),
            coordinate(row),
        ): row
        for row in drift_rows
    }
    if len(drift_by_key) != len(drift_rows):
        raise ValueError("Duplicate task-pair drift keys were found.")

    rows = []
    missing = []
    for conflict in conflict_rows:
        for prefix in ("old", "new"):
            full = int(conflict[f"{prefix}_probe_samples"])
            half_a = int(conflict[f"{prefix}_half_a_samples"])
            half_b = int(conflict[f"{prefix}_half_b_samples"])
            if half_a <= 0 or half_b <= 0 or half_a != half_b:
                raise ValueError(
                    f"Invalid {prefix}-task split-half sizes: "
                    f"full={full}, half_a={half_a}, half_b={half_b}"
                )
            if full != half_a + half_b:
                raise ValueError(
                    f"{prefix}-task probe size does not equal its two halves: "
                    f"{full} != {half_a} + {half_b}"
                )
        key = (
            int(conflict["repeat_id"]),
            int(conflict["seed"]),
            int(conflict["old_task"]),
            int(conflict["new_task"]),
            coordinate(conflict),
        )
        drift = drift_by_key.get(key)
        if drift is None:
            missing.append(key)
            continue
        coord = coordinate(conflict)
        rows.append(
            {
                "repeat_id": key[0],
                "seed": key[1],
                "old_task": key[2],
                "new_task": key[3],
                "layer": coord[0],
                "head": coord[1],
                "expert": coord[2],
                "old_probe_samples": int(conflict["old_probe_samples"]),
                "new_probe_samples": int(conflict["new_probe_samples"]),
                "shared_hard_route": bool(conflict["shared_hard_route"]),
                "old_reference_hard_usage": float(
                    conflict["old_reference_hard_usage"]
                ),
                "new_pretraining_hard_usage": float(
                    conflict["new_pretraining_hard_usage"]
                ),
                "old_reference_soft_usage": float(
                    conflict["old_reference_soft_usage"]
                ),
                "new_pretraining_soft_usage": float(
                    conflict["new_pretraining_soft_usage"]
                ),
                "valid_cross_task_gradient": bool(
                    conflict["cross_task"]["valid_gradient_pair"]
                ),
                "cross_task_cosine": float(conflict["cross_task"]["cosine"]),
                "cross_task_negative_cosine": float(
                    conflict["cross_task"]["negative_cosine"]
                ),
                "same_old_task_negative_cosine": float(
                    conflict["same_old_task"]["negative_cosine"]
                ),
                "same_new_task_negative_cosine": float(
                    conflict["same_new_task"]["negative_cosine"]
                ),
                "same_task_control_negative_cosine": float(
                    conflict["same_task_control_negative_cosine"]
                ),
                "excess_cross_task_conflict": float(
                    conflict["excess_cross_task_conflict"]
                ),
                "first_order_old_loss_change": float(
                    conflict["actual_update"]["first_order_old_loss_change"]
                ),
                "predicted_harm": float(
                    conflict["actual_update"]["predicted_harm"]
                ),
                "expert_update_norm": float(
                    conflict["actual_update"]["update_norm"]
                ),
                "observed_full_model_old_loss_change": float(
                    conflict["observed_full_model_old_loss_change"]
                ),
                "key_cross_task_negative_cosine": float(
                    conflict["key"]["cross_task"]["negative_cosine"]
                ),
                "value_cross_task_negative_cosine": float(
                    conflict["value"]["cross_task"]["negative_cosine"]
                ),
                "incremental_response_cosine_distance": float(
                    drift["incremental_response_cosine_distance"]
                ),
                "incremental_response_relative_l2": float(
                    drift["incremental_response_relative_l2"]
                ),
                "cumulative_response_cosine_distance": float(
                    drift["cumulative_response_cosine_distance"]
                ),
                "incremental_key_cosine_distance": float(
                    drift["incremental_key_cosine_distance"]
                ),
                "incremental_value_cosine_distance": float(
                    drift["incremental_value_cosine_distance"]
                ),
                "learning_boundary_forgetting": float(
                    drift["learning_boundary_forgetting"]
                ),
                "max_history_forgetting": float(
                    drift["max_history_forgetting"]
                ),
            }
        )
    if missing:
        raise ValueError(
            f"Missing task-pair drift records for {len(missing)} conflict rows; "
            f"first missing key: {missing[0]}"
        )
    return rows


def build_task_pair_summary(expert_rows):
    grouped = defaultdict(list)
    for row in expert_rows:
        grouped[
            (row["repeat_id"], row["seed"], row["old_task"], row["new_task"])
        ].append(row)
    summaries = []
    for (repeat_id, seed, old_task, new_task), rows in sorted(grouped.items()):
        shared = [row for row in rows if row["shared_hard_route"]]
        valid = [
            row
            for row in shared
            if row["valid_cross_task_gradient"]
            and math.isfinite(row["same_task_control_negative_cosine"])
        ]
        summaries.append(
            {
                "repeat_id": repeat_id,
                "seed": seed,
                "old_task": old_task,
                "new_task": new_task,
                "expert_units": len(rows),
                "shared_hard_route_units": len(shared),
                "valid_primary_units": len(valid),
                "shared_route_fraction": len(shared) / max(len(rows), 1),
                "mean_cross_task_negative_cosine": finite_mean(
                    row["cross_task_negative_cosine"] for row in valid
                ),
                "mean_same_task_control_negative_cosine": finite_mean(
                    row["same_task_control_negative_cosine"] for row in valid
                ),
                "median_excess_cross_task_conflict": finite_median(
                    row["excess_cross_task_conflict"] for row in valid
                ),
                "positive_excess_fraction": finite_mean(
                    row["excess_cross_task_conflict"] > 0.0 for row in valid
                ),
                "sum_shared_expert_first_order_old_loss_change": float(
                    sum(row["first_order_old_loss_change"] for row in shared)
                ),
                "sum_shared_expert_predicted_harm": float(
                    sum(row["predicted_harm"] for row in shared)
                ),
                "mean_incremental_response_cosine_distance": finite_mean(
                    row["incremental_response_cosine_distance"] for row in shared
                ),
                "mean_cumulative_response_cosine_distance": finite_mean(
                    row["cumulative_response_cosine_distance"] for row in shared
                ),
                "observed_full_model_old_loss_change": rows[0][
                    "observed_full_model_old_loss_change"
                ],
                "learning_boundary_forgetting": rows[0][
                    "learning_boundary_forgetting"
                ],
                "max_history_forgetting": rows[0]["max_history_forgetting"],
            }
        )
    return summaries


def build_layer_head_summary(expert_rows):
    grouped = defaultdict(list)
    for row in expert_rows:
        if (
            row["shared_hard_route"]
            and row["valid_cross_task_gradient"]
            and math.isfinite(row["excess_cross_task_conflict"])
        ):
            grouped[
                (row["repeat_id"], row["seed"], row["layer"], row["head"])
            ].append(row)
    return [
        {
            "repeat_id": key[0],
            "seed": key[1],
            "layer": key[2],
            "head": key[3],
            "task_pair_expert_units": len(rows),
            "median_excess_cross_task_conflict": finite_median(
                row["excess_cross_task_conflict"] for row in rows
            ),
            "positive_excess_fraction": finite_mean(
                row["excess_cross_task_conflict"] > 0.0 for row in rows
            ),
        }
        for key, rows in sorted(grouped.items())
    ]


def build_seed_endpoint_summary(task_pair_rows):
    grouped = defaultdict(list)
    for row in task_pair_rows:
        grouped[(row["repeat_id"], row["seed"])].append(row)
    summaries = []
    for (repeat_id, seed), rows in sorted(grouped.items()):
        harm_observed_rho, harm_pairs = safe_spearman(
            [
                row["sum_shared_expert_first_order_old_loss_change"]
                for row in rows
            ],
            [row["observed_full_model_old_loss_change"] for row in rows],
        )
        harm_drift_rho, drift_pairs = safe_spearman(
            [row["sum_shared_expert_predicted_harm"] for row in rows],
            [row["mean_incremental_response_cosine_distance"] for row in rows],
        )
        drift_forgetting_rho, forgetting_pairs = safe_spearman(
            [row["mean_incremental_response_cosine_distance"] for row in rows],
            [row["learning_boundary_forgetting"] for row in rows],
        )
        conflict_drift_rho, conflict_drift_pairs = safe_spearman(
            [row["median_excess_cross_task_conflict"] for row in rows],
            [row["mean_incremental_response_cosine_distance"] for row in rows],
        )
        summaries.append(
            {
                "repeat_id": repeat_id,
                "seed": seed,
                "task_pairs": len(rows),
                "median_cross_task_negative_cosine": finite_median(
                    row["mean_cross_task_negative_cosine"] for row in rows
                ),
                "median_same_task_control_negative_cosine": finite_median(
                    row["mean_same_task_control_negative_cosine"] for row in rows
                ),
                "primary_median_excess_conflict": finite_median(
                    row["median_excess_cross_task_conflict"] for row in rows
                ),
                "primary_positive_task_pair_fraction": finite_mean(
                    row["median_excess_cross_task_conflict"] > 0.0
                    for row in rows
                    if math.isfinite(row["median_excess_cross_task_conflict"])
                ),
                "harm_to_observed_loss_spearman": harm_observed_rho,
                "harm_to_observed_loss_task_pairs": harm_pairs,
                "harm_to_incremental_drift_spearman": harm_drift_rho,
                "harm_to_incremental_drift_task_pairs": drift_pairs,
                "conflict_to_incremental_drift_spearman": conflict_drift_rho,
                "conflict_to_incremental_drift_task_pairs": conflict_drift_pairs,
                "incremental_drift_to_forgetting_spearman": drift_forgetting_rho,
                "incremental_drift_to_forgetting_task_pairs": forgetting_pairs,
            }
        )
    return summaries


def build_across_seed_summary(seed_rows):
    endpoints = (
        "primary_median_excess_conflict",
        "harm_to_observed_loss_spearman",
        "harm_to_incremental_drift_spearman",
        "conflict_to_incremental_drift_spearman",
        "incremental_drift_to_forgetting_spearman",
    )
    rows = []
    for endpoint in endpoints:
        values = finite_values(row[endpoint] for row in seed_rows)
        rows.append(
            {
                "endpoint": endpoint,
                "valid_seeds": len(values),
                "mean": float(np.mean(values)) if values else math.nan,
                "sample_sd": (
                    float(np.std(values, ddof=1)) if len(values) >= 2 else math.nan
                ),
                "median": float(np.median(values)) if values else math.nan,
                "q25": float(np.quantile(values, 0.25)) if values else math.nan,
                "q75": float(np.quantile(values, 0.75)) if values else math.nan,
                "positive_seed_fraction": (
                    float(np.mean(np.asarray(values) > 0.0))
                    if values
                    else math.nan
                ),
            }
        )
    return rows


def write_report(path, seed_rows, across_seed_rows, usage_overall_rows):
    lines = [
        "# Within-expert cross-task interference report",
        "",
        "The primary endpoint is cross-task negative gradient cosine minus the same-task split-half negative-cosine baseline, restricted to task pairs that share the same hard-routed expert.",
        "",
        "| seed | task pairs | cross-task conflict | same-task control | excess conflict | positive task-pair fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {row['task_pairs']} | "
            f"{row['median_cross_task_negative_cosine']:.6f} | "
            f"{row['median_same_task_control_negative_cosine']:.6f} | "
            f"{row['primary_median_excess_conflict']:.6f} | "
            f"{row['primary_positive_task_pair_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Routing frequency by layer/head pool",
            "",
            "Expert identity is the tuple (layer, head, expert). The whole-run table below aggregates pool-level statistics and never merges equal expert indices across different layer/head pools.",
            "",
            "| scope | seed | layer-head pools | median active experts | median Top-k share | exact same Top-k pool fraction | Top-k share >=95% pool fraction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in usage_overall_rows:
        lines.append(
            f"| {row['routing_scope']} | {row['seed']} | "
            f"{row['layer_head_pools']} | "
            f"{row['median_active_experts_per_pool']:.2f} | "
            f"{row['median_topk_selection_share']:.4f} | "
            f"{row['exact_same_topk_pool_fraction']:.4f} | "
            f"{row['topk_share_ge_95pct_pool_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Across-seed summary",
            "",
            "| endpoint | valid seeds | median | IQR | positive seed fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in across_seed_rows:
        lines.append(
            f"| {row['endpoint']} | {row['valid_seeds']} | "
            f"{row['median']:.6f} | [{row['q25']:.6f}, {row['q75']:.6f}] | "
            f"{row['positive_seed_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "A positive primary endpoint shows that different tasks conflict inside the same shared expert more than stochastic gradients from two disjoint halves of the same task. It does not alone prove forgetting. The next links require actual expert-update harm, incremental task-conditioned response drift, and task-level forgetting to agree across independent seeds.",
            "",
            "Experts and task pairs within one run are nested observations, not independent experimental replicates. Paper-level inference must use seed-level effects (at least five seeds); do not use an expert-level p-value as the final significance test.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "expert_interference" / "analysis"
    expert_rows = build_expert_task_pair_rows(args.run_dir)
    audit_dir = args.run_dir / "expert_interference"
    usage_rows = list(read_jsonl(audit_dir / "expert_usage.jsonl.gz"))
    drift_rows = list(read_jsonl(audit_dir / "expert_functional_drift.jsonl.gz"))
    final_usage_rows = build_final_checkpoint_usage_rows(usage_rows, drift_rows)
    validate_usage_conservation(final_usage_rows)
    boundary_summaries = build_usage_summaries(
        usage_rows, routing_scope="learning_boundary"
    )
    final_summaries = build_usage_summaries(
        final_usage_rows, routing_scope="final_checkpoint"
    )
    coordinate_usage_rows = boundary_summaries[0] + final_summaries[0]
    pool_usage_rows = boundary_summaries[1] + final_summaries[1]
    usage_overall_rows = boundary_summaries[2] + final_summaries[2]
    task_pair_rows = build_task_pair_summary(expert_rows)
    layer_head_rows = build_layer_head_summary(expert_rows)
    seed_rows = build_seed_endpoint_summary(task_pair_rows)
    across_seed_rows = build_across_seed_summary(seed_rows)
    write_csv(output_dir / "expert_task_pair_long.csv", expert_rows)
    write_csv(output_dir / "expert_usage_layer_head.csv", coordinate_usage_rows)
    write_csv(output_dir / "layer_head_usage_summary.csv", pool_usage_rows)
    write_csv(output_dir / "overall_usage_summary.csv", usage_overall_rows)
    write_csv(output_dir / "task_pair_summary.csv", task_pair_rows)
    write_csv(output_dir / "layer_head_primary.csv", layer_head_rows)
    write_csv(output_dir / "seed_endpoint_summary.csv", seed_rows)
    write_csv(output_dir / "across_seed_summary.csv", across_seed_rows)
    write_report(
        output_dir / "within_expert_interference_report.md",
        seed_rows,
        across_seed_rows,
        usage_overall_rows,
    )
    print(
        f"Wrote {len(expert_rows)} expert-task-pair rows, "
        f"{len(task_pair_rows)} task-pair summaries, and {len(seed_rows)} seed summaries."
    )


if __name__ == "__main__":
    main()
