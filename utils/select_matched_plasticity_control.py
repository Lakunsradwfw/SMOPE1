"""Select epoch-reduced baseline controls that match freeze-run plasticity loss.

The match is computed seed by seed from the task-2..N accuracy-matrix diagonal.
After matching, a positive ``matched_retention_gain`` means the freeze condition
retains old tasks better than a non-freeze control with comparable plasticity.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.summarize_forgetting_audit import load_run, validate_manifests


CORE_KEYS = ("dataset", "max_task", "crct_epochs", "prompt_param", "rand_split")


def validate_core(reference, run):
    if reference["num_tasks"] != run["num_tasks"]:
        raise ValueError(
            f"Task-count mismatch: {run['run']}={run['num_tasks']} "
            f"vs {reference['run']}={reference['num_tasks']}"
        )
    if set(reference["seeds"]) != set(run["seeds"]):
        raise ValueError(
            f"Seed mismatch: {run['run']}={run['seeds']} "
            f"vs {reference['run']}={reference['seeds']}"
        )
    for key in CORE_KEYS:
        if reference["args"].get(key) != run["args"].get(key):
            raise ValueError(
                f"Configuration mismatch for {key}: {run['run']}="
                f"{run['args'].get(key)!r} vs {reference['args'].get(key)!r}"
            )


def plasticity_loss(run, baseline, seed):
    matrix = run["matrices"][seed]
    baseline_matrix = baseline["matrices"][seed]
    if run["num_tasks"] <= 1:
        return 0.0
    return mean(
        baseline_matrix[index][index] - matrix[index][index]
        for index in range(1, run["num_tasks"])
    )


def final_old_accuracy(run, seed):
    final_index = run["num_tasks"] - 1
    if final_index <= 0:
        return math.nan
    matrix = run["matrices"][seed]
    return mean(matrix[index][final_index] for index in range(final_index))


def score_candidate(target, candidate, baseline, tolerance):
    seed_rows = []
    for seed in baseline["seeds"]:
        target_loss = plasticity_loss(target, baseline, seed)
        candidate_loss = plasticity_loss(candidate, baseline, seed)
        error = candidate_loss - target_loss
        target_old = final_old_accuracy(target, seed)
        candidate_old = final_old_accuracy(candidate, seed)
        seed_rows.append(
            {
                "seed": seed,
                "target_plasticity_loss": target_loss,
                "candidate_plasticity_loss": candidate_loss,
                "plasticity_error": error,
                "abs_plasticity_error": abs(error),
                "target_final_old_accuracy": target_old,
                "candidate_final_old_accuracy": candidate_old,
                "matched_retention_gain": target_old - candidate_old,
            }
        )
    mean_abs_error = mean(row["abs_plasticity_error"] for row in seed_rows)
    max_abs_error = max(row["abs_plasticity_error"] for row in seed_rows)
    return {
        "target_run": target["run"],
        "target_condition": target["condition"],
        "candidate_run": candidate["run"],
        "candidate_main_epochs": int(candidate["args"].get("audit_main_epochs", 0)),
        "num_seeds": len(seed_rows),
        "target_mean_plasticity_loss": mean(
            row["target_plasticity_loss"] for row in seed_rows
        ),
        "candidate_mean_plasticity_loss": mean(
            row["candidate_plasticity_loss"] for row in seed_rows
        ),
        "mean_plasticity_error": mean(row["plasticity_error"] for row in seed_rows),
        "mean_abs_plasticity_error": mean_abs_error,
        "max_abs_plasticity_error": max_abs_error,
        "seeds_within_tolerance": sum(
            row["abs_plasticity_error"] <= tolerance for row in seed_rows
        ),
        "mean_matched_retention_gain": mean(
            row["matched_retention_gain"] for row in seed_rows
        ),
        "seed_rows": seed_rows,
    }


def write_outputs(output_dir, scores, selected, tolerance, strict_multiplier):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "matched_plasticity_candidates.csv"
    csv_fields = [key for key in scores[0] if key != "seed_rows"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in scores:
            writer.writerow({key: row[key] for key in csv_fields})

    report_path = output_dir / "matched_plasticity_report.md"
    lines = [
        "# Matched-plasticity control selection",
        "",
        f"Mean absolute seed-level tolerance: `{tolerance:.3f}` percentage points.",
        f"Maximum seed-level tolerance: `{tolerance * strict_multiplier:.3f}` points.",
        "",
        "A positive matched retention gain means the freeze target has higher final old-task accuracy than the selected non-freeze control at comparable plasticity.",
        "",
        "| target | selected control | epochs | target plasticity loss | control plasticity loss | mean abs error | max abs error | matched retention gain | accepted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected:
        accepted = (
            row["mean_abs_plasticity_error"] <= tolerance
            and row["max_abs_plasticity_error"] <= tolerance * strict_multiplier
        )
        lines.append(
            "| {target_condition} | {candidate_run} | {candidate_main_epochs} | "
            "{target_mean_plasticity_loss:.4f} | {candidate_mean_plasticity_loss:.4f} | "
            "{mean_abs_plasticity_error:.4f} | {max_abs_plasticity_error:.4f} | "
            "{mean_matched_retention_gain:.4f} | {accepted} |".format(
                **row, accepted="yes" if accepted else "no"
            )
        )
    lines.extend(
        [
            "",
            "Do not interpret an unaccepted nearest candidate as a matched control. Add a neighboring epoch count and rerun the selector.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--targets", nargs="+", required=True, type=Path)
    parser.add_argument("--candidates", nargs="+", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=0.35)
    parser.add_argument("--max_seed_error_multiplier", type=float, default=2.0)
    args = parser.parse_args()

    baseline = load_run(args.baseline)
    targets = [load_run(path) for path in args.targets]
    candidates = [load_run(path) for path in args.candidates]
    if baseline["condition"] != "baseline":
        raise ValueError("--baseline must have audit_freeze_component=none")
    for run in targets + candidates:
        validate_core(baseline, run)
    for candidate in candidates:
        if candidate["condition"] != "baseline":
            raise ValueError(f"Candidate {candidate['run']} must be a non-freeze baseline control.")
        if int(candidate["args"].get("audit_main_epochs", 0)) <= 0:
            raise ValueError(f"Candidate {candidate['run']} lacks audit_main_epochs > 0.")
    manifest_warnings = validate_manifests([baseline, *targets, *candidates], baseline)
    if manifest_warnings:
        raise ValueError("Manifest validation failed: " + "; ".join(manifest_warnings))

    scores = [
        score_candidate(target, candidate, baseline, args.tolerance)
        for target in targets
        for candidate in candidates
    ]
    selected = []
    for target in targets:
        target_scores = [row for row in scores if row["target_run"] == target["run"]]
        selected.append(
            min(
                target_scores,
                key=lambda row: (
                    row["mean_abs_plasticity_error"],
                    row["max_abs_plasticity_error"],
                    row["candidate_main_epochs"],
                ),
            )
        )

    csv_path, report_path = write_outputs(
        args.output_dir,
        scores,
        selected,
        args.tolerance,
        args.max_seed_error_multiplier,
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    failed = [
        row
        for row in selected
        if row["mean_abs_plasticity_error"] > args.tolerance
        or row["max_abs_plasticity_error"]
        > args.tolerance * args.max_seed_error_multiplier
    ]
    if failed:
        for row in failed:
            print(
                "NO ACCEPTED MATCH for "
                f"{row['target_condition']}: nearest={row['candidate_run']} "
                f"mean_abs_error={row['mean_abs_plasticity_error']:.4f}, "
                f"max_abs_error={row['max_abs_plasticity_error']:.4f}"
            )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
