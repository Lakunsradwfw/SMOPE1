"""Compare a protocol-matched single-GPU/DDP diagnostic pair.

This deliberately distinguishes per-rank wall time from useful sample
throughput.  A sharded DDP run may take longer per rank while still process
more samples per second; it is only a training-protocol comparison when the
effective global batch, optimizer updates, and global sample set also match.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def read_run(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    traces = sorted(directory.glob("rank-*.jsonl"))
    if not traces:
        raise FileNotFoundError(f"No rank-*.jsonl under {directory}")

    ranks = []
    all_sample_ids: list[str] = []
    errors = []
    for trace in traces:
        rows = _read_jsonl(trace)
        summaries = [row for row in rows if row.get("type") == "summary"]
        if len(summaries) != 1:
            raise ValueError(f"Expected one summary row in {trace}, got {len(summaries)}")
        microbatches = [row for row in rows if row.get("type") == "microbatch"]
        rank_errors = [row for row in rows if row.get("type") == "error"]
        sample_ids = [str(sample_id) for row in microbatches
                      for sample_id in row.get("sample_ids", [])]
        summary = dict(summaries[0])
        if summary.get("samples_seen") != len(sample_ids):
            errors.append(
                f"{trace.name}: summary samples_seen={summary.get('samples_seen')} "
                f"but trace contains {len(sample_ids)} sample IDs")
        if rank_errors:
            errors.append(f"{trace.name}: contains {len(rank_errors)} error row(s)")
        ranks.append({"trace": str(trace), "summary": summary, "sample_ids": sample_ids})
        all_sample_ids.extend(sample_ids)

    unique_ids = set(all_sample_ids)
    wall_seconds = max(float(rank["summary"]["wall_seconds"]) for rank in ranks)
    samples_seen = sum(int(rank["summary"]["samples_seen"]) for rank in ranks)
    return {
        "directory": str(directory),
        "rank_count": len(ranks),
        "ranks": ranks,
        "wall_seconds": wall_seconds,
        "samples_seen": samples_seen,
        "unique_sample_ids": sorted(unique_ids),
        "duplicate_sample_count": len(all_sample_ids) - len(unique_ids),
        "samples_per_second": samples_seen / wall_seconds,
        "effective_global_batch_sizes": sorted({
            int(rank["summary"]["effective_global_batch_size"])
            for rank in ranks
        }),
        "optimizer_updates": sorted({
            int(rank["summary"]["optimizer_updates"]) for rank in ranks
        }),
        "nonfinite_trainable_parameters": sum(
            int(rank["summary"].get("nonfinite_trainable_parameters", 0))
            for rank in ranks),
        "errors": errors,
    }


def compare_runs(single: dict[str, Any], dual: dict[str, Any],
                 min_speedup: float = 1.5) -> dict[str, Any]:
    single_ids = set(single["unique_sample_ids"])
    dual_ids = set(dual["unique_sample_ids"])
    throughput_speedup = dual["samples_per_second"] / single["samples_per_second"]
    wall_time_ratio = dual["wall_seconds"] / single["wall_seconds"]
    issues = list(single["errors"]) + list(dual["errors"])

    if single["rank_count"] != 1:
        issues.append(f"single run has {single['rank_count']} ranks, expected 1")
    if dual["rank_count"] != 2:
        issues.append(f"dual run has {dual['rank_count']} ranks, expected 2")
    if single["effective_global_batch_sizes"] != dual["effective_global_batch_sizes"]:
        issues.append(
            "effective global batch mismatch: "
            f"single={single['effective_global_batch_sizes']}, "
            f"dual={dual['effective_global_batch_sizes']}")
    if single["optimizer_updates"] != dual["optimizer_updates"]:
        issues.append(
            f"optimizer update mismatch: single={single['optimizer_updates']}, "
            f"dual={dual['optimizer_updates']}")
    if single_ids != dual_ids:
        issues.append(
            f"global sample set mismatch: single_only={len(single_ids - dual_ids)}, "
            f"dual_only={len(dual_ids - single_ids)}")
    if single["duplicate_sample_count"] or dual["duplicate_sample_count"]:
        issues.append(
            f"duplicate samples: single={single['duplicate_sample_count']}, "
            f"dual={dual['duplicate_sample_count']}")
    if single["nonfinite_trainable_parameters"] or dual["nonfinite_trainable_parameters"]:
        issues.append(
            "non-finite trainable parameters: "
            f"single={single['nonfinite_trainable_parameters']}, "
            f"dual={dual['nonfinite_trainable_parameters']}")

    protocol_comparable = not issues
    return {
        "single": {
            key: single[key] for key in (
                "directory", "rank_count", "wall_seconds", "samples_seen",
                "duplicate_sample_count", "samples_per_second",
                "effective_global_batch_sizes", "optimizer_updates",
                "nonfinite_trainable_parameters")
        },
        "dual": {
            key: dual[key] for key in (
                "directory", "rank_count", "wall_seconds", "samples_seen",
                "duplicate_sample_count", "samples_per_second",
                "effective_global_batch_sizes", "optimizer_updates",
                "nonfinite_trainable_parameters")
        },
        "wall_time_ratio_dual_over_single": wall_time_ratio,
        "useful_sample_throughput_speedup": throughput_speedup,
        "minimum_required_speedup": min_speedup,
        "protocol_comparable": protocol_comparable,
        "issues": issues,
        "passes_speedup_gate": protocol_comparable and throughput_speedup >= min_speedup,
    }


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", required=True, help="single-GPU diagnostic directory")
    parser.add_argument("--dual", required=True, help="two-rank DDP diagnostic directory")
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--output", default=None, help="optional JSON output path")
    args = parser.parse_args(argv)
    if args.min_speedup <= 1:
        parser.error("--min-speedup must be greater than 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    result = compare_runs(read_run(args.single), read_run(args.dual), args.min_speedup)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
