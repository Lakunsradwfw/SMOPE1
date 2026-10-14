"""Deterministic, auditable filtering for over-length CoIN training samples."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .data import CoINCollator, CoINDataset, ImageUnavailableError
from .report import write_json


FILTER_FORMAT_VERSION = 1
FILTER_SHARD_FORMAT_VERSION = 1
PROCESSOR_FILES = (
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processor_signature(path: str | Path) -> dict[str, str]:
    root = Path(path).resolve()
    return {name: _sha256(root / name) for name in PROCESSOR_FILES if (root / name).is_file()}


def filter_signature(dataset: CoINDataset, processor_path: str | Path, *,
                     max_length: int, max_visual_tokens: int, length_bucket: int,
                     image_transport: str, image_base_url: str | None) -> dict[str, Any]:
    return {
        "format_version": FILTER_FORMAT_VERSION,
        "task": dataset.task,
        "split": dataset.split,
        "source": str(dataset.path),
        "source_size_bytes": dataset.source_size,
        "source_mtime_ns": dataset.source_mtime_ns,
        "processor": processor_signature(processor_path),
        "max_length": max_length,
        "max_visual_tokens": max_visual_tokens,
        "length_bucket": length_bucket,
        "image_transport": image_transport,
        "image_base_url": image_base_url if image_transport == "http" else None,
    }


def signature_digest(signature: dict[str, Any]) -> str:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def filter_cache_path(cache: str | Path, signature: dict[str, Any]) -> Path:
    digest = signature_digest(signature)[:16]
    return Path(cache) / f"{signature['task']}-{signature['split']}-{digest}.json"


def impact_level(rate: float) -> str:
    if rate >= 0.10:
        return "high"
    if rate >= 0.01:
        return "material"
    return "low"


def measure_filter_range(dataset: CoINDataset, collator: CoINCollator, start: int = 0,
                         end: int | None = None,
                         progress: Callable[[dict[str, Any]], None] | None = None,
                         ) -> tuple[list[int], list[dict[str, Any]]]:
    """Measure one source-index range without treating transient I/O errors as data."""
    end = len(dataset.samples) if end is None else end
    if not 0 <= start <= end <= len(dataset.samples):
        raise ValueError(f"Invalid filter range [{start}, {end}) for {len(dataset.samples)} samples")
    valid_indices: list[int] = []
    excluded: list[dict[str, Any]] = []
    for index in range(start, end):
        sample = dataset.samples[index]
        try:
            measurement = collator.measure(sample)
        except ImageUnavailableError as error:
            measurement = {
                "task": sample.task, "sample_id": sample.sample_id,
                "prompt_len": None, "answer_len": None, "full_len": None,
                "reason": "image_unavailable", "image_location": error.location,
                "http_status": error.status,
            }
        if measurement["reason"] is None:
            valid_indices.append(index)
        else:
            excluded.append({"source_index": index, **measurement})
        processed = index - start + 1
        if progress is not None and (processed % 1000 == 0 or index + 1 == end):
            progress({"type": "filter_scan_progress", "task": dataset.task,
                      "range_start": start, "range_end": end,
                      "processed": processed, "total": end - start,
                      "excluded_examples": len(excluded)})
    return valid_indices, excluded


def _complete_manifest(dataset: CoINDataset, signature: dict[str, Any],
                       valid_indices: list[int], excluded: list[dict[str, Any]]) -> dict[str, Any]:
    original = len(dataset.samples)
    rate = len(excluded) / original if original else 0.0
    return {
        "format_version": FILTER_FORMAT_VERSION,
        "signature": signature,
        "signature_sha256": signature_digest(signature),
        "task": dataset.task,
        "split": dataset.split,
        "original_examples": original,
        "valid_examples": len(valid_indices),
        "excluded_examples": len(excluded),
        "excluded_rate": rate,
        "impact": impact_level(rate),
        "valid_indices": valid_indices,
        "excluded_samples": excluded,
    }


def build_filter_manifest(dataset: CoINDataset, collator: CoINCollator,
                          signature: dict[str, Any], path: str | Path,
                          progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Inspect the complete source split and atomically persist valid source indices."""
    valid_indices, excluded = measure_filter_range(dataset, collator, progress=progress)
    manifest = _complete_manifest(dataset, signature, valid_indices, excluded)
    write_json(path, manifest)
    return manifest


def filter_shard_path(cache: str | Path, signature: dict[str, Any],
                      start: int, end: int) -> Path:
    digest = signature_digest(signature)[:16]
    return Path(cache) / ".parts" / f"{signature['task']}-{digest}-{start:09d}-{end:09d}.json"


def build_filter_shard(dataset: CoINDataset, collator: CoINCollator,
                       signature: dict[str, Any], start: int, end: int,
                       path: str | Path,
                       progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    valid_indices, excluded = measure_filter_range(dataset, collator, start, end, progress)
    shard = {
        "shard_format_version": FILTER_SHARD_FORMAT_VERSION,
        "signature_sha256": signature_digest(signature),
        "start_index": start, "end_index": end,
        "valid_indices": valid_indices, "excluded_samples": excluded,
    }
    write_json(path, shard)
    return shard


def load_filter_shard(path: str | Path, signature: dict[str, Any],
                      start: int, end: int) -> dict[str, Any]:
    path = Path(path)
    shard = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "shard_format_version": FILTER_SHARD_FORMAT_VERSION,
        "signature_sha256": signature_digest(signature),
        "start_index": start, "end_index": end,
    }
    if any(shard.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Stale or incompatible filter shard: {path}")
    valid = shard.get("valid_indices")
    excluded = shard.get("excluded_samples")
    if not isinstance(valid, list) or not isinstance(excluded, list):
        raise ValueError(f"Incomplete filter shard: {path}")
    if any(not isinstance(index, int) or not start <= index < end for index in valid):
        raise ValueError(f"Out-of-range valid index in filter shard: {path}")
    if any(not isinstance(row, dict) or
           not isinstance(row.get("source_index"), int) or
           not start <= row["source_index"] < end for row in excluded):
        raise ValueError(f"Out-of-range excluded index in filter shard: {path}")
    if sorted(valid + [row["source_index"] for row in excluded]) != list(range(start, end)):
        raise ValueError(f"Filter shard does not cover its range exactly once: {path}")
    return shard


def merge_filter_shards(dataset: CoINDataset, signature: dict[str, Any],
                        shards: list[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    ordered = sorted(shards, key=lambda shard: shard["start_index"])
    cursor = 0
    expected_digest = signature_digest(signature)
    valid_indices: list[int] = []
    excluded: list[dict[str, Any]] = []
    for shard in ordered:
        if (shard.get("shard_format_version") != FILTER_SHARD_FORMAT_VERSION or
                shard.get("signature_sha256") != expected_digest):
            raise ValueError("Cannot merge a stale or incompatible filter shard")
        if shard["start_index"] != cursor or shard["end_index"] <= cursor:
            raise ValueError("Filter shards must be contiguous, non-overlapping and source ordered")
        cursor = shard["end_index"]
        valid_indices.extend(shard["valid_indices"])
        excluded.extend(shard["excluded_samples"])
    if cursor != len(dataset.samples):
        raise ValueError(f"Filter shards cover [0, {cursor}), expected [0, {len(dataset.samples)})")
    manifest = _complete_manifest(dataset, signature, valid_indices, excluded)
    write_json(path, manifest)
    return manifest


def load_filter_manifest(path: str | Path, signature: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FILTER_FORMAT_VERSION:
        raise ValueError(f"Unsupported filter manifest format: {path}")
    if manifest.get("signature") != signature:
        raise ValueError(f"Stale or incompatible filter manifest: {path}")
    if manifest.get("signature_sha256") != signature_digest(signature):
        raise ValueError(f"Incorrect filter manifest signature digest: {path}")
    valid = manifest.get("valid_indices")
    excluded = manifest.get("excluded_samples")
    if not isinstance(valid, list) or not isinstance(excluded, list):
        raise ValueError(f"Incomplete filter manifest: {path}")
    if manifest.get("valid_examples") != len(valid) or manifest.get("excluded_examples") != len(excluded):
        raise ValueError(f"Inconsistent filter manifest counts: {path}")
    original = manifest.get("original_examples")
    excluded_indices = ([row.get("source_index") for row in excluded]
                        if all(isinstance(row, dict) for row in excluded) else [None])
    all_indices = valid + excluded_indices
    indices_are_ints = all(isinstance(index, int) for index in all_indices)
    if (not isinstance(original, int) or original < 0 or not indices_are_ints or
            valid != sorted(set(valid)) or
            excluded_indices != sorted(set(excluded_indices)) or
            any(not 0 <= index < original for index in all_indices) or
            set(valid).intersection(excluded_indices) or
            len(valid) + len(excluded_indices) != original):
        raise ValueError(f"Filter manifest does not cover its source exactly once: {path}")
    expected_rate = len(excluded) / original if original else 0.0
    if (manifest.get("excluded_rate") != expected_rate or
            manifest.get("impact") != impact_level(expected_rate)):
        raise ValueError(f"Incorrect filter manifest rate or impact level: {path}")
    return manifest


def ensure_filter_manifest(dataset: CoINDataset, collator: CoINCollator,
                           cache: str | Path, processor_path: str | Path, *,
                           max_length: int, max_visual_tokens: int, length_bucket: int,
                           image_transport: str, image_base_url: str | None,
                           progress: Callable[[dict[str, Any]], None] | None = None,
                           ) -> tuple[dict[str, Any], Path, bool]:
    signature = filter_signature(
        dataset, processor_path, max_length=max_length,
        max_visual_tokens=max_visual_tokens, length_bucket=length_bucket,
        image_transport=image_transport, image_base_url=image_base_url)
    path = filter_cache_path(cache, signature)
    if path.is_file():
        return load_filter_manifest(path, signature), path, False
    return build_filter_manifest(dataset, collator, signature, path, progress), path, True


def validate_filter_impact(manifest: dict[str, Any], *, mode: str,
                           allow_high_filter_rate: bool) -> None:
    if mode == "full" and manifest["impact"] == "high" and not allow_high_filter_rate:
        rate = 100 * float(manifest["excluded_rate"])
        raise RuntimeError(
            f"{manifest['task']} filters {rate:.2f}% of training samples; "
            "review dataset_manifest.json and pass --allow-high-filter-rate to acknowledge it")


def manifest_for_run(manifest: dict[str, Any], cache_path: str | Path,
                     sampled_examples: int, acknowledged: bool) -> dict[str, Any]:
    """Paper-facing manifest payload; excludes the large valid-index implementation detail."""
    return {
        "cache": str(Path(cache_path).resolve()),
        "signature_sha256": manifest["signature_sha256"],
        "original_examples": manifest["original_examples"],
        "valid_examples": manifest["valid_examples"],
        "sampled_examples": sampled_examples,
        "excluded_examples": manifest["excluded_examples"],
        "excluded_rate": manifest["excluded_rate"],
        "impact": manifest["impact"],
        "high_filter_rate_acknowledged": bool(acknowledged),
        "excluded_samples": manifest["excluded_samples"],
    }
