"""Build full-split CoIN training-length manifests without loading the 9B model."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any

from .data import TASK_ORDER, CoINCollator, CoINDataset, ImageFetcher
from .filtering import (build_filter_shard, ensure_filter_manifest, filter_cache_path,
                        filter_shard_path, filter_signature, load_filter_manifest,
                        load_filter_shard, merge_filter_shards)
from .train import DEFAULT_IMAGE_BASE_URL


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="pretrained/Qwen3.5-9B")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--coin-root", default="datas/CoIN")
    parser.add_argument("--tasks", default=",".join(TASK_ORDER))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--image-transport", choices=("mount", "http"), default="http")
    parser.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL)
    parser.add_argument("--filter-cache", default="outputs/coin-qwen3.5-9b/_filter_cache")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--length-bucket", type=int, default=128)
    parser.add_argument("--scan-workers", type=int, default=8,
                        help="完整 split 的并行扫描进程数；HTTP 通道最多 14")
    args = parser.parse_args(argv)
    args.tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = set(args.tasks) - set(TASK_ORDER)
    if not args.tasks or unknown:
        parser.error(f"Invalid --tasks; unknown={sorted(unknown)}")
    if args.scan_workers <= 0:
        parser.error("--scan-workers must be positive")
    if args.image_transport == "http" and args.scan_workers > 14:
        parser.error("--scan-workers must be <=14 for HTTP to avoid overloading WebDAV")
    return args


def _ranges(total: int, workers: int) -> list[tuple[int, int]]:
    width = math.ceil(total / min(total, workers))
    return [(start, min(start + width, total)) for start in range(0, total, width)]


def _build_shard_worker(payload: dict[str, Any]) -> str:
    """Spawn-safe worker; each process owns its processor and HTTP connection."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(payload["processor_path"], local_files_only=True)
    processor.tokenizer.padding_side = "right"
    fetcher = ImageFetcher(
        payload["image_transport"], payload["image_base_url"], payload["coin_root"])
    collator = CoINCollator(
        processor, payload["max_length"], payload["max_visual_tokens"], fetcher,
        bucket=payload["length_bucket"])
    dataset = CoINDataset(
        payload["coin_root"], payload["task"], payload["train_split"], 0,
        verify_images=False)
    signature = filter_signature(
        dataset, payload["processor_path"], max_length=payload["max_length"],
        max_visual_tokens=payload["max_visual_tokens"],
        length_bucket=payload["length_bucket"],
        image_transport=payload["image_transport"], image_base_url=payload["image_base_url"])
    if signature != payload["signature"]:
        raise RuntimeError("Dataset or processor changed while filter shards were running")
    start, end = payload["start"], payload["end"]
    path = Path(payload["path"])
    build_filter_shard(
        dataset, collator, signature, start, end, path,
        progress=lambda row: print(json.dumps(
            {**row, "shard": f"{start}:{end}"}, ensure_ascii=False), flush=True))
    return str(path)


def ensure_parallel_manifest(args: argparse.Namespace, task: str,
                             processor_path: str) -> tuple[dict[str, Any], Path, bool]:
    dataset = CoINDataset(
        args.coin_root, task, args.train_split, 0, verify_images=False)
    signature = filter_signature(
        dataset, processor_path, max_length=args.max_length,
        max_visual_tokens=args.max_visual_tokens, length_bucket=args.length_bucket,
        image_transport=args.image_transport, image_base_url=args.image_base_url)
    final_path = filter_cache_path(args.filter_cache, signature)
    if final_path.is_file():
        return load_filter_manifest(final_path, signature), final_path, False

    plans = []
    shards: list[dict[str, Any]] = []
    for start, end in _ranges(len(dataset), args.scan_workers):
        path = filter_shard_path(args.filter_cache, signature, start, end)
        if path.is_file():
            try:
                shards.append(load_filter_shard(path, signature, start, end))
                print(json.dumps({"type": "filter_shard_reused", "task": task,
                                  "start": start, "end": end, "path": str(path)},
                                 ensure_ascii=False), flush=True)
                continue
            except (ValueError, json.JSONDecodeError):
                # The worker atomically overwrites an incomplete or stale part.
                pass
        plans.append((start, end, path))

    if plans:
        common = {
            "processor_path": processor_path, "coin_root": args.coin_root,
            "task": task, "train_split": args.train_split,
            "image_transport": args.image_transport, "image_base_url": args.image_base_url,
            "max_length": args.max_length, "max_visual_tokens": args.max_visual_tokens,
            "length_bucket": args.length_bucket, "signature": signature,
        }
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(args.scan_workers, len(plans)),
                                 mp_context=context) as pool:
            futures = {
                pool.submit(_build_shard_worker, {
                    **common, "start": start, "end": end, "path": str(path)}):
                (start, end, path)
                for start, end, path in plans
            }
            for future in as_completed(futures):
                start, end, path = futures[future]
                future.result()
                shards.append(load_filter_shard(path, signature, start, end))
                print(json.dumps({"type": "filter_shard_complete", "task": task,
                                  "start": start, "end": end, "path": str(path)},
                                 ensure_ascii=False), flush=True)
    return merge_filter_shards(dataset, signature, shards, final_path), final_path, True


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    from transformers import AutoProcessor

    processor_path = args.processor_path or args.model_path
    processor = collator = None
    if args.scan_workers == 1:
        processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
        processor.tokenizer.padding_side = "right"
        fetcher = ImageFetcher(args.image_transport, args.image_base_url, args.coin_root)
        collator = CoINCollator(
            processor, args.max_length, args.max_visual_tokens, fetcher,
            bucket=args.length_bucket)
    summaries = []
    for task in args.tasks:
        if args.scan_workers == 1:
            dataset = CoINDataset(
                args.coin_root, task, args.train_split, 0, verify_images=False)
            manifest, path, built = ensure_filter_manifest(
                dataset, collator, args.filter_cache, processor_path,
                max_length=args.max_length, max_visual_tokens=args.max_visual_tokens,
                length_bucket=args.length_bucket, image_transport=args.image_transport,
                image_base_url=args.image_base_url,
                progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True))
        else:
            manifest, path, built = ensure_parallel_manifest(args, task, processor_path)
        summary = {"task": task, "path": str(path), "cache_built": built,
                   "scan_workers": args.scan_workers,
                   **{key: manifest[key] for key in (
                       "original_examples", "valid_examples", "excluded_examples",
                       "excluded_rate", "impact")}}
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(json.dumps({"type": "filter_preparation_complete", "tasks": summaries},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
