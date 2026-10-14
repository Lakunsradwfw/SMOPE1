"""Build full-split CoIN training-length manifests without loading the 9B model."""
from __future__ import annotations

import argparse
import json

from .data import TASK_ORDER, CoINCollator, CoINDataset, ImageFetcher
from .filtering import ensure_filter_manifest
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
    args = parser.parse_args(argv)
    args.tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = set(args.tasks) - set(TASK_ORDER)
    if not args.tasks or unknown:
        parser.error(f"Invalid --tasks; unknown={sorted(unknown)}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    from transformers import AutoProcessor

    processor_path = args.processor_path or args.model_path
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    fetcher = ImageFetcher(args.image_transport, args.image_base_url, args.coin_root)
    collator = CoINCollator(
        processor, args.max_length, args.max_visual_tokens, fetcher, bucket=args.length_bucket)
    summaries = []
    for task in args.tasks:
        dataset = CoINDataset(
            args.coin_root, task, args.train_split, 0,
            verify_images=args.image_transport == "mount")
        manifest, path, built = ensure_filter_manifest(
            dataset, collator, args.filter_cache, processor_path,
            max_length=args.max_length, max_visual_tokens=args.max_visual_tokens,
            length_bucket=args.length_bucket, image_transport=args.image_transport,
            image_base_url=args.image_base_url,
            progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True))
        summary = {"task": task, "path": str(path), "cache_built": built,
                   **{key: manifest[key] for key in (
                       "original_examples", "valid_examples", "excluded_examples",
                       "excluded_rate", "impact")}}
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(json.dumps({"type": "filter_preparation_complete", "tasks": summaries},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
