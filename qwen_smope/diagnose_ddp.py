"""One-variable-at-a-time H100 diagnostic for CoIN training and DDP."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

from .data import CoINCollator, CoINDataset, ImageFetcher
from .filtering import ensure_filter_manifest
from .model import load_model
from .report import append_jsonl, write_json
from .train import DEFAULT_IMAGE_BASE_URL, transfer


VARIANTS = ("single", "multiprocess-independent", "ddp-same-batch", "ddp-sharded")


def validate_variant_world(variant: str, world: int) -> None:
    expected = 1 if variant == "single" else 2
    if world != expected:
        raise ValueError(f"{variant} requires WORLD_SIZE={expected}, got {world}")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="pretrained/Qwen3.5-9B")
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--coin-root", default="datas/CoIN")
    parser.add_argument("--task", default="VQAv2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--image-transport", choices=("mount", "http"), default="http")
    parser.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL)
    parser.add_argument("--filter-cache", default="outputs/coin-qwen3.5-9b/_filter_cache")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=None,
                        help="可选的受控对照断言；必须等于 batch*accumulation*WORLD_SIZE")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--length-bucket", type=int, default=128)
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.accumulation, args.max_length,
           args.max_visual_tokens) <= 0:
        parser.error("steps/batch/accumulation/length arguments must be positive")
    return args


def _sampler(dataset, variant: str, rank: int, world: int):
    # A/B/C intentionally see the same source-ordered batches.  This leaves
    # process count and DDP wrapping as the only changed variables.  Sharding
    # is introduced only in D, after the earlier controls pass.
    if variant in {"single", "multiprocess-independent", "ddp-same-batch"}:
        return SequentialSampler(dataset)
    return DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False)


def _sync_cuda(device: torch.device) -> float:
    torch.cuda.synchronize(device)
    return time.perf_counter()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_global_batch(args: argparse.Namespace, world: int,
                          variant: str | None = None) -> int:
    # Independent processes do not form one optimizer/global batch.
    multiplier = 1 if variant == "multiprocess-independent" else world
    effective = args.batch_size * args.accumulation * multiplier
    if args.global_batch_size is not None and args.global_batch_size != effective:
        raise ValueError(
            f"Expected global batch {args.global_batch_size}, got {effective} "
            f"({args.batch_size}*{args.accumulation}*{multiplier})")
    return effective


def main(argv: list[str] | None = None) -> None:
    args = arguments(argv)
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    validate_variant_world(args.variant, world)
    effective_global_batch = validate_global_batch(args, world, args.variant)
    batch_size_scope = ("per_process" if args.variant == "multiprocess-independent"
                        else "global")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The DDP diagnostic requires a BF16-capable CUDA GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))

    output = Path(args.output)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Diagnostic output is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    trace_path = output / f"rank-{rank}.jsonl"

    from transformers import AutoProcessor
    processor_path = args.processor_path or args.model_path
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    fetcher = ImageFetcher(args.image_transport, args.image_base_url, args.coin_root)
    collator = CoINCollator(
        processor, args.max_length, args.max_visual_tokens, fetcher, bucket=args.length_bucket)
    dataset = CoINDataset(
        args.coin_root, args.task, args.train_split, 0,
        verify_images=False)
    if rank == 0:
        manifest, cache_path, _ = ensure_filter_manifest(
            dataset, collator, args.filter_cache, processor_path,
            max_length=args.max_length, max_visual_tokens=args.max_visual_tokens,
            length_bucket=args.length_bucket, image_transport=args.image_transport,
            image_base_url=args.image_base_url,
            progress=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True))
    if dist.is_initialized():
        dist.barrier()
    if rank != 0:
        manifest, cache_path, _ = ensure_filter_manifest(
            dataset, collator, args.filter_cache, processor_path,
            max_length=args.max_length, max_visual_tokens=args.max_visual_tokens,
            length_bucket=args.length_bucket, image_transport=args.image_transport,
            image_base_url=args.image_base_url)
    dataset.apply_training_filter(manifest["valid_indices"], 0, manifest)
    sampler = _sampler(dataset, args.variant, rank, world)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=collator, num_workers=0, pin_memory=True)

    # Identical initialization is part of the B/C control.  DDP also broadcasts
    # rank 0, but making the pre-wrap state identical keeps B comparable.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = load_model(args.model_path, device, method="smope")
    network = (DDP(model, device_ids=[local_rank], broadcast_buffers=False)
               if args.variant.startswith("ddp-") else model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    run_started = _sync_cuda(device)
    samples_seen = 0
    optimizer_updates = 0

    try:
        for index in range(args.steps):
            data_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            data_finished = time.perf_counter()
            inputs = transfer(batch["inputs"], device)
            router_mask = transfer(batch["router_mask"], device)
            transfer_finished = _sync_cuda(device)
            group_start = (index // args.accumulation) * args.accumulation
            group_end = min(group_start + args.accumulation, args.steps)
            update = index + 1 == group_end
            context = (network.no_sync() if isinstance(network, DDP) and not update
                       else contextlib.nullcontext())
            with context:
                result = network(inputs, router_mask, dense=True)
                forward_finished = _sync_cuda(device)
                (result["loss"] / (group_end - group_start)).backward()
                backward_finished = _sync_cuda(device)
            optimizer_seconds = 0.0
            if update:
                optimizer_started = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                optimizer_finished = _sync_cuda(device)
                optimizer_seconds = optimizer_finished - optimizer_started
            row = {
                "type": "microbatch", "variant": args.variant, "rank": rank,
                "index": index, "update": update,
                "timestamp_utc": _timestamp(),
                "sample_ids": [item["id"] for item in batch["metadata"]],
                "sequence_width": int(inputs["input_ids"].shape[-1]),
                "data_wait_seconds": data_finished - data_started,
                "transfer_seconds": transfer_finished - data_finished,
                "forward_seconds": forward_finished - transfer_finished,
                "backward_seconds": backward_finished - forward_finished,
                "optimizer_seconds": optimizer_seconds,
                "synchronization": (
                    "ddp-allreduce" if isinstance(network, DDP) and update else
                    "ddp-no-sync" if isinstance(network, DDP) else "independent"),
                "loss": float(result["loss"].detach()),
                "monotonic_seconds": time.perf_counter() - run_started,
            }
            append_jsonl(trace_path, row)
            samples_seen += len(batch["metadata"])
    except Exception as error:
        append_jsonl(trace_path, {"type": "error", "variant": args.variant,
                                  "rank": rank, "error": repr(error),
                                  "timestamp_utc": _timestamp(),
                                  "traceback": traceback.format_exc(),
                                  "monotonic_seconds": time.perf_counter() - run_started})
        raise
    finally:
        elapsed = _sync_cuda(device) - run_started
        summary = {"type": "summary", "variant": args.variant,
                   "rank": rank, "steps": args.steps,
                   "timestamp_utc": _timestamp(), "wall_seconds": elapsed,
                   "samples_seen": samples_seen, "optimizer_updates": optimizer_updates,
                   "effective_global_batch_size": effective_global_batch,
                   "batch_size_scope": batch_size_scope,
                   "filter_manifest": str(cache_path)}
        try:
            state_stats = torch.zeros(2, dtype=torch.float64, device=device)
            nonfinite_parameters = 0
            with torch.no_grad():
                for parameter in model.parameters():
                    if not parameter.requires_grad:
                        continue
                    value = parameter.detach().float()
                    nonfinite_parameters += int((~torch.isfinite(value)).sum().item())
                    state_stats[0] += value.sum().double()
                    state_stats[1] += value.square().sum().double()
            summary.update({
                "trainable_parameter_sum": float(state_stats[0]),
                "trainable_parameter_l2": float(state_stats[1].sqrt()),
                "nonfinite_trainable_parameters": nonfinite_parameters,
            })
        except Exception as state_error:
            summary["trainable_state_error"] = repr(state_error)
        append_jsonl(trace_path, summary)
        if rank == 0:
            write_json(output / "configuration.json", {
                "arguments": vars(args), "world_size": world,
                "effective_global_batch_size": effective_global_batch,
                "batch_size_scope": batch_size_scope,
                "filter_manifest": str(cache_path),
                "torch_distributed_debug": os.getenv("TORCH_DISTRIBUTED_DEBUG"),
                "nccl_debug": os.getenv("NCCL_DEBUG"),
            })
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
