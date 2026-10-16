"""Run Qwen3.5 on CoIN: zero-shot, sequential LoRA, or generative SMoPE."""
from __future__ import annotations

import argparse
import contextlib
from datetime import timedelta
import gc
import hashlib
from importlib import metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .data import (TASK_ORDER, CoINCollator, CoINDataset, CoINEvalCollator,
                   ExactShard, ImageFetcher, check_model, resolve_split_path)
from .filtering import (FILTER_FORMAT_VERSION, ensure_filter_manifest, manifest_for_run,
                        processor_signature, validate_filter_impact)


DEFAULT_IMAGE_BASE_URL = (
    "http://10.0.255.254:4918/Storage(default)/ProjectGroup(group_h100)"
    "/zq/SMOPE/datas/CoIN/cl_dataset/")
from .model import QwenCoINModel, load_model
from .report import event, write_json, write_jsonl, write_matrix
from .scoring import score_prediction


WORKER_START_METHODS = tuple(
    method for method in ("forkserver", "spawn") if method in mp.get_all_start_methods())
DEFAULT_WORKER_START_METHOD = "forkserver" if "forkserver" in WORKER_START_METHODS else "spawn"
OBJECT_COLLECTIVE_TIMEOUT = timedelta(minutes=10)
_OBJECT_COLLECTIVE_GROUP = None


def arguments(argv: list[str] | None = None, world_size: int | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("zero_shot", "lora", "smope"), required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="smoke")
    parser.add_argument("--model-path", default="pretrained/Qwen3.5-9B")
    parser.add_argument("--processor-path", default=None,
                        help="processor/tokenizer 所在目录，默认与 --model-path 相同。"
                             "指向本地副本可避免在 davfs 上读取 tokenizer.json")
    parser.add_argument("--coin-root", default="datas/CoIN")
    parser.add_argument("--image-transport", choices=("mount", "http"), default="mount",
                        help="图像读取通道。mount 走 davfs 挂载；http 直连 WebDAV 服务器，"
                             "实测快约 60 倍（挂载 233MB/582s vs HTTP 2.4s）")
    parser.add_argument("--image-base-url", default=DEFAULT_IMAGE_BASE_URL,
                        help="image-transport=http 时的 cl_dataset 根 URL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default=",".join(TASK_ORDER))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dense-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1,
                        help="生成评测 batch；大于 1 使用左填充，正式采用前须与 batch=1 对照")
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=None,
                        help="默认按 global-batch-size/(batch-size*world-size) 推导")
    # 注意: 每个 DataLoader 只要 workers>0, 就要付一次 forkserver 启动 + 各 worker
    # import torch/transformers 并反序列化 processor 的固定代价, 实测约 24~38s,
    # **且几乎不随 worker 数变化**。make_loader 在训练、route_scan 及每次 evaluate_task
    # 都被调用, 所以 smoke 模式下这笔开销会完全淹没真实计算
    # (实测 train_sparse: workers=8 时 23.46s, workers=0 时 0.45s)。
    # 但正式 run 总共只创建约 52 个 DataLoader, 合计约半小时, 相对 20~30 小时的
    # 训练可以忽略, 而 workers>0 能让 HTTP 取图与计算重叠。取图中位数 4:
    # 4 线程实测 84 文件/s, 已远超需要。做 smoke 排查时可加 --workers 0 看真实耗时。
    parser.add_argument("--workers", type=int, default=None,
                        help="默认 preflight/smoke=0，full=4")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True, help="默认开启；--no-gradient-checkpointing 仅用于 H100 吞吐/显存对照")
    parser.add_argument("--worker-start-method", choices=WORKER_START_METHODS,
                        default=DEFAULT_WORKER_START_METHOD)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--length-bucket", type=int, default=128,
                        help="把 batch 宽度向上对齐到该倍数, 收敛 Triton 需要编译的形状数。"
                             "0 或 1 关闭。实测变长输入下每个新宽度要编译一整族内核, "
                             "导致 >17s/步（固定形状只要 0.33s/步）")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--eval-sample-seed", type=int, default=42)
    parser.add_argument("--routing-scan-samples", type=int, default=0)
    parser.add_argument("--routing-scan-seed", type=int, default=42,
                        help="路由扫描限量时的固定随机采样种子；0 个样本上限表示全量")
    parser.add_argument("--routing-scan-batch-size", type=int, default=None,
                        help="路由统计只做前向，默认复用 --batch-size；可设为 1 做逐样本等价性复核")
    parser.add_argument("--filter-cache", default="outputs/coin-qwen3.5-9b/_filter_cache",
                        help="完整训练 split 的确定性长度过滤清单目录")
    parser.add_argument("--allow-high-filter-rate", action="store_true",
                        help="确认并允许 full 模式运行剔除率达到 10%% 的任务")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--experts", type=int, default=25)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--route-mix", type=float, default=0.75)
    parser.add_argument("--route-temperature", type=float, default=0.07)
    parser.add_argument("--router-weight", type=float, default=1e-5)
    parser.add_argument("--old-weight", type=float, default=1e-5)
    parser.add_argument("--balance-weight", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)
    args.tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not args.tasks or len(set(args.tasks)) != len(args.tasks):
        parser.error("--tasks must contain unique task names")
    unknown = set(args.tasks) - set(TASK_ORDER)
    if unknown:
        parser.error(f"Unknown CoIN tasks: {sorted(unknown)}")
    positive = ("epochs", "batch_size", "eval_batch_size", "global_batch_size", "max_length",
                "max_new_tokens", "max_visual_tokens", "lora_rank", "lora_alpha",
                "experts", "topk", "log_every")
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error(f"These arguments must be positive: {positive}")
    if args.workers is None:
        args.workers = 4 if args.mode == "full" else 0
    if args.routing_scan_batch_size is None:
        args.routing_scan_batch_size = args.batch_size
    if args.topk > args.experts or args.workers < 0 or args.dense_epochs < 0:
        parser.error("Invalid topk/experts/workers/dense-epochs")
    if args.routing_scan_batch_size <= 0:
        parser.error("--routing-scan-batch-size must be positive")
    if args.learning_rate is None:
        args.learning_rate = 2e-4 if args.method == "lora" else 1e-3
    world = world_size if world_size is not None else int(os.getenv("WORLD_SIZE", "1"))
    denominator = args.batch_size * world
    if args.global_batch_size % denominator:
        parser.error("--global-batch-size must be divisible by --batch-size * WORLD_SIZE")
    inferred_accumulation = args.global_batch_size // denominator
    if args.accumulation is None:
        args.accumulation = inferred_accumulation
    elif args.accumulation != inferred_accumulation:
        parser.error("--accumulation conflicts with --global-batch-size and WORLD_SIZE")
    args.effective_global_batch_size = args.batch_size * world * args.accumulation
    if args.mode == "smoke":
        args.epochs = args.dense_epochs = 1
        args.max_train_samples = args.max_train_samples or 8
        args.max_eval_samples = args.max_eval_samples or 4
        args.routing_scan_samples = args.routing_scan_samples or 8
    elif args.mode == "preflight":
        args.max_train_samples = args.max_train_samples or 1
        args.max_eval_samples = args.max_eval_samples or 1
        args.routing_scan_samples = args.routing_scan_samples or 1
    return args


def sync() -> None:
    if dist.is_initialized():
        dist.barrier()


def initialize_distributed(world: int):
    """Use NCCL for tensors and a blocking CPU/Gloo group for Python objects.

    ``all_gather_object`` on an NCCL group serializes through CUDA tensors.  That
    made tiny prediction/metadata gathers depend on outstanding CUDA stream
    state and was the blocked path in a late-run two-rank H100 hang.  Keeping
    tensor collectives on NCCL while moving Python objects to Gloo preserves the
    training protocol and removes that CUDA synchronization path.
    """
    global _OBJECT_COLLECTIVE_GROUP
    if world <= 1:
        _OBJECT_COLLECTIVE_GROUP = None
        return None
    if not dist.is_gloo_available():
        raise RuntimeError("DDP requires PyTorch with Gloo support for safe object collectives")
    dist.init_process_group("nccl", timeout=timedelta(minutes=90))
    _OBJECT_COLLECTIVE_GROUP = dist.new_group(
        backend="gloo", timeout=OBJECT_COLLECTIVE_TIMEOUT)
    return _OBJECT_COLLECTIVE_GROUP


def all_gather_objects(value: Any, *, label: str, world: int) -> list[Any]:
    """Gather small Python values on Gloo and fail on collective-order drift."""
    if not dist.is_initialized():
        return [value]
    if _OBJECT_COLLECTIVE_GROUP is None:
        raise RuntimeError("Object collective group was not initialized")
    gathered = [None] * world
    envelope = {"label": label, "payload": value}
    dist.all_gather_object(gathered, envelope, group=_OBJECT_COLLECTIVE_GROUP)
    labels = [item.get("label") if isinstance(item, dict) else None for item in gathered]
    if any(item is None or labels[index] != label for index, item in enumerate(gathered)):
        raise RuntimeError(
            f"Object collective order mismatch at {label!r}; rank labels={labels!r}")
    return [item["payload"] for item in gathered]


def reduce(tensor: torch.Tensor, op=None) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op or dist.ReduceOp.SUM)
    return tensor


def transfer(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: transfer(item, device) for key, item in value.items()}
    return value


def make_loader(dataset, collator, args, rank: int, world: int, train: bool = False,
                batch_size: int | None = None):
    sampler = (DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
               if train else ExactShard(dataset, rank, world))
    context = args.worker_start_method if args.workers > 0 else None
    effective_batch_size = (batch_size if batch_size is not None
                            else args.batch_size if train else args.eval_batch_size)
    return DataLoader(dataset, batch_size=effective_batch_size,
                      sampler=sampler, collate_fn=collator, num_workers=args.workers,
                      pin_memory=True, multiprocessing_context=context), sampler


def image_fetcher(args) -> ImageFetcher:
    return ImageFetcher(args.image_transport, args.image_base_url, args.coin_root)


def train_dataset(args, task: str) -> CoINDataset:
    """Legacy loader used by lightweight tests; production calls prepare_train_dataset."""
    return CoINDataset(args.coin_root, task, args.train_split, args.max_train_samples,
                       verify_images=args.image_transport == "mount")


def prepare_train_dataset(args, task: str, collator: CoINCollator,
                          rank: int, world: int) -> CoINDataset:
    """Filter the complete source split, then take the requested valid prefix."""
    dataset = CoINDataset(args.coin_root, task, args.train_split, 0,
                          # The filter must record permanent missing images in
                          # its manifest instead of failing during path probing.
                          verify_images=False)
    processor_path = args.processor_path or args.model_path
    if rank == 0:
        manifest, cache_path, built = ensure_filter_manifest(
            dataset, collator, args.filter_cache, processor_path,
            max_length=args.max_length, max_visual_tokens=args.max_visual_tokens,
            length_bucket=args.length_bucket, image_transport=args.image_transport,
            image_base_url=args.image_base_url,
            progress=lambda row: event(args.output, row))
    else:
        manifest = cache_path = built = None
    sync()
    if rank != 0:
        manifest, cache_path, built = ensure_filter_manifest(
            dataset, collator, args.filter_cache, processor_path,
            max_length=args.max_length, max_visual_tokens=args.max_visual_tokens,
            length_bucket=args.length_bucket, image_transport=args.image_transport,
            image_base_url=args.image_base_url)
    validate_filter_impact(
        manifest, mode=args.mode, allow_high_filter_rate=args.allow_high_filter_rate)
    dataset.apply_training_filter(manifest["valid_indices"], args.max_train_samples, manifest)
    dataset.filter_manifest = manifest_for_run(
        manifest, cache_path, len(dataset), args.allow_high_filter_rate)
    if rank == 0:
        event(args.output, {"type": "training_filter", "task": task,
                            "cache": str(cache_path), "cache_built": built,
                            **{key: manifest[key] for key in (
                                "original_examples", "valid_examples", "excluded_examples",
                                "excluded_rate", "impact")}})
    return dataset


def eval_dataset(args, task: str) -> CoINDataset:
    """Held-out split, sampled with a seed derived from the task rather than --seed.

    Every method, seed sweep and continual-learning round must score the identical
    held-out examples, otherwise the accuracy matrix compares different data.
    """
    return CoINDataset(
        args.coin_root, task, args.eval_split, args.max_eval_samples,
        sample_seed=f"coin-eval-{args.eval_sample_seed}-{task}-{args.eval_split}",
        verify_images=args.image_transport == "mount")


def optimizer_for(model: QwenCoINModel, args) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"Method {args.method} has no trainable parameters")
    return torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)


def scheduler_for(optimizer, updates: int, warmup_ratio: float):
    warmup = int(updates * warmup_ratio)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(updates - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path: Path, model: QwenCoINModel, next_task: int,
                    matrix: list[list[float]], signature: dict, rank: int) -> None:
    """Persist the lightweight adapter, task progress and accuracy matrix.

    This is the resume point: it runs once per completed task, so a crashed run
    loses at most the task in progress. Upstream had this body pasted into
    ``record_dataset`` (where ``signature``/``next_task``/``matrix``/``model`` are
    not defined) and left this function as a bare ``sync()``, so no checkpoint was
    ever written and ``--resume`` could never have worked.
    """
    if rank == 0:
        state = {"format_version": 2, "signature": signature, "next_task": next_task,
                 "matrix": matrix, "adapter": model.lightweight_state()}
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
    sync()


def record_dataset(directory: str | Path, dataset: CoINDataset, role: str, rank: int) -> None:
    """Record which split file backed a role, for later provenance checks."""
    if rank != 0:
        return
    path = Path(directory) / "dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    entry = {
        "source": str(dataset.path), "source_size_bytes": dataset.source_size,
        "source_mtime_ns": dataset.source_mtime_ns, "examples_loaded": len(dataset),
    }
    if dataset.filter_manifest is not None:
        entry["training_filter"] = dataset.filter_manifest
    manifest[f"{role}:{dataset.task}:{dataset.split}"] = entry
    write_json(path, manifest)


@contextlib.contextmanager
def stage(args, name: str, task: int, device: torch.device, rank: int, world: int):
    sync()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
        reduce(elapsed, dist.ReduceOp.MAX)
        local = {"rank": rank, "allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                 "reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
        memories = all_gather_objects(
            local, label=f"stage:{task}:{name}:memory", world=world)
        if rank == 0:
            event(args.output, {"type": "stage", "stage": name, "task": task + 1,
                                "seconds": float(elapsed), "memory": memories})


def train_epoch(network, model, loader, sampler, optimizer, scheduler, args,
                device, task_index: int, epoch: int, dense: bool, rank: int, world: int) -> dict:
    model.train()
    sampler.set_epoch(task_index * 10000 + epoch + (0 if dense else 1000))
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(7, dtype=torch.float64, device=device)
    steps = len(loader)
    input_wait_seconds = 0.0
    optimizer_updates = 0
    width_counts: dict[int, dict[str, int]] = {}
    iterator = iter(loader)
    for index in range(steps):
        wait_started = time.perf_counter()
        batch = next(iterator)
        input_wait_seconds += time.perf_counter() - wait_started
        inputs = transfer(batch["inputs"], device)
        router_mask = transfer(batch["router_mask"], device)
        group_start = (index // args.accumulation) * args.accumulation
        group_end = min(group_start + args.accumulation, steps)
        group_size = group_end - group_start
        update = index + 1 == group_end
        no_sync = network.no_sync() if isinstance(network, DDP) and not update else contextlib.nullcontext()
        with no_sync:
            result = network(inputs, router_mask, dense=dense)
            if not bool(torch.isfinite(result["loss"])):
                raise FloatingPointError(f"Nonfinite loss at task={task_index + 1}, epoch={epoch + 1}")
            (result["loss"] / group_size).backward()
        tokens = int((inputs["labels"] != -100).sum())
        samples = int(inputs["input_ids"].shape[0])
        width = int(inputs["input_ids"].shape[1])
        width_entry = width_counts.setdefault(width, {"microbatches": 0, "samples": 0})
        width_entry["microbatches"] += 1
        width_entry["samples"] += samples
        totals += torch.tensor([
            float(result["lm_loss"].detach()) * tokens,
            float(result["router_loss"].detach()) * samples,
            float(result["old_loss"].detach()) * samples,
            float(result["balance_loss"].detach()) * samples,
            float(result["loss"].detach()) * samples, tokens, samples,
        ], dtype=torch.float64, device=device)
        if update:
            norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], args.clip_grad)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
            if rank == 0 and ((index + 1) % (args.log_every * args.accumulation) == 0 or index + 1 == steps):
                event(args.output, {"type": "step", "task": task_index + 1, "epoch": epoch + 1,
                                    "dense": dense, "batch": index + 1, "loss": float(result["loss"].detach()),
                                    "gradient_norm": float(norm), "lr": scheduler.get_last_lr()})
    reduce(totals)
    wait = torch.tensor(input_wait_seconds, dtype=torch.float64, device=device)
    reduce(wait, dist.ReduceOp.MAX)
    gathered_widths = all_gather_objects(
        width_counts,
        label=f"train:{task_index}:{epoch}:{'dense' if dense else 'sparse'}:widths",
        world=world)
    merged_widths: dict[int, dict[str, int]] = {}
    for rank_widths in gathered_widths:
        for width, values in rank_widths.items():
            entry = merged_widths.setdefault(int(width), {"microbatches": 0, "samples": 0})
            entry["microbatches"] += int(values["microbatches"])
            entry["samples"] += int(values["samples"])
    return {"lm_loss_per_token": float(totals[0] / max(float(totals[5]), 1.0)),
            "router_loss": float(totals[1] / max(float(totals[6]), 1.0)),
            "old_loss": float(totals[2] / max(float(totals[6]), 1.0)),
            "balance_loss": float(totals[3] / max(float(totals[6]), 1.0)),
            "loss": float(totals[4] / max(float(totals[6]), 1.0)),
            "supervised_tokens": int(totals[5]), "samples_with_ddp_padding": int(totals[6]),
            # Time blocked in DataLoader.__next__. With workers=0 this includes HTTP image
            # fetch and processor work; with workers>0 it is the non-overlapped input wait.
            # It is intentionally measured without per-step CUDA synchronisation so the
            # diagnostic does not distort the throughput it is meant to explain.
            "input_wait_seconds": float(wait),
            "optimizer_updates": optimizer_updates,
            "sequence_widths": {str(width): merged_widths[width]
                                for width in sorted(merged_widths)}}


def routing_scan_indices(length: int, limit: int, seed: int,
                         task: str, split: str) -> list[int]:
    """Choose a reproducible source-ordered random subset, never a biased prefix."""
    if limit <= 0 or limit >= length:
        return list(range(length))
    generator = random.Random(f"coin-route-{seed}-{task}-{split}")
    # One fixed permutation makes the N=1000 subset a true prefix of the
    # N=5000 subset, so stability tests change sample count without also
    # changing every selected record.
    indices = list(range(length))
    generator.shuffle(indices)
    return sorted(indices[:limit])


@torch.no_grad()
def scan_routes(model: QwenCoINModel, dataset, collator, args, device, rank, world) -> list[dict]:
    task, split, original_count = dataset.task, dataset.split, len(dataset)
    indices = routing_scan_indices(
        original_count, args.routing_scan_samples, args.routing_scan_seed, task, split)
    if len(indices) < original_count:
        dataset = Subset(dataset, indices)
    scan_metadata = {
        "sampling": "full" if len(indices) == original_count else "fixed_random",
        "sample_count": len(indices),
        "source_count": original_count,
        "sample_seed": args.routing_scan_seed,
        "source_indices_sha256": hashlib.sha256(
            ",".join(map(str, indices)).encode("ascii")).hexdigest(),
    }
    # Route observation is a regular masked forward, not autoregressive generation.
    # Reusing the training batch avoids a full extra pass over every task at
    # eval_batch_size=1 while preserving the exact sample set and token masks.
    loader, _ = make_loader(dataset, collator, args, rank, world, train=False,
                            batch_size=args.routing_scan_batch_size)
    model.eval()
    counts = [torch.zeros_like(expert.frequency) for expert in model.expert_modules()]
    for batch in loader:
        model.observe_routes(transfer(batch["inputs"], device), transfer(batch["router_mask"], device))
        for expert, count in zip(model.expert_modules(), counts):
            indices = expert.last_indices.permute(1, 0, 2).reshape(count.shape[0], -1)
            count.scatter_add_(1, indices, torch.ones_like(indices))
    result = []
    for layer, expert, count in zip(model.layer_ids, model.expert_modules(), counts):
        reduce(count)
        expert.frequency.add_(count)
        frequency = expert.frequency.float()
        probability = frequency / frequency.sum(-1, keepdim=True).clamp_min(1)
        result.append({"layer": layer, "scan": scan_metadata,
                       "task_frequency": count.cpu().tolist(),
                       "cumulative_frequency": expert.frequency.cpu().tolist(),
                       "coverage": (frequency > 0).float().mean(-1).cpu().tolist(),
                       "entropy": (-(probability * probability.clamp_min(1e-30).log()).sum(-1)).cpu().tolist(),
                       "protected": expert.used.cpu().tolist(),
                       "route_mix": expert.route_mix_logit.sigmoid().detach().cpu().tolist(),
                       "temperature": expert.log_temperature.exp().detach().cpu().tolist()})
    return result


@torch.no_grad()
def evaluate_task(model: QwenCoINModel, dataset, collator, processor, args, device,
                  rank: int, world: int, stage_name: str) -> float:
    loader, _ = make_loader(dataset, collator, args, rank, world, train=False)
    model.eval()
    local_rows: list[dict] = []
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    tokenizer = processor.tokenizer
    for batch in loader:
        inputs = transfer(batch["inputs"], device)
        input_width = inputs["input_ids"].shape[1]
        generated = model.generate(
            inputs, transfer(batch["router_mask"], device), max_new_tokens=args.max_new_tokens,
            do_sample=False, num_beams=1, use_cache=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        predictions = processor.batch_decode(generated[:, input_width:], skip_special_tokens=True)
        for prediction, metadata in zip(predictions, batch["metadata"]):
            scored = score_prediction(metadata["task"], prediction, metadata["references"])
            totals += torch.tensor([scored["score"], 1.0], dtype=torch.float64, device=device)
            local_rows.append({**metadata, "prediction": prediction, **scored})
    reduce(totals)
    gathered = all_gather_objects(
        local_rows, label=f"eval:{stage_name}:{dataset.task}:predictions", world=world)
    if rank == 0:
        rows = sorted((row for part in gathered for row in part), key=lambda row: row["id"])
        write_jsonl(Path(args.output) / "predictions" / stage_name / f"{dataset.task}.jsonl", rows)
    return 100.0 * float(totals[0]) / max(float(totals[1]), 1.0)


def model_signature(args, config: dict, world: int) -> dict:
    identity = {key: value for key, value in vars(args).items()
                if key not in {"output", "resume", "workers", "worker_start_method",
                               "log_every", "model_path", "processor_path", "coin_root",
                               "filter_cache", "allow_high_filter_rate",
                               # I/O plumbing: same bytes either way, so switching
                               # transport must not invalidate a resume.
                               "image_transport", "image_base_url",
                               # Padding width only; masked positions contribute
                               # nothing to the loss, so it is not experiment state.
                               "length_bucket"}}
    index = Path(args.model_path) / "model.safetensors.index.json"
    files = []
    split_roles = [("eval", args.eval_split)]
    if args.method != "zero_shot":
        split_roles.append(("train", args.train_split))
    root = Path(args.coin_root).resolve()
    for task in args.tasks:
        for role, split in split_roles:
            # Resolved, not assumed: training reads Instructions_Qwen while the
            # held-out instructions come from Instructions_Original.
            path, kind = resolve_split_path(root, task, split)
            stat = path.stat()
            files.append({"role": role, "task": task, "split": split, "format": kind,
                          "source": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"arguments": identity, "world_size": world, "model_config": config,
            "filter_format_version": FILTER_FORMAT_VERSION,
            "processor": processor_signature(args.processor_path or args.model_path),
            "dataset_files": files,
            "weight_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None}


def main(argv: list[str] | None = None) -> None:
    run_started = time.perf_counter()
    rank = int(os.getenv("RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    args = arguments(argv, world)
    if not torch.cuda.is_available():
        raise RuntimeError("CoIN Qwen3.5 runs require CUDA; unit tests remain CPU-only")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable GPU is required")
    initialize_distributed(world)
    peft_version = None
    if args.method == "lora":
        try:
            peft_version = metadata.version("peft")
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "method=lora requires peft==0.18.1; run scripts/qwen/setup.sh") from error
        if peft_version != "0.18.1":
            raise RuntimeError(
                f"method=lora requires peft==0.18.1, found {peft_version!r}")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    output = Path(args.output)
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError(f"Output {output} is not empty; use --resume or another directory")
        output.mkdir(parents=True, exist_ok=True)
    sync()
    config = check_model(args.model_path)
    signature = model_signature(args, config, world)
    checkpoint_path = output / "latest_adapter.pt"
    checkpoint = None
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No lightweight checkpoint to resume: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 2:
            raise ValueError("Unsupported lightweight checkpoint format")
        if checkpoint["signature"] != signature:
            raise ValueError("Resume requires identical method, task order, model and training arguments")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.processor_path or args.model_path,
                                              local_files_only=True)
    processor.tokenizer.padding_side = "right"
    fetcher = image_fetcher(args)
    train_collator = CoINCollator(processor, args.max_length, args.max_visual_tokens, fetcher,
                                  bucket=args.length_bucket)
    eval_collator = CoINEvalCollator(processor, args.max_length, args.max_visual_tokens, fetcher,
                                     bucket=args.length_bucket)
    preview_index = 0 if checkpoint is None else min(checkpoint["next_task"], len(args.tasks) - 1)
    preview_task = args.tasks[preview_index]
    preview_role = "eval" if args.method == "zero_shot" else "train"
    # Parse one real split and process one image before allocating 9B parameters.
    # Remaining task files were already checked by model_signature and are loaded lazily.
    preview_set = (eval_dataset(args, preview_task) if preview_role == "eval" else
                   prepare_train_dataset(args, preview_task, train_collator, rank, world))
    record_dataset(output, preview_set, preview_role, rank)
    (eval_collator if args.method == "zero_shot" else train_collator)([preview_set[0]])

    if rank == 0:
        import transformers
        try:
            commit = subprocess.check_output(
                ["git", "-c", f"safe.directory={Path.cwd()}", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        write_json(output / ("resume_environment.json" if args.resume else "environment.json"),
                   {"arguments": vars(args), "signature": signature, "git_commit": commit,
                    "python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "cuda": torch.version.cuda,
                    "peft": peft_version,
                    "gpu": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
                    "distributed_collectives": {
                        "tensor": "nccl" if world > 1 else None,
                        "python_object": "gloo" if world > 1 else None,
                        "object_timeout_seconds": (
                            int(OBJECT_COLLECTIVE_TIMEOUT.total_seconds()) if world > 1 else None),
                    },
                    "protocol": "CoIN task accuracy; no raw-example replay; no base weights saved"})
    with stage(args, "model_loading", -1, device, rank, world):
        model = load_model(
            args.model_path, device, method=args.method, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            gradient_checkpointing=args.gradient_checkpointing,
            experts=args.experts, topk=args.topk, epsilon=args.epsilon,
            router_weight=args.router_weight, old_weight=args.old_weight,
            balance_weight=args.balance_weight, route_mix=args.route_mix,
            route_temperature=args.route_temperature)
    if checkpoint:
        model.load_lightweight_state(checkpoint["adapter"])
    if rank == 0:
        event(args.output, {"type": "model", "method": args.method,
                            "full_attention_layers": model.layer_ids,
                            "total_parameters": sum(p.numel() for p in model.parameters()),
                            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                            **model.loading_profile})

    # Real model preflight before a long run.
    with stage(args, "preflight", -1, device, rank, world):
        if args.method == "zero_shot":
            model.eval()
            batch = eval_collator([preview_set[0]])
            model.generate(transfer(batch["inputs"], device), transfer(batch["router_mask"], device),
                           max_new_tokens=2, do_sample=False, num_beams=1, use_cache=True,
                           pad_token_id=processor.tokenizer.pad_token_id,
                           eos_token_id=processor.tokenizer.eos_token_id)
        else:
            batch = train_collator([preview_set[0]])
            model.train()
            result = model(transfer(batch["inputs"], device), transfer(batch["router_mask"], device),
                           dense=args.method == "smope")
            result["loss"].backward()
            trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
            if any(parameter.grad is None or not bool(parameter.grad.isfinite().all()) for _, parameter in trainable):
                raise RuntimeError("Preflight found missing or nonfinite adapter gradients")
            model.zero_grad(set_to_none=True)
    del batch
    if args.method != "zero_shot":
        del result, trainable
    torch.cuda.empty_cache()
    if args.mode == "preflight":
        if rank == 0:
            event(args.output, {"type": "preflight_passed"})
        return

    if args.method == "zero_shot":
        scores = {}
        for task_index, task in enumerate(args.tasks):
            dataset = preview_set if task_index == preview_index else eval_dataset(args, task)
            record_dataset(output, dataset, "eval", rank)
            with stage(args, "zero_shot_evaluation", args.tasks.index(task), device, rank, world):
                scores[task] = evaluate_task(model, dataset, eval_collator, processor,
                                             args, device, rank, world, "zero_shot")
            if dataset is preview_set:
                preview_set = None
            del dataset
            gc.collect()
        if rank == 0:
            summary = {"task_accuracy": scores, "average_accuracy": float(np.mean(list(scores.values())))}
            write_json(output / "summary.json", summary)
            event(args.output, {"type": "run_complete", **summary,
                                "wall_seconds": time.perf_counter() - run_started})
        return

    network = DDP(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    matrix = checkpoint["matrix"] if checkpoint else []
    start_task = checkpoint["next_task"] if checkpoint else 0
    for task_index in range(start_task, len(args.tasks)):
        task = args.tasks[task_index]
        train_set = (preview_set if task_index == preview_index
                     else prepare_train_dataset(args, task, train_collator, rank, world))
        if train_set is preview_set:
            preview_set = None
        record_dataset(output, train_set, "train", rank)
        loader, sampler = make_loader(train_set, train_collator, args, rank, world, train=True)
        phases = [("dense", args.dense_epochs if args.method == "smope" and task_index == 0 else 0),
                  ("sparse" if args.method == "smope" else "lora", args.epochs)]
        for phase, epochs in phases:
            if not epochs:
                continue
            optimizer = optimizer_for(model, args)
            updates = epochs * math.ceil(len(loader) / args.accumulation)
            scheduler = scheduler_for(optimizer, updates, args.warmup_ratio)
            for epoch in range(epochs):
                with stage(args, f"train_{phase}", task_index, device, rank, world):
                    stats = train_epoch(network, model, loader, sampler, optimizer, scheduler,
                                        args, device, task_index, epoch, phase == "dense", rank, world)
                if rank == 0:
                    event(args.output, {"type": "epoch", "task": task, "task_index": task_index + 1,
                                        "phase": phase, "epoch": epoch + 1, **stats})
        if args.method == "smope":
            with stage(args, "route_scan", task_index, device, rank, world):
                routing = scan_routes(model, train_set, train_collator, args, device, rank, world)
            if rank == 0:
                write_json(output / "routing" / f"after_{task_index + 1:02d}_{task}.json", routing)
        del loader, sampler, train_set
        gc.collect()
        row = []
        for seen_task in args.tasks[:task_index + 1]:
            eval_set = eval_dataset(args, seen_task)
            record_dataset(output, eval_set, "eval", rank)
            with stage(args, f"evaluate_{seen_task}", task_index, device, rank, world):
                row.append(evaluate_task(model, eval_set, eval_collator, processor, args,
                                         device, rank, world, f"after_{task_index + 1:02d}_{task}"))
            del eval_set
            gc.collect()
        matrix.append(row)
        if args.method == "smope":
            for expert in model.expert_modules():
                expert.next_task()
        save_checkpoint(checkpoint_path, model, task_index + 1, matrix, signature, rank)
        if rank == 0:
            summary = write_matrix(output, args.tasks, matrix)
            event(args.output, {"type": "task_complete", "task": task,
                                "task_index": task_index + 1, **summary})
    if rank == 0:
        summary = write_matrix(output, args.tasks, matrix)
        event(args.output, {"type": "run_complete", "tasks": len(args.tasks), **summary,
                            "wall_seconds": time.perf_counter() - run_started})


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM: reduce --max-length, --max-visual-tokens, or --batch-size and use a new output.",
              flush=True)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
