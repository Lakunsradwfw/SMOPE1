"""Run Qwen3.5 on CoIN: zero-shot, sequential LoRA, or generative SMoPE."""
from __future__ import annotations

import argparse
import contextlib
from datetime import timedelta
import gc
import hashlib
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
                   ExactShard, check_model)
from .model import QwenCoINModel, load_model
from .report import event, write_json, write_jsonl, write_matrix
from .scoring import score_prediction


WORKER_START_METHODS = tuple(
    method for method in ("forkserver", "spawn") if method in mp.get_all_start_methods())
DEFAULT_WORKER_START_METHOD = "forkserver" if "forkserver" in WORKER_START_METHODS else "spawn"


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("zero_shot", "lora", "smope"), required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="smoke")
    parser.add_argument("--model-path", default="pretrained/Qwen3.5-9B")
    parser.add_argument("--coin-root", default="datas/CoIN")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default=",".join(TASK_ORDER))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dense-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker-start-method", choices=WORKER_START_METHODS,
                        default=DEFAULT_WORKER_START_METHOD)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--routing-scan-samples", type=int, default=0)
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
    positive = ("epochs", "batch_size", "eval_batch_size", "accumulation", "max_length",
                "max_new_tokens", "max_visual_tokens", "lora_rank", "lora_alpha",
                "experts", "topk", "log_every")
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error(f"These arguments must be positive: {positive}")
    if args.topk > args.experts or args.workers < 0 or args.dense_epochs < 0:
        parser.error("Invalid topk/experts/workers/dense-epochs")
    if args.eval_batch_size != 1:
        parser.error("Qwen decoder evaluation currently requires --eval-batch-size 1")
    if args.learning_rate is None:
        args.learning_rate = 2e-4 if args.method == "lora" else 1e-3
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


def make_loader(dataset, collator, args, rank: int, world: int, train: bool = False):
    sampler = (DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
               if train else ExactShard(dataset, rank, world))
    context = args.worker_start_method if args.workers > 0 else None
    return DataLoader(dataset, batch_size=args.batch_size if train else args.eval_batch_size,
                      sampler=sampler, collate_fn=collator, num_workers=args.workers,
                      pin_memory=True, multiprocessing_context=context), sampler


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
    sync()


def record_dataset(directory: str | Path, dataset: CoINDataset, role: str, rank: int) -> None:
    if rank != 0:
        return
    path = Path(directory) / "dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    manifest[f"{role}:{dataset.task}:{dataset.split}"] = {
        "source": str(dataset.path), "source_size_bytes": dataset.source_size,
        "source_mtime_ns": dataset.source_mtime_ns, "examples_loaded": len(dataset),
    }
    write_json(path, manifest)
    if rank == 0:
        state = {"format_version": 2, "signature": signature, "next_task": next_task,
                 "matrix": matrix, "adapter": model.lightweight_state()}
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
    sync()


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
        memories = [None] * world
        local = {"rank": rank, "allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                 "reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
        if dist.is_initialized():
            dist.all_gather_object(memories, local)
        else:
            memories = [local]
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
    for index, batch in enumerate(loader):
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
            if rank == 0 and ((index + 1) % (args.log_every * args.accumulation) == 0 or index + 1 == steps):
                event(args.output, {"type": "step", "task": task_index + 1, "epoch": epoch + 1,
                                    "dense": dense, "batch": index + 1, "loss": float(result["loss"].detach()),
                                    "gradient_norm": float(norm), "lr": scheduler.get_last_lr()})
    reduce(totals)
    return {"lm_loss_per_token": float(totals[0] / max(float(totals[5]), 1.0)),
            "router_loss": float(totals[1] / max(float(totals[6]), 1.0)),
            "old_loss": float(totals[2] / max(float(totals[6]), 1.0)),
            "balance_loss": float(totals[3] / max(float(totals[6]), 1.0)),
            "loss": float(totals[4] / max(float(totals[6]), 1.0)),
            "supervised_tokens": int(totals[5]), "samples_with_ddp_padding": int(totals[6])}


@torch.no_grad()
def scan_routes(model: QwenCoINModel, dataset, collator, args, device, rank, world) -> list[dict]:
    if args.routing_scan_samples > 0:
        dataset = Subset(dataset, range(min(len(dataset), args.routing_scan_samples)))
    loader, _ = make_loader(dataset, collator, args, rank, world, train=False)
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
        result.append({"layer": layer, "task_frequency": count.cpu().tolist(),
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
    gathered = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(gathered, local_rows)
    else:
        gathered = [local_rows]
    if rank == 0:
        rows = sorted((row for part in gathered for row in part), key=lambda row: row["id"])
        write_jsonl(Path(args.output) / "predictions" / stage_name / f"{dataset.task}.jsonl", rows)
    return 100.0 * float(totals[0]) / max(float(totals[1]), 1.0)


def model_signature(args, config: dict, world: int) -> dict:
    identity = {key: value for key, value in vars(args).items()
                if key not in {"output", "resume", "workers", "worker_start_method",
                               "log_every", "model_path", "coin_root"}}
    index = Path(args.model_path) / "model.safetensors.index.json"
    files = []
    split_roles = [("eval", args.eval_split)]
    if args.method != "zero_shot":
        split_roles.append(("train", args.train_split))
    root = Path(args.coin_root).resolve()
    for task in args.tasks:
        for role, split in split_roles:
            path = root / "Instructions_Qwen" / task / f"{split}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing CoIN {role} split: {path}")
            stat = path.stat()
            files.append({"role": role, "task": task, "split": split,
                          "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {"arguments": identity, "world_size": world, "model_config": config,
            "dataset_files": files,
            "weight_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest() if index.is_file() else None}


def main(argv: list[str] | None = None) -> None:
    run_started = time.perf_counter()
    args = arguments(argv)
    rank = int(os.getenv("RANK", 0))
    world = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("CoIN Qwen3.5 runs require CUDA; unit tests remain CPU-only")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable GPU is required")
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=90))
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
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    train_collator = CoINCollator(processor, args.max_length, args.max_visual_tokens)
    eval_collator = CoINEvalCollator(processor, args.max_length, args.max_visual_tokens)
    preview_index = 0 if checkpoint is None else min(checkpoint["next_task"], len(args.tasks) - 1)
    preview_task = args.tasks[preview_index]
    preview_role = "eval" if args.method == "zero_shot" else "train"
    preview_split = args.eval_split if args.method == "zero_shot" else args.train_split
    preview_limit = args.max_eval_samples if args.method == "zero_shot" else args.max_train_samples
    # Parse one real split and process one image before allocating 9B parameters.
    # Remaining task files were already checked by model_signature and are loaded lazily.
    preview_set = CoINDataset(args.coin_root, preview_task, preview_split, preview_limit)
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
                    "gpu": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
                    "protocol": "CoIN task accuracy; no raw-example replay; no base weights saved"})
    with stage(args, "model_loading", -1, device, rank, world):
        model = load_model(
            args.model_path, device, method=args.method, lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
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
            dataset = preview_set if task_index == preview_index else CoINDataset(
                args.coin_root, task, args.eval_split, args.max_eval_samples)
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
        train_set = preview_set if task_index == preview_index else CoINDataset(
            args.coin_root, task, args.train_split, args.max_train_samples)
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
            eval_set = CoINDataset(args.coin_root, seen_task, args.eval_split, args.max_eval_samples)
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
