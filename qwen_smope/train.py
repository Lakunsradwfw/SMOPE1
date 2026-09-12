"""Run with torchrun --standalone --nproc_per_node=2 -m qwen_smope.train."""
import argparse
import contextlib
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import DATASETS, Collator, ExactShard, View, build_datasets, check_model
from .model import load_model
from .report import event, summarize, write_json


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--method", choices=["head_only", "smope"], default="smope")
    p.add_argument("--mode", choices=["smoke", "full", "preflight"], default="smoke")
    p.add_argument("--model-path", default="pretrained/Qwen3.5-9B-Base")
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--dense-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max-visual-tokens", type=int, default=256)
    p.add_argument("--experts", type=int, default=25)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--epsilon", type=float, default=0.4)
    p.add_argument("--router-weight", type=float, default=1e-5)
    p.add_argument("--old-weight", type=float, default=1e-5)
    p.add_argument("--prompt-lr", type=float, default=1e-3)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--clip-grad", type=float, default=1.)
    p.add_argument("--correction-epochs", type=int, default=5)
    p.add_argument("--correction-lr", type=float, default=1e-3)
    p.add_argument("--prototype-samples", type=int, default=16)
    p.add_argument("--log-every", type=int, default=10)
    a = p.parse_args()
    for name in ("epochs", "batch_size", "accumulation", "max_visual_tokens", "experts", "topk", "prototype_samples", "log_every"):
        if getattr(a, name) <= 0:
            p.error(f"{name} must be positive")
    if a.topk > a.experts or a.workers < 0 or a.dense_epochs < 0 or a.correction_epochs < 0:
        p.error("Invalid topk/workers/epoch settings")
    if a.mode == "smoke":
        a.epochs, a.dense_epochs, a.correction_epochs = 1, 1, min(1, a.correction_epochs)
    return a


def sync():
    if dist.is_initialized():
        dist.barrier()


def reduce(tensor, op=dist.ReduceOp.SUM):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def transfer(inputs, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in inputs.items()}


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def save_checkpoint(path, model, progress, optimizer, scheduler, matrix, prototypes, signature, rank, world):
    local = rng_state()
    states = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(states, local)
    else:
        states = [local]
    if rank == 0:
        state = dict(format_version=1, signature=signature, adapter=model.lightweight_state(),
                     progress=progress, optimizer=optimizer.state_dict() if optimizer else None,
                     scheduler=scheduler.state_dict() if scheduler else None, matrix=matrix,
                     prototypes={k: tuple(v.cpu() for v in values) for k, values in prototypes.items()}, rng=states)
        temp = path.with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(path)
    sync()


def make_loader(view, collate, args, rank, world, train=False):
    sampler = DistributedSampler(view, num_replicas=world, rank=rank, seed=args.seed, shuffle=True) if train else ExactShard(view, rank, world)
    # Dataset order and processor are deterministic. Keep workers off CUDA.
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(view, batch_size=args.batch_size, sampler=sampler, collate_fn=collate,
                      num_workers=args.workers, pin_memory=True, generator=generator), sampler


def optimizer_for(model, args):
    groups = [dict(params=list(model.classifier.parameters()), lr=args.head_lr)]
    experts = [p for m in model.expert_modules() for p in m.parameters()]
    if experts:
        groups.append(dict(params=experts, lr=args.prompt_lr))
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def train_epoch(network, model, loader, optimizer, scheduler, args, device, start, end,
                dense, task, epoch, phase, rank, world):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums = torch.zeros(6, device=device, dtype=torch.float64)
    steps = len(loader)
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = transfer(inputs, device), labels.to(device)
        group_start = (i // args.accumulation) * args.accumulation
        group_end = min(group_start + args.accumulation, steps)
        # Weight unequal final microbatches by actual sample count in this accumulation window.
        local_samples = len(loader.sampler)
        group_samples = min(group_end * args.batch_size, local_samples) - group_start * args.batch_size
        update = i + 1 == group_end
        ctx = contextlib.nullcontext() if update or not isinstance(network, DDP) else network.no_sync()
        with ctx:
            logits, router, old, _ = network(inputs, dense=dense)
            ce = F.cross_entropy(logits[:, start:end], labels - start)
            loss = ce + router + old
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at task={task} phase={phase} epoch={epoch} batch={i}")
            (loss * (labels.numel() / group_samples)).backward()
        correct = (logits[:, start:end].argmax(-1) == labels - start).sum()
        sums += torch.stack((ce.detach().double() * labels.numel(), router.detach().double() * labels.numel(),
                             old.detach().double() * labels.numel(), loss.detach().double() * labels.numel(),
                             correct.double(), torch.tensor(labels.numel(), device=device, dtype=torch.float64)))
        if update:
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_grad)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if (i + 1) % (args.log_every * args.accumulation) == 0 or i + 1 == steps:
                record = torch.stack([loss.detach().double(), norm.detach().double()])
                reduce(record)
                if rank == 0:
                    event(args.output, dict(type="step", task=task + 1, phase=phase, epoch=epoch + 1,
                         batch=i + 1, optimizer_step=scheduler.last_epoch,
                         loss=float(record[0] / world), gradient_norm=float(record[1] / world),
                         lr=[g["lr"] for g in optimizer.param_groups]))
    reduce(sums)
    n = max(float(sums[5]), 1)
    return dict(ce=float(sums[0]) / n, router=float(sums[1]) / n, old_router=float(sums[2]) / n,
                loss=float(sums[3]) / n, train_accuracy=100 * float(sums[4]) / n,
                samples_including_distributed_padding=int(n))


@torch.no_grad()
def scan(model, loader, device, classes, task_start, task_end, prototypes):
    model.eval()
    dim = model.classifier.in_features
    sums = torch.zeros(classes, dim, device=device, dtype=torch.float64)
    squares, counts = torch.zeros_like(sums), torch.zeros(classes, device=device, dtype=torch.float64)
    frequencies = [torch.zeros_like(m.frequency) for m in model.expert_modules()]
    for inputs, labels in loader:
        labels = labels.to(device)
        _, _, _, features = model(transfer(inputs, device))
        features = features.double()
        sums.index_add_(0, labels, features)
        squares.index_add_(0, labels, features.square())
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        for m, freq in zip(model.expert_modules(), frequencies):
            ix = m.last_indices.permute(1, 0, 2).reshape(freq.shape[0], -1)
            freq.scatter_add_(1, ix, torch.ones_like(ix))
    for value in (sums, squares, counts):
        reduce(value)
    for m, freq in zip(model.expert_modules(), frequencies):
        reduce(freq)
        m.frequency.add_(freq)
    for c in range(task_start, task_end):
        if counts[c] == 0:
            raise ValueError(f"No training observations for class {c}")
        mean = sums[c] / counts[c]
        variance = ((squares[c] - sums[c].square() / counts[c]) / (counts[c] - 1).clamp_min(1)).clamp_min(1e-4)
        prototypes[c] = (mean.float().cpu(), variance.float().cpu())


def correct_classifier(model, prototypes, end, args, device, rank):
    # Rank zero performs cheap feature-only correction, then broadcasts the head.
    model.eval()
    if rank == 0 and args.correction_epochs:
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.correction_lr)
        means = torch.stack([prototypes[c][0] for c in range(end)]).to(device)
        stds = torch.stack([prototypes[c][1].sqrt() for c in range(end)]).to(device)
        for epoch in range(args.correction_epochs):
            # Bounded batches; no C x D x D covariance tensors or full model checkpoint.
            labels = torch.arange(end, device=device).repeat_interleave(args.prototype_samples)
            labels = labels[torch.randperm(labels.numel(), device=device)]
            total, n = 0., 0
            for y in labels.split(128):
                features = means[y] + torch.randn(len(y), means.shape[1], device=device) * stds[y]
                logits = model.classifier(features)
                loss = F.cross_entropy(logits[:, :end], y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total, n = total + float(loss.detach()) * len(y), n + len(y)
            event(args.output, dict(type="correction", seen_classes=end, epoch=epoch + 1, loss=total / n))
    if dist.is_initialized():
        for p in model.classifier.parameters():
            dist.broadcast(p.data, 0)


@torch.no_grad()
def evaluate(model, loader, device, classes, end, width):
    model.eval()
    confusion = torch.zeros(classes, classes, dtype=torch.long, device=device)
    for inputs, labels in loader:
        logits, _, _, _ = model(transfer(inputs, device))
        pred = logits[:, :end].argmax(-1)
        labels = labels.to(device)
        confusion += torch.bincount(labels * classes + pred, minlength=classes * classes).reshape(classes, classes)
    reduce(confusion)
    total, correct = confusion.sum(-1), confusion.diag()
    row = []
    for start in range(0, end, width):
        if total[start:start + width].sum() == 0:
            raise ValueError("Empty test task")
        row.append(100 * float(correct[start:start + width].sum()) / float(total[start:start + width].sum()))
    return row, confusion.cpu(), total.cpu(), correct.cpu()


@contextlib.contextmanager
def stage(args, name, task, device, rank, world):
    sync()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - t, device=device, dtype=torch.float64)
        reduce(elapsed, dist.ReduceOp.MAX)
        memory = [None] * world
        local = dict(rank=rank, allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                     reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30)
        if dist.is_initialized():
            dist.all_gather_object(memory, local)
        else:
            memory = [local]
        if rank == 0:
            event(args.output, dict(type="stage", stage=name, task=task + 1,
                                   seconds=float(elapsed), memory=memory))


def main():
    run_started = time.perf_counter()
    args = arguments()
    rank, world, local_rank = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("Server training requires CUDA; CPU unit tests: python -m unittest discover -s tests_qwen")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required (A100 supported)")
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    output = Path(args.output)
    if rank == 0:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError(f"Output {output} is not empty; use --resume or a new output directory")
        output.mkdir(parents=True, exist_ok=True)
    sync()
    config = check_model(args.model_path)
    train, test, tasks = build_datasets(args.dataset, args.data_root, args.seed)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    
    if getattr(processor, "chat_template", None) is None:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "chat_template", None):
            processor.chat_template = tokenizer.chat_template
    processor.tokenizer.padding_side = "right"
    collate = Collator(processor, args.max_visual_tokens)
    classes, width = DATASETS[args.dataset][0], len(tasks[0])
    # Fail on corrupt images/processor before allocating base-model GPU memory.
    collate([train[0][:2]])
    identity = {k: v for k, v in vars(args).items() if k not in ("output", "resume", "workers", "log_every", "model_path", "data_root")}
    index_path = Path(args.model_path) / "model.safetensors.index.json"
    signature = dict(arguments=identity, world_size=world, tasks=tasks, config=config,
                     weight_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None)
    checkpoint_path = output / "latest_adapter.pt"
    checkpoint = None
    if args.resume:
        # Only load checkpoints created by this experiment, not untrusted pickle files.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature:
            raise ValueError("Resume requires identical model config, class order, world size and training arguments")
    if rank == 0:
        import transformers
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        write_json(output / ("resume_environment.json" if args.resume else "environment.json"),
                   dict(arguments=vars(args), signature=signature, git_commit=commit,
                        torch=torch.__version__, transformers=transformers.__version__,
                        python=platform.python_version(), cuda=torch.version.cuda,
                        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                        effective_batch=args.batch_size * args.accumulation * world,
                        protocol="class incremental; no image replay; diagonal prototype correction"))
    with stage(args, "model_loading", -1, device, rank, world):
        model = load_model(args.model_path, classes, device, method=args.method, experts=args.experts,
                           topk=args.topk, epsilon=args.epsilon, router_weight=args.router_weight, old_weight=args.old_weight)
    if rank == 0:
        event(args.output, dict(type="loading_profile", **model.loading_profile))
    if checkpoint:
        model.load_lightweight_state(checkpoint["adapter"])
    network = DDP(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
    if rank == 0:
        event(args.output, dict(type="model", full_attention_layers=model.layer_ids,
                               total_parameters=sum(p.numel() for p in model.parameters()),
                               trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # A real multimodal forward/backward without an optimizer update precedes every run.
    with stage(args, "preflight_forward_backward", -1, device, rank, world):
        model.train()
        inputs, labels = collate([train[i][:2] for i in range(args.batch_size)])
        logits, r, o, _ = network(transfer(inputs, device))
        loss = F.cross_entropy(logits, labels.to(device)) + r + o
        loss.backward()
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if any(p.grad is None or not bool(p.grad.isfinite().all()) for _, p in trainable):
            raise RuntimeError("Preflight failed: missing/nonfinite trainable gradients")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError("Preflight failed: base-model gradients unexpectedly enabled")
        model.zero_grad(set_to_none=True)
    if args.mode == "preflight":
        if rank == 0:
            event(args.output, dict(type="preflight_passed"))
        return
    matrix = checkpoint["matrix"] if checkpoint else []
    prototypes = checkpoint["prototypes"] if checkpoint else {}
    progress = checkpoint["progress"] if checkpoint else dict(task=0, phase="dense", epoch=0)
    if checkpoint:
        restore_rng(checkpoint["rng"][rank])
    limit = 2 if args.mode == "smoke" else len(tasks)
    per_class = 2 if args.mode == "smoke" else 0
    for task in range(progress["task"], limit):
        start, end = task * width, (task + 1) * width
        current = View(train, task, per_class=per_class)
        loader, sampler = make_loader(current, collate, args, rank, world, train=True)
        if len(loader) == 0:
            raise ValueError("Empty training loader")
        # Equal classifier optimization budget; head_only has no prompt experts.
        phases = [("dense", args.dense_epochs if task == 0 else 0), ("train", args.epochs)]
        for phase, epochs in phases:
            if task == progress["task"] and ["dense", "train", "post"].index(phase) < ["dense", "train", "post"].index(progress["phase"]):
                continue
            if not epochs:
                continue
            optimizer = optimizer_for(model, args)
            total_updates = epochs * math.ceil(len(loader) / args.accumulation)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_updates, eta_min=1e-6)
            begin = 0
            if checkpoint and task == progress["task"] and phase == progress["phase"]:
                begin = progress["epoch"]
                if checkpoint["optimizer"]:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    scheduler.load_state_dict(checkpoint["scheduler"])
            for epoch in range(begin, epochs):
                sampler.set_epoch(task * 100000 + (0 if phase == "dense" else 10000) + epoch)
                sync()
                t = time.perf_counter()
                with stage(args, phase, task, device, rank, world):
                    result = train_epoch(network, model, loader, optimizer, scheduler, args, device,
                                         start, end, phase == "dense", task, epoch, phase, rank, world)
                if rank == 0:
                    event(args.output, dict(type="epoch", task=task + 1, phase=phase, epoch=epoch + 1,
                          samples_per_second=result["samples_including_distributed_padding"] / (time.perf_counter() - t), **result))
                save_checkpoint(checkpoint_path, model, dict(task=task, phase=phase, epoch=epoch + 1),
                                optimizer, scheduler, matrix, prototypes, signature, rank, world)
        # Postprocessing is replayed from the last epoch checkpoint after interruption.
        exact, _ = make_loader(current, collate, args, rank, world)
        with stage(args, "expert_frequency_and_prototype_scan", task, device, rank, world):
            scan(model, exact, device, classes, start, end, prototypes)
        with stage(args, "classifier_correction", task, device, rank, world):
            correct_classifier(model, prototypes, end, args, device, rank)
        test_loader, _ = make_loader(View(test, task, cumulative=True, per_class=per_class), collate, args, rank, world)
        with stage(args, "evaluation", task, device, rank, world):
            row, confusion, totals, correct = evaluate(model, test_loader, device, classes, end, width)
        matrix.append(row)
        if rank == 0:
            write_json(output / f"per_class_task_{task + 1}.json",
                       [dict(mapped_class=c, original_class=tasks[c // width][c % width],
                             correct=int(correct[c]), total=int(totals[c]),
                             accuracy=100 * int(correct[c]) / max(int(totals[c]), 1)) for c in range(end)])
            np.save(output / f"confusion_task_{task + 1}.npy", confusion.numpy())
            usage = []
            for layer, expert in zip(model.layer_ids, model.expert_modules()):
                freq = expert.frequency.cpu().numpy()
                probs = freq / np.maximum(freq.sum(-1, keepdims=True), 1)
                usage.append(dict(layer=layer, frequency=freq.tolist(), coverage=(freq > 0).mean(-1).tolist(),
                                  entropy=(-(probs * np.log(probs.clip(1e-30))).sum(-1)).tolist(), used=expert.used.cpu().tolist()))
            write_json(output / f"experts_task_{task + 1}.json", usage)
            result = summarize(output, matrix)
            event(args.output, dict(type="task_complete", task=task + 1, **result))
        for expert in model.expert_modules():
            expert.next_task()
        save_checkpoint(checkpoint_path, model, dict(task=task + 1, phase="dense", epoch=0),
                        None, None, matrix, prototypes, signature, rank, world)
        checkpoint = None
        progress = dict(task=task + 1, phase="dense", epoch=0)
    if rank == 0:
        event(args.output, dict(type="run_complete", tasks=limit, wall_seconds=time.perf_counter() - run_started))


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM: reduce --max-visual-tokens or --batch-size, then start a NEW run. "
              "DDP replicates weights on each GPU; two 40GB cards are not one 80GB card.", flush=True)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
