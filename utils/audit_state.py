"""State helpers for safe, temporary component-restoration experiments."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from contextlib import contextmanager

import torch


EXPERT_PATTERN = re.compile(r"^prompt\.e_p(?P<kind>[kv])_(?P<layer>\d+)_(?P<expert>\d+)_(?P<head>\d+)$")


def state_checksum(state_dict):
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def restore_prompt_auxiliary_state(prompt, state):
    prompt.task_count = int(state.get("task_count", prompt.task_count))
    prompt.num_samples = int(state.get("num_samples", prompt.num_samples))
    if "used_frequently" in state:
        prompt.used_frequently = state["used_frequently"]
    for name, value in state.get("frequencies", {}).items():
        setattr(prompt, name, int(value))


def expert_coordinate(parameter_name):
    match = EXPERT_PATTERN.match(parameter_name)
    if match is None:
        return None
    return (
        int(match.group("layer")),
        int(match.group("head")),
        int(match.group("expert")),
    )


def selected_experts_from_snapshot(snapshot, scope, usage_coverage=0.90):
    if scope == "full_pool":
        return None
    counts = defaultdict(int)
    for batch in snapshot["batches"]:
        for layer, layer_state in batch["layers"].items():
            indices = layer_state["indices"]
            for head in range(indices.size(1)):
                values, occurrences = torch.unique(
                    indices[:, head].reshape(-1), return_counts=True
                )
                for expert, count in zip(values.tolist(), occurrences.tolist()):
                    counts[(int(layer), head, int(expert))] += int(count)
    if scope == "used_experts":
        return {coordinate for coordinate, count in counts.items() if count > 0}
    if scope != "high_frequency_experts":
        raise ValueError(f"Unknown restore scope: {scope}")
    by_layer_head = defaultdict(list)
    for (layer, head, expert), count in counts.items():
        by_layer_head[(layer, head)].append((count, expert))
    selected = set()
    for (layer, head), ranked in by_layer_head.items():
        ranked.sort(reverse=True)
        total = sum(count for count, _ in ranked)
        cumulative = 0
        for count, expert in ranked:
            selected.add((layer, head, expert))
            cumulative += count
            if total and cumulative / total >= float(usage_coverage):
                break
    return selected


def build_restoration_updates(
    final_state,
    historical_state,
    components,
    *,
    classifier_rows,
    selected_experts=None,
):
    """Return full-tensor updates; row/expert scoping is applied before return."""
    components = set(components)
    updates = {}
    if "full_historical" in components:
        return {name: value.detach().clone() for name, value in historical_state.items()}
    for name, historical in historical_state.items():
        coordinate = expert_coordinate(name)
        if coordinate is not None:
            kind = "key" if ".e_pk_" in name else "value"
            if kind not in components:
                continue
            if selected_experts is not None and coordinate not in selected_experts:
                continue
            updates[name] = historical.detach().clone()
        elif name in {"last.weight", "last.bias"} and "classifier" in components:
            value = final_state[name].detach().clone()
            value[:classifier_rows].copy_(historical[:classifier_rows])
            updates[name] = value
    return updates


@contextmanager
def temporary_state_updates(module, updates):
    state = module.state_dict()
    unknown = sorted(set(updates) - set(state))
    if unknown:
        raise KeyError(f"Unknown state keys: {unknown}")
    backup = {name: state[name].detach().clone() for name in updates}
    with torch.no_grad():
        for name, value in updates.items():
            if tuple(value.shape) != tuple(state[name].shape):
                raise ValueError(
                    f"State shape mismatch for {name}: {tuple(value.shape)} != {tuple(state[name].shape)}"
                )
            state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            current = module.state_dict()
            for name, value in backup.items():
                current[name].copy_(value)
