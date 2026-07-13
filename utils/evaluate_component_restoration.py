"""Evaluate historical component restoration in the final SMoPE model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer import Trainer
from utils.audit_metrics import SCHEMA_VERSION, safe_recovery_ratio
from utils.audit_state import (
    build_restoration_updates,
    restore_prompt_auxiliary_state,
    selected_experts_from_snapshot,
    state_checksum,
    temporary_state_updates,
)


DEFAULT_COMPONENTS = [
    "final",
    "router_identity",
    "router_identity_logits",
    "key",
    "value",
    "classifier",
    "full_historical",
]
DEFAULT_COMBINATIONS = [
    "key+value",
    "router_identity+value",
    "router_identity+classifier",
    "key+value+classifier",
]


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_checkpoint_index(run_dir, repeat_id):
    index = {}
    root = Path(run_dir) / "models" / f"repeat-{repeat_id}"
    for path in root.glob("task-*/checkpoint.pt"):
        checkpoint = torch.load(path, map_location="cpu")
        index[int(checkpoint["task"])] = (path, checkpoint)
    if not index:
        raise FileNotFoundError(
            f"No audit checkpoint.pt files under {root}; rerun with --audit_save_full_checkpoints."
        )
    return index


def load_router_reference(run_dir, repeat_id, historical_task):
    path = (
        Path(run_dir)
        / "forgetting_audit"
        / f"repeat-{repeat_id}"
        / "router_reference"
        / f"checkpoint-{historical_task}_eval-task-{historical_task}.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing historical router snapshot: {path}")
    return torch.load(path, map_location="cpu")


def restore_prompt_auxiliary_backup(prompt):
    frequencies = {}
    for layer in prompt.e_layers:
        for expert in range(prompt.num_experts):
            for head in range(prompt.num_heads):
                name = f"e_freq_{layer}_{expert}_{head}"
                frequencies[name] = int(getattr(prompt, name))
    return {
        "task_count": int(prompt.task_count),
        "num_samples": int(prompt.num_samples),
        "used_frequently": copy.deepcopy(prompt.used_frequently),
        "frequencies": frequencies,
    }


def evaluate_task(trainer, eval_task, valid_out_dim, snapshot, replay_mode=None):
    model = trainer.learner.model
    was_training = model.training
    model.eval()
    correct = total = seen = 0
    with torch.no_grad():
        for batch_index, (inputs, targets, _) in enumerate(
            trainer._audit_task_loader(eval_task - 1)
        ):
            if batch_index >= len(snapshot["batches"]):
                break
            reference = snapshot["batches"][batch_index]
            expected_size = int(reference["targets"].numel())
            inputs = inputs[:expected_size]
            targets = targets[:expected_size]
            if inputs.size(0) != expected_size:
                raise ValueError("Restoration loader has fewer samples than the snapshot.")
            sample_ids = trainer._stable_sample_ids(inputs, targets, seen)
            if reference.get("sample_ids") and sample_ids != reference["sample_ids"]:
                raise ValueError("Restoration samples do not match the router snapshot.")
            forced_indices = None
            forced_logits = None
            if replay_mode in {"identity", "identity_prompt_logits"}:
                forced_indices = {
                    int(layer): state["indices"]
                    for layer, state in reference["layers"].items()
                }
            if replay_mode == "identity_prompt_logits":
                forced_logits = {}
                for layer, state in reference["layers"].items():
                    if "selected_raw_logits" not in state:
                        raise ValueError(
                            "Router snapshot lacks selected_raw_logits for logits replay."
                        )
                    forced_logits[int(layer)] = state["selected_raw_logits"]
            if trainer.learner.gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()
            logits = model(
                inputs,
                forced_prompt_indices=forced_indices,
                forced_prompt_logits=forced_logits,
            )[:, :valid_out_dim]
            correct += int((logits.argmax(dim=1) == targets).sum())
            total += int(targets.numel())
            seen += expected_size
    model.train(was_training)
    if total == 0:
        raise ValueError(f"No restoration samples for eval task {eval_task}.")
    return 100.0 * correct / total, total


def forgetting_gap(run_dir, eval_task, final_checkpoint, repeat_index):
    history = load_yaml(Path(run_dir) / "results-acc" / "pt.yaml")["history"]
    at_learning = float(history[eval_task - 1][eval_task - 1][repeat_index])
    final = float(history[eval_task - 1][final_checkpoint - 1][repeat_index])
    return at_learning - final


def parse_condition(condition):
    tokens = set(condition.split("+"))
    replay_mode = None
    if "router_identity_logits" in tokens:
        replay_mode = "identity_prompt_logits"
        tokens.remove("router_identity_logits")
    elif "router_identity" in tokens:
        replay_mode = "identity"
        tokens.remove("router_identity")
    components = {token for token in tokens if token not in {"final"}}
    return replay_mode, components


def make_trainer(run_dir, run_args, seed, repeat_index, gpuid):
    values = dict(run_args)
    defaults = {
        "audit_router_replay_modes": ["identity", "identity_prompt_logits"],
        "audit_freeze_until_task": 0,
        "audit_expert_usage": False,
        "audit_expert_usage_coverage": 0.90,
        "audit_gradient_direction": False,
        "audit_gradient_tasks": "all_old",
        "audit_gradient_max_samples": 256,
        "audit_gradient_components": ["key", "value", "classifier"],
        "audit_router_margin_threshold": 0.01,
        "audit_save_full_checkpoints": False,
        "audit_sample_manifest": "",
        "audit_save_logits": False,
    }
    for key, value in defaults.items():
        values.setdefault(key, value)
    values.update(
        {
            "log_dir": str(run_dir),
            "gpuid": [gpuid],
            "overwrite": 0,
            "audit_router": False,
            "audit_expert_usage": False,
            "audit_gradient_direction": False,
            "audit_save_full_checkpoints": False,
            "audit_freeze_component": "none",
        }
    )
    return Trainer(
        SimpleNamespace(**values),
        seed,
        ["acc", "time", "fr"],
        ["global", "pt"],
        repeat_index,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--final_checkpoint", type=int)
    parser.add_argument("--historical_checkpoints", nargs="+", type=int)
    parser.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS)
    parser.add_argument("--combinations", nargs="+", default=DEFAULT_COMBINATIONS)
    parser.add_argument(
        "--restore_scope",
        nargs="+",
        choices=["full_pool", "used_experts", "high_frequency_experts"],
        default=["full_pool", "used_experts", "high_frequency_experts"],
    )
    parser.add_argument("--usage_coverage", type=float, default=0.90)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_args = load_yaml(args.run_dir / "args.yaml")
    seeds = list(run_args.get("seeds") or range(int(run_args.get("repeat", 1))))
    output = args.output or args.run_dir / "forgetting_audit" / "component_restoration.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    all_conditions = list(dict.fromkeys(args.components + args.combinations))

    for repeat_index, seed in enumerate(seeds):
        repeat_id = repeat_index + 1
        checkpoints = load_checkpoint_index(args.run_dir, repeat_id)
        final_checkpoint = args.final_checkpoint or max(checkpoints)
        if final_checkpoint not in checkpoints:
            raise ValueError(f"Final checkpoint {final_checkpoint} is unavailable.")
        historical_tasks = args.historical_checkpoints or [
            task for task in sorted(checkpoints) if task < final_checkpoint
        ]
        trainer = make_trainer(args.run_dir, run_args, int(seed), repeat_index, args.gpuid)
        core = trainer._audit_core_model()
        final_payload = checkpoints[final_checkpoint][1]
        core.load_state_dict(final_payload["model_state_dict"], strict=True)
        restore_prompt_auxiliary_state(
            core.prompt, final_payload.get("prompt_auxiliary_state", {})
        )
        trainer.learner.valid_out_dim = int(final_payload["valid_out_dim"])
        trainer.learner.last_valid_out_dim = int(final_payload["last_valid_out_dim"])
        core.task_id = final_checkpoint - 1
        final_checksum = state_checksum(core.state_dict())

        for historical_task in historical_tasks:
            if historical_task not in checkpoints:
                raise ValueError(f"Historical checkpoint {historical_task} is unavailable.")
            historical_payload = checkpoints[historical_task][1]
            snapshot = load_router_reference(
                args.run_dir, repeat_id, historical_task
            )
            baseline_accuracy, samples = evaluate_task(
                trainer,
                historical_task,
                int(final_payload["valid_out_dim"]),
                snapshot,
                replay_mode=None,
            )
            gap = forgetting_gap(
                args.run_dir, historical_task, final_checkpoint, repeat_index
            )
            gains_by_scope = {}
            for condition in all_conditions:
                replay_mode, components = parse_condition(condition)
                scopes = (
                    args.restore_scope
                    if components.intersection({"key", "value"})
                    else ["full_pool"]
                )
                for scope in scopes:
                    selected = selected_experts_from_snapshot(
                        snapshot, scope, args.usage_coverage
                    )
                    updates = build_restoration_updates(
                        final_payload["model_state_dict"],
                        historical_payload["model_state_dict"],
                        components,
                        classifier_rows=int(historical_payload["valid_out_dim"]),
                        selected_experts=selected,
                    )
                    prompt_backup = restore_prompt_auxiliary_backup(core.prompt)
                    with temporary_state_updates(core, updates):
                        valid_out_dim = int(final_payload["valid_out_dim"])
                        if "full_historical" in components:
                            restore_prompt_auxiliary_state(
                                core.prompt,
                                historical_payload.get("prompt_auxiliary_state", {}),
                            )
                            valid_out_dim = int(historical_payload["valid_out_dim"])
                            core.task_id = historical_task - 1
                        restored_accuracy, restored_samples = evaluate_task(
                            trainer,
                            historical_task,
                            valid_out_dim,
                            snapshot,
                            replay_mode=replay_mode,
                        )
                    restore_prompt_auxiliary_state(core.prompt, prompt_backup)
                    core.task_id = final_checkpoint - 1
                    current_state = core.state_dict()
                    if any(
                        not torch.equal(
                            current_state[name].detach().cpu(),
                            final_payload["model_state_dict"][name],
                        )
                        for name in updates
                    ):
                        raise RuntimeError(
                            f"Model state did not roll back after {condition}/{scope}."
                        )
                    gain = restored_accuracy - baseline_accuracy
                    key = (scope, condition)
                    gains_by_scope[key] = gain
                    record = {
                        "event": "component_restoration",
                        "schema_version": SCHEMA_VERSION,
                        "repeat_id": repeat_id,
                        "seed": int(seed),
                        "final_checkpoint": final_checkpoint,
                        "historical_checkpoint": historical_task,
                        "eval_task": historical_task,
                        "restore_condition": condition,
                        "restore_scope": scope,
                        "usage_coverage": args.usage_coverage,
                        "samples": restored_samples,
                        "baseline_final_accuracy": baseline_accuracy,
                        "restored_accuracy": restored_accuracy,
                        "restoration_gain": gain,
                        "forgetting_gap": gap,
                        "restoration_ratio": safe_recovery_ratio(gain, gap),
                        "interaction_term": None,
                    }
                    interaction_pairs = {
                        "key+value": ("key", "value"),
                        "router_identity+value": ("router_identity", "value"),
                        "router_identity+classifier": ("router_identity", "classifier"),
                    }
                    if condition in interaction_pairs:
                        left, right = interaction_pairs[condition]
                        left_gain = gains_by_scope.get(
                            (scope, left), gains_by_scope.get(("full_pool", left))
                        )
                        right_gain = gains_by_scope.get(
                            (scope, right), gains_by_scope.get(("full_pool", right))
                        )
                        if left_gain is not None and right_gain is not None:
                            record["interaction_term"] = (
                                gain
                                - left_gain
                                - right_gain
                            )
                    with output.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, allow_nan=True) + "\n")
        if state_checksum(core.state_dict()) != final_checksum:
            raise RuntimeError("Final model checksum changed during restoration audit.")
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Wrote component restoration results to {output}")


if __name__ == "__main__":
    main()
