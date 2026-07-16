import os
import sys
import datetime
import json
import hashlib
from tqdm import tqdm, trange
import math
from timm.utils import accuracy
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
from typing import Iterable
import learners
from utils import utils_tap
from utils.schedulers import CosineSchedulerIter
from utils.audit_metrics import (
    SCHEMA_VERSION,
    route_set_metrics,
    vector_direction_metrics,
)


class Trainer:
    def __init__(self, args, seed, metric_keys, save_keys, round_id):

        # process inputs
        self.seed = seed
        self.round_id = round_id
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers

        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == "CIFAR100":
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32, 32, 3]
        elif args.dataset == "ImageNet_R":
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224, 224, 3]
            self.top_k = 1
        elif args.dataset == "CUB200":
            Dataset = dataloaders.iCUB200
            num_classes = 200
            self.dataset_size = [224, 224, 3]
            self.top_k = 1
        else:
            raise ValueError("Dataset not implemented!")

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if args.rand_split:
            print("=============================================")
            print("Shuffling....")
            print("pre-shuffle:" + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            print("post-shuffle:" + str(class_order))
            print("=============================================")
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (
            args.max_task == -1 or len(self.tasks) < args.max_task
        ):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p : p + inc])
            self.tasks_logits.append(class_order_logits[p : p + inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i + 1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1  # number of transforms per image
        if args.model_name.startswith("vit"):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(
            dataset=args.dataset,
            phase="train",
            aug=args.train_aug,
            resize_imnet=resize_imnet,
        )
        test_transform = dataloaders.utils.get_transform(
            dataset=args.dataset,
            phase="test",
            aug=args.train_aug,
            resize_imnet=resize_imnet,
        )
        self.train_dataset = Dataset(
            args.dataroot,
            train=True,
            lab=True,
            tasks=self.tasks,
            download_flag=True,
            transform=train_transform,
            seed=self.seed,
            rand_split=args.rand_split,
            validation=args.validation,
        )
        self.test_dataset = Dataset(
            args.dataroot,
            train=False,
            tasks=self.tasks,
            download_flag=False,
            transform=test_transform,
            seed=self.seed,
            rand_split=args.rand_split,
            validation=args.validation,
        )
        self.audit_gradient_dataset = None
        if bool(args.audit_gradient_direction):
            # Gradient diagnostics must not use test-set gradients.  This is a
            # deterministic, non-augmented view of the training archive and is
            # used only for measurements, never for optimizer decisions.
            self.audit_gradient_dataset = Dataset(
                args.dataroot,
                train=True,
                lab=True,
                tasks=self.tasks,
                download_flag=False,
                transform=test_transform,
                seed=self.seed,
                rand_split=args.rand_split,
                validation=False,
            )

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {
            "num_classes": num_classes,
            "lr": args.lr,
            "debug_mode": args.debug_mode == 1,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "schedule": args.schedule,
            "schedule_type": args.schedule_type,
            "iter_step": args.iter_step,  # add this for coswm
            "model_type": args.model_type,
            "model_name": args.model_name,
            "optimizer": args.optimizer,
            "gpuid": args.gpuid,
            "memory": args.memory,
            "temp": args.temp,
            "out_dim": num_classes,
            "overwrite": args.overwrite == 1,
            "DW": args.DW,
            "batch_size": args.batch_size,
            "upper_bound_flag": args.upper_bound_flag,
            "tasks": self.tasks_logits,
            "top_k": self.top_k,
            "prompt_param": [self.num_tasks, args.prompt_param],
            "pretrained_weight": args.pretrained_weight,
            "audit_freeze_component": args.audit_freeze_component,
            "audit_freeze_from_task": args.audit_freeze_from_task,
            "audit_freeze_until_task": args.audit_freeze_until_task,
            "audit_main_epochs": args.audit_main_epochs,
        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](
            self.learner_config
        )

        self.learner.print_model()

        # storing class mean and covariance
        # self.learner.cls_mean = dict()
        # self.learner.cls_cov = dict()
        self.num_classes = num_classes
        self.adaptive_pred = args.adaptive_pred
        self.n_centroids = args.n_centroids
        self.ca_method = args.ca_method
        self.crct_epochs = args.crct_epochs
        self.ca_lr = args.ca_lr
        self.ca_weight_decay = args.ca_weight_decay
        self.ca_batch_size_ratio = args.ca_batch_size_ratio
        self.audit_router = bool(args.audit_router)
        self.audit_save_logits = bool(args.audit_save_logits)
        self.audit_router_replay_modes = set(args.audit_router_replay_modes)
        self.audit_router_max_samples = max(0, int(args.audit_router_max_samples))
        self.audit_expert_usage = bool(args.audit_expert_usage)
        self.audit_expert_usage_coverage = float(args.audit_expert_usage_coverage)
        self.audit_gradient_direction = bool(args.audit_gradient_direction)
        self.audit_gradient_tasks = args.audit_gradient_tasks
        self.audit_gradient_max_samples = max(1, int(args.audit_gradient_max_samples))
        self.audit_gradient_components = tuple(args.audit_gradient_components)
        self.audit_router_margin_threshold = float(args.audit_router_margin_threshold)
        self.audit_save_full_checkpoints = bool(args.audit_save_full_checkpoints)
        self.audit_sample_manifest_source = str(args.audit_sample_manifest or "")
        self.audit_checkpoints = {
            checkpoint
            for checkpoint in args.audit_checkpoints
            if 1 <= checkpoint <= self.max_task
        }
        self.audit_checkpoints.add(self.max_task)
        self.router_references = {}
        self.component_references = {}
        self.audit_manifest_entries = {}
        self.audit_dir = os.path.join(
            self.log_dir, "forgetting_audit", f"repeat-{self.round_id + 1}"
        )
        self.audit_enabled = bool(
            self.audit_router
            or self.audit_expert_usage
            or self.audit_gradient_direction
            or self.audit_save_full_checkpoints
            or args.audit_freeze_component != "none"
        )
        if self.audit_enabled:
            os.makedirs(self.audit_dir, exist_ok=True)
            if self.round_id == 0 and args.overwrite:
                for filename in (
                    "router_audit.jsonl",
                    "component_drift.jsonl",
                    "gradient_direction.jsonl",
                ):
                    path = os.path.join(self.log_dir, "forgetting_audit", filename)
                    if os.path.exists(path):
                        os.remove(path)
                manifest_path = os.path.join(
                    self.log_dir, "forgetting_audit", "audit_sample_manifest.json"
                )
                source_path = (
                    os.path.abspath(self.audit_sample_manifest_source)
                    if self.audit_sample_manifest_source
                    else ""
                )
                if (
                    os.path.exists(manifest_path)
                    and os.path.abspath(manifest_path) != source_path
                ):
                    os.remove(manifest_path)
            self._write_expert_structure()

    def task_eval(self, t_index, local=False, task="acc"):

        val_name = self.task_names[t_index]
        print(f"validation split name (local {local}):", val_name)

        # eval
        self.test_dataset.load_dataset(
            t_index, train=True
        )  # train=True, only load task i data; else, load task 0~i data
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.workers,
            pin_memory=True,
        )
        if local:
            return self.learner.validation(
                test_loader, task_in=self.tasks_logits[t_index], task_metric=task
            )
        else:
            return self.learner.validation(test_loader, task_metric=task)

    def _audit_core_model(self):
        model = self.learner.model
        return model.module if hasattr(model, "module") else model

    def _audit_task_loader(self, task_index):
        self.test_dataset.load_dataset(task_index, train=True)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def _audit_limit_batch(self, inputs, targets, seen):
        if self.audit_router_max_samples <= 0:
            return inputs, targets
        remaining = self.audit_router_max_samples - seen
        if remaining <= 0:
            return None, None
        return inputs[:remaining], targets[:remaining]

    def _stable_sample_ids(self, inputs, targets, sample_start):
        targets = targets.detach().cpu()
        raw_data = getattr(self.test_dataset, "data", None)
        sample_ids = []
        for offset in range(targets.size(0)):
            dataset_index = sample_start + offset
            digest = hashlib.sha256()
            raw_sample = raw_data[dataset_index] if raw_data is not None else None
            if isinstance(raw_sample, (str, os.PathLike)):
                digest.update(os.fspath(raw_sample).replace("\\", "/").encode("utf-8"))
            elif raw_sample is not None:
                digest.update(np.asarray(raw_sample).tobytes())
            else:
                digest.update(inputs[offset].detach().cpu().contiguous().numpy().tobytes())
            digest.update(str(int(targets[offset])).encode("ascii"))
            sample_ids.append(
                f"{dataset_index}:{int(targets[offset])}:{digest.hexdigest()[:20]}"
            )
        return sample_ids

    def _write_expert_structure(self):
        core = self._audit_core_model()
        prompt = core.prompt
        structure = {
            "schema_version": SCHEMA_VERSION,
            "expert_sharing": "global_shared",
            "task_private_experts": False,
            "num_prompt_layers": len(prompt.e_layers),
            "prompt_layers": list(prompt.e_layers),
            "num_heads": int(prompt.num_heads),
            "num_experts_per_layer_head": int(prompt.num_experts),
            "topk": int(prompt.topk),
            "active_fraction": float(prompt.topk / prompt.num_experts),
            "key_dim_per_head": int(prompt.head_dim),
            "key_parameter_count": int(
                sum(
                    parameter.numel()
                    for name, parameter in prompt.named_parameters()
                    if "e_pk_" in name
                )
            ),
            "value_parameter_count": int(
                sum(
                    parameter.numel()
                    for name, parameter in prompt.named_parameters()
                    if "e_pv_" in name
                )
            ),
        }
        path = os.path.join(self.log_dir, "forgetting_audit", "expert_structure.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != structure:
                raise ValueError("Expert structure changed within the same audit run.")
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(structure, handle, ensure_ascii=False, indent=2)

    def _router_batch_record(self, prompt_scores, inputs, targets, sample_start):
        topk = self._audit_core_model().prompt.topk
        layers = {}
        for layer_id, layer_scores in enumerate(prompt_scores):
            if layer_scores is None or layer_scores[1] is None:
                continue
            raw_logits, selection_logits = layer_scores
            values, indices = torch.topk(selection_logits, topk, dim=-1)
            selected_raw_logits = torch.gather(raw_logits, -1, indices)
            layer = {
                "indices": indices.detach().cpu(),
                "selected_raw_logits": selected_raw_logits.detach().cpu(),
                "selected_selection_logits": values.detach().cpu(),
                # Compatibility alias for snapshots produced by the first audit.
                "scores": values.detach().cpu(),
            }
            if self.audit_save_logits:
                layer["raw_logits"] = raw_logits.detach().cpu()
                layer["selection_logits"] = selection_logits.detach().cpu()
            layers[layer_id] = layer
        return {
            "sample_start": int(sample_start),
            "sample_indices": torch.arange(
                sample_start, sample_start + targets.size(0), dtype=torch.long
            ),
            "dataset_indices": torch.arange(
                sample_start, sample_start + targets.size(0), dtype=torch.long
            ),
            "sample_ids": self._stable_sample_ids(inputs, targets, sample_start),
            "targets": targets.detach().cpu(),
            "class_ids": targets.detach().cpu(),
            "layers": layers,
        }

    def _save_router_snapshot(self, kind, checkpoint_task, eval_task, batches):
        directory = os.path.join(self.audit_dir, kind)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(
            directory,
            f"checkpoint-{checkpoint_task}_eval-task-{eval_task}.pt",
        )
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "repeat_id": self.round_id + 1,
                "seed": self.seed,
                "checkpoint_task": checkpoint_task,
                "eval_task": eval_task,
                "topk": self._audit_core_model().prompt.topk,
                "save_full_router_logits": self.audit_save_logits,
                "batches": batches,
            },
            path,
        )

    def _record_audit_manifest(self, task_index, batches, source):
        entry = {
            "repeat_id": self.round_id + 1,
            "seed": self.seed,
            "eval_task": task_index + 1,
            "source": source,
            "samples": sum(len(batch["sample_ids"]) for batch in batches),
            "dataset_indices": [
                int(value)
                for batch in batches
                for value in batch["dataset_indices"].tolist()
            ],
            "sample_ids": [
                value for batch in batches for value in batch["sample_ids"]
            ],
            "class_ids": [
                int(value)
                for batch in batches
                for value in batch["class_ids"].tolist()
            ],
        }
        if self.audit_sample_manifest_source:
            source_path = self.audit_sample_manifest_source
            if not os.path.exists(source_path):
                raise FileNotFoundError(
                    f"Audit sample manifest does not exist: {source_path}"
                )
            with open(source_path, "r", encoding="utf-8") as handle:
                reference_manifest = json.load(handle)
            matches = [
                candidate
                for candidate in reference_manifest.get("entries", [])
                if int(candidate["seed"]) == int(self.seed)
                and int(candidate["eval_task"]) == task_index + 1
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Reference manifest must contain exactly one matching "
                    f"seed={self.seed}, eval_task={task_index + 1}; found {len(matches)}."
                )
            reference = matches[0]
            for field in ("dataset_indices", "sample_ids", "class_ids"):
                if list(reference[field]) != list(entry[field]):
                    raise ValueError(
                        "Audit sample manifest mismatch for "
                        f"seed={self.seed}, eval_task={task_index + 1}, field={field}."
                    )

        path = os.path.join(
            self.log_dir, "forgetting_audit", "audit_sample_manifest.json"
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        else:
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "selection": "deterministic_first_n_in_task_order",
                "entries": [],
            }
        key = (entry["repeat_id"], entry["seed"], entry["eval_task"])
        manifest["entries"] = [
            candidate
            for candidate in manifest.get("entries", [])
            if (
                int(candidate["repeat_id"]),
                int(candidate["seed"]),
                int(candidate["eval_task"]),
            )
            != key
        ]
        manifest["entries"].append(entry)
        manifest["entries"].sort(
            key=lambda item: (
                int(item["repeat_id"]),
                int(item["seed"]),
                int(item["eval_task"]),
            )
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    def _capture_router_reference(self, task_index):
        model = self.learner.model
        was_training = model.training
        model.eval()
        batches = []
        seen = 0
        with torch.no_grad():
            for inputs, targets, _ in self._audit_task_loader(task_index):
                inputs, targets = self._audit_limit_batch(inputs, targets, seen)
                if inputs is None or inputs.size(0) == 0:
                    break
                if self.learner.gpu:
                    inputs = inputs.cuda()
                scores = model(inputs, train=False, return_attn=True)
                batch = self._router_batch_record(scores, inputs, targets, seen)
                batches.append(batch)
                seen += targets.size(0)
        model.train(was_training)
        self.router_references[task_index] = batches
        self._save_router_snapshot(
            "router_reference", task_index + 1, task_index + 1, batches
        )
        self._record_audit_manifest(task_index, batches, source="test")

    @staticmethod
    def _snapshot_component_state(core):
        prompt = core.prompt
        return {
            "key": {
                name: parameter.detach().cpu().clone()
                for name, parameter in prompt.named_parameters()
                if "e_pk_" in name
            },
            "value": {
                name: parameter.detach().cpu().clone()
                for name, parameter in prompt.named_parameters()
                if "e_pv_" in name
            },
            "classifier": {
                name: value.detach().cpu().clone()
                for name, value in core.last.state_dict().items()
            },
        }

    def _capture_component_reference(self, task_index):
        state = self._snapshot_component_state(self._audit_core_model())
        self.component_references[task_index] = state
        directory = os.path.join(self.audit_dir, "component_reference")
        os.makedirs(directory, exist_ok=True)
        torch.save(state, os.path.join(directory, f"task-{task_index + 1}.pt"))

    @staticmethod
    def _prompt_auxiliary_state(prompt):
        frequencies = {}
        for layer in prompt.e_layers:
            for expert in range(prompt.num_experts):
                for head in range(prompt.num_heads):
                    name = f"e_freq_{layer}_{expert}_{head}"
                    frequencies[name] = int(getattr(prompt, name))
        return {
            "task_count": int(prompt.task_count),
            "num_samples": int(prompt.num_samples),
            "used_frequently": prompt.used_frequently,
            "frequencies": frequencies,
        }

    def _save_full_audit_checkpoint(self, model_save_dir, task_index):
        core = self._audit_core_model()
        model_state = {
            name: value.detach().cpu().clone()
            for name, value in core.state_dict().items()
        }
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "model_state_dict": model_state,
            "prompt_auxiliary_state": self._prompt_auxiliary_state(core.prompt),
            "task": task_index + 1,
            "task_name": self.task_names[task_index],
            "valid_out_dim": int(self.learner.valid_out_dim),
            "last_valid_out_dim": int(self.learner.last_valid_out_dim),
            "repeat_id": self.round_id + 1,
            "seed": self.seed,
            "config": dict(self.learner_config),
        }
        torch.save(checkpoint, os.path.join(model_save_dir, "checkpoint.pt"))

    @staticmethod
    def _component_distance(current, reference):
        squared = 0.0
        reference_squared = 0.0
        for name, old_value in reference.items():
            new_value = current[name].to(dtype=torch.float64)
            old_value = old_value.to(dtype=torch.float64)
            squared += float(torch.sum((new_value - old_value) ** 2))
            reference_squared += float(torch.sum(old_value**2))
        l2 = math.sqrt(squared)
        relative = l2 / max(math.sqrt(reference_squared), 1e-12)
        return {"l2": l2, "relative_l2": relative}

    @staticmethod
    def _classifier_rows(state, row_count):
        return {name: value[:row_count] for name, value in state.items()}

    @staticmethod
    def _expert_coordinates(name):
        parts = name.split("_")
        if len(parts) != 5 or parts[0] != "e" or parts[1] not in {"pk", "pv"}:
            return None
        return int(parts[2]), int(parts[4]), int(parts[3])

    @staticmethod
    def _reference_usage_counts(reference_batches):
        counts = {}
        for batch in reference_batches:
            for layer_id, layer in batch["layers"].items():
                indices = layer["indices"].reshape(
                    layer["indices"].size(0), layer["indices"].size(1), -1
                )
                for head_id in range(indices.size(1)):
                    values, occurrences = torch.unique(
                        indices[:, head_id, :].reshape(-1), return_counts=True
                    )
                    for expert, count in zip(values.tolist(), occurrences.tolist()):
                        counts[(int(layer_id), head_id, int(expert))] = int(count)
        return counts

    def _component_scope_distances(self, current, reference, reference_batches):
        usage_counts = self._reference_usage_counts(reference_batches)
        all_names = list(reference)
        used_names = [
            name
            for name in all_names
            if usage_counts.get(self._expert_coordinates(name), 0) > 0
        ]
        selected_coordinates = set()
        prompt = self._audit_core_model().prompt
        for layer_id in prompt.e_layers:
            for head_id in range(prompt.num_heads):
                ranked = sorted(
                    (
                        (usage_counts.get((layer_id, head_id, expert), 0), expert)
                        for expert in range(prompt.num_experts)
                    ),
                    reverse=True,
                )
                total = sum(count for count, _ in ranked)
                cumulative = 0
                for count, expert in ranked:
                    if count <= 0:
                        break
                    selected_coordinates.add((layer_id, head_id, expert))
                    cumulative += count
                    if total and cumulative / total >= self.audit_expert_usage_coverage:
                        break
        high_frequency_names = [
            name
            for name in all_names
            if self._expert_coordinates(name) in selected_coordinates
        ]

        def distance_for(names):
            if not names:
                return {"l2": 0.0, "relative_l2": 0.0}
            return self._component_distance(
                {name: current[name] for name in names},
                {name: reference[name] for name in names},
            )

        total_usage = max(sum(usage_counts.values()), 1)
        weighted_squared = 0.0
        per_expert = []
        for name in all_names:
            coordinate = self._expert_coordinates(name)
            diff = (
                current[name].to(torch.float64) - reference[name].to(torch.float64)
            )
            l2 = float(torch.linalg.vector_norm(diff))
            count = usage_counts.get(coordinate, 0)
            weighted_squared += (count / total_usage) * (l2**2)
            layer, head, expert = coordinate
            per_expert.append(
                {
                    "parameter": name,
                    "layer": layer,
                    "head": head,
                    "expert": expert,
                    "usage_count": count,
                    "l2": l2,
                }
            )
        global_distance = distance_for(all_names)
        global_distance["usage_weighted_l2"] = math.sqrt(weighted_squared)
        global_distance["old_task_used_only_l2"] = distance_for(used_names)["l2"]
        return {
            "global_pool": global_distance,
            "old_task_used_experts": distance_for(used_names),
            "old_task_high_frequency_experts": distance_for(high_frequency_names),
            "per_expert": per_expert,
        }

    def _write_component_drift(self, checkpoint_index):
        current = self._snapshot_component_state(self._audit_core_model())
        path = os.path.join(self.log_dir, "forgetting_audit", "component_drift.jsonl")
        for task_index in range(checkpoint_index + 1):
            reference = self.component_references[task_index]
            classifier_rows = sum(
                len(self.tasks_logits[seen_task_index])
                for seen_task_index in range(task_index + 1)
            )
            current_classifier = self._classifier_rows(
                current["classifier"], classifier_rows
            )
            reference_classifier = self._classifier_rows(
                reference["classifier"], classifier_rows
            )
            key_scopes = self._component_scope_distances(
                current["key"],
                reference["key"],
                self.router_references.get(task_index, []),
            )
            value_scopes = self._component_scope_distances(
                current["value"],
                reference["value"],
                self.router_references.get(task_index, []),
            )
            classifier_distance = self._component_distance(
                current_classifier, reference_classifier
            )
            record = {
                "event": "component_drift",
                "schema_version": SCHEMA_VERSION,
                "repeat_id": self.round_id + 1,
                "seed": self.seed,
                "checkpoint_task": checkpoint_index + 1,
                "reference_task": task_index + 1,
                "classifier_rows": classifier_rows,
                "key": key_scopes["global_pool"],
                "value": value_scopes["global_pool"],
                "classifier": classifier_distance,
                "components": {
                    "key": key_scopes,
                    "value": value_scopes,
                    "classifier": {"global_pool": classifier_distance},
                },
            }
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _update_route_drift(current, reference, accumulator):
        metrics = route_set_metrics(current, reference)
        accumulator["changed"] += metrics["changed"]
        accumulator["total"] += metrics["decisions"]
        accumulator["jaccard_sum"] += metrics["jaccard_sum"]

    def _audit_router_task(self, checkpoint_index, eval_task_index):
        model = self.learner.model
        was_training = model.training
        model.eval()
        reference_batches = self.router_references[eval_task_index]
        current_batches = []
        per_layer = {}
        current_correct = identity_correct = identity_logits_correct = total = seen = 0
        run_identity_logits = "identity_prompt_logits" in self.audit_router_replay_modes
        run_identity = (
            "identity" in self.audit_router_replay_modes or run_identity_logits
        )

        with torch.no_grad():
            for batch_index, (inputs, targets, _) in enumerate(
                self._audit_task_loader(eval_task_index)
            ):
                inputs, targets = self._audit_limit_batch(inputs, targets, seen)
                if inputs is None or inputs.size(0) == 0:
                    break
                if batch_index >= len(reference_batches):
                    raise ValueError("Historical router reference has too few batches.")
                reference_batch = reference_batches[batch_index]
                if targets.size(0) != reference_batch["targets"].size(0):
                    raise ValueError("Historical router batch size changed during replay.")
                if not torch.equal(targets.cpu(), reference_batch["targets"]):
                    raise ValueError(
                        "Historical router sample order changed during replay."
                    )
                if self.learner.gpu:
                    inputs = inputs.cuda()
                    targets_device = targets.cuda()
                else:
                    targets_device = targets

                scores = model(inputs, train=False, return_attn=True)
                current_batch = self._router_batch_record(
                    scores, inputs, targets, seen
                )
                if current_batch["sample_ids"] != reference_batch.get(
                    "sample_ids", current_batch["sample_ids"]
                ):
                    raise ValueError(
                        "Historical router sample IDs changed during replay."
                    )
                current_batches.append(current_batch)
                forced = {}
                forced_logits = {}
                for layer_id, reference_layer in reference_batch["layers"].items():
                    forced[layer_id] = reference_layer["indices"]
                    if run_identity_logits:
                        if "selected_raw_logits" not in reference_layer:
                            raise ValueError(
                                "Historical snapshot lacks selected_raw_logits "
                                "required for identity_prompt_logits replay."
                            )
                        forced_logits[layer_id] = reference_layer[
                            "selected_raw_logits"
                        ]
                    layer_accumulator = per_layer.setdefault(
                        layer_id, {"changed": 0, "total": 0, "jaccard_sum": 0.0}
                    )
                    self._update_route_drift(
                        current_batch["layers"][layer_id]["indices"],
                        reference_layer["indices"],
                        layer_accumulator,
                    )

                current_logits = model(inputs)[:, : self.learner.valid_out_dim]
                current_correct += int(
                    (current_logits.argmax(dim=1) == targets_device).sum().item()
                )
                if run_identity:
                    identity_logits = model(
                        inputs, forced_prompt_indices=forced
                    )[:, : self.learner.valid_out_dim]
                    identity_correct += int(
                        (identity_logits.argmax(dim=1) == targets_device).sum().item()
                    )
                if run_identity_logits:
                    identity_prompt_logits = model(
                        inputs,
                        forced_prompt_indices=forced,
                        forced_prompt_logits=forced_logits,
                    )[:, : self.learner.valid_out_dim]
                    identity_logits_correct += int(
                        (
                            identity_prompt_logits.argmax(dim=1) == targets_device
                        ).sum().item()
                    )
                total += int(targets_device.numel())
                seen += targets.size(0)

        model.train(was_training)
        self._save_router_snapshot(
            "router_current",
            checkpoint_index + 1,
            eval_task_index + 1,
            current_batches,
        )
        changed = sum(item["changed"] for item in per_layer.values())
        decisions = sum(item["total"] for item in per_layer.values())
        jaccard_sum = sum(item["jaccard_sum"] for item in per_layer.values())
        current_accuracy = 100.0 * current_correct / max(total, 1)
        identity_accuracy = (
            100.0 * identity_correct / max(total, 1) if run_identity else None
        )
        identity_prompt_logits_accuracy = (
            100.0 * identity_logits_correct / max(total, 1)
            if run_identity_logits
            else None
        )
        identity_gain = (
            identity_accuracy - current_accuracy
            if identity_accuracy is not None
            else None
        )
        total_gain = (
            identity_prompt_logits_accuracy - current_accuracy
            if identity_prompt_logits_accuracy is not None
            else None
        )
        record = {
            "event": "historical_router_replay",
            "schema_version": SCHEMA_VERSION,
            "repeat_id": self.round_id + 1,
            "seed": self.seed,
            "checkpoint_task": checkpoint_index + 1,
            "eval_task": eval_task_index + 1,
            "samples": total,
            "current_accuracy": current_accuracy,
            "identity_replay_accuracy": identity_accuracy,
            "identity_prompt_logits_replay_accuracy": identity_prompt_logits_accuracy,
            "identity_replay_gain": identity_gain,
            "additional_prompt_logit_gain": (
                identity_prompt_logits_accuracy - identity_accuracy
                if identity_prompt_logits_accuracy is not None
                and identity_accuracy is not None
                else None
            ),
            "total_router_replay_gain": total_gain,
            "within_condition_router_change_rate": changed / max(decisions, 1),
            "within_condition_topk_jaccard": jaccard_sum / max(decisions, 1),
            # Compatibility aliases for the initial audit schema.
            "historical_router_accuracy": identity_accuracy,
            "replay_gain": identity_gain,
            "router_change_rate": changed / max(decisions, 1),
            "mean_topk_jaccard": jaccard_sum / max(decisions, 1),
            "per_layer": {
                str(layer_id): {
                    "within_condition_router_change_rate": item["changed"]
                    / max(item["total"], 1),
                    "within_condition_topk_jaccard": item["jaccard_sum"]
                    / max(item["total"], 1),
                }
                for layer_id, item in per_layer.items()
            },
        }
        path = os.path.join(self.log_dir, "forgetting_audit", "router_audit.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            "[forgetting-audit] "
            f"T{checkpoint_index + 1} eval T{eval_task_index + 1}: "
            f"route_change={record['within_condition_router_change_rate']:.4f}, "
            f"identity_gain={identity_gain if identity_gain is not None else float('nan'):.3f}, "
            f"identity+logits_gain={total_gain if total_gain is not None else float('nan'):.3f}"
        )

    def _run_forgetting_audit(self, checkpoint_index):
        self._write_component_drift(checkpoint_index)
        for eval_task_index in range(checkpoint_index + 1):
            self._audit_router_task(checkpoint_index, eval_task_index)

    def _audit_gradient_task_loader(self, task_index):
        if self.audit_gradient_dataset is None:
            raise RuntimeError("Gradient audit dataset was not initialized.")
        self.audit_gradient_dataset.load_dataset(task_index, train=True)
        return DataLoader(
            self.audit_gradient_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def _audit_component_named_parameters(self, components=None):
        requested = set(components or self.audit_gradient_components)
        groups = {component: [] for component in requested}
        for name, parameter in self._audit_core_model().named_parameters():
            if "key" in requested and "prompt.e_pk_" in name:
                groups["key"].append((name, parameter))
            elif "value" in requested and "prompt.e_pv_" in name:
                groups["value"].append((name, parameter))
            elif "classifier" in requested and name.startswith("last."):
                groups["classifier"].append((name, parameter))
        return groups

    @staticmethod
    def _flatten_component_tensors(named_tensors, component, classifier_rows):
        flattened = []
        for name, tensor in named_tensors:
            value = tensor.detach().cpu().to(torch.float64)
            if component == "classifier" and classifier_rows > 0:
                value = value[:classifier_rows]
            flattened.append(value.reshape(-1))
        if not flattened:
            return torch.zeros(0, dtype=torch.float64)
        return torch.cat(flattened)

    def _snapshot_audit_parameters(self, components, classifier_rows):
        groups = self._audit_component_named_parameters(components)
        return {
            component: self._flatten_component_tensors(
                named_parameters, component, classifier_rows
            )
            for component, named_parameters in groups.items()
        }

    def _gradient_vectors_for_task(self, task_index, components, classifier_rows):
        groups = self._audit_component_named_parameters(components)
        flat_named = [
            (component, name, parameter)
            for component, named_parameters in groups.items()
            for name, parameter in named_parameters
        ]
        if not flat_named:
            return {component: torch.zeros(0, dtype=torch.float64) for component in components}
        parameters = [parameter for _, _, parameter in flat_named]
        original_requires_grad = [parameter.requires_grad for parameter in parameters]
        for parameter in parameters:
            parameter.requires_grad_(True)

        accumulators = [torch.zeros_like(parameter, device="cpu", dtype=torch.float64) for parameter in parameters]
        seen = 0
        model = self.learner.model
        was_training = model.training
        model.eval()
        try:
            for inputs, targets, _ in self._audit_gradient_task_loader(task_index):
                remaining = self.audit_gradient_max_samples - seen
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                targets = targets[:remaining]
                if self.learner.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                logits = model(inputs)[:, : self.learner.valid_out_dim]
                loss = torch.nn.functional.cross_entropy(
                    logits, targets.long(), reduction="sum"
                )
                gradients = torch.autograd.grad(
                    loss,
                    parameters,
                    allow_unused=True,
                    retain_graph=False,
                    create_graph=False,
                )
                for index, gradient in enumerate(gradients):
                    if gradient is not None:
                        accumulators[index].add_(
                            gradient.detach().cpu().to(torch.float64)
                        )
                seen += int(targets.numel())
        finally:
            model.train(was_training)
            for parameter, requires_grad in zip(parameters, original_requires_grad):
                parameter.requires_grad_(requires_grad)

        if seen == 0:
            raise ValueError(f"Gradient audit task {task_index + 1} has no samples.")
        by_component = {component: [] for component in components}
        for (component, name, _), accumulator in zip(flat_named, accumulators):
            value = accumulator / seen
            if component == "classifier" and classifier_rows > 0:
                value = value[:classifier_rows]
            by_component[component].append((name, value))
        return {
            component: self._flatten_component_tensors(
                tensors, component, classifier_rows
            )
            for component, tensors in by_component.items()
        }

    def _gradient_route_state(self, task_index):
        model = self.learner.model
        was_training = model.training
        model.eval()
        per_layer = {}
        seen = 0
        with torch.no_grad():
            for inputs, targets, _ in self._audit_gradient_task_loader(task_index):
                remaining = self.audit_gradient_max_samples - seen
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                if self.learner.gpu:
                    inputs = inputs.cuda()
                scores = model(inputs, train=False, return_attn=True)
                for layer_id, (_, selection_logits) in enumerate(scores):
                    if selection_logits is None:
                        continue
                    sorted_scores, sorted_indices = torch.sort(
                        selection_logits, dim=-1, descending=True
                    )
                    topk = self._audit_core_model().prompt.topk
                    state = per_layer.setdefault(
                        layer_id, {"indices": [], "margins": []}
                    )
                    state["indices"].append(
                        sorted_indices[..., :topk].detach().cpu()
                    )
                    state["margins"].append(
                        (
                            sorted_scores[..., topk - 1]
                            - sorted_scores[..., topk]
                        ).detach().cpu()
                    )
                seen += int(inputs.size(0))
        model.train(was_training)
        return {
            layer_id: {
                "indices": torch.cat(state["indices"], dim=0),
                "margins": torch.cat(state["margins"], dim=0),
            }
            for layer_id, state in per_layer.items()
        }

    def _prepare_gradient_audit(self, train_task_index, components, include_routes=True):
        classifier_rows = int(self.learner.last_valid_out_dim)
        before = self._snapshot_audit_parameters(components, classifier_rows)
        new_gradients = self._gradient_vectors_for_task(
            train_task_index, components, classifier_rows
        )
        old_gradients = {}
        route_before = {}
        for old_task_index in range(train_task_index):
            old_gradients[old_task_index] = self._gradient_vectors_for_task(
                old_task_index, components, classifier_rows
            )
            if include_routes:
                route_before[old_task_index] = self._gradient_route_state(old_task_index)
        if self.audit_gradient_tasks == "aggregate_old" and old_gradients:
            aggregate = {}
            for component in components:
                aggregate[component] = torch.stack(
                    [value[component] for value in old_gradients.values()]
                ).mean(dim=0)
            old_gradients = {-1: aggregate}
        return {
            "classifier_rows": classifier_rows,
            "before": before,
            "old_gradients": old_gradients,
            "new_gradients": new_gradients,
            "route_before": route_before,
            "components": tuple(components),
        }

    def _write_gradient_audit(
        self, train_task_index, prepared, training_phase, include_routes=True
    ):
        after = self._snapshot_audit_parameters(
            prepared["components"], prepared["classifier_rows"]
        )
        route_after = {}
        if include_routes:
            for old_task_index in prepared["route_before"]:
                route_after[old_task_index] = self._gradient_route_state(old_task_index)
        path = os.path.join(
            self.log_dir, "forgetting_audit", "gradient_direction.jsonl"
        )
        for old_task_index, old_components in prepared["old_gradients"].items():
            route_metrics = {
                "mean_router_margin": math.nan,
                "p10_router_margin": math.nan,
                "near_boundary_rate": math.nan,
                "route_flip_rate": math.nan,
            }
            if include_routes and old_task_index >= 0:
                margins = []
                changed = decisions = 0
                for layer_id, before_state in prepared["route_before"][old_task_index].items():
                    margins.append(before_state["margins"].reshape(-1).to(torch.float64))
                    metrics = route_set_metrics(
                        route_after[old_task_index][layer_id]["indices"],
                        before_state["indices"],
                    )
                    changed += metrics["changed"]
                    decisions += metrics["decisions"]
                if margins:
                    margin_values = torch.cat(margins)
                    route_metrics = {
                        "mean_router_margin": float(margin_values.mean()),
                        "p10_router_margin": float(torch.quantile(margin_values, 0.10)),
                        "near_boundary_rate": float(
                            (margin_values < self.audit_router_margin_threshold)
                            .to(torch.float64)
                            .mean()
                        ),
                        "route_flip_rate": changed / max(decisions, 1),
                    }
            for component in prepared["components"]:
                update = after[component] - prepared["before"][component]
                metrics = vector_direction_metrics(
                    old_components[component],
                    prepared["new_gradients"][component],
                    update,
                )
                record = {
                    "event": "gradient_direction",
                    "schema_version": SCHEMA_VERSION,
                    "repeat_id": self.round_id + 1,
                    "seed": self.seed,
                    "train_task": train_task_index + 1,
                    "eval_old_task": (
                        "ALL_OLD" if old_task_index < 0 else old_task_index + 1
                    ),
                    "component": component,
                    "training_phase": training_phase,
                    "parameter_scope": (
                        "old_seen_rows" if component == "classifier" else "global_pool"
                    ),
                    **metrics,
                    **route_metrics,
                }
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, allow_nan=True) + "\n")

    def train(self, avg_metrics):

        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys:
            temp_table[mkey] = []
        temp_dir = self.log_dir + "/temp/"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # for each task
        for i in trange(self.max_task, desc="Task"):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print("======================", train_name, "=======================")

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[
                    self.learner_name
                ](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=int(self.workers),
                pin_memory=True,
            )

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()  # reinit all the prompt?

            # learn
            self.test_dataset.load_dataset(i, train=False)

            test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.workers,
                pin_memory=True,
            )
            model_save_dir = (
                self.model_top_dir
                + "/models/repeat-"
                + str(self.round_id + 1)
                + "/task-"
                + self.task_names[i]
                + "/"
            )
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            main_gradient_audit = None
            if self.audit_gradient_direction and i > 0:
                main_gradient_audit = self._prepare_gradient_audit(
                    i, self.audit_gradient_components, include_routes=True
                )
            avg_train_time, re_train = self.learner.learn_batch(
                train_loader, self.train_dataset, model_save_dir, test_loader
            )
            if main_gradient_audit is not None:
                self._write_gradient_audit(
                    i,
                    main_gradient_audit,
                    "main_prompt_training",
                    include_routes=True,
                )

            correction_gradient_audit = None
            if self.adaptive_pred:
                # compute mean and variance
                self._compute_mean(model=self.learner.model, class_mask=self.tasks[i])

                # pseudo replay
                if i > 0:
                    if self.audit_gradient_direction:
                        correction_gradient_audit = self._prepare_gradient_audit(
                            i, ("classifier",), include_routes=False
                        )
                    self.train_task_adaptive_prediction(
                        model=self.learner.model, class_mask=self.tasks, task_id=i
                    )
                    if correction_gradient_audit is not None:
                        self._write_gradient_audit(
                            i,
                            correction_gradient_audit,
                            "classifier_correction",
                            include_routes=False,
                        )

            # save model
            if re_train:
                self.learner.save_model(model_save_dir)
            if self.audit_save_full_checkpoints:
                self._save_full_audit_checkpoint(model_save_dir, i)

            if self.audit_router:
                # Capture each task's own post-training route as the historical
                # reference.  Later checkpoints replay these exact top-k choices.
                self._capture_component_reference(i)
                self._capture_router_reference(i)
                if (i + 1) in self.audit_checkpoints:
                    self._run_forgetting_audit(i)

            # evaluate acc -> NO NEED
            acc_table = []
            # acc_table_ssl = []
            self.reset_cluster_labels = True
            for j in range(i + 1):
                acc_table.append(
                    self.task_eval(j)
                )  # eval each task one-by-one, after learning a new task; on train dataset
            temp_table["acc"].append(np.mean(np.asarray(acc_table)))

            # self.learner.model.prompt.print_freq()

            print(f"Averaged accuracy for task {i + 1}: {temp_table['acc'][-1]}")

            if avg_train_time is not None:
                avg_metrics["time"]["global"][
                    i, self.round_id
                ] = avg_train_time  # time/epoch for each task
            # why not use avg_metrics to save other metrics such as 'acc'?

        return avg_metrics

    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict["global"]  # avg_metrics['acc']['global'] after training
        avg_acc_pt = acc_dict["pt"]
        # avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i + 1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]  # metric_table['acc']
                avg_acc_pt[j, i, self.round_id] = acc_table[val_name][
                    train_name
                ]  # metric_table['acc']
            avg_acc_history[i] = cls_acc_sum / (
                i + 1
            )  # metric_table['acc'], FAA of every task

        # Gather the final avg accuracy
        avg_acc_all[
            :, self.round_id
        ] = avg_acc_history  # metric_table['acc'] FAA? 'global'<-'pt'

        # repack dictionary and return
        return {"global": avg_acc_all, "pt": avg_acc_pt}

    def summarize_fr(self, fr_dict, acc_matrix):

        # unpack dictionary
        avg_fr_all = fr_dict["global"]

        avg_fr_history = [0] * self.max_task
        for task_id in range(self.max_task):
            if task_id > 0:
                avg_fr_history[task_id] = np.mean(
                    (np.max(acc_matrix[:, :task_id], axis=1) - acc_matrix[:, task_id])[
                        :task_id
                    ]
                )

        # Gather the final forgetting rate
        avg_fr_all[:, self.round_id] = avg_fr_history
        # repack dictionary and return
        return {"global": avg_fr_all}

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](
            self.learner_config
        )

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}

        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                # try:
                #     if self.learner.model.module.prompt is not None:
                #         self.learner.model.module.prompt.process_task_count()
                # except:
                if self.learner.model.prompt is not None:
                    self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = (
                self.model_top_dir
                + "/models/repeat-"
                + str(self.round_id + 1)
                + "/task-"
                + self.task_names[i]
                + "/"
            )
            self.learner.task_count = i
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc - three-level dict
            metric_table["acc"][
                self.task_names[i]
            ] = OrderedDict()  # 'acc' is a two-level dict
            # metric_table_local['acc'][self.task_names[i]] = OrderedDict() # local evaluation
            self.reset_cluster_labels = True
            accs = []
            for j in range(i + 1):
                val_name = self.task_names[j]
                metric_table["acc"][val_name][self.task_names[i]] = self.task_eval(j)
                accs.append(metric_table["acc"][val_name][self.task_names[i]])
            print(f"Averaged accuracy for task {i + 1}: {np.mean(accs)}")

        # summarize metrics
        avg_metrics["acc"] = self.summarize_acc(
            avg_metrics["acc"], metric_table["acc"], metric_table_local["acc"]
        )
        avg_metrics["fr"] = self.summarize_fr(
            avg_metrics["fr"], avg_metrics["acc"]["pt"][:, :, self.round_id]
        )  # can use avg_metrics['acc']['pt-local'] for DIL

        return avg_metrics

    @torch.no_grad()
    def _compute_mean(self, model: torch.nn.Module, class_mask=None):
        model.eval()

        for cls_id in class_mask:
            self.train_dataset.load_class(cls_id)
            data_loader_cls = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=self.workers,
                pin_memory=True,
            )
            features_per_cls = []
            for i, (inputs, targets, task) in enumerate(data_loader_cls):
                # send data to gpu
                if self.learner.gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                features = model(inputs, return_pre_logits=True, train=False)
                features_per_cls.append(features)
            features_per_cls = torch.cat(features_per_cls, dim=0)

            if self.ca_method == "covariance":
                self.learner.cls_mean[cls_id] = features_per_cls.mean(dim=0)
                self.learner.cls_cov[cls_id] = torch.cov(features_per_cls.T) + (
                    torch.eye(self.learner.cls_mean[cls_id].shape[-1]) * 1e-4
                ).to(inputs.device)
            elif self.ca_method == "multi-centroid":
                from sklearn.cluster import KMeans

                n_clusters = self.n_centroids  # default 10
                features_per_cls = features_per_cls.cpu().numpy()
                kmeans = KMeans(
                    n_clusters=n_clusters, n_init="auto", random_state=self.seed
                )
                kmeans.fit(features_per_cls)
                cluster_labels = kmeans.labels_
                cluster_means = []
                cluster_vars = []
                for i in range(n_clusters):
                    cluster_data = features_per_cls[cluster_labels == i]
                    cluster_mean = torch.tensor(
                        np.mean(cluster_data, axis=0), dtype=torch.float64
                    ).to(inputs.device)
                    cluster_var = torch.tensor(
                        np.var(cluster_data, axis=0), dtype=torch.float64
                    ).to(inputs.device)
                    cluster_means.append(cluster_mean)
                    cluster_vars.append(cluster_var)

                self.learner.cls_mean[cls_id] = cluster_means
                self.learner.cls_cov[cls_id] = cluster_vars
            else:
                raise NotImplementedError

    def train_task_adaptive_prediction(
        self, model: torch.nn.Module, class_mask=None, task_id=-1
    ):
        model.train()
        run_epochs = self.crct_epochs
        crct_num = 0
        valid_out_dim = self.learner.valid_out_dim
        ca_lr = self.ca_lr
        weight_decay = self.ca_weight_decay
        batch_size = self.batch_size
        param_list = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad
            and "prompt" not in n
        ]
        network_params = [
            {"params": param_list, "lr": ca_lr, "weight_decay": weight_decay}
        ]

        optimizer = torch.optim.AdamW(
            network_params, lr=ca_lr / 10, weight_decay=weight_decay
        )  # ****

        criterion = torch.nn.CrossEntropyLoss()
        if self.learner.gpu:
            criterion = criterion.cuda()

        for i in range(task_id):  # only take part of the samples after random permute
            crct_num += len(class_mask[i])

        scheduler_cfg = {
            "base_value": [ca_lr / 10],
            "final_value": [1e-6],
            "optimizer": optimizer,
            "iter_step": crct_num,
            "n_epochs": run_epochs,
            "last_epoch": -1,
            "warmup_epochs": 0,
            "start_warmup_value": 0,
            "freeze_iters": 0,
        }
        scheduler = CosineSchedulerIter(**scheduler_cfg)

        for epoch in range(run_epochs):

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = int(batch_size * self.ca_batch_size_ratio)  # default 5

            metric_logger = utils_tap.MetricLogger(delimiter="  ")
            metric_logger.add_meter(
                "Lr", utils_tap.SmoothedValue(window_size=1, fmt="{value:.6f}")
            )
            metric_logger.add_meter(
                "Loss", utils_tap.SmoothedValue(window_size=1, fmt="{value:.4f}")
            )

            if self.ca_method == "covariance":
                for i in range(task_id + 1):
                    for c_id in class_mask[i]:
                        mapped_c_id = self.train_dataset.class_mapping[c_id]
                        mean = self.learner.cls_mean[c_id].detach()
                        cov = self.learner.cls_cov[c_id].detach()
                        m = MultivariateNormal(mean.float(), cov.float())
                        sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                        sampled_data.append(sampled_data_single)

                        sampled_label.extend([mapped_c_id] * num_sampled_pcls)
            elif self.ca_method == "multi-centroid":
                for i in range(task_id + 1):
                    for c_id in class_mask[i]:
                        mapped_c_id = self.train_dataset.class_mapping[c_id]
                        for cluster in range(len(self.learner.cls_mean[c_id])):
                            mean = self.learner.cls_mean[c_id][cluster]
                            var = self.learner.cls_cov[c_id][cluster]
                            if var.mean() == 0:
                                continue
                            m = MultivariateNormal(
                                mean.float(),
                                (
                                    torch.diag(var)
                                    + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)
                                ).float(),
                            )
                            sampled_data_single = m.sample(
                                sample_shape=(num_sampled_pcls,)
                            )
                            sampled_data.append(sampled_data_single)
                            sampled_label.extend([mapped_c_id] * num_sampled_pcls)
            else:
                raise NotImplementedError

            sampled_data = torch.cat(sampled_data, dim=0).float().cuda()
            sampled_label = torch.tensor(sampled_label).long().to(sampled_data.device)
            # print(sampled_data.shape)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls : (_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls : (_iter + 1) * num_sampled_pcls]

                try:
                    logits = model.module.forward_fc(inp)
                except:
                    logits = model.forward_fc(inp)

                logits = logits[:, :valid_out_dim]

                loss = criterion(logits, tgt)  # base criterion (CrossEntropyLoss)
                acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()))
                    sys.exit(1)

                optimizer.zero_grad()
                loss.backward()
                old_class_dim = sum(
                    len(self.tasks_logits[old_task_id])
                    for old_task_id in range(task_id)
                )
                freeze_classifier = (
                    self.learner_config.get("audit_freeze_component")
                    == "classifier"
                    and (task_id + 1)
                    >= int(self.learner_config.get("audit_freeze_from_task", 2))
                    and old_class_dim > 0
                )
                frozen_weight = frozen_bias = None
                if freeze_classifier:
                    core = model.module if hasattr(model, "module") else model
                    frozen_weight = core.last.weight[:old_class_dim].detach().clone()
                    frozen_bias = core.last.bias[:old_class_dim].detach().clone()
                    if core.last.weight.grad is not None:
                        core.last.weight.grad[:old_class_dim].zero_()
                    if core.last.bias.grad is not None:
                        core.last.bias.grad[:old_class_dim].zero_()
                optimizer.step()
                if freeze_classifier:
                    with torch.no_grad():
                        core.last.weight[:old_class_dim].copy_(frozen_weight)
                        core.last.bias[:old_class_dim].copy_(frozen_bias)
                scheduler.step()  # step inside loop for Iter scheduler

                metric_logger.update(Loss=loss.item())
                metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
                metric_logger.meters["Acc@1"].update(acc1.item(), n=inp.shape[0])
                metric_logger.meters["Acc@5"].update(acc5.item(), n=inp.shape[0])

            print("Averaged stats:", metric_logger)
