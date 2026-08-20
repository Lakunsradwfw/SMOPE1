from contextlib import contextmanager
import csv
import json
import os
from statistics import mean, pstdev
import time

import torch


class FlopsProfiler:
    """Profile one representative batch for each requested execution path."""

    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.records = {}

    def has(self, name):
        return name in self.records

    @contextmanager
    def profile_once(self, name, samples):
        if not self.enabled or self.has(name):
            yield
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        wall_start = time.perf_counter()
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            yield
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - wall_start
        total_flops = float(sum(event.flops for event in prof.key_averages()))
        self.records[name] = {
            "batch_flops": total_flops,
            "samples": int(samples),
            "flops_per_sample": total_flops / samples if samples else 0.0,
            "profiled_wall_seconds": wall_seconds,
        }
        print(
            "FLOPs profile {}: {:.6f} GFLOPs/sample (supported PyTorch ops)".format(
                name, self.records[name]["flops_per_sample"] / 1e9
            )
        )


def _numeric_summary(records, key):
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return {"mean": None, "std": None, "values": []}
    return {"mean": mean(values), "std": pstdev(values), "values": values}


def write_efficiency_records(log_dir, records):
    """Write raw per-trial data plus aggregate JSON/CSV summaries."""
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "efficiency_trials.json"), "w") as handle:
        json.dump(records, handle, indent=2, sort_keys=True)

    scalar_keys = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if isinstance(value, (int, float)) and key not in {"trial", "seed"}
        }
    )
    summary = {
        "schema_version": 1,
        "mode": records[0]["mode"] if records else None,
        "dataset": records[0]["dataset"] if records else None,
        "num_trials": len(records),
        "metrics": {key: _numeric_summary(records, key) for key in scalar_keys},
        "formula_inputs": {
            "S_CL_train": "baseline.cl_train_seconds / optimized.cl_train_seconds",
            "S_route": "baseline.routed_training_seconds / optimized.routed_training_seconds",
            "p": "baseline.routed_training_seconds / baseline.cl_train_seconds",
            "S_overall": "1 / ((1 - p) + p / S_route)",
            "S_infer": "baseline.inference_seconds_per_sample / optimized.inference_seconds_per_sample",
            "FLOPs_reduction": "1 - optimized.flops_per_sample / baseline.flops_per_sample",
        },
    }
    with open(os.path.join(log_dir, "efficiency_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if records:
        fieldnames = sorted({key for record in records for key in record})
        with open(os.path.join(log_dir, "efficiency_trials.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
