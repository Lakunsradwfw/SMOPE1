from collections import OrderedDict
from contextlib import contextmanager
import time

import torch


class StageTimer:
    """Accumulate wall-clock time for the expensive experiment stages."""

    STAGES = OrderedDict(
        [
            (
                "dense_initialization",
                "Dense initialization training (no Top-K routing)",
            ),
            (
                "routed_training",
                "Routed training (fwd+bwd + online Top-K experts)",
            ),
            (
                "expert_frequency_scan",
                "Post-training expert-frequency scan",
            ),
            ("prototype_statistics", "Prototype statistics (_compute_mean)"),
            ("prototype_replay", "Prototype replay training (crct)"),
            ("task_evaluation", "Task evaluation (task_eval)"),
        ]
    )

    def __init__(self):
        self.elapsed = OrderedDict((name, 0.0) for name in self.STAGES)
        self.calls = OrderedDict((name, 0) for name in self.STAGES)

    @staticmethod
    def _synchronize_cuda():
        # CUDA kernels are asynchronous, so a plain CPU timer would otherwise
        # charge their execution to whichever stage happens to synchronize next.
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    @contextmanager
    def measure(self, stage):
        if stage not in self.elapsed:
            raise KeyError("Unknown timing stage: {}".format(stage))

        self._synchronize_cuda()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize_cuda()
            self.elapsed[stage] += time.perf_counter() - start
            self.calls[stage] += 1

    def print_report(self, trial_time=None, trial_id=None):
        measured_time = sum(self.elapsed.values())
        title = "=== Stage timing"
        if trial_id is not None:
            title += " for trial {}".format(trial_id)
        print(title + " ===")
        print(
            "{:<56} {:>12} {:>10} {:>10} {:>8}".format(
                "Stage", "Seconds", "Measured%", "Trial%", "Calls"
            )
        )

        for name, label in self.STAGES.items():
            seconds = self.elapsed[name]
            measured_pct = 100.0 * seconds / measured_time if measured_time else 0.0
            trial_pct = 100.0 * seconds / trial_time if trial_time else 0.0
            print(
                "{:<56} {:>12.2f} {:>9.2f}% {:>9.2f}% {:>8d}".format(
                    label,
                    seconds,
                    measured_pct,
                    trial_pct,
                    self.calls[name],
                )
            )

        print("-" * 102)
        print("Measured stages total: {:.2f}s".format(measured_time))
        if trial_time is not None:
            other_time = max(0.0, trial_time - measured_time)
            other_pct = 100.0 * other_time / trial_time if trial_time else 0.0
            print(
                "Other/setup/I/O: {:.2f}s ({:.2f}% of trial)".format(
                    other_time, other_pct
                )
            )
            print("Trial wall-clock total: {:.2f}s".format(trial_time))

        routed_time = self.elapsed["routed_training"]
        dense_time = self.elapsed["dense_initialization"]
        optimization_time = dense_time + routed_time
        routed_trial_pct = 100.0 * routed_time / trial_time if trial_time else 0.0
        optimization_trial_pct = (
            100.0 * optimization_time / trial_time if trial_time else 0.0
        )
        print("=== Requested routing-training timing ===")
        print(
            "Routed training only (includes online Top-K routing/expert selection): "
            "{:.2f}s ({:.2f}% of trial)".format(routed_time, routed_trial_pct)
        )
        print(
            "SMoPE optimization total (dense initialization + routed training): "
            "{:.2f}s ({:.2f}% of trial)".format(
                optimization_time, optimization_trial_pct
            )
        )
        print(
            "Post-training expert-frequency scan is auxiliary statistics and is "
            "excluded from routed training."
        )
