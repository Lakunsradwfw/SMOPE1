from collections import OrderedDict
from contextlib import contextmanager
import time

import torch


class StageTimer:
    """CUDA-safe timers and counters for reproducible efficiency reports.

    Some entries are intentionally nested (for example ``inference_forward`` is
    inside an evaluation stage), so their values must not be blindly summed.
    ``cl_train`` is derived in ``run.py`` from the training phase wall time after
    subtracting intermediate evaluation.
    """

    STAGES = OrderedDict(
        [
            ("dense_initialization", "Dense initialization training"),
            ("routed_training", "Routed training"),
            ("expert_frequency_scan", "Expert-frequency scan"),
            ("prototype_statistics", "Prototype statistics"),
            ("prototype_replay", "Prototype replay training"),
            ("intermediate_evaluation", "Evaluation during continual training"),
            ("final_evaluation", "Final checkpoint evaluation"),
            ("inference_forward", "Model forward calls during evaluation"),
        ]
    )

    def __init__(self):
        self.elapsed = OrderedDict((name, 0.0) for name in self.STAGES)
        self.calls = OrderedDict((name, 0) for name in self.STAGES)
        self.samples = OrderedDict((name, 0) for name in self.STAGES)
        self._pending_cuda_events = []

    @staticmethod
    def synchronize_cuda():
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    @staticmethod
    def wall_time():
        StageTimer.synchronize_cuda()
        return time.perf_counter()

    @contextmanager
    def measure(self, stage, samples=0):
        if stage not in self.elapsed:
            raise KeyError("Unknown timing stage: {}".format(stage))

        self.synchronize_cuda()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize_cuda()
            self.elapsed[stage] += time.perf_counter() - start
            self.calls[stage] += 1
            self.samples[stage] += int(samples)

    @contextmanager
    def measure_device(self, stage, samples=0):
        """Measure device work without synchronizing the CUDA stream per call.

        CUDA event pairs are collected here and resolved together by ``snapshot``.
        This keeps high-frequency measurements, such as validation forwards, from
        changing the execution pipeline that is being benchmarked.
        """
        if stage not in self.elapsed:
            raise KeyError("Unknown timing stage: {}".format(stage))

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._pending_cuda_events.append((stage, start, end))
                self.calls[stage] += 1
                self.samples[stage] += int(samples)
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self.elapsed[stage] += time.perf_counter() - start
            self.calls[stage] += 1
            self.samples[stage] += int(samples)

    def flush_device_events(self):
        if not self._pending_cuda_events:
            return
        torch.cuda.synchronize()
        for stage, start, end in self._pending_cuda_events:
            self.elapsed[stage] += start.elapsed_time(end) / 1000.0
        self._pending_cuda_events.clear()

    def add_samples(self, stage, count):
        if stage not in self.samples:
            raise KeyError("Unknown timing stage: {}".format(stage))
        self.samples[stage] += int(count)

    def snapshot(self):
        self.flush_device_events()
        return {
            "seconds": dict(self.elapsed),
            "calls": dict(self.calls),
            "samples": dict(self.samples),
        }

    def print_report(self, derived=None, trial_id=None):
        self.flush_device_events()
        title = "=== Efficiency timing"
        if trial_id is not None:
            title += " for trial {}".format(trial_id)
        print(title + " ===")
        print(
            "{:<52} {:>12} {:>10} {:>12}".format(
                "Stage", "Seconds", "Calls", "Samples"
            )
        )

        for name, label in self.STAGES.items():
            seconds = self.elapsed[name]
            print(
                "{:<52} {:>12.4f} {:>10d} {:>12d}".format(
                    label,
                    seconds,
                    self.calls[name],
                    self.samples[name],
                )
            )
        if derived:
            print("-" * 90)
            for key, value in derived.items():
                if isinstance(value, (float, int)):
                    print("{:<52} {:>12.4f}".format(key, value))
        print(
            "Note: inference_forward is nested inside evaluation and is not added "
            "to other stages."
        )
