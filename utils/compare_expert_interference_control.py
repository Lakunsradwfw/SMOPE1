"""Pass/fail check that mechanism probes do not change benchmark metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def load_metric(run_dir, metric):
    path = Path(run_dir) / f"results-{metric}" / "global.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return np.asarray(yaml.safe_load(handle)["history"], dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control_dir", type=Path)
    parser.add_argument("mechanism_dir", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    control_acc = load_metric(args.control_dir, "acc")
    mechanism_acc = load_metric(args.mechanism_dir, "acc")
    control_fr = load_metric(args.control_dir, "fr")
    mechanism_fr = load_metric(args.mechanism_dir, "fr")
    if control_acc.shape != mechanism_acc.shape or control_fr.shape != mechanism_fr.shape:
        raise ValueError("Control and mechanism result shapes do not match.")

    rows = []
    for repeat_index in range(control_acc.shape[1]):
        rows.append(
            {
                "repeat_id": repeat_index + 1,
                "control_faa": control_acc[-1, repeat_index],
                "mechanism_faa": mechanism_acc[-1, repeat_index],
                "abs_delta_faa": abs(
                    mechanism_acc[-1, repeat_index] - control_acc[-1, repeat_index]
                ),
                "control_caa": control_acc[:, repeat_index].mean(),
                "mechanism_caa": mechanism_acc[:, repeat_index].mean(),
                "abs_delta_caa": abs(
                    mechanism_acc[:, repeat_index].mean()
                    - control_acc[:, repeat_index].mean()
                ),
                "control_fr": control_fr[-1, repeat_index],
                "mechanism_fr": mechanism_fr[-1, repeat_index],
                "abs_delta_fr": abs(
                    mechanism_fr[-1, repeat_index] - control_fr[-1, repeat_index]
                ),
            }
        )
    output = args.output or args.mechanism_dir / "expert_interference" / "analysis" / "instrumentation_control.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    maximum = max(
        max(row["abs_delta_faa"], row["abs_delta_caa"], row["abs_delta_fr"])
        for row in rows
    )
    print(f"maximum absolute FAA/CAA/FR delta: {maximum:.12g}")
    if maximum > args.tolerance:
        raise SystemExit(
            f"FAIL: instrumentation delta {maximum:.12g} exceeds {args.tolerance:.12g}"
        )
    print(f"PASS: all deltas are <= {args.tolerance:.12g}")


if __name__ == "__main__":
    main()
