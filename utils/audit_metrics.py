"""Pure metrics used by the forgetting-source audit infrastructure."""

from __future__ import annotations

import math
from statistics import mean, median, pstdev

import torch


SCHEMA_VERSION = "2.0"


def safe_recovery_ratio(gain, forgetting_gap, epsilon=1e-12):
    """Return an unclipped recovery ratio, or NaN for a zero denominator."""
    if forgetting_gap is None or not math.isfinite(float(forgetting_gap)):
        return math.nan
    if float(forgetting_gap) <= float(epsilon):
        return math.nan
    return float(gain) / float(forgetting_gap)


def clipped_recovery_ratio(value):
    if value is None or not math.isfinite(float(value)):
        return math.nan
    return min(1.0, max(0.0, float(value)))


def route_set_metrics(current, reference):
    """Compare unordered top-k routes and return rate/Jaccard/count totals."""
    current = torch.as_tensor(current).detach().cpu()
    reference = torch.as_tensor(reference).detach().cpu()
    if tuple(current.shape) != tuple(reference.shape):
        raise ValueError(
            f"Route shapes differ: {tuple(current.shape)} != {tuple(reference.shape)}"
        )
    if current.ndim < 1:
        raise ValueError("Routes must have a top-k dimension.")
    changed = ~(
        torch.sort(current, dim=-1).values
        == torch.sort(reference, dim=-1).values
    ).all(dim=-1)
    intersection = (
        current.unsqueeze(-1) == reference.unsqueeze(-2)
    ).any(dim=-1).sum(dim=-1).to(torch.float64)
    union = current.size(-1) * 2 - intersection
    decisions = int(changed.numel())
    return {
        "changed": int(changed.sum()),
        "decisions": decisions,
        "change_rate": float(changed.to(torch.float64).mean()) if decisions else math.nan,
        "jaccard_sum": float((intersection / union.clamp(min=1)).sum()),
        "mean_jaccard": (
            float((intersection / union.clamp(min=1)).mean())
            if decisions
            else math.nan
        ),
    }


def vector_direction_metrics(old_gradient, new_gradient, update):
    """Compute signed first-order and cosine metrics without dividing by zero."""
    old = torch.as_tensor(old_gradient, dtype=torch.float64).reshape(-1)
    new = torch.as_tensor(new_gradient, dtype=torch.float64).reshape(-1)
    delta = torch.as_tensor(update, dtype=torch.float64).reshape(-1)
    if old.numel() != new.numel() or old.numel() != delta.numel():
        raise ValueError("Gradient and update vectors must have equal sizes.")

    old_norm = float(torch.linalg.vector_norm(old))
    new_norm = float(torch.linalg.vector_norm(new))
    update_norm = float(torch.linalg.vector_norm(delta))
    first_order = float(torch.dot(old, delta))

    def cosine(a, b, norm_a, norm_b):
        if norm_a == 0.0 or norm_b == 0.0:
            return math.nan
        return float(torch.dot(a, b) / (norm_a * norm_b))

    return {
        "gradient_norm_old": old_norm,
        "gradient_norm_new": new_norm,
        "update_norm": update_norm,
        "first_order_old_loss_change": first_order,
        "update_old_gradient_cosine": cosine(old, delta, old_norm, update_norm),
        "old_new_gradient_cosine": cosine(old, new, old_norm, new_norm),
        "descent_direction_old_alignment": cosine(
            old, -new, old_norm, new_norm
        ),
    }


def paired_effect_summary(deltas):
    values = [float(value) for value in deltas if math.isfinite(float(value))]
    if not values:
        return {
            "paired_mean_delta": math.nan,
            "paired_std_delta": math.nan,
            "paired_median_delta": math.nan,
            "paired_min_delta": math.nan,
            "paired_max_delta": math.nan,
            "same_direction_count": 0,
            "num_pairs": 0,
        }
    mean_delta = mean(values)
    if mean_delta > 0:
        same_direction = sum(value > 0 for value in values)
    elif mean_delta < 0:
        same_direction = sum(value < 0 for value in values)
    else:
        same_direction = sum(value == 0 for value in values)
    return {
        "paired_mean_delta": mean_delta,
        "paired_std_delta": pstdev(values),
        "paired_median_delta": median(values),
        "paired_min_delta": min(values),
        "paired_max_delta": max(values),
        "same_direction_count": same_direction,
        "num_pairs": len(values),
    }


def interpretation_flags(
    *,
    delta_faa,
    delta_fr,
    delta_final_old_accuracy,
    mean_plasticity_loss,
    plasticity_warn_threshold=2.0,
    retention_gain_threshold=1.0,
):
    """Generate conservative flags; these are warnings, not paper conclusions."""
    flags = []
    lower_fr = float(delta_fr) < 0.0
    retention_gain = float(delta_final_old_accuracy)
    plasticity = float(mean_plasticity_loss)
    if lower_fr and retention_gain <= 0.0:
        flags.append("LOWER_FR_WITHOUT_RETENTION_GAIN")
    if lower_fr and plasticity >= float(plasticity_warn_threshold):
        flags.append("LOWER_FR_WITH_LARGE_PLASTICITY_LOSS")
    if (
        retention_gain >= float(retention_gain_threshold)
        and plasticity < float(plasticity_warn_threshold)
    ):
        flags.append("RETENTION_GAIN_WITH_SMALL_PLASTICITY_LOSS")
    if float(delta_faa) > 0.0 and retention_gain > 0.0:
        flags.append("IMPROVED_FAA_AND_RETENTION")
    return flags
