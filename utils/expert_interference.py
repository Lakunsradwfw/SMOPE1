"""Small, testable metrics for the expert-interference mechanism audit."""

from __future__ import annotations

import math

import numpy as np
import torch


def stratified_probe_indices(targets, max_samples, seed):
    """Select a deterministic, approximately class-balanced probe subset."""
    targets = np.asarray(targets)
    if max_samples <= 0 or max_samples >= len(targets):
        return list(range(len(targets)))
    classes = np.unique(targets)
    rng = np.random.RandomState(int(seed))
    buckets = {}
    for class_id in classes:
        indices = np.flatnonzero(targets == class_id)
        buckets[class_id] = rng.permutation(indices).tolist()

    selected = []
    while len(selected) < max_samples:
        progressed = False
        for class_id in classes:
            bucket = buckets[class_id]
            if bucket and len(selected) < max_samples:
                selected.append(int(bucket.pop()))
                progressed = True
        if not progressed:
            break
    return selected


def gradient_conflict_summary(old_gradients, new_gradient, epsilon=1e-12):
    """Summarize old/new task gradient conflict for one expert tensor.

    Zero-norm task pairs are excluded instead of being treated as orthogonal;
    this prevents inactive experts from diluting the conflict estimate.
    """
    new = torch.as_tensor(new_gradient, dtype=torch.float64).reshape(-1)
    new_norm = float(torch.linalg.vector_norm(new))
    cosines = []
    old_norms = []
    for old_gradient in old_gradients:
        old = torch.as_tensor(old_gradient, dtype=torch.float64).reshape(-1)
        old_norm = float(torch.linalg.vector_norm(old))
        old_norms.append(old_norm)
        if old_norm <= epsilon or new_norm <= epsilon:
            continue
        cosine = float(torch.dot(old, new) / (old_norm * new_norm))
        cosines.append(max(-1.0, min(1.0, cosine)))

    if not cosines:
        return {
            "new_gradient_norm": new_norm,
            "mean_old_gradient_norm": (
                float(np.mean(old_norms)) if old_norms else math.nan
            ),
            "old_task_pairs": len(old_norms),
            "valid_gradient_pairs": 0,
            "mean_cosine": math.nan,
            "conflict_score": math.nan,
            "negative_pair_rate": math.nan,
            "mean_negative_cosine": math.nan,
        }

    return {
        "new_gradient_norm": new_norm,
        "mean_old_gradient_norm": float(np.mean(old_norms)),
        "old_task_pairs": len(old_norms),
        "valid_gradient_pairs": len(cosines),
        "mean_cosine": float(np.mean(cosines)),
        "conflict_score": float(1.0 - np.mean(cosines)),
        "negative_pair_rate": float(np.mean([value < 0.0 for value in cosines])),
        "mean_negative_cosine": float(
            np.mean([max(0.0, -value) for value in cosines])
        ),
    }


def gradient_pair_metrics(first_gradient, second_gradient, epsilon=1e-12):
    """Metrics for one explicitly identified task pair in one expert tensor."""
    first = torch.as_tensor(first_gradient, dtype=torch.float64).reshape(-1)
    second = torch.as_tensor(second_gradient, dtype=torch.float64).reshape(-1)
    if first.numel() != second.numel():
        raise ValueError("Gradient vectors must have equal sizes.")
    first_norm = float(torch.linalg.vector_norm(first))
    second_norm = float(torch.linalg.vector_norm(second))
    valid = first_norm > epsilon and second_norm > epsilon
    if valid:
        cosine = float(torch.dot(first, second) / (first_norm * second_norm))
        cosine = max(-1.0, min(1.0, cosine))
    else:
        cosine = math.nan
    return {
        "first_gradient_norm": first_norm,
        "second_gradient_norm": second_norm,
        "valid_gradient_pair": bool(valid),
        "cosine": cosine,
        "negative_cosine": max(0.0, -cosine) if valid else math.nan,
    }


def update_harm_metrics(old_gradient, parameter_update, epsilon=1e-12):
    """First-order old-loss change attributable to one expert update."""
    old = torch.as_tensor(old_gradient, dtype=torch.float64).reshape(-1)
    update = torch.as_tensor(parameter_update, dtype=torch.float64).reshape(-1)
    if old.numel() != update.numel():
        raise ValueError("Gradient and update vectors must have equal sizes.")
    old_norm = float(torch.linalg.vector_norm(old))
    update_norm = float(torch.linalg.vector_norm(update))
    first_order_change = float(torch.dot(old, update))
    if old_norm <= epsilon or update_norm <= epsilon:
        cosine = math.nan
    else:
        cosine = float(torch.dot(old, update) / (old_norm * update_norm))
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "old_gradient_norm": old_norm,
        "update_norm": update_norm,
        "first_order_old_loss_change": first_order_change,
        "old_gradient_update_cosine": cosine,
        "predicted_harm": max(0.0, first_order_change),
    }


def tensor_drift_metrics(current, reference, epsilon=1e-12):
    """Return magnitude and angular drift without hiding degenerate vectors."""
    current = torch.as_tensor(current, dtype=torch.float64).reshape(-1)
    reference = torch.as_tensor(reference, dtype=torch.float64).reshape(-1)
    difference = current - reference
    l2 = float(torch.linalg.vector_norm(difference))
    reference_norm = float(torch.linalg.vector_norm(reference))
    current_norm = float(torch.linalg.vector_norm(current))
    if reference_norm <= epsilon or current_norm <= epsilon:
        cosine_distance = math.nan
    else:
        cosine = float(torch.dot(current, reference) / (current_norm * reference_norm))
        cosine_distance = 1.0 - max(-1.0, min(1.0, cosine))
    return {
        "l2": l2,
        "relative_l2": l2 / max(reference_norm, epsilon),
        "cosine_distance": cosine_distance,
        "reference_norm": reference_norm,
        "current_norm": current_norm,
    }
