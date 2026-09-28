"""Deterministic scorers matching the task rules used by CoIN."""
from __future__ import annotations

import re
import string
from typing import Iterable

ARTICLES = {"a", "an", "the"}
NUMBER_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}


def final_answer(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text


def normalize_answer(text: str) -> str:
    text = final_answer(text).lower().replace("\n", " ")
    text = text.translate(str.maketrans({character: " " for character in string.punctuation}))
    return " ".join(word for word in text.split() if word not in ARTICLES)


def normalize_vqa_answer(text: str) -> str:
    """EvalAI-style normalization needed by the official TextVQA scorer."""
    text = final_answer(text).lower().replace("\n", " ").replace("\t", " ")
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    punctuation = r"[;\/\[\]\"{}()=+\\_\-><@`,?!]"
    text = re.sub(punctuation, " ", text)
    words = [NUMBER_MAP.get(word, word) for word in text.split() if word not in ARTICLES]
    return " ".join(words)


def _choice(text: str) -> str | None:
    match = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", final_answer(text).upper())
    return match.group(1) if match else None


def _box(text: str) -> tuple[float, float, float, float] | None:
    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", final_answer(text))
    if len(values) < 4:
        return None
    box = tuple(float(value) for value in values[:4])
    if max(abs(value) for value in box) > 1.5:
        box = tuple(value / 1000.0 for value in box)
    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def box_iou(left: tuple[float, float, float, float],
            right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def score_prediction(task: str, prediction: str, references: str | Iterable[str]) -> dict:
    refs = [references] if isinstance(references, str) else list(references)
    if not refs:
        raise ValueError("At least one reference answer is required")
    if task == "ScienceQA":
        predicted = _choice(prediction)
        targets = [_choice(reference) for reference in refs]
        score = float(predicted is not None and predicted in targets)
        detail = {"parsed_prediction": predicted, "parsed_references": targets}
    elif task == "Grounding":
        predicted_box = _box(prediction)
        target_boxes = [box for reference in refs if (box := _box(reference)) is not None]
        best_iou = max((box_iou(predicted_box, target) for target in target_boxes), default=0.0) \
            if predicted_box is not None else 0.0
        score = float(best_iou > 0.5)
        detail = {"parsed_prediction": predicted_box, "best_iou": best_iou}
    elif task == "TextVQA" and len(refs) > 1:
        predicted = normalize_vqa_answer(prediction)
        targets = [normalize_vqa_answer(reference) for reference in refs]
        # The official scorer leaves each annotator out in turn and averages
        # min(matches / 3, 1), rather than using a single exact target.
        scores = []
        for index in range(len(targets)):
            matches = sum(predicted == target for other, target in enumerate(targets) if other != index)
            scores.append(min(matches / 3.0, 1.0))
        score = sum(scores) / len(scores)
        detail = {"normalized_prediction": predicted, "normalized_references": targets}
    elif task == "ImageNet":
        predicted = final_answer(prediction).strip().casefold()
        targets = [final_answer(reference).strip().casefold() for reference in refs]
        score = float(any(predicted and target and (predicted in target or target in predicted)
                          for target in targets))
        detail = {"normalized_prediction": predicted, "normalized_references": targets}
    else:
        # GQA, VizWiz, VQAv2 and OCRVQA use case-insensitive exact
        # instruction-answer accuracy in the CoIN reference implementation.
        predicted = final_answer(prediction).strip().casefold()
        targets = [final_answer(reference).strip().casefold() for reference in refs]
        score = float(predicted in targets)
        detail = {"normalized_prediction": predicted, "normalized_references": targets}
    return {"score": score, "correct": bool(score >= 0.5), **detail}
