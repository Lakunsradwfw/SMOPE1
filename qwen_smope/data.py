"""CoIN datasets and Qwen3.5 multimodal collators.

The source JSON remains untouched. Legacy ``<img>path</img>`` markers are
resolved at runtime and converted to the official structured image input.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset, Sampler


TASK_ORDER = (
    "ScienceQA", "TextVQA", "ImageNet", "GQA",
    "VizWiz", "Grounding", "VQAv2", "OCRVQA",
)
IMAGE_PATTERN = re.compile(r"(?:Picture\s+\d+:\s*)?<img>(.*?)</img>", re.IGNORECASE | re.DOTALL)
ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def check_model(path: str | Path) -> dict[str, Any]:
    """Fail before allocating a GPU when the local post-trained model is incomplete."""
    path = Path(path)
    if "base" in path.name.casefold():
        raise ValueError(f"Expected post-trained Qwen3.5-9B, not a Base checkpoint: {path}")
    required = ["config.json", "tokenizer_config.json", "preprocessor_config.json",
                "tokenizer.json", "chat_template.jinja"]
    missing = [name for name in required if not (path / name).is_file()]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        missing.extend(name for name in sorted(shards) if not (path / name).is_file())
    elif not (path / "model.safetensors").is_file():
        missing.append("model.safetensors or complete sharded model")
    if missing:
        raise FileNotFoundError(f"Incomplete local model at {path}: {missing}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError("CoIN requires the dense multimodal Qwen3.5 model")
    architectures = set(config.get("architectures", ()))
    if architectures and "Qwen3_5ForConditionalGeneration" not in architectures:
        raise ValueError(f"Expected Qwen3_5ForConditionalGeneration, got {sorted(architectures)}")
    return config


def _safe_path(root: Path, raw: str) -> Path:
    value = raw.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    candidate = (root / value).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"Image path escapes CoIN root: {raw!r}")
    return candidate


def _normalise_conversations(record: dict[str, Any]) -> list[dict[str, str]]:
    raw = record.get("conversations")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Each CoIN sample must contain at least one conversation turn")
    result: list[dict[str, str]] = []
    for turn in raw:
        role = ROLE_MAP.get(str(turn.get("from", "")).lower())
        value = turn.get("value")
        if role is None or not isinstance(value, str):
            raise ValueError(f"Invalid conversation turn: {turn!r}")
        result.append({"role": role, "text": value})
    if not any(turn["role"] == "user" for turn in result):
        raise ValueError("A CoIN sample must contain a user turn")
    return result


@dataclass(frozen=True)
class CoINSample:
    sample_id: str
    task: str
    image_path: Path
    history: tuple[dict[str, str], ...]
    answer: str
    references: tuple[str, ...]

    def prompt_messages(self, image: Any) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        image_inserted = False
        for turn in self.history:
            text = IMAGE_PATTERN.sub("", turn["text"]).strip()
            if turn["role"] == "user" and not image_inserted:
                content: Any = [{"type": "image", "image": image}, {"type": "text", "text": text}]
                image_inserted = True
            else:
                content = text
            messages.append({"role": turn["role"], "content": content})
        if not image_inserted:
            raise ValueError(f"Sample {self.sample_id} has no user turn for image insertion")
        return messages


class CoINDataset(Dataset):
    """One CoIN task split from ``Instructions_Qwen/<task>/<split>.json``."""

    def __init__(self, root: str | Path, task: str, split: str,
                 max_samples: int = 0, require_images: bool = True) -> None:
        self.root = Path(root).resolve()
        self.task = task
        path = self.root / "Instructions_Qwen" / task / f"{split}.json"
        if not path.is_file():
            available = sorted(p.name for p in path.parent.glob("*.json")) if path.parent.is_dir() else []
            raise FileNotFoundError(f"Missing CoIN split {path}; available={available}")
        self.path = path.resolve()
        self.split = split
        self.source_size = self.path.stat().st_size
        self.source_mtime_ns = self.path.stat().st_mtime_ns
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array")
        if max_samples > 0:
            records = records[:max_samples]
        textvqa_references = self._textvqa_references(split) if task == "TextVQA" else {}
        self.samples = [self._convert(record, index, require_images, textvqa_references)
                        for index, record in enumerate(records)]
        if not self.samples:
            raise ValueError(f"Empty CoIN split: {path}")

    def _textvqa_references(self, split: str) -> dict[str, tuple[str, ...]]:
        directory = self.root / "cl_dataset" / "TextVQA"
        candidates = [directory / f"TextVQA_0.5.1_{split}.json"]
        # CoIN calls the held-out instruction split test while the official
        # TextVQA annotations call the same answer-bearing split val.
        if split == "test":
            candidates.append(directory / "TextVQA_0.5.1_val.json")
        for path in candidates:
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("data", payload) if isinstance(payload, dict) else payload
            result = {}
            for row in rows:
                answers = row.get("answers")
                if isinstance(answers, list) and answers:
                    values = [item.get("answer", "") if isinstance(item, dict) else item
                              for item in answers]
                    result[str(row.get("question_id", row.get("id")))] = tuple(
                        str(value) for value in values if str(value).strip())
            if result:
                return result
        return {}

    @staticmethod
    def _record_references(record: dict[str, Any], fallback: str | None) -> tuple[str, ...]:
        raw = record.get("answers")
        if isinstance(raw, list):
            values = [item.get("answer", "") if isinstance(item, dict) else item for item in raw]
            references = tuple(str(value) for value in values if str(value).strip())
            if references:
                return references
        answer = record.get("answer")
        if answer is not None and str(answer).strip():
            return (str(answer),)
        bbox = record.get("answer_bbox")
        if bbox is not None and str(bbox).strip():
            return (str(bbox),)
        if fallback:
            return (fallback,)
        raise ValueError("Evaluation sample has no assistant answer, answer, answers, or answer_bbox")

    def _convert(self, record: dict[str, Any], index: int, require_images: bool,
                 textvqa_references: dict[str, tuple[str, ...]]) -> CoINSample:
        turns = _normalise_conversations(record)
        inline = [match.group(1) for turn in turns for match in IMAGE_PATTERN.finditer(turn["text"])]
        raw_image = inline[0] if inline else record.get("image")
        if not raw_image:
            raise ValueError(f"{self.task}[{index}] has no image path")
        if len(set(inline)) > 1:
            raise ValueError(f"{self.task}[{index}] references multiple images; unsupported")
        image_path = _safe_path(self.root, str(raw_image))
        if not image_path.is_file() and record.get("image"):
            fallback = _safe_path(self.root / "cl_dataset", str(record["image"]))
            if fallback.is_file():
                image_path = fallback
        if require_images and not image_path.is_file():
            raise FileNotFoundError(f"Missing image for {self.task}[{index}]: {image_path}")
        sample_id = str(record.get("question_id", record.get("id", f"{self.task}-{index}")))
        has_assistant_answer = turns[-1]["role"] == "assistant" and bool(turns[-1]["text"].strip())
        conversation_answer = turns[-1]["text"].strip() if has_assistant_answer else None
        references = textvqa_references.get(sample_id)
        if references is None:
            references = self._record_references(record, conversation_answer)
        answer = conversation_answer or references[0]
        if self.task == "ScienceQA" and answer.isdigit() and 0 <= int(answer) < 5:
            answer = chr(ord("A") + int(answer))
            references = (answer,)
        history = turns[:-1] if has_assistant_answer else turns
        return CoINSample(sample_id=sample_id, task=self.task, image_path=image_path,
                          history=tuple(history), answer=answer, references=references)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CoINSample:
        return self.samples[index]


class ExactShard(Sampler[int]):
    """Distributed evaluation/route scan without padding or duplicated samples."""

    def __init__(self, dataset: Dataset, rank: int = 0, world: int = 1):
        self.indices = list(range(rank, len(dataset), world))

    def __iter__(self) -> Iterable[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class CoINCollator:
    """Build answer-only SFT batches and a leak-free prefix routing mask."""

    def __init__(self, processor: Any, max_length: int = 1024, max_visual_tokens: int = 256):
        self.processor = processor
        self.max_length = max_length
        image_processor = processor.image_processor
        merge = image_processor.merge_size
        factor = image_processor.patch_size * merge
        self.image_size = {
            "shortest_edge": min(image_processor.size["shortest_edge"], max_visual_tokens * factor * factor),
            "longest_edge": max_visual_tokens * factor * factor,
        }

    def _encode(self, texts: list[str], images: list[Any]) -> dict[str, torch.Tensor]:
        encoded = self.processor(
            text=texts, images=images, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt", images_kwargs={"size": self.image_size},
        )
        encoded.pop("token_type_ids", None)
        return dict(encoded)

    def __call__(self, batch: list[CoINSample]) -> dict[str, Any]:
        from PIL import Image

        images, prompt_texts, full_texts = [], [], []
        for sample in batch:
            image = Image.open(sample.image_path).convert("RGB")
            messages = sample.prompt_messages(image)
            prompt_texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            full_texts.append(self.processor.apply_chat_template(
                messages + [{"role": "assistant", "content": sample.answer}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False))
            images.append(image)
        prompt = self._encode(prompt_texts, images)
        full = self._encode(full_texts, images)
        prompt_lengths = prompt["attention_mask"].sum(-1).long()
        labels = full["input_ids"].clone()
        router_mask = torch.zeros_like(full["attention_mask"], dtype=torch.bool)
        for row, length in enumerate(prompt_lengths.tolist()):
            if length >= int(full["attention_mask"][row].sum()):
                raise ValueError(f"Answer was truncated or empty for sample {batch[row].sample_id}")
            if not torch.equal(full["input_ids"][row, :length], prompt["input_ids"][row, :length]):
                raise ValueError("Prompt tokens are not a prefix of the supervised conversation")
            labels[row, :length] = -100
            router_mask[row, :length] = True
        labels.masked_fill_(~full["attention_mask"].bool(), -100)
        full["labels"] = labels
        return {"inputs": full, "router_mask": router_mask,
                "metadata": [{"id": s.sample_id, "task": s.task, "answer": s.answer,
                              "references": list(s.references)} for s in batch]}


class CoINEvalCollator(CoINCollator):
    def __call__(self, batch: list[CoINSample]) -> dict[str, Any]:
        from PIL import Image

        images, texts = [], []
        for sample in batch:
            image = Image.open(sample.image_path).convert("RGB")
            texts.append(self.processor.apply_chat_template(
                sample.prompt_messages(image), tokenize=False,
                add_generation_prompt=True, enable_thinking=False))
            images.append(image)
        inputs = self._encode(texts, images)
        return {"inputs": inputs, "router_mask": inputs["attention_mask"].bool(),
                "metadata": [{"id": s.sample_id, "task": s.task, "answer": s.answer,
                              "references": list(s.references)} for s in batch]}


def build_task_datasets(root: str | Path, tasks: Iterable[str] = TASK_ORDER,
                        train_split: str = "train", eval_split: str = "test",
                        max_train_samples: int = 0, max_eval_samples: int = 0):
    train = {task: CoINDataset(root, task, train_split, max_train_samples) for task in tasks}
    evaluation = {task: CoINDataset(root, task, eval_split, max_eval_samples) for task in tasks}
    return train, evaluation
