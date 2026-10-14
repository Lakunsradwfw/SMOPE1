"""CoIN datasets and Qwen3.5 multimodal collators.

The source JSON remains untouched. Legacy ``<img>path</img>`` markers are
resolved at runtime and converted to the official structured image input.
"""
from __future__ import annotations

from dataclasses import dataclass
import http.client
from io import BytesIO
import json
from pathlib import Path
import random
import re
import threading
import time
from typing import Any, Iterable
import urllib.parse

import torch
from torch.utils.data import Dataset, Sampler


TASK_ORDER = (
    "ScienceQA", "TextVQA", "ImageNet", "GQA",
    "VizWiz", "Grounding", "VQAv2", "OCRVQA",
)
IMAGE_PATTERN = re.compile(r"(?:Picture\s+\d+:\s*)?<img>(.*?)</img>", re.IGNORECASE | re.DOTALL)
ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}

# CoIN ships the held-out instructions in ``Instructions_Original`` (flat
# ``question_id``/``image``/``text``/``answer`` records) rather than in the Qwen
# layer, whose per-task directory only holds ``train.json``. The split that
# carries gold answers differs per task because the official test files for
# TextVQA, VizWiz and VQAv2 are unlabelled (and their image directories were
# never downloaded).
EVAL_SPLIT_FILES = {
    "ScienceQA": "test",
    "TextVQA": "val",
    "ImageNet": "test",
    "GQA": "test",
    "VizWiz": "val",
    "Grounding": "test",
    "VQAv2": "val",
    "OCRVQA": "test",
}
# ``Instructions_Original`` writes TextVQA val image paths against
# ``train_images`` while the files actually live in ``val_images``.
IMAGE_DIRECTORY_ALIASES = {("TextVQA", "train_images"): "val_images"}


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
    image_path: Path | None
    history: tuple[dict[str, str], ...]
    answer: str
    references: tuple[str, ...]

    @property
    def has_image(self) -> bool:
        return self.image_path is not None

    def prompt_messages(self, image: Any = None) -> list[dict[str, Any]]:
        """Render turns; the image is attached only when the sample carries one.

        ScienceQA is roughly half text-only questions in the official release, so
        an unconditional ``{"type": "image"}`` item makes the processor reject
        those records.
        """
        if self.image_path is not None and image is None:
            raise ValueError(f"Sample {self.sample_id} needs its image but none was supplied")
        messages: list[dict[str, Any]] = []
        image_inserted = False
        for turn in self.history:
            text = IMAGE_PATTERN.sub("", turn["text"]).strip()
            if turn["role"] == "user" and self.image_path is not None and not image_inserted:
                content: Any = [{"type": "image", "image": image}, {"type": "text", "text": text}]
                image_inserted = True
            else:
                content = text
            messages.append({"role": turn["role"], "content": content})
        if self.image_path is not None and not image_inserted:
            raise ValueError(f"Sample {self.sample_id} has no user turn for image insertion")
        if not any(turn["role"] == "user" for turn in self.history):
            raise ValueError(f"Sample {self.sample_id} has no user turn")
        return messages


class ImageUnavailableError(FileNotFoundError):
    """A permanent missing-image response that may be audited and filtered."""

    def __init__(self, location: str, status: int | None = None) -> None:
        self.location = location
        self.status = status
        detail = f"HTTP {status} for {location}" if status is not None else f"Missing image: {location}"
        super().__init__(detail)


class ImageFetcher:
    """Loads sample images from the filesystem or straight over HTTP.

    The davfs mount cannot carry this workload: it serialises metadata inside one
    client process, re-fetches whole directories every ``dir_refresh`` seconds
    (``GQA/images`` holds 148854 entries), reads in 16 KiB requests, and reports
    stale negatives for trees written by an out-of-band HTTP client. Measured on
    this box: 233 MB in 582 s through the mount versus 2.4 s (97 MB/s) over HTTP.
    """

    def __init__(self, kind: str = "mount", base_url: str | None = None,
                 coin_root: str | Path | None = None, *, timeout_seconds: float = 30,
                 max_attempts: int = 4) -> None:
        if kind not in ("mount", "http"):
            raise ValueError(f"Unknown image transport: {kind!r}")
        if kind == "http" and not base_url:
            raise ValueError("HTTP image transport needs --image-base-url")
        self.kind = kind
        self.coin_root = Path(coin_root).resolve() if coin_root else None
        if timeout_seconds <= 0 or max_attempts <= 0:
            raise ValueError("timeout_seconds and max_attempts must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._local = threading.local()
        if kind == "http":
            parsed = urllib.parse.urlsplit(base_url)
            if not parsed.hostname:
                raise ValueError(f"Bad --image-base-url: {base_url!r}")
            self.host = parsed.hostname
            self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
            self.secure = parsed.scheme == "https"
            self.prefix = urllib.parse.quote(parsed.path.rstrip("/") + "/", safe="/%()")

    def _connection(self) -> http.client.HTTPConnection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            factory = http.client.HTTPSConnection if self.secure else http.client.HTTPConnection
            connection = self._local.connection = factory(
                self.host, self.port, timeout=self.timeout_seconds)
        return connection

    def __getstate__(self) -> dict[str, Any]:
        """``threading.local`` is not picklable, and DataLoader pickles the collator
        (which holds this fetcher) into every worker process."""
        state = self.__dict__.copy()
        state.pop("_local", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._local = threading.local()

    def _url(self, path: Path) -> str:
        if self.coin_root is None:
            raise ValueError("HTTP image transport needs the CoIN root to build URLs")
        relative = path.resolve().relative_to(self.coin_root / "cl_dataset")
        # HTTP paths always use POSIX separators.  ``str(Path)`` leaks Windows
        # backslashes into the URL and percent-encodes them as ``%5C``.
        return self.prefix + urllib.parse.quote(relative.as_posix(), safe="/")

    def __call__(self, sample: CoINSample) -> Any:
        from PIL import Image

        if not sample.has_image:
            return None
        if self.kind == "mount":
            try:
                return Image.open(sample.image_path).convert("RGB")
            except FileNotFoundError as error:
                raise ImageUnavailableError(str(sample.image_path)) from error
        url = self._url(sample.image_path)
        transient_statuses = {408, 425, 429, 500, 502, 503, 504}
        for attempt in range(self.max_attempts):
            try:
                connection = self._connection()
                connection.request("GET", url)
                response = connection.getresponse()
                body = response.read()
                if response.status in {404, 410}:
                    raise ImageUnavailableError(url, response.status)
                if response.status != 200:
                    if response.status not in transient_statuses and response.status < 500:
                        raise RuntimeError(f"Non-retryable HTTP {response.status} for {url}")
                    raise http.client.HTTPException(f"HTTP {response.status} for {url}")
                return Image.open(BytesIO(body)).convert("RGB")
            except ImageUnavailableError:
                self._local.connection = None
                raise
            except (http.client.HTTPException, OSError):
                self._local.connection = None
                if attempt + 1 == self.max_attempts:
                    raise
                time.sleep(min(0.5 * 2**attempt, 4.0))
        raise AssertionError("unreachable")


def subsample_records(records: list[Any], limit: int, seed: str) -> list[Any]:
    """Deterministic fixed-seed sample, returned in source order.

    Continual evaluation scores every task once per completed task, so the
    held-out set has to be identical across methods, seeds and rounds; the seed
    is therefore derived from the task and split rather than from ``--seed``.
    """
    if limit <= 0 or len(records) <= limit:
        return records
    generator = random.Random(seed)
    return [records[index] for index in sorted(generator.sample(range(len(records)), limit))]


def resolve_split_path(root: str | Path, task: str, split: str) -> tuple[Path, str]:
    """Return the ``(path, format)`` backing a CoIN task split.

    Training lives in the Qwen conversation layer. Held-out instructions are the
    flat ``Instructions_Original`` records, so evaluation prefers that layer and
    falls back to ``EVAL_SPLIT_FILES`` for tasks whose official test file carries
    no gold answers.
    """
    root = Path(root).resolve()
    qwen = root / "Instructions_Qwen" / task / f"{split}.json"
    original = root / "Instructions_Original" / task / f"{split}.json"
    candidates: list[tuple[Path, str]] = []
    if split in ("train", "train_new"):
        candidates.append((qwen, "qwen"))
    else:
        # The declared held-out file wins over the requested name: for TextVQA,
        # VizWiz and VQAv2 the ``test`` file exists but carries neither gold
        # answers nor a downloadable image directory.
        declared = EVAL_SPLIT_FILES.get(task)
        if declared:
            candidates.append((root / "Instructions_Original" / task / f"{declared}.json", "original"))
        if split != declared:
            candidates.append((original, "original"))
        candidates.append((qwen, "qwen"))
    for path, kind in candidates:
        if path.is_file():
            return path, kind
    available = sorted(
        f"{layer.name}/{task}/{item.name}"
        for layer in (qwen.parent.parent, original.parent.parent) if layer.is_dir()
        for item in layer.glob("*.json"))
    requested = ", ".join(str(path) for path, _ in candidates)
    raise FileNotFoundError(
        f"Missing CoIN split for {task}/{split}; tried {requested}; available={available}")


class CoINDataset(Dataset):
    """One CoIN task split.

    Training reads the Qwen conversation layer
    (``Instructions_Qwen/<task>/<split>.json``). Held-out instructions live in
    ``Instructions_Original/<task>/<split>.json`` as flat records, so evaluation
    resolves that layer first and wraps each record into a Qwen conversation at
    load time; the source files stay untouched.
    """

    def __init__(self, root: str | Path, task: str, split: str,
                 max_samples: int = 0, require_images: bool = True,
                 sample_seed: str | None = None, verify_images: bool = True) -> None:
        self.root = Path(root).resolve()
        self.task = task
        # With HTTP image transport the mount must not be touched at all: a stat
        # would be both slow and untrustworthy. Paths are still derived (so the
        # fetcher can build URLs) but never probed.
        self.verify_images = verify_images
        path, self.format = resolve_split_path(self.root, task, split)
        self.path = path.resolve()
        self.split = split
        self.source_size = self.path.stat().st_size
        self.source_mtime_ns = self.path.stat().st_mtime_ns
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array")
        if max_samples > 0:
            # Held-out splits are sampled, not truncated: taking the first N of a
            # 214k-record VQAv2 val file would score one contiguous slice of it.
            records = (subsample_records(records, max_samples, sample_seed)
                       if sample_seed is not None else records[:max_samples])
        self.sampled = len(records)
        convert = self._convert if self.format == "qwen" else self._convert_flat
        # Key the official TextVQA annotations off the file actually loaded, so a
        # held-out ``val`` split picks up ``TextVQA_0.5.1_val.json`` directly.
        textvqa_references = self._textvqa_references(self.path.stem) if task == "TextVQA" else {}
        self.samples = [convert(record, index, require_images, textvqa_references)
                        for index, record in enumerate(records)]
        self.original_examples = len(self.samples)
        self.valid_examples = len(self.samples)
        self.filter_manifest: dict[str, Any] | None = None
        if not self.samples:
            raise ValueError(f"Empty CoIN split: {path}")

    def apply_training_filter(self, valid_indices: list[int], max_samples: int,
                              manifest: dict[str, Any]) -> None:
        """Apply a full-split validity list before taking the training prefix.

        ``valid_indices`` always addresses the complete source split.  Keeping
        filtering separate from ``max_samples`` makes smoke/full and all methods
        draw prefixes from the same deterministic valid sequence.
        """
        if self.split not in ("train", "train_new"):
            raise ValueError("Length filtering is training-only")
        if any(index < 0 or index >= len(self.samples) for index in valid_indices):
            raise ValueError("Filter manifest contains an out-of-range source index")
        if valid_indices != sorted(set(valid_indices)):
            raise ValueError("Filter manifest indices must be unique and source ordered")
        selected = [self.samples[index] for index in valid_indices]
        self.valid_examples = len(selected)
        if max_samples > 0:
            selected = selected[:max_samples]
        self.samples = selected
        self.sampled = len(selected)
        self.filter_manifest = manifest
        if not self.samples:
            raise ValueError(f"Length filter removed every sample from {self.task}/{self.split}")

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

    def _image_candidates(self, raw: str, aliases: bool = True) -> list[str]:
        """Root-relative candidate paths, applying known directory aliases first.

        ``aliases`` must be true only for the ``Instructions_Original`` layer. The
        TextVQA repair exists because that layer's ``val.json`` points at
        ``train_images`` for files that live in ``val_images``; the training layer
        spells ``train_images`` correctly and rewriting it there would send every
        training image to a 404.
        """
        cleaned = str(raw).strip().replace("\\", "/")
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        bases = [] if cleaned.startswith("cl_dataset/") else ["cl_dataset/" + cleaned]
        bases.append(cleaned)
        candidates: list[str] = []
        for base in bases:
            if aliases:
                segments = base.split("/")
                for position, segment in enumerate(segments):
                    replacement = IMAGE_DIRECTORY_ALIASES.get((self.task, segment))
                    if replacement:
                        aliased = "/".join(segments[:position] + [replacement]
                                           + segments[position + 1:])
                        if aliased not in candidates:
                            candidates.append(aliased)
            if base not in candidates:
                candidates.append(base)
        return candidates

    def _resolve_image(self, raw: Any, index: int, require_images: bool,
                       aliases: bool = True) -> Path | None:
        """Resolve an image reference, or ``None`` for a genuinely text-only sample.

        ScienceQA is about half text-only questions in the official release, so an
        absent ``image`` is a property of the dataset rather than a download gap.
        """
        if raw is None or not str(raw).strip():
            return None
        candidates = self._image_candidates(str(raw), aliases)
        if not self.verify_images:
            # The aliased candidate comes first, which is also the correct one for
            # TextVQA's ``train_images`` -> ``val_images`` repair.
            return _safe_path(self.root, candidates[0])
        for candidate in candidates:
            path = _safe_path(self.root, candidate)
            if path.is_file():
                return path
        missing = _safe_path(self.root, candidates[0])
        if require_images:
            raise FileNotFoundError(f"Missing image for {self.task}[{index}]: {missing}")
        return None

    @staticmethod
    def _flat_references(record: dict[str, Any]) -> tuple[str, ...]:
        """Gold answer of an ``Instructions_Original`` record.

        ``answer_bbox`` is preferred over ``origin_answer_bbox`` because Grounding
        boxes are scored in normalised coordinates, matching the training targets.
        """
        for key in ("answer", "answers", "answer_bbox"):
            raw = record.get(key)
            if raw is None:
                continue
            if isinstance(raw, list):
                values = tuple(str(item.get("answer", "") if isinstance(item, dict) else item)
                               for item in raw)
                values = tuple(value for value in values if value.strip())
            elif str(raw).strip():
                values = (str(raw).strip(),)
            else:
                continue
            if values:
                return values
        raise ValueError("Evaluation sample has no answer, answers, or answer_bbox")

    def _convert_flat(self, record: dict[str, Any], index: int, require_images: bool,
                      textvqa_references: dict[str, tuple[str, ...]]) -> CoINSample:
        """Wrap a flat ``Instructions_Original`` record into Qwen conversation form."""
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{self.task}[{index}] has no instruction text")
        image_path = self._resolve_image(record.get("image"), index, require_images)
        sample_id = str(record.get("question_id", record.get("id", f"{self.task}-{index}")))
        references = textvqa_references.get(sample_id) or self._flat_references(record)
        answer = references[0]
        if self.task == "ScienceQA" and answer.isdigit() and 0 <= int(answer) < 5:
            answer = chr(ord("A") + int(answer))
            references = (answer,)
        return CoINSample(sample_id=sample_id, task=self.task, image_path=image_path,
                          history=({"role": "user", "text": text.strip()},),
                          answer=answer, references=references)

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
        if len(set(inline)) > 1:
            raise ValueError(f"{self.task}[{index}] references multiple images; unsupported")
        raw_image = inline[0] if inline else record.get("image")
        # Qwen 训练层不做目录别名修正: 它的路径本来就是对的
        image_path = self._resolve_image(raw_image, index, require_images, aliases=False)
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

    def __init__(self, processor: Any, max_length: int = 1024, max_visual_tokens: int = 256,
                 fetcher: ImageFetcher | None = None, bucket: int = 128):
        self.processor = processor
        self.max_length = max_length
        self.bucket = bucket
        self.fetcher = fetcher or ImageFetcher("mount")
        image_processor = processor.image_processor
        merge = image_processor.merge_size
        factor = image_processor.patch_size * merge
        self.image_size = {
            "shortest_edge": min(image_processor.size["shortest_edge"], max_visual_tokens * factor * factor),
            "longest_edge": max_visual_tokens * factor * factor,
        }

    def _encode(self, texts: list[str], images: list[list[Any]] | None, *,
                truncate: bool = True, bucketed: bool = True) -> dict[str, torch.Tensor]:
        # ``images`` is one list per sample, empty for text-only records; the
        # processor rejects an empty flat list, so a fully text-only batch passes
        # ``None``.
        #
        # ``pad_to_multiple_of`` 把宽度收敛到有限的几个桶。Triton 为每个新宽度编译
        # 一整族内核（实测磁盘缓存 4894 个条目仍不够：同一批数据第二次运行的
        # preflight 还要 20s，而 torch 回退只要 1.5s），变长输入会让每个 batch 都
        # 触发新编译，实测 >17s/步；分桶后 preflight 降到 0.3s。
        # 必须让 processor 自己 padding：它还会返回 mm_token_type_ids 等张量，
        # 事后手动只补 input_ids/attention_mask 会造成形状不一致
        # （实测 IndexError: mask [384] vs tensor [326]）。
        bucket = self.bucket if bucketed and 1 < self.bucket <= self.max_length else None
        options = dict(text=texts, images=images, padding=True,
                       pad_to_multiple_of=bucket, truncation=truncate,
                       return_tensors="pt", images_kwargs={"size": self.image_size})
        if truncate:
            options["max_length"] = self.max_length
        encoded = self.processor(**options)
        encoded.pop("token_type_ids", None)
        return dict(encoded)

    def measure(self, sample: CoINSample) -> dict[str, Any]:
        """Measure an untruncated training example with the real multimodal path."""
        images = self._batch_images([sample])
        sample_images = images[0] if images else []
        messages = sample.prompt_messages(sample_images[0] if sample_images else None)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full_text = self.processor.apply_chat_template(
            messages + [{"role": "assistant", "content": sample.answer}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prompt = self._encode([prompt_text], images, truncate=False, bucketed=False)
        full = self._encode([full_text], images, truncate=False, bucketed=False)
        prompt_len = int(prompt["attention_mask"][0].sum())
        full_len = int(full["attention_mask"][0].sum())
        prefix_matches = (prompt_len <= full_len and torch.equal(
            full["input_ids"][0, :prompt_len], prompt["input_ids"][0, :prompt_len]))
        answer_len = max(full_len - prompt_len, 0) if prefix_matches else 0
        reason = None
        if not prefix_matches:
            reason = "prompt_not_prefix"
        elif answer_len <= 0:
            reason = "empty_answer_tokens"
        elif full_len > self.max_length:
            reason = "full_length_exceeds_max_length"
        return {"task": sample.task, "sample_id": sample.sample_id,
                "prompt_len": prompt_len, "answer_len": answer_len,
                "full_len": full_len, "reason": reason}

    def _batch_images(self, batch: list[CoINSample]) -> list[list[Any]] | None:
        images = [[self.fetcher(sample)] if sample.has_image else [] for sample in batch]
        return images if any(images) else None

    def __call__(self, batch: list[CoINSample]) -> dict[str, Any]:
        images = self._batch_images(batch)
        prompt_texts, full_texts = [], []
        for sample, sample_images in zip(batch, images or [[]] * len(batch)):
            image = sample_images[0] if sample_images else None
            messages = sample.prompt_messages(image)
            prompt_texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            full_texts.append(self.processor.apply_chat_template(
                messages + [{"role": "assistant", "content": sample.answer}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False))
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
        images = self._batch_images(batch)
        texts = []
        for sample, sample_images in zip(batch, images or [[]] * len(batch)):
            texts.append(self.processor.apply_chat_template(
                sample.prompt_messages(sample_images[0] if sample_images else None), tokenize=False,
                add_generation_prompt=True, enable_thinking=False))
        # Decoder-only batched generation must be left padded: generation
        # continues from the final column, which is padding for shorter prompts
        # under the right-padding convention used by supervised training.
        tokenizer = self.processor.tokenizer
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = self._encode(texts, images)
        finally:
            tokenizer.padding_side = original_padding_side
        return {"inputs": inputs, "router_mask": inputs["attention_mask"].bool(),
                "metadata": [{"id": s.sample_id, "task": s.task, "answer": s.answer,
                              "references": list(s.references)} for s in batch]}


def build_task_datasets(root: str | Path, tasks: Iterable[str] = TASK_ORDER,
                        train_split: str = "train", eval_split: str = "test",
                        max_train_samples: int = 0, max_eval_samples: int = 0):
    train = {task: CoINDataset(root, task, train_split, max_train_samples) for task in tasks}
    evaluation = {task: CoINDataset(root, task, eval_split, max_eval_samples) for task in tasks}
    return train, evaluation
