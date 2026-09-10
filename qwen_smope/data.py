"""Reuse legacy split readers, but feed unnormalised PIL images to Qwen."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

DATASETS = {"cifar100": (100, "iCIFAR100"), "cub200": (200, "iCUB200"),
            "imagenet-r": (200, "iIMAGENET_R")}


def check_model(path):
    path = Path(path)
    required = ["config.json", "tokenizer_config.json", "preprocessor_config.json"]
    missing = [f for f in required if not (path / f).is_file()]
    if not (path / "tokenizer.json").is_file():
        missing.append("tokenizer.json")
    if not ((path / "chat_template.jinja").is_file() or
            (path / "chat_template.json").is_file() or
            (path / "tokenizer_config.json").is_file() and
            json.loads((path / "tokenizer_config.json").read_text()).get("chat_template")):
        missing.append("chat_template.jinja (or template in tokenizer config)")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        missing.extend(s for s in sorted(shards) if not (path / s).is_file())
    elif not (path / "model.safetensors").is_file():
        missing.append("model.safetensors or complete sharded model")
    if missing:
        raise FileNotFoundError(f"Incomplete local model at {path}: {missing}")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Expected multimodal dense Qwen3.5, not text-only or MoE weights")
    return config


def build_datasets(name, root, seed):
    from dataloaders import dataloader
    count, cls = DATASETS[name]
    # Match legacy class permutation; archive membership remains unchanged.
    order = list(range(count))
    import random
    random.Random(seed).shuffle(order)
    width = count // 10
    tasks = [order[i:i + width] for i in range(0, count, width)]
    factory = getattr(dataloader, cls)
    args = dict(root=root, tasks=tasks, seed=seed, transform=None, download_flag=False)
    train, test = factory(train=True, **args), factory(train=False, **args)
    # Detect missing files before loading 9B weights.
    if name != "cifar100":
        missing = [str(p) for d in (train, test) for p in d.data if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} images, first: {missing[:3]}")
    return train, test, tasks


class View(Dataset):
    def __init__(self, dataset, task, cumulative=False, per_class=0):
        self.dataset = dataset
        labels = [c for t in dataset.tasks[:task + 1] for c in t] if cumulative else dataset.tasks[task]
        indices = []
        for c in labels:
            ix = np.flatnonzero(dataset.targets == c)
            indices.extend(ix[:per_class] if per_class else ix)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label, _ = self.dataset[self.indices[i]]
        return image.convert("RGB"), label


class ExactShard(Sampler):
    """No padding/duplicates for eval, frequency and prototype scans."""
    def __init__(self, dataset, rank=0, world=1):
        self.indices = list(range(rank, len(dataset), world))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class Collator:
    def __init__(self, processor, max_visual_tokens=256):
        self.processor = processor
        image_processor = processor.image_processor
        self.merge_size = image_processor.merge_size
        self.max_visual_tokens = max_visual_tokens
        factor = image_processor.patch_size * self.merge_size
        self.max_pixels = max_visual_tokens * factor * factor
        self.min_pixels = min(image_processor.size["shortest_edge"], self.max_pixels)

    def __call__(self, batch):
        images, labels = zip(*batch)
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"},
             {"type": "text", "text": "Classify this image."}]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[text] * len(images), images=list(images), padding=True,
                                return_tensors="pt", images_kwargs={"size": {
                                    "shortest_edge": self.min_pixels, "longest_edge": self.max_pixels}})
        visual_tokens = inputs["image_grid_thw"].prod(-1) // (self.merge_size ** 2)
        if bool((visual_tokens > self.max_visual_tokens).any()):
            raise ValueError(f"Processor exceeded visual token budget: {visual_tokens.tolist()}")
        inputs.pop("token_type_ids", None)
        return dict(inputs), torch.tensor(labels, dtype=torch.long)
