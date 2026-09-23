"""Optional real-processor check, using local tokenizer/config files (no model weights)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from PIL import Image
from transformers import AutoProcessor
from qwen_smope.data import Collator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", nargs="?", default="pretrained/Qwen3.5-9B")
    a = p.parse_args()
    processor = AutoProcessor.from_pretrained(a.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    for budget in (64, 128, 256):
        items = [(Image.new("RGB", size), i) for i, size in enumerate(((32, 32), (1200, 800), (800, 1200)))]
        batch, labels = Collator(processor, budget)(items)
        grids = batch["image_grid_thw"]
        tokens = grids.prod(-1) // processor.image_processor.merge_size ** 2
        assert int(labels.numel()) == 3 and int(tokens.max()) <= budget
        print(f"PASS budget={budget}, actual visual tokens={tokens.tolist()}, sequence={batch['input_ids'].shape}")


if __name__ == "__main__":
    main()
