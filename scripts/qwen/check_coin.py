"""Validate CoIN paths and Qwen3.5 processor output without loading model weights."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers import AutoProcessor
from qwen_smope.data import TASK_ORDER, CoINCollator, CoINDataset, CoINEvalCollator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="pretrained/Qwen3.5-9B")
    parser.add_argument("--coin-root", default="datas/CoIN")
    parser.add_argument("--eval-split", default="test")
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    train_collator = CoINCollator(processor)
    eval_collator = CoINEvalCollator(processor)
    for task in TASK_ORDER:
        train = CoINDataset(args.coin_root, task, "train", max_samples=1)
        evaluation = CoINDataset(args.coin_root, task, args.eval_split, max_samples=1)
        train_batch, eval_batch = train_collator([train[0]]), eval_collator([evaluation[0]])
        supervised = int((train_batch["inputs"]["labels"] != -100).sum())
        assert supervised > 0 and bool(train_batch["router_mask"].any())
        print(f"PASS {task}: train_seq={train_batch['inputs']['input_ids'].shape[-1]} "
              f"answer_tokens={supervised} eval_seq={eval_batch['inputs']['input_ids'].shape[-1]}")


if __name__ == "__main__":
    main()
