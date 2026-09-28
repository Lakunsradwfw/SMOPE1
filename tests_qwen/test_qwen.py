import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch.nn import functional as F

from qwen_smope.data import CoINCollator, CoINDataset, ExactShard, check_model
from qwen_smope.model import Experts, QwenCoINModel
from qwen_smope.report import continual_metrics, write_matrix
from qwen_smope.scoring import box_iou, normalize_answer, normalize_vqa_answer, score_prediction
from qwen_smope.train import arguments


class FakeProcessor:
    def __init__(self):
        self.image_processor = SimpleNamespace(
            merge_size=2, patch_size=2, size={"shortest_edge": 16})
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, padding_side="right")
        self.vocabulary = {"<image>": 3}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        values = ["<image>"]
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = " ".join(part.get("text", "") for part in content if part["type"] == "text")
            values.extend((message["role"], content))
        if add_generation_prompt:
            values.append("assistant")
        return " ".join(values)

    def __call__(self, text, images, padding, truncation, max_length, return_tensors, **kwargs):
        encoded = []
        for value in text:
            ids = []
            for token in value.split():
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary) + 3
                ids.append(self.vocabulary[token])
            encoded.append(ids[:max_length])
        width = max(map(len, encoded))
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for row, ids in enumerate(encoded):
            input_ids[row, :len(ids)] = torch.tensor(ids)
            attention[row, :len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention,
                "pixel_values": torch.ones(len(encoded), 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * len(encoded))}


def make_coin(root: Path):
    image = root / "cl_dataset" / "ScienceQA" / "images" / "train" / "1" / "image.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(image)
    directory = root / "Instructions_Qwen" / "ScienceQA"
    directory.mkdir(parents=True)
    record = [{"id": "1", "image": "ScienceQA/images/train/1/image.png",
               "conversations": [
                   {"from": "user", "value": "Picture 1: <img>./cl_dataset/ScienceQA/images/train/1/image.png</img>\nQuestion A B"},
                   {"from": "assistant", "value": "A"}]}]
    (directory / "train.json").write_text(json.dumps(record), encoding="utf-8")


class DataTests(unittest.TestCase):
    def test_dataset_resolves_legacy_image_and_collator_masks_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_coin(root)
            dataset = CoINDataset(root, "ScienceQA", "train")
            self.assertTrue(dataset[0].image_path.is_file())
            self.assertNotIn("<img>", dataset[0].prompt_messages("image")[0]["content"][1]["text"])
            batch = CoINCollator(FakeProcessor(), max_length=64)([dataset[0]])
            labels, route = batch["inputs"]["labels"], batch["router_mask"]
            self.assertTrue(bool((labels[route] == -100).all()))
            self.assertTrue(bool((labels[~route] != -100).any()))
            self.assertEqual(int((labels != -100).sum()), 1)

    def test_textvqa_uses_official_multi_answer_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "cl_dataset" / "TextVQA" / "val" / "image.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "blue").save(image)
            raw = {"data": [{"question_id": 7, "answers": ["two"] * 10}]}
            (root / "cl_dataset" / "TextVQA" / "TextVQA_0.5.1_val.json").write_text(
                json.dumps(raw), encoding="utf-8")
            directory = root / "Instructions_Qwen" / "TextVQA"
            directory.mkdir(parents=True)
            rows = [{"question_id": 7, "image": "TextVQA/val/image.jpg",
                     "conversations": [
                         {"from": "user", "value": "<img>./cl_dataset/TextVQA/val/image.jpg</img> How many?"}]}]
            (directory / "test.json").write_text(json.dumps(rows), encoding="utf-8")
            sample = CoINDataset(root, "TextVQA", "test")[0]
            self.assertEqual(sample.references, tuple(["two"] * 10))
            self.assertEqual(sample.answer, "two")

    def test_exact_shards_have_no_duplicates(self):
        shards = [list(ExactShard(range(9), rank, 3)) for rank in range(3)]
        self.assertEqual(sorted(sum(shards, [])), list(range(9)))
        self.assertEqual(sum(map(len, shards)), len(set(sum(shards, []))))

    def test_model_check_rejects_base_and_accepts_post_trained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
                (root / name).write_text("{}")
            (root / "chat_template.jinja").write_text("{{ messages }}")
            (root / "model.safetensors").write_bytes(b"test")
            (root / "config.json").write_text(json.dumps(
                {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"]}))
            with self.assertRaises(ValueError):
                check_model(root)
            (root / "config.json").write_text(json.dumps(
                {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}))
            self.assertEqual(check_model(root)["model_type"], "qwen3_5")


class ExpertTests(unittest.TestCase):
    def test_eval_does_not_penalize_zero_frequency_experts(self):
        expert = Experts(1, 3, 2, topk=1, epsilon=0.4)
        with torch.no_grad():
            expert.pk.copy_(torch.tensor([[[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]]))
            expert.frequency[0, 1:] = 1
            expert.used[0, 0] = True
        query = torch.tensor([[[[1.0, 0.0]]]])
        prefix = torch.ones(1, 1, dtype=torch.bool)

        expert.select(query, prefix, training=False)
        torch.testing.assert_close(expert.last_labels, expert.last_scores)
        self.assertEqual(int(expert.last_indices.item()), 0)

        expert.select(query, prefix, training=True)
        self.assertEqual(int(expert.last_indices.item()), 1)

    def test_batched_old_loss_matches_loop_value_and_gradient(self):
        torch.manual_seed(12)
        for has_old in (False, True):
            expert = Experts(3, 5, 8, topk=2)
            expert.old_pk.normal_()
            expert.has_old.fill_(has_old)
            expert.used[0, :2] = True
            expert.used[1, :] = True

            actual = expert.losses()[1]
            expected = expert.pk.sum() * 0
            if has_old:
                anchors = F.normalize(expert.old_pk.float(), dim=-1)
                current = F.normalize(expert.pk.float(), dim=-1)
                for head in range(3):
                    active = expert.used[head]
                    if bool(active.any()):
                        selected = anchors[head, active]
                        targets = (selected @ anchors[head].T).topk(2, -1).indices
                        logp = F.log_softmax(selected @ current[head].T, -1)
                        expected = expected - logp.gather(-1, targets).sum(-1).mean()

            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual, expert.pk)[0],
                torch.autograd.grad(expected, expert.pk)[0],
            )

    def test_answer_tokens_do_not_change_route_and_decode_reuses_it(self):
        torch.manual_seed(2)
        expert = Experts(3, 7, 8, topk=2)
        query = torch.randn(2, 3, 6, 8)
        prefix = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
        score, value = expert.select(query, prefix, freeze=True)
        indices = expert.last_indices.clone()
        changed = query.clone()
        changed[:, :, 4:] = torch.randn_like(changed[:, :, 4:]) * 100
        expert.clear_route()
        changed_score, changed_value = expert.select(changed, prefix, freeze=True)
        torch.testing.assert_close(score, changed_score)
        torch.testing.assert_close(value, changed_value)
        self.assertTrue(torch.equal(indices, expert.last_indices))
        decode_score, decode_value = expert.select(torch.randn(2, 3, 1, 8), None)
        torch.testing.assert_close(decode_score, changed_score)
        torch.testing.assert_close(decode_value, changed_value)

    def test_losses_and_task_state_have_finite_gradients(self):
        expert = Experts(2, 5, 4, topk=2)
        query = torch.randn(3, 2, 4, 4)
        expert.select(query, torch.ones(3, 4, dtype=torch.bool), training=True)
        expert.frequency[:, :2] = 4
        expert.next_task()
        expert.select(query, torch.ones(3, 4, dtype=torch.bool), training=True)
        loss = sum(expert.losses())
        loss.backward()
        self.assertTrue(expert.pk.grad.isfinite().all())
        self.assertTrue(expert.route_mix_logit.grad.isfinite().all())
        self.assertTrue(expert.log_temperature.grad.isfinite().all())

    def test_lightweight_lora_state_excludes_backbone_weights(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(3, 3)
                self.lora_A = torch.nn.Linear(3, 2, bias=False)

        model = QwenCoINModel(TinyBackbone(), method="lora")
        state = model.lightweight_state()
        self.assertTrue(state)
        self.assertTrue(all("lora_" in key for key in state))
        self.assertFalse(any("base" in key for key in state))

    def test_eval_skips_auxiliary_expert_losses(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.expert = Experts(2, 5, 4, topk=2)

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(loss=input_ids.float().sum() * 0)

        model = QwenCoINModel(TinyBackbone(), method="zero_shot").eval()
        inputs = {"input_ids": torch.ones(1, 3, dtype=torch.long)}
        router_mask = torch.ones(1, 3, dtype=torch.bool)
        with patch.object(Experts, "losses", side_effect=AssertionError("unused eval loss")):
            result = model(inputs, router_mask)
        self.assertEqual(float(result["router_loss"]), 0.0)
        self.assertEqual(float(result["old_loss"]), 0.0)
        self.assertEqual(float(result["balance_loss"]), 0.0)


class ScoringReportTests(unittest.TestCase):
    def test_task_scorers(self):
        self.assertEqual(normalize_answer("The, RED car!"), "red car")
        self.assertEqual(normalize_vqa_answer("Two, cats!"), "2 cats")
        self.assertTrue(score_prediction("ScienceQA", "Answer: A", "A")["correct"])
        textvqa = score_prediction("TextVQA", "two", ["2"] * 3 + ["three"] * 7)
        self.assertAlmostEqual(textvqa["score"], 0.9)
        self.assertTrue(score_prediction("GQA", "Blue", "blue")["correct"])
        self.assertTrue(score_prediction("ImageNet", "golden retriever", "golden retriever dog")["correct"])
        self.assertTrue(score_prediction("Grounding", "[0, 0, 1, 1]", "[0, 0, 1, 1]")["correct"])
        self.assertAlmostEqual(box_iou((0, 0, 1, 1), (0.5, 0.5, 1, 1)), 0.25)

    def test_coin_metrics_and_matrix(self):
        result = continual_metrics([[80.0], [70.0, 90.0]])
        self.assertEqual(result["final_average_accuracy"], 80.0)
        self.assertEqual(result["mean_average_accuracy"], 80.0)
        self.assertEqual(result["new_task_accuracy"], 85.0)
        self.assertEqual(result["backward_transfer"], -5.0)
        self.assertEqual(result["forgetting"], 10.0)
        with tempfile.TemporaryDirectory() as temporary:
            write_matrix(temporary, ["a", "b"], [[80.0], [70.0, 90.0]])
            self.assertTrue((Path(temporary) / "accuracy_matrix.csv").is_file())

    def test_cli_defaults_to_post_trained_model(self):
        args = arguments(["--method", "smope", "--output", "out"])
        self.assertEqual(args.model_path, "pretrained/Qwen3.5-9B")
        self.assertNotIn("Base", args.model_path)


if __name__ == "__main__":
    unittest.main()
