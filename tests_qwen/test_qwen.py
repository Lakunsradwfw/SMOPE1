import json
from datetime import timedelta
from io import BytesIO
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch.nn import functional as F

import qwen_smope.train as train_module
from qwen_smope.data import (CoINCollator, CoINDataset, CoINEvalCollator, CoINSample, ExactShard,
                             ImageFetcher, ImageUnavailableError, check_model,
                             subsample_records)
from qwen_smope.compare_ddp import compare_runs, read_run
from qwen_smope.diagnose_ddp import validate_global_batch, validate_variant_world
from qwen_smope.filtering import (build_filter_manifest, build_filter_shard,
                                  ensure_filter_manifest, filter_signature, impact_level,
                                  manifest_for_run, merge_filter_shards,
                                  validate_filter_impact)
from qwen_smope.model import Experts, QwenCoINModel
from qwen_smope.prepare_data import arguments as prepare_arguments
from qwen_smope.report import continual_metrics, write_matrix
from qwen_smope.scoring import box_iou, normalize_answer, normalize_vqa_answer, score_prediction
from qwen_smope.train import (all_gather_objects, arguments, eval_dataset, record_dataset,
                              routing_scan_indices, save_checkpoint, train_dataset, train_epoch)


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

    def __call__(self, text, images, padding, truncation, return_tensors,
                 max_length=None, **kwargs):
        encoded = []
        for value in text:
            ids = []
            for token in value.split():
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary) + 3
                ids.append(self.vocabulary[token])
            encoded.append(ids[:max_length] if truncation and max_length is not None else ids)
        width = max(map(len, encoded))
        # 真实 processor 支持 pad_to_multiple_of；这里如实模拟，否则关于分桶宽度
        # 的断言毫无意义
        multiple = kwargs.get("pad_to_multiple_of")
        if multiple:
            width = -(-width // multiple) * multiple
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for row, ids in enumerate(encoded):
            start = width - len(ids) if self.tokenizer.padding_side == "left" else 0
            input_ids[row, start:start + len(ids)] = torch.tensor(ids)
            attention[row, start:start + len(ids)] = 1
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
    def test_eval_collator_left_pads_batches_and_restores_training_padding(self):
        processor = FakeProcessor()
        collator = CoINEvalCollator(processor, max_length=32, max_visual_tokens=4,
                                    fetcher=ImageFetcher("mount"), bucket=1)
        short = CoINSample("short", "GQA", None,
                           ({"role": "user", "text": "one"},), "a", ("a",))
        long = CoINSample("long", "GQA", None,
                          ({"role": "user", "text": "one two three"},), "a", ("a",))
        batch = collator([short, long])
        mask = batch["inputs"]["attention_mask"]
        self.assertEqual(processor.tokenizer.padding_side, "right")
        self.assertEqual(mask[0, 0].item(), 0)
        self.assertEqual(mask[0, -1].item(), 1)
        self.assertTrue(torch.equal(batch["router_mask"], mask.bool()))

    def test_limited_route_scan_is_seeded_random_and_source_ordered(self):
        first = routing_scan_indices(100, 12, 42, "GQA", "train")
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 12)
        self.assertNotEqual(first, list(range(12)))
        self.assertEqual(first, routing_scan_indices(100, 12, 42, "GQA", "train"))
        self.assertNotEqual(first, routing_scan_indices(100, 12, 43, "GQA", "train"))
        larger = routing_scan_indices(100, 30, 42, "GQA", "train")
        self.assertTrue(set(first).issubset(larger))
        self.assertEqual(routing_scan_indices(5, 0, 42, "GQA", "train"), list(range(5)))

    def test_missing_images_are_audited_but_transient_errors_abort_the_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "Instructions_Qwen" / "OCRVQA"
            directory.mkdir(parents=True)
            rows = [{"id": "missing", "conversations": [
                {"from": "user", "value": "question"},
                {"from": "assistant", "value": "answer"}]}]
            (directory / "train.json").write_text(json.dumps(rows), encoding="utf-8")
            dataset = CoINDataset(root, "OCRVQA", "train", verify_images=False)
            signature = {"format_version": 1, "task": "OCRVQA", "split": "train"}

            class MissingCollator:
                def measure(self, sample):
                    raise ImageUnavailableError("OCRVQA/missing.jpg", 404)

            manifest = build_filter_manifest(
                dataset, MissingCollator(), signature, root / "missing.json")
            row = manifest["excluded_samples"][0]
            self.assertEqual(row["reason"], "image_unavailable")
            self.assertEqual(row["http_status"], 404)
            self.assertIsNone(row["prompt_len"])

            class TransientCollator:
                def measure(self, sample):
                    raise TimeoutError("temporary WebDAV timeout")

            with self.assertRaises(TimeoutError):
                build_filter_manifest(
                    dataset, TransientCollator(), signature, root / "transient.json")
            self.assertFalse((root / "transient.json").exists())

    def test_filter_shards_merge_byte_equivalent_to_single_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "Instructions_Qwen" / "VQAv2"
            directory.mkdir(parents=True)
            rows = [{"id": str(index), "conversations": [
                {"from": "user", "value": "long " * (20 if index == 2 else 1)},
                {"from": "assistant", "value": "yes"}]}
                    for index in range(6)]
            (directory / "train.json").write_text(json.dumps(rows), encoding="utf-8")
            dataset = CoINDataset(root, "VQAv2", "train", verify_images=False)
            collator = CoINCollator(
                FakeProcessor(), max_length=8, max_visual_tokens=4,
                fetcher=ImageFetcher("mount"), bucket=1)
            signature = filter_signature(
                dataset, root, max_length=8, max_visual_tokens=4, length_bucket=1,
                image_transport="mount", image_base_url=None)
            single_path, merged_path = root / "single.json", root / "merged.json"
            build_filter_manifest(dataset, collator, signature, single_path)
            shards = [
                build_filter_shard(dataset, collator, signature, 0, 3, root / "part-0.json"),
                build_filter_shard(dataset, collator, signature, 3, 6, root / "part-1.json"),
            ]
            merge_filter_shards(dataset, signature, shards, merged_path)
            self.assertEqual(single_path.read_bytes(), merged_path.read_bytes())

    def test_http_fetch_retries_transient_status_but_not_permanent_404(self):
        class Response:
            def __init__(self, status, body=b""):
                self.status, self.body = status, body

            def read(self):
                return self.body

        class Connection:
            def __init__(self, response):
                self.response = response

            def request(self, method, url):
                self.requested = (method, url)

            def getresponse(self):
                return self.response

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = CoINSample("x", "OCRVQA", root / "cl_dataset" / "x.jpg",
                                ({"role": "user", "text": "q"},), "a", ("a",))
            fetcher = ImageFetcher("http", "http://example/base/", root,
                                   timeout_seconds=1, max_attempts=2)
            with patch.object(fetcher, "_connection",
                              return_value=Connection(Response(404))) as connection_mock:
                with self.assertRaises(ImageUnavailableError):
                    fetcher(sample)
                self.assertEqual(connection_mock.call_count, 1)

            image = Image.new("RGB", (2, 2), "red")
            encoded = BytesIO()
            image.save(encoded, format="PNG")
            connections = [Connection(Response(503)), Connection(Response(200, encoded.getvalue()))]
            with patch.object(fetcher, "_connection", side_effect=connections), \
                    patch("qwen_smope.data.time.sleep"):
                self.assertEqual(fetcher(sample).size, (2, 2))

    def test_full_split_filter_precedes_training_prefix_and_records_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            directory = root / "Instructions_Qwen" / "VQAv2"
            directory.mkdir(parents=True)
            rows = []
            for sample_id, question in [
                    ("524377", "too " * 20), ("good-1", "short"),
                    ("good-2", "shorter"), ("good-3", "last")]:
                rows.append({"id": sample_id, "conversations": [
                    {"from": "user", "value": question},
                    {"from": "assistant", "value": "yes"}]})
            (directory / "train.json").write_text(json.dumps(rows), encoding="utf-8")
            processor = FakeProcessor()
            collator = CoINCollator(processor, max_length=8, max_visual_tokens=4,
                                    fetcher=ImageFetcher("mount"), bucket=1)
            dataset = CoINDataset(root, "VQAv2", "train", verify_images=False)
            manifest, cache_path, built = ensure_filter_manifest(
                dataset, collator, root / "cache", root,
                max_length=8, max_visual_tokens=4, length_bucket=1,
                image_transport="mount", image_base_url=None)
            self.assertTrue(built)
            self.assertEqual(manifest["excluded_samples"][0]["sample_id"], "524377")
            self.assertGreater(manifest["excluded_samples"][0]["prompt_len"], 8)
            self.assertGreater(manifest["excluded_samples"][0]["answer_len"], 0)

            # Filtering is applied to all four records before taking two valid samples.
            dataset.apply_training_filter(manifest["valid_indices"], 2, manifest)
            self.assertEqual([sample.sample_id for sample in dataset.samples], ["good-1", "good-2"])
            dataset.filter_manifest = manifest_for_run(manifest, cache_path, len(dataset), False)
            output = root / "output"
            record_dataset(output, dataset, "train", rank=0)
            recorded = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
            details = recorded["train:VQAv2:train"]["training_filter"]
            self.assertEqual(details["sampled_examples"], 2)
            self.assertEqual(details["excluded_samples"][0]["answer_len"], 1)

            fresh = CoINDataset(root, "VQAv2", "train", verify_images=False)
            again, again_path, rebuilt = ensure_filter_manifest(
                fresh, collator, root / "cache", root,
                max_length=8, max_visual_tokens=4, length_bucket=1,
                image_transport="mount", image_base_url=None)
            self.assertFalse(rebuilt)
            self.assertEqual(again_path, cache_path)
            self.assertEqual(again["signature_sha256"], manifest["signature_sha256"])

            # Source metadata is part of the cache key.  Changing the complete
            # split must create a new cache instead of reusing stale indices.
            rows.append({"id": "good-4", "conversations": [
                {"from": "user", "value": "new short"},
                {"from": "assistant", "value": "yes"}]})
            (directory / "train.json").write_text(json.dumps(rows), encoding="utf-8")
            changed = CoINDataset(root, "VQAv2", "train", verify_images=False)
            changed_manifest, changed_path, changed_built = ensure_filter_manifest(
                changed, collator, root / "cache", root,
                max_length=8, max_visual_tokens=4, length_bucket=1,
                image_transport="mount", image_base_url=None)
            self.assertTrue(changed_built)
            self.assertNotEqual(changed_path, cache_path)
            self.assertEqual(changed_manifest["original_examples"], 5)

            (root / "tokenizer_config.json").write_text(
                '{"revision":"changed"}', encoding="utf-8")
            processor_manifest, processor_path, processor_built = ensure_filter_manifest(
                changed, collator, root / "cache", root,
                max_length=8, max_visual_tokens=4, length_bucket=1,
                image_transport="mount", image_base_url=None)
            self.assertTrue(processor_built)
            self.assertNotEqual(processor_path, changed_path)
            self.assertNotEqual(processor_manifest["signature_sha256"],
                                changed_manifest["signature_sha256"])

    def test_filter_impact_levels_and_full_run_gate(self):
        self.assertEqual(impact_level(0.009), "low")
        self.assertEqual(impact_level(0.01), "material")
        self.assertEqual(impact_level(0.10), "high")
        high = {"task": "VQAv2", "impact": "high", "excluded_rate": 0.10}
        with self.assertRaises(RuntimeError):
            validate_filter_impact(high, mode="full", allow_high_filter_rate=False)
        validate_filter_impact(high, mode="full", allow_high_filter_rate=True)
        validate_filter_impact(high, mode="smoke", allow_high_filter_rate=False)

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

    def test_eval_wraps_original_flat_records_into_qwen_conversations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "cl_dataset" / "GQA" / "images" / "7.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "green").save(image)
            directory = root / "Instructions_Original" / "GQA"
            directory.mkdir(parents=True)
            rows = [{"question_id": "7", "image": "./GQA/images/7.jpg",
                     "text": "Is it overcast?\nAnswer with a single word or phrase.",
                     "answer": "No"}]
            (directory / "test.json").write_text(json.dumps(rows), encoding="utf-8")
            sample = CoINDataset(root, "GQA", "test")[0]
            self.assertEqual(sample.sample_id, "7")
            self.assertEqual(sample.answer, "No")
            self.assertEqual(len(sample.history), 1)
            self.assertEqual(sample.history[0]["role"], "user")
            self.assertEqual(sample.history[0]["text"], rows[0]["text"])
            self.assertTrue(sample.image_path.is_file())
            messages = sample.prompt_messages("image")
            self.assertEqual(messages[0]["content"][0], {"type": "image", "image": "image"})
            self.assertEqual(messages[0]["content"][1]["text"], rows[0]["text"])

    def test_text_only_records_omit_the_image_content_item(self):
        """ScienceQA ships roughly half its questions without an image."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "cl_dataset" / "ScienceQA" / "images" / "test" / "5" / "image.png"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "red").save(image)
            directory = root / "Instructions_Original" / "ScienceQA"
            directory.mkdir(parents=True)
            rows = [{"question_id": "4", "text": "Which figure of speech?\nA. chiasmus\nB. apostrophe",
                     "answer": "B"},
                    {"question_id": "5", "image": "ScienceQA/images/test/5/image.png",
                     "text": "Which colony is shown?", "answer": "B"}]
            (directory / "test.json").write_text(json.dumps(rows), encoding="utf-8")
            missing = CoINDataset(root, "ScienceQA", "test", max_samples=1)
            self.assertIsNone(missing[0].image_path)
            self.assertFalse(missing[0].has_image)
            messages = missing[0].prompt_messages()
            self.assertIsInstance(messages[0]["content"], str)
            self.assertNotIn("image", json.dumps(messages))
            # A text-only batch must reach the processor as images=None, since an
            # empty flat image list raises IndexError.
            collator = CoINCollator(FakeProcessor(), max_length=64)
            self.assertIsNone(collator._batch_images([missing[0]]))
            batch = collator([missing[0]])
            self.assertEqual(int((batch["inputs"]["labels"] != -100).sum()), 1)
            sample = CoINDataset(root, "ScienceQA", "test",
                                 require_images=False)[1]
            self.assertTrue(sample.has_image)
            self.assertIsInstance(sample.prompt_messages("picture")[0]["content"], list)

    def test_textvqa_held_out_split_repairs_the_image_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "cl_dataset" / "TextVQA" / "val_images" / "abc.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "blue").save(image)
            # Instructions_Original points val records at train_images.
            directory = root / "Instructions_Original" / "TextVQA"
            directory.mkdir(parents=True)
            rows = [{"question_id": "1", "image": "TextVQA/train_images/abc.jpg",
                     "text": "what brand of phone?", "answer": "nokia"}]
            (directory / "val.json").write_text(json.dumps(rows), encoding="utf-8")
            # test.json exists but carries neither answers nor images.
            (directory / "test.json").write_text(json.dumps(
                [{"question_id": "2", "image": "TextVQA/test_images/x.jpg", "text": "q"}]), encoding="utf-8")
            sample = CoINDataset(root, "TextVQA", "test")[0]
            self.assertEqual(sample.answer, "nokia")
            self.assertEqual(sample.image_path.name, "abc.jpg")
            self.assertEqual(sample.image_path.parent.name, "val_images")

    def test_flat_records_prefer_normalised_grounding_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "Instructions_Original" / "Grounding"
            directory.mkdir(parents=True)
            rows = [{"question_id": "34136", "text": "Provide the bounding box of: tennis player.",
                     "origin_answer_bbox": "[242.94,155.68,86.22,162.53]",
                     "answer_bbox": "[0.38,0.41,0.51,0.67]"}]
            (directory / "test.json").write_text(json.dumps(rows), encoding="utf-8")
            sample = CoINDataset(root, "Grounding", "test", require_images=False)[0]
            self.assertEqual(sample.references, ("[0.38,0.41,0.51,0.67]",))
            self.assertEqual(score_prediction("Grounding", "[0.38,0.41,0.51,0.67]",
                                              sample.references)["correct"], True)

    def test_eval_subsample_is_seeded_stable_and_not_a_prefix(self):
        records = [{"question_id": str(index), "text": "q", "answer": "a"} for index in range(60)]
        first = subsample_records(records, 12, "coin-eval-42-GQA-test")
        again = subsample_records(records, 12, "coin-eval-42-GQA-test")
        other = subsample_records(records, 12, "coin-eval-42-VQAv2-test")
        ids = [int(row["question_id"]) for row in first]
        self.assertEqual(len(ids), 12)
        self.assertEqual(len(set(ids)), 12)
        self.assertEqual(ids, sorted(ids), "采样结果应保持源文件顺序")
        self.assertEqual([r["question_id"] for r in first], [r["question_id"] for r in again])
        self.assertNotEqual([r["question_id"] for r in first], [r["question_id"] for r in other])
        self.assertNotEqual(ids, list(range(12)), "不能退化成取前 N 条")
        self.assertEqual(len(subsample_records(records, 0, "s")), 60)
        self.assertEqual(len(subsample_records(records, 60, "s")), 60)
        self.assertEqual(len(subsample_records(records, 999, "s")), 60)

    def test_eval_dataset_seed_is_independent_of_the_run_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "Instructions_Original" / "GQA"
            directory.mkdir(parents=True)
            rows = [{"question_id": str(index), "text": f"q{index}", "answer": "No"}
                    for index in range(40)]
            (directory / "test.json").write_text(json.dumps(rows), encoding="utf-8")
            first = CoINDataset(root, "GQA", "test", 8, sample_seed="coin-eval-42-GQA-test")
            again = CoINDataset(root, "GQA", "test", 8, sample_seed="coin-eval-42-GQA-test")
            self.assertEqual([s.sample_id for s in first.samples],
                             [s.sample_id for s in again.samples])
            # 无 sample_seed 时保持旧的"取前 N 条"语义（smoke/preflight 依赖）
            prefix = CoINDataset(root, "GQA", "test", 8)
            self.assertEqual([s.sample_id for s in prefix.samples], [str(i) for i in range(8)])

    def test_http_transport_builds_urls_without_touching_the_filesystem(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "Instructions_Original" / "TextVQA"
            directory.mkdir(parents=True)
            rows = [{"question_id": "1", "image": "TextVQA/train_images/abc.jpg",
                     "text": "what brand of phone?", "answer": "nokia"}]
            (directory / "val.json").write_text(json.dumps(rows), encoding="utf-8")
            # verify_images=False 只推导路径, 从不 stat: 用 HTTP 时挂载必须完全不被触碰
            dataset = CoINDataset(root, "TextVQA", "test", verify_images=False)
            sample = dataset[0]
            self.assertFalse((root / "cl_dataset").exists())
            self.assertEqual(sample.image_path.parent.name, "val_images")
            fetcher = ImageFetcher("http", "http://10.0.0.1:4918/base/cl_dataset/", root)
            self.assertEqual(fetcher._url(sample.image_path),
                             "/base/cl_dataset/TextVQA/val_images/abc.jpg")
            with self.assertRaises(ValueError):
                ImageFetcher("nonsense")
            with self.assertRaises(ValueError):
                ImageFetcher("http", None)
            text_only = CoINSample(sample_id="x", task="ScienceQA", image_path=None,
                                   history=(), answer="A", references=("A",))
            # 无图样本任何通道都返回 None, 不产生 {"type": "image"}
            self.assertIsNone(ImageFetcher("mount")(text_only))

    def test_fetcher_and_collator_survive_pickling_for_dataloader_workers(self):
        """DataLoader 在 num_workers>0 时会把 collate_fn 序列化进每个子进程。

        ImageFetcher 持有 threading.local（不可 pickle），必须靠 __getstate__ 剥掉、
        __setstate__ 重建，否则多 worker 训练会直接崩。
        """
        fetcher = ImageFetcher("http", "http://10.0.0.1:4918/base/cl_dataset/", "/tmp/coin-root")
        restored = pickle.loads(pickle.dumps(fetcher))
        self.assertEqual(restored.kind, "http")
        self.assertEqual((restored.host, restored.port), ("10.0.0.1", 4918))
        self.assertEqual(restored.prefix, "/base/cl_dataset/")
        # 还原后线程本地连接是空的, 会在首次使用时重新建立
        self.assertIsNone(getattr(restored._local, "connection", None))
        pickle.loads(pickle.dumps(ImageFetcher("mount")))
        collator = CoINCollator(FakeProcessor(), max_length=64, fetcher=fetcher)
        clone = pickle.loads(pickle.dumps(collator))
        self.assertEqual(clone.fetcher.kind, "http")

    def test_record_dataset_writes_manifest_and_checkpoint_round_trips(self):
        """上游把 save_checkpoint 的函数体误粘进了 record_dataset。

        record_dataset 因此引用未定义的 signature/next_task/matrix/model 而崩溃；
        save_checkpoint 退化成空壳，latest_adapter.pt 从不写入，--resume 永远不可用。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "Instructions_Original" / "GQA"
            directory.mkdir(parents=True)
            (directory / "test.json").write_text(json.dumps(
                [{"question_id": "7", "text": "q", "answer": "No"}]), encoding="utf-8")
            output = root / "out"
            output.mkdir()
            dataset = CoINDataset(root, "GQA", "test", verify_images=False)

            # 只写清单，不碰 checkpoint；不应引用任何未定义名字
            record_dataset(output, dataset, "eval", 0)
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertEqual(manifest["eval:GQA:test"]["examples_loaded"], 1)
            self.assertEqual(manifest["eval:GQA:test"]["source"], str(dataset.path))

            class StubModel:
                def lightweight_state(self):
                    return {"adapter": torch.zeros(2)}

            path = output / "latest_adapter.pt"
            self.assertFalse(path.exists())
            save_checkpoint(path, StubModel(), 3, [[1.0, 2.0]], {"k": "v"}, 0)
            self.assertTrue(path.is_file(), "save_checkpoint 必须真的写出 latest_adapter.pt")
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(state["format_version"], 2)
            self.assertEqual(state["next_task"], 3)
            self.assertEqual(state["matrix"], [[1.0, 2.0]])
            self.assertEqual(state["signature"], {"k": "v"})
            self.assertIn("adapter", state)

    def test_train_and_eval_wrappers_skip_image_probes_under_http_transport(self):
        """复现真实故障：指令集在本地 coin_local，而 cl_dataset 根本不在那里。

        任何绕过 train_dataset()/eval_dataset() 直接构造 CoINDataset 的调用点，
        都会因为 verify_images 默认 True 而去做 stat，然后在缺图的本地根上崩溃。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir = root / "Instructions_Qwen" / "GQA"
            train_dir.mkdir(parents=True)
            (train_dir / "train.json").write_text(json.dumps([{
                "id": "1", "image": "GQA/images/1.jpg",
                "conversations": [
                    {"from": "user", "value": "<img>./cl_dataset/GQA/images/1.jpg</img> Q"},
                    {"from": "assistant", "value": "Yes"}]}]), encoding="utf-8")
            eval_dir = root / "Instructions_Original" / "GQA"
            eval_dir.mkdir(parents=True)
            (eval_dir / "test.json").write_text(json.dumps(
                [{"question_id": "2", "image": "GQA/images/2.jpg",
                  "text": "Q", "answer": "No"}]), encoding="utf-8")
            self.assertFalse((root / "cl_dataset").exists())

            args = SimpleNamespace(
                coin_root=str(root), train_split="train", eval_split="test",
                max_train_samples=0, max_eval_samples=0, eval_sample_seed=42,
                image_transport="http", image_base_url="http://10.0.0.1:4918/cl_dataset/")
            train = train_dataset(args, "GQA")
            evaluation = eval_dataset(args, "GQA")
            self.assertTrue(train[0].has_image)
            self.assertEqual(evaluation[0].answer, "No")
            # mount 通道仍应做存在性校验并因此报错——这正是我们选择显式开关的原因
            with self.assertRaises(FileNotFoundError):
                CoINDataset(root, "GQA", "test")

    def test_textvqa_alias_applies_only_to_the_eval_layer(self):
        """别名修正只属于 Instructions_Original 层。

        该层 val.json 指向 train_images 而文件在 val_images，所以要改写；但 Qwen
        训练层的 train_images 路径本来就是对的——改写会把 34602 条训练样本送去 404。
        这在 HTTP 通道下尤其致命：不做存在性探测，直接取候选[0]。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_dir = root / "Instructions_Qwen" / "TextVQA"
            train_dir.mkdir(parents=True)
            (train_dir / "train.json").write_text(json.dumps([{
                "id": "1", "image": "TextVQA/train_images/abc.jpg",
                "conversations": [
                    {"from": "user",
                     "value": "<img>./cl_dataset/TextVQA/train_images/abc.jpg</img> q"},
                    {"from": "assistant", "value": "nokia"}]}]), encoding="utf-8")
            trained = CoINDataset(root, "TextVQA", "train", verify_images=False)[0]
            self.assertEqual(trained.image_path.parent.name, "train_images")

            eval_dir = root / "Instructions_Original" / "TextVQA"
            eval_dir.mkdir(parents=True)
            (eval_dir / "val.json").write_text(json.dumps(
                [{"question_id": "2", "image": "TextVQA/train_images/def.jpg",
                  "text": "q", "answer": "nokia"}]), encoding="utf-8")
            evaluated = CoINDataset(root, "TextVQA", "test", verify_images=False)[0]
            self.assertEqual(evaluated.image_path.parent.name, "val_images")

    def test_batch_width_is_bucketed_so_triton_compiles_finitely_many_shapes(self):
        """变长输入会让每个 batch 触发一整族内核编译(实测 >17s/步)。

        把宽度向上对齐到 bucket 后, 形状空间被收敛到有限几个, 编译只发生有限次。
        对齐只追加**右侧 masked padding**, 所以语义不变。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "cl_dataset" / "GQA" / "images" / "1.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "green").save(image)
            directory = root / "Instructions_Original" / "GQA"
            directory.mkdir(parents=True)
            rows = [{"question_id": "1", "image": "./GQA/images/1.jpg", "text": "Q", "answer": "No"},
                    {"question_id": "2", "image": "./GQA/images/1.jpg",
                     "text": "Q " * 200, "answer": "No"}]
            (directory / "test.json").write_text(json.dumps(rows), encoding="utf-8")
            dataset = CoINDataset(root, "GQA", "test", verify_images=False)
            fetcher = ImageFetcher("mount")

            collator = CoINCollator(FakeProcessor(), max_length=512, fetcher=fetcher, bucket=128)
            for sample in dataset.samples:
                batch = collator([sample])
                width = batch["inputs"]["input_ids"].shape[-1]
                self.assertEqual(width % 128, 0, f"宽度 {width} 未对齐到 128")
                # 追加的必须是 masked padding, 否则会污染 loss
                self.assertEqual(int(batch["inputs"]["labels"][0, -1]), -100)

            # bucket=1 时必须退化成旧行为(不额外 pad)
            plain = CoINCollator(FakeProcessor(), max_length=512, fetcher=fetcher, bucket=1)
            self.assertLess(plain([dataset.samples[0]])["inputs"]["input_ids"].shape[-1], 128)

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
    def test_distributed_initialization_adds_a_gloo_object_group(self):
        group = object()
        with patch.object(train_module, "_OBJECT_COLLECTIVE_GROUP", None), \
                patch.object(train_module.dist, "is_gloo_available", return_value=True), \
                patch.object(train_module.dist, "init_process_group") as initialize, \
                patch.object(train_module.dist, "new_group", return_value=group) as new_group:
            self.assertIs(train_module.initialize_distributed(2), group)
        initialize.assert_called_once_with("nccl", timeout=timedelta(minutes=90))
        new_group.assert_called_once_with(
            backend="gloo", timeout=train_module.OBJECT_COLLECTIVE_TIMEOUT)

    def test_object_collective_uses_gloo_group_and_validates_label(self):
        group = object()

        def matching(output, envelope, group=None):
            self.assertIs(group, train_module._OBJECT_COLLECTIVE_GROUP)
            output[:] = [envelope, {"label": envelope["label"], "payload": "rank-1"}]

        with patch.object(train_module, "_OBJECT_COLLECTIVE_GROUP", group), \
                patch.object(train_module.dist, "is_initialized", return_value=True), \
                patch.object(train_module.dist, "all_gather_object", side_effect=matching):
            self.assertEqual(
                all_gather_objects("rank-0", label="eval:8:OCRVQA", world=2),
                ["rank-0", "rank-1"])

        def mismatched(output, envelope, group=None):
            output[:] = [envelope, {"label": "stage:8:memory", "payload": "rank-1"}]

        with patch.object(train_module, "_OBJECT_COLLECTIVE_GROUP", group), \
                patch.object(train_module.dist, "is_initialized", return_value=True), \
                patch.object(train_module.dist, "all_gather_object", side_effect=mismatched):
            with self.assertRaisesRegex(RuntimeError, "order mismatch"):
                all_gather_objects("rank-0", label="eval:8:OCRVQA", world=2)

    def test_train_epoch_reports_input_wait_and_sequence_widths(self):
        class TinyNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, inputs, router_mask, dense=False):
                loss = self.weight.square() * inputs["input_ids"].float().mean()
                zero = loss * 0
                return {"loss": loss, "lm_loss": loss, "router_loss": zero,
                        "old_loss": zero, "balance_loss": zero}

        class Sampler:
            def set_epoch(self, value):
                self.epoch = value

        batches = []
        for samples, width in ((2, 4), (1, 8)):
            ids = torch.ones(samples, width, dtype=torch.long)
            batches.append({"inputs": {"input_ids": ids, "labels": ids.clone()},
                            "router_mask": torch.ones_like(ids, dtype=torch.bool)})
        network = TinyNetwork()
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        args = SimpleNamespace(accumulation=2, clip_grad=1.0, log_every=100, output="unused")
        with patch("qwen_smope.train.event"):
            stats = train_epoch(network, network, batches, Sampler(), optimizer, scheduler,
                                args, torch.device("cpu"), 0, 0, False, 0, 1)
        self.assertEqual(stats["samples_with_ddp_padding"], 3)
        self.assertEqual(stats["optimizer_updates"], 1)
        self.assertGreaterEqual(stats["input_wait_seconds"], 0.0)
        self.assertEqual(stats["sequence_widths"], {
            "4": {"microbatches": 1, "samples": 2},
            "8": {"microbatches": 1, "samples": 1},
        })

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
        args = arguments(["--method", "smope", "--output", "out"], world_size=1)
        self.assertEqual(args.model_path, "pretrained/Qwen3.5-9B")
        self.assertNotIn("Base", args.model_path)
        self.assertEqual(args.accumulation, 4)
        self.assertEqual(args.effective_global_batch_size, 16)
        self.assertEqual(args.workers, 0)
        self.assertEqual(args.routing_scan_batch_size, args.batch_size)
        self.assertTrue(args.gradient_checkpointing)

        distributed = arguments(["--method", "smope", "--output", "out"], world_size=2)
        self.assertEqual(distributed.accumulation, 2)
        full = arguments(["--method", "smope", "--mode", "full", "--output", "out"],
                         world_size=1)
        self.assertEqual(full.workers, 4)
        explicit_scan_batch = arguments([
            "--method", "smope", "--output", "out", "--routing-scan-batch-size", "1"],
            world_size=1)
        self.assertEqual(explicit_scan_batch.routing_scan_batch_size, 1)
        batched_eval = arguments([
            "--method", "smope", "--output", "out", "--eval-batch-size", "4"],
            world_size=1)
        self.assertEqual(batched_eval.eval_batch_size, 4)
        no_checkpointing = arguments([
            "--method", "smope", "--output", "out", "--no-gradient-checkpointing"],
            world_size=1)
        self.assertFalse(no_checkpointing.gradient_checkpointing)
        with self.assertRaises(SystemExit):
            arguments(["--method", "smope", "--output", "out",
                       "--routing-scan-batch-size", "0"], world_size=1)
        with self.assertRaises(SystemExit):
            arguments(["--method", "smope", "--output", "out", "--accumulation", "2"],
                      world_size=1)

    def test_diagnostic_variant_world_contract(self):
        validate_variant_world("single", 1)
        validate_variant_world("multiprocess-independent", 2)
        validate_variant_world("ddp-same-batch", 2)
        validate_variant_world("ddp-sharded", 2)
        with self.assertRaises(ValueError):
            validate_variant_world("single", 2)
        with self.assertRaises(ValueError):
            validate_variant_world("ddp-sharded", 1)
        args = SimpleNamespace(batch_size=4, accumulation=4, global_batch_size=16)
        self.assertEqual(validate_global_batch(args, 1), 16)
        args.accumulation = 2
        with self.assertRaises(ValueError):
            validate_global_batch(args, 1)

    def test_ddp_comparison_uses_throughput_and_rejects_protocol_mismatch(self):
        def write_trace(directory, rank, ids, wall, global_batch, updates):
            directory.mkdir(parents=True, exist_ok=True)
            rows = [{"type": "microbatch", "sample_ids": ids}]
            rows.append({
                "type": "summary", "rank": rank, "wall_seconds": wall,
                "samples_seen": len(ids), "optimizer_updates": updates,
                "effective_global_batch_size": global_batch,
                "nonfinite_trainable_parameters": 0,
            })
            (directory / f"rank-{rank}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_trace(root / "single", 0, ["a", "b", "c", "d"], 4.0, 16, 1)
            write_trace(root / "dual", 0, ["a", "c"], 2.0, 16, 1)
            write_trace(root / "dual", 1, ["b", "d"], 2.0, 16, 1)
            result = compare_runs(read_run(root / "single"), read_run(root / "dual"), 1.5)
            self.assertTrue(result["protocol_comparable"])
            self.assertTrue(result["passes_speedup_gate"])
            self.assertAlmostEqual(result["useful_sample_throughput_speedup"], 2.0)

            # A faster-looking run with a different global batch is not a valid
            # training-protocol comparison.
            write_trace(root / "mismatch", 0, ["a", "c"], 2.0, 32, 1)
            write_trace(root / "mismatch", 1, ["b", "d"], 2.0, 32, 1)
            mismatch = compare_runs(
                read_run(root / "single"), read_run(root / "mismatch"), 1.5)
            self.assertFalse(mismatch["protocol_comparable"])
            self.assertFalse(mismatch["passes_speedup_gate"])
            self.assertIn("effective global batch mismatch", mismatch["issues"][0])

    def test_parallel_filter_cli_caps_http_concurrency(self):
        args = prepare_arguments(["--scan-workers", "14"])
        self.assertEqual(args.scan_workers, 14)
        with self.assertRaises(SystemExit):
            prepare_arguments(["--scan-workers", "15"])


if __name__ == "__main__":
    unittest.main()
