import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from qwen_smope.data import ExactShard, View, check_model
from qwen_smope.model import Experts, QwenClassifier, load_model
from qwen_smope.report import metrics, summarize


def tiny_model(method="smope"):
    from transformers import Qwen3_5Config, Qwen3_5Model
    config = Qwen3_5Config(
        text_config=dict(vocab_size=100, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, layer_types=["linear_attention", "full_attention"] * 2,
                         linear_num_key_heads=2, linear_num_value_heads=4,
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         pad_token_id=0, rope_parameters=dict(rope_type="default", rope_theta=10000.,
                                                            partial_rotary_factor=0.5, mrope_section=[1, 1, 0])),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
                           out_hidden_size=32, num_position_embeddings=16),
        image_token_id=90, vision_start_token_id=91, vision_end_token_id=92, video_token_id=93)
    config._attn_implementation = "eager"
    return QwenClassifier(Qwen3_5Model(config), 6, method=method, experts=5, topk=2)


def tiny_inputs():
    return dict(input_ids=torch.tensor([[1, 91, 90, 92, 5, 6]]),
                attention_mask=torch.ones(1, 6, dtype=torch.long),
                mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0, 0]]),
                image_grid_thw=torch.tensor([[1, 2, 2]]), pixel_values=torch.randn(4, 12))


class ExpertTests(unittest.TestCase):
    def test_batched_old_loss_matches_loop_value_and_gradient(self):
        torch.manual_seed(12)
        for has_old in (False, True):
            e = Experts(3, 5, 8, 2)
            e.old_pk.normal_()
            e.has_old.fill_(has_old)
            # Unequal counts and an empty head catch incorrect normalization.
            e.used[0, :2] = True
            e.used[1, :] = True
            actual = e.losses()[1]
            expected = e.pk.sum() * 0
            if has_old:
                for h in range(3):
                    anchors = e.old_pk[h, e.used[h]]
                    if anchors.numel():
                        targets = (anchors @ e.old_pk[h].T).topk(2, -1).indices
                        logp = F.log_softmax(anchors @ e.pk[h].T, -1)
                        expected = expected - logp.gather(-1, targets).sum(-1).mean()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual, e.pk)[0],
                                       torch.autograd.grad(expected, e.pk)[0])

    def test_topk_padding_and_old_state(self):
        e = Experts(2, 5, 8, 2)
        q = torch.randn(1, 2, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        s1, v1 = e.select(q, mask)
        ix = e.last_indices.clone()
        s2, v2 = e.select(torch.cat([q, torch.randn(1, 2, 2, 8)], 2),
                          torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool))
        torch.testing.assert_close(s1, s2)
        torch.testing.assert_close(v1, v2)
        self.assertTrue(torch.equal(ix, e.last_indices))
        self.assertEqual(tuple(ix.shape), (1, 2, 2))
        e.frequency[:, 0] = 5
        e.next_task()
        self.assertTrue(bool(e.has_old))
        self.assertTrue(bool(e.used[:, 0].all()))
        r, o = e.losses()
        (r + o + v2.sum()).backward()
        self.assertTrue(e.pk.grad.isfinite().all())
        e2 = Experts(2, 5, 8, 2)
        e2.load_state_dict(e.state_dict())
        self.assertTrue(torch.equal(e2.frequency, e.frequency))
        torch.testing.assert_close(e2.old_pk, e.old_pk)

    def test_zero_frequency_not_marked_used(self):
        e = Experts(2, 5, 8, 2)
        e.next_task()
        self.assertFalse(e.used.any())
        _, old = e.losses()
        self.assertTrue(torch.isfinite(old))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_disabled_matches_original(self):
        model = tiny_model().eval()
        inputs = tiny_inputs()
        for m in model.backbone.modules():
            if hasattr(m, "experts_enabled"):
                m.experts_enabled = False
        with torch.no_grad():
            actual = model(inputs)[0]
            expected = model.classifier(model.backbone(**inputs, use_cache=False).last_hidden_state[:, -1].float())
        torch.testing.assert_close(actual, expected)

    def test_eval_skips_losses_without_changing_predictions(self):
        from unittest.mock import patch
        model = tiny_model()
        inputs = tiny_inputs()
        with torch.no_grad():
            expected = model(inputs)[0]
            model.eval()
            with patch.object(Experts, "losses", side_effect=AssertionError("unused eval loss")):
                actual, router, old, _ = model(inputs)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(float(router), 0.)
        self.assertEqual(float(old), 0.)

    def test_multimodal_backward_checkpoint_and_restore(self):
        model = tiny_model()
        inputs = tiny_inputs()
        model.train()
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        for dense in (True, False):
            logits, router, old, _ = model(inputs, dense=dense)
            loss = F.cross_entropy(logits, torch.tensor([2])) + router + old
            loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(p.grad.isfinite().all(), name)
                else:
                    self.assertIsNone(p.grad, name)
            optim.step()
            optim.zero_grad()
        model.eval()
        with torch.no_grad():
            expected = model(inputs)[0]
        adapter = model.lightweight_state()
        self.assertTrue(all(k.startswith("classifier.") or ".expert." in k for k in adapter))
        self.assertLess(sum(v.numel() for v in adapter.values()), sum(p.numel() for p in model.parameters()))
        model.classifier.weight.data.zero_()
        model.load_lightweight_state(adapter)
        with torch.no_grad():
            torch.testing.assert_close(model(inputs)[0], expected)
        with self.assertRaises(ValueError):
            model.load_lightweight_state({})

    def test_head_only_has_no_experts(self):
        model = tiny_model("head_only")
        logits = model(tiny_inputs())[0]
        logits.sum().backward()
        self.assertEqual(model.expert_modules(), [])
        self.assertTrue(all(n.startswith("classifier.") for n, p in model.named_parameters() if p.requires_grad))

    def test_huggingface_conditional_weight_loading(self):
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Model
        config = tiny_model("head_only").backbone.config
        source = Qwen3_5ForConditionalGeneration(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            # Emulate an official multimodal checkpoint, including the unused language head.
            source.save_pretrained(directory, safe_serialization=True)
            loaded = Qwen3_5Model.from_pretrained(directory, local_files_only=True, attn_implementation="eager").eval()
            inputs = tiny_inputs()
            with torch.no_grad():
                torch.testing.assert_close(source.model(**inputs, use_cache=False).last_hidden_state,
                                           loaded(**inputs, use_cache=False).last_hidden_state)
            # Exercise the actual server loader and mixed BF16-base/FP32-adapter backward.
            adapted = load_model(directory, 6, torch.device("cpu"), experts=5, topk=2).train()
            logits, router, old, _ = adapted(inputs)
            (F.cross_entropy(logits, torch.tensor([2])) + router + old).backward()
            self.assertEqual(adapted.backbone.language_model.embed_tokens.weight.dtype, torch.bfloat16)
            self.assertEqual(adapted.classifier.weight.dtype, torch.float32)
            for name, param in adapted.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(param.grad.isfinite().all(), name)

    def test_padded_classifier_matches_single(self):
        model = tiny_model().eval()
        inputs = tiny_inputs()
        padded = dict(inputs)
        padded["input_ids"] = F.pad(inputs["input_ids"], (0, 2))
        padded["attention_mask"] = F.pad(inputs["attention_mask"], (0, 2))
        padded["mm_token_type_ids"] = F.pad(inputs["mm_token_type_ids"], (0, 2))
        with torch.no_grad():
            a, b = model(inputs)[0], model(padded)[0]
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


class DataReportTests(unittest.TestCase):
    def test_exact_shards(self):
        for size in (1, 7, 8):
            shards = [list(ExactShard(range(size), r, 2)) for r in range(2)]
            self.assertEqual(sorted(shards[0] + shards[1]), list(range(size)))
            self.assertFalse(set(shards[0]) & set(shards[1]))

    def test_metrics_and_report(self):
        m = metrics([[80, 0], [70, 90]])
        self.assertEqual(m["final_average_accuracy"], 80)
        self.assertEqual(m["forgetting"], 10)
        self.assertEqual(m["backward_transfer"], -10)
        with tempfile.TemporaryDirectory() as tmp:
            summarize(tmp, [[80], [70, 90]])
            self.assertTrue((Path(tmp) / "report.md").is_file())
            self.assertNotIn("nan", (Path(tmp) / "summary.json").read_text())

    def test_missing_weights_fail_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                check_model(tmp)


if __name__ == "__main__":
    unittest.main()
