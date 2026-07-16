import json

import torch
from torch import nn

from learners.prompt import OnePrompt as OnePromptLearner
from models.vit import Attention, VisionTransformer
from trainer import Trainer
from utils.audit_metrics import safe_recovery_ratio, vector_direction_metrics
from utils.audit_state import temporary_state_updates
from utils.analyze_expert_interference import (
    build_final_checkpoint_usage_rows,
    build_seed_endpoint_summary,
    build_task_pair_summary,
    build_usage_summaries,
)
from utils.expert_interference import (
    gradient_conflict_summary,
    gradient_pair_metrics,
    stratified_probe_indices,
    tensor_drift_metrics,
    update_harm_metrics,
)
from utils.summarize_forgetting_audit import (
    build_accuracy_rows,
    build_condition_rows,
)
from utils.summarize_expert_usage import summarize_expert_usage


def test_historical_router_replay_accepts_exact_topk_indices():
    torch.manual_seed(7)
    attention = Attention(dim=8, num_heads=2, qkv_bias=True)
    attention.eval()
    inputs = torch.randn(3, 5, 8)
    prompt = [
        torch.randn(3, 2, 4, 4),
        torch.randn(3, 2, 4, 4),
        torch.zeros(2, 4),
    ]

    automatic, scores = attention(
        inputs, prompt=prompt, topk=2, reduce_query=True
    )
    historical = torch.topk(scores[1], k=2, dim=-1).indices.cpu()
    replayed, _ = attention(
        inputs,
        prompt=prompt,
        topk=2,
        reduce_query=True,
        forced_indices=historical,
    )
    assert torch.allclose(automatic, replayed)


def test_expert_state_probe_is_non_invasive_and_has_complete_pool():
    torch.manual_seed(17)
    attention = Attention(dim=8, num_heads=2, qkv_bias=True)
    attention.eval()
    inputs = torch.randn(3, 5, 8)
    prompt = [
        torch.randn(3, 2, 4, 4),
        torch.randn(3, 2, 4, 4),
        torch.zeros(2, 4),
    ]
    ordinary, ordinary_scores = attention(
        inputs, prompt=prompt, topk=2, reduce_query=True
    )
    probed, probed_scores, expert_state = attention(
        inputs,
        prompt=prompt,
        topk=2,
        reduce_query=True,
        return_expert_state=True,
    )
    assert torch.allclose(ordinary, probed)
    assert torch.allclose(ordinary_scores[0], probed_scores[0])
    assert expert_state["response"].shape == (3, 2, 4, 4)
    assert expert_state["query"].shape == (3, 2, 4)
    assert expert_state["indices"].shape == (3, 2, 1, 2)
    assert expert_state["router_probability"].shape == (3, 2, 4)
    assert torch.allclose(
        expert_state["router_probability"].sum(dim=-1), torch.ones(3, 2)
    )


def test_gradient_conflict_excludes_inactive_pairs():
    summary = gradient_conflict_summary(
        [torch.tensor([0.0, 0.0]), torch.tensor([-1.0, 0.0])],
        torch.tensor([1.0, 0.0]),
    )
    assert summary["old_task_pairs"] == 2
    assert summary["valid_gradient_pairs"] == 1
    assert summary["negative_pair_rate"] == 1.0
    assert summary["mean_negative_cosine"] == 1.0


def test_task_pair_conflict_and_update_harm_have_correct_signs():
    conflict = gradient_pair_metrics(
        torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    )
    assert conflict["valid_gradient_pair"]
    assert conflict["cosine"] == -1.0
    assert conflict["negative_cosine"] == 1.0

    harmful_update = update_harm_metrics(
        torch.tensor([1.0, 0.0]), torch.tensor([0.25, 0.0])
    )
    assert harmful_update["first_order_old_loss_change"] == 0.25
    assert harmful_update["predicted_harm"] == 0.25


def test_same_task_split_half_control_can_remove_gradient_noise_baseline():
    cross = gradient_pair_metrics(
        torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    )
    same_old = gradient_pair_metrics(
        torch.tensor([1.0, 0.0]), torch.tensor([0.8, 0.2])
    )
    same_new = gradient_pair_metrics(
        torch.tensor([-1.0, 0.0]), torch.tensor([-0.8, 0.2])
    )
    control = (same_old["negative_cosine"] + same_new["negative_cosine"]) / 2
    assert control == 0.0
    assert cross["negative_cosine"] - control == 1.0


def test_stratified_probe_is_balanced_and_deterministic():
    targets = [0] * 10 + [1] * 10 + [2] * 10
    first = stratified_probe_indices(targets, 12, seed=9)
    second = stratified_probe_indices(targets, 12, seed=9)
    assert first == second
    selected_targets = [targets[index] for index in first]
    assert {class_id: selected_targets.count(class_id) for class_id in range(3)} == {
        0: 4,
        1: 4,
        2: 4,
    }


def test_tensor_drift_reports_angular_and_relative_change():
    metrics = tensor_drift_metrics(
        torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])
    )
    assert abs(metrics["cosine_distance"] - 1.0) < 1e-12
    assert abs(metrics["relative_l2"] - 2**0.5) < 1e-12


def test_fixed_query_response_substitutes_only_current_expert_parameters():
    queries = {0: torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])}
    parameters = {
        (0, 0, 0): {
            "key": torch.tensor([[1.0, 0.0]]),
            "value": torch.tensor([[2.0, 0.0]]),
        },
        (0, 0, 1): {
            "key": torch.tensor([[0.0, 1.0]]),
            "value": torch.tensor([[0.0, 4.0]]),
        },
    }
    responses = Trainer._mechanism_responses_from_fixed_queries(
        queries, parameters
    )
    probability = torch.softmax(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1)
    assert torch.allclose(
        responses[(0, 0, 0)],
        torch.tensor([2.0 * probability[:, 0].mean(), 0.0]),
    )
    assert torch.allclose(
        responses[(0, 0, 1)],
        torch.tensor([0.0, 4.0 * probability[:, 1].mean()]),
    )


def test_task_pair_analysis_uses_shared_experts_and_seed_level_pairs():
    base = {
        "repeat_id": 1,
        "seed": 7,
        "old_task": 1,
        "new_task": 2,
        "valid_cross_task_gradient": True,
        "same_task_control_negative_cosine": 0.1,
        "first_order_old_loss_change": 0.2,
        "predicted_harm": 0.2,
        "incremental_response_cosine_distance": 0.3,
        "cumulative_response_cosine_distance": 0.4,
        "observed_full_model_old_loss_change": 0.5,
        "learning_boundary_forgetting": 0.6,
        "max_history_forgetting": 0.7,
    }
    expert_rows = [
        {
            **base,
            "shared_hard_route": True,
            "cross_task_negative_cosine": 0.5,
            "excess_cross_task_conflict": 0.4,
        },
        {
            **base,
            "shared_hard_route": False,
            "cross_task_negative_cosine": 1.0,
            "excess_cross_task_conflict": 0.9,
        },
    ]
    task_pairs = build_task_pair_summary(expert_rows)
    assert len(task_pairs) == 1
    assert task_pairs[0]["shared_hard_route_units"] == 1
    assert task_pairs[0]["median_excess_cross_task_conflict"] == 0.4
    seed_rows = build_seed_endpoint_summary(task_pairs)
    assert seed_rows[0]["task_pairs"] == 1
    assert seed_rows[0]["primary_median_excess_conflict"] == 0.4


def test_usage_summary_preserves_layer_head_expert_identity():
    usage_rows = []
    for task in (1, 2):
        for layer, selected in ((0, {0, 1}), (1, {1, 2})):
            for expert in (0, 1, 2):
                usage_rows.append(
                    {
                        "repeat_id": 1,
                        "seed": 9,
                        "task": task,
                        "layer": layer,
                        "head": 0,
                        "expert": expert,
                        "samples": 10,
                        "topk": 2,
                        "usage_count": 10 if expert in selected else 0,
                    }
                )
    coordinate_rows, pool_rows, overall_rows = build_usage_summaries(usage_rows)
    assert len(coordinate_rows) == 6
    assert len(pool_rows) == 2
    assert all(row["exact_same_topk_all_tasks"] for row in pool_rows)
    assert all(row["topk_selection_share"] == 1.0 for row in pool_rows)
    assert overall_rows[0]["layer_head_pools"] == 2
    assert overall_rows[0]["coordinate_experts"] == 6
    assert sum(
        row["global_coordinate_selection_share"] for row in coordinate_rows
    ) == 1.0


def test_final_checkpoint_usage_uses_old_task_drift_and_final_task_reference():
    reference_rows = [
        {
            "repeat_id": 1,
            "seed": 4,
            "task": task,
            "layer": 0,
            "head": 0,
            "expert": expert,
            "samples": 10,
            "topk": 1,
            "usage_count": int(expert == task - 1) * 10,
        }
        for task in (1, 2)
        for expert in (0, 1)
    ]
    drift_rows = [
        {
            "repeat_id": 1,
            "seed": 4,
            "old_task": 1,
            "new_task": 2,
            "layer": 0,
            "head": 0,
            "expert": expert,
            "samples": 10,
            "topk": 1,
            "old_current_usage_count": int(expert == 1) * 10,
        }
        for expert in (0, 1)
    ]
    final_rows = build_final_checkpoint_usage_rows(reference_rows, drift_rows)
    assert len(final_rows) == 4
    task_one = {row["expert"]: row["usage_count"] for row in final_rows if row["task"] == 1}
    assert task_one == {0: 0, 1: 10}


def test_identity_logits_replay_matches_historical_forward_at_same_checkpoint():
    torch.manual_seed(11)
    attention = Attention(dim=8, num_heads=2, qkv_bias=True)
    attention.eval()
    inputs = torch.randn(3, 5, 8)
    prompt = [
        torch.randn(3, 2, 4, 4),
        torch.randn(3, 2, 4, 4),
        torch.zeros(2, 4),
    ]
    automatic, scores = attention(inputs, prompt=prompt, topk=2, reduce_query=True)
    indices = torch.topk(scores[1], k=2, dim=-1).indices
    selected_raw_logits = torch.gather(scores[0], -1, indices)
    replayed, _ = attention(
        inputs,
        prompt=prompt,
        topk=2,
        reduce_query=True,
        forced_indices=indices,
        forced_prompt_logits=selected_raw_logits,
    )
    assert torch.allclose(automatic, replayed)


def test_forced_prompt_logits_shape_validation():
    attention = Attention(dim=8, num_heads=2, qkv_bias=True)
    inputs = torch.randn(2, 5, 8)
    prompt = [
        torch.randn(2, 2, 4, 4),
        torch.randn(2, 2, 4, 4),
        torch.zeros(2, 4),
    ]
    _, scores = attention(inputs, prompt=prompt, topk=2, reduce_query=True)
    indices = torch.topk(scores[1], k=2, dim=-1).indices
    try:
        attention(
            inputs,
            prompt=prompt,
            topk=2,
            reduce_query=True,
            forced_indices=indices,
            forced_prompt_logits=torch.zeros(2, 2, 1, 1),
        )
    except ValueError as error:
        assert "prompt logits" in str(error)
    else:
        raise AssertionError("Invalid historical prompt-logit shape was accepted.")


def test_vit_prompt_loss_accumulator_is_device_safe_for_backward():
    model = VisionTransformer(
        img_size=8,
        patch_size=4,
        embed_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2,
    )
    output, prompt_loss, _ = model(torch.randn(2, 3, 8, 8), train=True)
    (output.sum() + prompt_loss.sum()).backward()
    assert model.patch_embed.proj.weight.grad is not None


def test_freeze_filters_key_and_value_independently():
    learner = object.__new__(OnePromptLearner)
    learner.task_count = 1
    learner.config = {
        "audit_freeze_component": "key",
        "audit_freeze_from_task": 2,
    }
    assert not learner._keep_prompt_parameter("e_pk_0_0_0")
    assert learner._keep_prompt_parameter("e_pv_0_0_0")

    learner.config["audit_freeze_component"] = "value"
    assert learner._keep_prompt_parameter("e_pk_0_0_0")
    assert not learner._keep_prompt_parameter("e_pv_0_0_0")

    learner.config["audit_freeze_component"] = "key_value"
    assert not learner._keep_prompt_parameter("e_pk_0_0_0")
    assert not learner._keep_prompt_parameter("e_pv_0_0_0")


def test_key_value_freeze_sets_both_requires_grad_false():
    class TinyPrompt(nn.Module):
        def __init__(self):
            super().__init__()
            self.e_pk_0_0_0 = nn.Parameter(torch.ones(1))
            self.e_pv_0_0_0 = nn.Parameter(torch.ones(1))
            self.other = nn.Parameter(torch.ones(1))

    class TinyCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt = TinyPrompt()

    learner = object.__new__(OnePromptLearner)
    nn.Module.__init__(learner)
    learner.model = TinyCore()
    learner.task_count = 1
    learner.config = {
        "audit_freeze_component": "key_value",
        "audit_freeze_from_task": 2,
        "audit_freeze_until_task": 0,
    }
    learner._configure_audit_freeze_for_current_task()
    assert not learner.model.prompt.e_pk_0_0_0.requires_grad
    assert not learner.model.prompt.e_pv_0_0_0.requires_grad
    assert learner.model.prompt.other.requires_grad
    learner.model.prompt.other.sum().backward()
    assert learner.model.prompt.e_pk_0_0_0.grad is None
    assert learner.model.prompt.e_pv_0_0_0.grad is None


def test_audit_main_epochs_preserves_default_and_supports_override():
    learner = object.__new__(OnePromptLearner)
    learner.config = {"schedule": [80, 0, 20], "audit_main_epochs": 0}
    assert learner._audit_main_epoch_count() == 20
    assert learner._audit_main_epoch_count(epoch_factor=0.5) == 10
    learner.config["audit_main_epochs"] = 17
    assert learner._audit_main_epoch_count() == 17
    assert learner._audit_main_epoch_count(epoch_factor=0.5) == 8


def test_key_freeze_sets_requires_grad_false_and_grad_stays_none():
    class TinyPrompt(nn.Module):
        def __init__(self):
            super().__init__()
            self.e_pk_0_0_0 = nn.Parameter(torch.ones(1))
            self.e_pv_0_0_0 = nn.Parameter(torch.ones(1))

    class TinyCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt = TinyPrompt()

    learner = object.__new__(OnePromptLearner)
    nn.Module.__init__(learner)
    learner.model = TinyCore()
    learner.task_count = 1
    learner.config = {
        "audit_freeze_component": "key",
        "audit_freeze_from_task": 2,
        "audit_freeze_until_task": 0,
    }
    learner._configure_audit_freeze_for_current_task()
    key = learner.model.prompt.e_pk_0_0_0
    value = learner.model.prompt.e_pv_0_0_0
    assert not key.requires_grad
    assert value.requires_grad
    value.sum().backward()
    assert key.grad is None


def test_router_change_rate_is_topk_set_based():
    accumulator = {"changed": 0, "total": 0, "jaccard_sum": 0.0}
    reference = torch.tensor([[[[1, 2]], [[3, 4]]]])
    current = torch.tensor([[[[2, 1]], [[3, 5]]]])
    Trainer._update_route_drift(current, reference, accumulator)
    assert accumulator["changed"] == 1
    assert accumulator["total"] == 2
    assert abs(accumulator["jaccard_sum"] - (1.0 + 1.0 / 3.0)) < 1e-6


def test_classifier_intervention_keeps_old_rows_exact_and_learns_new_rows():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.last = nn.Linear(2, 4)

        def forward(self, inputs, **_):
            return self.last(inputs), torch.zeros(1)

    learner = object.__new__(OnePromptLearner)
    nn.Module.__init__(learner)
    learner.model = TinyModel()
    learner.config = {
        "audit_freeze_component": "classifier",
        "audit_freeze_from_task": 2,
    }
    learner.task_count = 1
    learner.last_valid_out_dim = 2
    learner.valid_out_dim = 4
    learner.cls_mean = {}
    learner.dw_k = torch.ones(5)
    learner.criterion_fn = nn.CrossEntropyLoss(reduction="none")
    learner.optimizer = torch.optim.AdamW(
        learner.model.last.parameters(), lr=0.1, weight_decay=0.1
    )
    old_rows = learner.model.last.weight[:2].detach().clone()
    old_bias = learner.model.last.bias[:2].detach().clone()
    new_rows = learner.model.last.weight[2:].detach().clone()

    learner.update_model(torch.randn(8, 2), torch.tensor([2, 3] * 4))

    assert torch.equal(old_rows, learner.model.last.weight[:2])
    assert torch.equal(old_bias, learner.model.last.bias[:2])
    assert not torch.equal(new_rows, learner.model.last.weight[2:])


def test_classifier_drift_ignores_future_class_rows():
    reference = {
        "weight": torch.zeros(4, 2),
        "bias": torch.zeros(4),
    }
    current = {
        "weight": reference["weight"].clone(),
        "bias": reference["bias"].clone(),
    }
    current["weight"][2:] = 10
    current["bias"][2:] = 10
    distance = Trainer._component_distance(
        Trainer._classifier_rows(current, 2),
        Trainer._classifier_rows(reference, 2),
    )
    assert distance == {"l2": 0.0, "relative_l2": 0.0}


def test_negative_replay_gain_and_zero_forgetting_gap_are_preserved():
    assert safe_recovery_ratio(-2.0, 4.0) == -0.5
    assert torch.isnan(torch.tensor(safe_recovery_ratio(1.0, 0.0)))


def test_gradient_direction_known_harmful_update_is_positive():
    metrics = vector_direction_metrics(
        old_gradient=torch.tensor([1.0, 0.0]),
        new_gradient=torch.tensor([-1.0, 0.0]),
        update=torch.tensor([0.5, 0.0]),
    )
    assert metrics["first_order_old_loss_change"] > 0
    assert metrics["update_old_gradient_cosine"] == 1.0


def test_temporary_component_restore_rolls_back_state():
    module = nn.Linear(2, 2)
    original = module.weight.detach().clone()
    replacement = torch.full_like(original, 7.0)
    with temporary_state_updates(module, {"weight": replacement}):
        assert torch.equal(module.weight, replacement)
    assert torch.equal(module.weight, original)


def test_accuracy_matrix_pairing_exposes_plasticity_and_retention():
    baseline = {
        "run": "baseline",
        "condition": "baseline",
        "args": {"audit_freeze_component": "none"},
        "num_tasks": 2,
        "seeds": [0],
        "matrices": {0: [[90.0, 80.0], [0.0, 85.0]]},
        "global_history": [[90.0], [82.5]],
        "fr_history": [[0.0], [10.0]],
    }
    frozen = {
        "run": "freeze-key",
        "condition": "key",
        "args": {"audit_freeze_component": "key"},
        "num_tasks": 2,
        "seeds": [0],
        "matrices": {0: [[90.0, 85.0], [0.0, 80.0]]},
        "global_history": [[90.0], [82.5]],
        "fr_history": [[0.0], [5.0]],
    }
    accuracy_rows = build_accuracy_rows([baseline, frozen], baseline)
    frozen_task_one = next(
        row
        for row in accuracy_rows
        if row["run"] == "freeze-key"
        and row["eval_task"] == 1
        and row["checkpoint_task"] == 2
    )
    assert frozen_task_one["plasticity_loss"] == 0.0
    assert frozen_task_one["final_retention_gain"] == 5.0

    condition_rows = build_condition_rows([baseline, frozen], baseline, 2.0, 1.0)
    frozen_summary = next(row for row in condition_rows if row["run"] == "freeze-key")
    assert frozen_summary["mean_new_task_plasticity_loss"] == 5.0
    assert "LOWER_FR_WITH_LARGE_PLASTICITY_LOSS" in frozen_summary["interpretation_flags"]


def test_expert_usage_count_conservation(tmp_path):
    audit_dir = tmp_path / "run" / "forgetting_audit"
    snapshot_dir = audit_dir / "repeat-1" / "router_current"
    snapshot_dir.mkdir(parents=True)
    with (audit_dir / "expert_structure.json").open("w", encoding="utf-8") as handle:
        json.dump({"num_experts_per_layer_head": 3}, handle)
    snapshot = {
        "repeat_id": 1,
        "seed": 0,
        "checkpoint_task": 1,
        "eval_task": 1,
        "topk": 2,
        "batches": [
            {
                "targets": torch.tensor([0, 1]),
                "layers": {
                    0: {
                        "indices": torch.tensor([[[[0, 1]]], [[[1, 2]]]])
                    }
                },
            }
        ],
    }
    torch.save(snapshot, snapshot_dir / "checkpoint-1_eval-task-1.pt")
    usage_rows, _ = summarize_expert_usage(tmp_path / "run")
    assert sum(row["usage_count"] for row in usage_rows) == 2 * 2
    assert abs(sum(row["usage_rate"] for row in usage_rows) - 1.0) < 1e-12
