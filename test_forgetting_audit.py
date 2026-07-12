import torch
from torch import nn

from learners.prompt import OnePrompt as OnePromptLearner
from models.vit import Attention
from trainer import Trainer


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
