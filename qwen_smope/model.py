"""Qwen3.5-native SMoPE and LoRA adapters for generative CoIN training."""
from __future__ import annotations

import math
import time
from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class Experts(nn.Module):
    """Per-query-head K/V experts with prompt-conditioned, fixed generation routes."""

    def __init__(self, heads: int, count: int, dim: int, topk: int = 5,
                 epsilon: float = 0.4, route_mix: float = 0.75,
                 temperature: float = 0.07):
        super().__init__()
        if not 1 <= topk <= count:
            raise ValueError("Require 1 <= topk <= experts")
        if not 0 < route_mix < 1 or temperature <= 0:
            raise ValueError("route_mix must be in (0, 1), temperature must be positive")
        self.topk, self.epsilon = topk, epsilon
        self.pk = nn.Parameter(torch.empty(heads, count, dim))
        self.pv = nn.Parameter(torch.empty(heads, count, dim))
        nn.init.normal_(self.pk, std=0.02)
        nn.init.normal_(self.pv, std=0.02)
        self.route_mix_logit = nn.Parameter(torch.full((heads, 1), math.log(route_mix / (1 - route_mix))))
        self.log_temperature = nn.Parameter(torch.full((heads, 1), math.log(temperature)))
        self.register_buffer("frequency", torch.zeros(heads, count, dtype=torch.long))
        self.register_buffer("used", torch.zeros(heads, count, dtype=torch.bool))
        self.register_buffer("old_pk", torch.zeros(heads, count, dim))
        self.register_buffer("has_old", torch.tensor(False))
        self.last_scores = self.last_labels = self.last_indices = None
        self.frozen_indices = self.frozen_scores = None

    def clear_route(self) -> None:
        self.frozen_indices = self.frozen_scores = None

    @torch.no_grad()
    def next_task(self) -> None:
        self.old_pk.copy_(self.pk)
        self.has_old.fill_(True)
        mean = self.frequency.sum(-1, keepdim=True) / (self.frequency > 0).sum(-1, keepdim=True).clamp_min(1)
        self.used.logical_or_((self.frequency >= mean) & (self.frequency > 0))
        self.clear_route()

    def _query(self, q: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
        if prefix_mask.shape != (q.shape[0], q.shape[2]):
            raise ValueError(f"routing mask {tuple(prefix_mask.shape)} does not match Q {tuple(q.shape)}")
        mask = prefix_mask[:, None, :, None].to(q.dtype)
        mean = (q * mask).sum(2) / mask.sum(2).clamp_min(1)
        positions = torch.arange(q.shape[2], device=q.device).expand(q.shape[0], -1)
        last = positions.masked_fill(~prefix_mask, -1).max(-1).values
        if bool((last < 0).any()):
            raise ValueError("Every sample needs at least one routing-prefix token")
        last_q = q[torch.arange(q.shape[0], device=q.device), :, last, :]
        mix = self.route_mix_logit.sigmoid()[None, :, :]
        return F.normalize((mix * last_q.float() + (1 - mix) * mean.float()), dim=-1)

    def _selected(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = indices.shape[0]
        pk = self.pk.unsqueeze(0).expand(batch, -1, -1, -1)
        pv = self.pv.unsqueeze(0).expand(batch, -1, -1, -1)
        gather = indices[..., None].expand(-1, -1, -1, pk.shape[-1])
        return pk.gather(2, gather), pv.gather(2, gather)

    def select(self, q: torch.Tensor, prefix_mask: torch.Tensor | None,
               dense: bool = False, training: bool = False,
               freeze: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if self.frozen_indices is not None:
            _, pv = self._selected(self.frozen_indices)
            return self.frozen_scores, pv
        if prefix_mask is None:
            raise ValueError("The generation prefill must provide a routing prefix mask")
        route_q = self._query(q, prefix_mask)
        keys = F.normalize(self.pk.float(), dim=-1)
        temperature = self.log_temperature.exp().clamp(0.01, 1.0)[None, :, :]
        scores = torch.einsum("bhd,hed->bhe", route_q, keys) / temperature
        span = scores.detach().amax(-1, keepdim=True) - scores.detach().amin(-1, keepdim=True)
        if training:
            labels = scores - span * self.used.float()[None, :, :] * self.epsilon
        else:
            # Evaluation and the post-task route scan must follow learned scores.
            # Penalizing zero-frequency experts here prevents newly trained
            # experts from ever being observed and can lock routing to task 1.
            labels = scores
        indices = labels.topk(self.topk, dim=-1).indices
        self.last_scores, self.last_labels, self.last_indices = scores, labels.detach(), indices.detach()
        if dense:
            prompt_scores = scores[:, :, None, :]
            pv = self.pv.unsqueeze(0).expand(q.shape[0], -1, -1, -1)
        else:
            pk, pv = self._selected(indices)
            # Use the fixed, prefix-derived score for all prompt and answer tokens.
            prompt_scores = torch.einsum("bhd,bhkd->bhk", route_q, F.normalize(pk.float(), dim=-1))
            prompt_scores = prompt_scores[:, :, None, :] / temperature[:, :, None, :]
        if freeze:
            self.frozen_indices = indices.detach()
            self.frozen_scores = prompt_scores.detach()
        return prompt_scores, pv

    def losses(self, dense: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = self.pk.sum() * 0
        router = old = balance = zero
        if self.last_scores is not None and not dense:
            scores = self.last_scores
            targets = self.last_labels.topk(self.topk, -1).indices
            selected = torch.zeros_like(scores).scatter(-1, targets, 1)
            span = scores.detach().amax(-1, True) - scores.detach().amin(-1, True)
            adjusted = scores + (1 - selected) * self.epsilon * span
            router = -F.log_softmax(adjusted, -1).gather(-1, targets).sum((1, 2)).mean()
            mean_probability = scores.softmax(-1).mean(0)
            balance = scores.shape[-1] * mean_probability.square().sum(-1).mean()
        anchors = F.normalize(self.old_pk.detach().float(), dim=-1)
        current = F.normalize(self.pk.float(), dim=-1)
        targets = (anchors @ anchors.transpose(-1, -2)).topk(self.topk, -1).indices
        logp = F.log_softmax(anchors @ current.transpose(-1, -2), -1)
        per_anchor = -logp.gather(-1, targets).sum(-1)
        active = self.used.to(per_anchor.dtype) * self.has_old.to(per_anchor.dtype)
        old = ((per_anchor * active).sum(-1) / active.sum(-1).clamp_min(1)).sum()
        return router, old, balance


def expert_attention(self, hidden_states, position_embeddings, attention_mask=None,
                     past_key_values=None, **kwargs):
    """Drop-in eager full attention with independent virtual Prompt K/V."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

    if not self.experts_enabled:
        return self.original_forward(hidden_states, position_embeddings, attention_mask,
                                     past_key_values=past_key_values, **kwargs)
    input_shape = hidden_states.shape[:-1]
    q, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    q = self.q_norm(q).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(*input_shape, -1, self.head_dim)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(*input_shape, -1, self.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position")}
        k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
    k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
    ordinary = (q.float() @ k.float().transpose(-1, -2)) * self.scaling
    if attention_mask is not None:
        ordinary = ordinary + attention_mask[..., :q.shape[2], :k.shape[2]]
    prompt_scores, pv = self.expert.select(
        q, self.routing_mask if q.shape[2] > 1 else None,
        dense=self.dense, training=self.training, freeze=self.freeze_routes)
    scores = torch.cat((prompt_scores.expand(-1, -1, q.shape[2], -1), ordinary), -1)
    weights = scores.softmax(-1).to(v.dtype)
    weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
    values = torch.cat((pv.to(v.dtype), v), 2)
    output = (weights @ values).transpose(1, 2).reshape(*input_shape, -1).contiguous()
    output = self.o_proj(output * torch.sigmoid(gate))
    return output, None


class QwenCoINModel(nn.Module):
    def __init__(self, backbone: nn.Module, method: str = "smope", experts: int = 25,
                 topk: int = 5, epsilon: float = 0.4, router_weight: float = 1e-5,
                 old_weight: float = 1e-5, balance_weight: float = 1e-3,
                 route_mix: float = 0.75, route_temperature: float = 0.07):
        super().__init__()
        self.backbone = backbone
        self.method = method
        self.router_weight, self.old_weight, self.balance_weight = router_weight, old_weight, balance_weight
        self.layer_ids: list[int] = []
        if method == "smope":
            for name, module in backbone.named_modules():
                if module.__class__.__name__ == "Qwen3_5Attention" and "language_model" in name:
                    module.expert = Experts(module.config.num_attention_heads, experts, module.head_dim,
                                            topk, epsilon, route_mix, route_temperature)
                    module.original_forward = module.forward
                    module.forward = MethodType(expert_attention, module)
                    module.experts_enabled, module.dense = True, False
                    module.routing_mask, module.freeze_routes = None, False
                    self.layer_ids.append(module.layer_idx)
            if not self.layer_ids:
                raise ValueError("No Qwen3_5Attention layers found; check the Transformers/model version")

    def expert_modules(self) -> list[Experts]:
        return [module for module in self.modules() if isinstance(module, Experts)]

    def clear_routes(self) -> None:
        for expert in self.expert_modules():
            expert.clear_route()

    def _context(self, router_mask: torch.Tensor | None, dense: bool, freeze: bool) -> None:
        for module in self.backbone.modules():
            if hasattr(module, "experts_enabled"):
                module.routing_mask, module.dense, module.freeze_routes = router_mask, dense, freeze

    def forward(self, inputs: dict[str, torch.Tensor], router_mask: torch.Tensor,
                dense: bool = False) -> dict[str, torch.Tensor]:
        self.clear_routes()
        self._context(router_mask, dense, False)
        outputs = self.backbone(**inputs, use_cache=False, return_dict=True)
        base_loss = outputs.loss
        router = old = balance = base_loss * 0
        if self.training:
            for expert in self.expert_modules():
                value = expert.losses(dense)
                router, old, balance = router + value[0], old + value[1], balance + value[2]
        return {"loss": base_loss + router * self.router_weight + old * self.old_weight +
                        balance * self.balance_weight,
                "lm_loss": base_loss, "router_loss": router * self.router_weight,
                "old_loss": old * self.old_weight, "balance_loss": balance * self.balance_weight}

    @torch.no_grad()
    def observe_routes(self, inputs: dict[str, torch.Tensor], router_mask: torch.Tensor) -> None:
        """Populate selected expert indices without materializing full-sequence LM logits."""
        self.clear_routes()
        self._context(router_mask, False, False)
        route_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        self.backbone(**route_inputs, use_cache=False, return_dict=True, logits_to_keep=1)

    @torch.no_grad()
    def generate(self, inputs: dict[str, torch.Tensor], router_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        if kwargs.get("num_beams", 1) != 1:
            raise ValueError("Fixed SMoPE routes currently require num_beams=1")
        self.clear_routes()
        self._context(router_mask, False, True)
        return self.backbone.generate(**inputs, **kwargs)

    def lightweight_state(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        if self.method == "smope":
            selected = {key for key in state if ".expert." in key}
        elif self.method == "lora":
            selected = {key for key in state if "lora_" in key}
        else:
            selected = set()
        return {key: state[key].detach().cpu().clone() for key in sorted(selected)}

    def load_lightweight_state(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.lightweight_state())
        if set(state) != expected:
            missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
            raise ValueError(f"Incompatible adapter state; missing={missing[:5]}, extra={extra[:5]}")
        self.load_state_dict(state, strict=False)


def load_model(path: str, device: torch.device | str, method: str = "smope",
               lora_rank: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
               **kwargs: Any) -> QwenCoINModel:
    import transformers
    from transformers import Qwen3_5ForConditionalGeneration

    if transformers.__version__ != "5.3.0":
        raise RuntimeError("This implementation is pinned to transformers==5.3.0")
    started = time.perf_counter()
    backbone, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="eager", output_loading_info=True)
    loaded = time.perf_counter()
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Incomplete/incompatible Qwen3.5 weights: {info}")
    backbone.requires_grad_(False)
    if method == "lora":
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(backbone, config)
    model = QwenCoINModel(backbone, method=method, **kwargs)
    adapted = time.perf_counter()
    model.to(device)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    moved = time.perf_counter()
    if method in {"smope", "lora"}:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        backbone.config.use_cache = False
    from transformers.models.qwen3_5 import modeling_qwen3_5 as implementation
    model.loading_profile = {
        "pretrained_cpu_seconds": loaded - started,
        "adapter_init_seconds": adapted - loaded,
        "move_to_device_seconds": moved - adapted,
        "linear_attention_kernels": {
            name: getattr(implementation, name, None) is not None
            for name in (
                "causal_conv1d_fn",
                "causal_conv1d_update",
                "chunk_gated_delta_rule",
                "fused_recurrent_gated_delta_rule",
            )
        },
    }
    return model
