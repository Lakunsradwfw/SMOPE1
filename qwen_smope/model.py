"""Per-query-head KV prompt experts for Qwen3.5 full attention.

Classification only: sample-level mean-query routing sees the entire input.
It is deliberately NOT a causal generation/cache implementation.
"""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


class Experts(nn.Module):
    def __init__(self, heads, count, dim, topk=5, epsilon=0.4):
        super().__init__()
        if not 1 <= topk <= count:
            raise ValueError("Require 1 <= topk <= experts")
        self.topk, self.epsilon = topk, epsilon
        self.pk = nn.Parameter(torch.empty(heads, count, dim))
        self.pv = nn.Parameter(torch.empty(heads, count, dim))
        nn.init.uniform_(self.pk)
        nn.init.uniform_(self.pv)
        self.register_buffer("frequency", torch.zeros(heads, count, dtype=torch.long))
        self.register_buffer("used", torch.zeros(heads, count, dtype=torch.bool))
        self.register_buffer("old_pk", torch.zeros(heads, count, dim))
        self.register_buffer("has_old", torch.tensor(False))
        self.last_scores = self.last_labels = self.last_indices = None

    @torch.no_grad()
    def next_task(self):
        self.old_pk.copy_(self.pk)
        self.has_old.fill_(True)
        mean = self.frequency.sum(-1, keepdim=True) / (self.frequency > 0).sum(-1, keepdim=True).clamp_min(1)
        self.used.logical_or_((self.frequency >= mean) & (self.frequency > 0))

    def select(self, q, valid, dense=False, training=False):
        weights = valid[:, None, :, None].to(q.dtype)
        mean_q = (q * weights).sum(2, keepdim=True) / weights.sum(2, keepdim=True).clamp_min(1)
        scores = mean_q.float() @ self.pk.float().transpose(-1, -2)
        span = (scores.detach().amax(-1, keepdim=True) - scores.detach().amin(-1, keepdim=True))
        penalty = self.used.float() * self.epsilon if training else (self.frequency == 0).float() * 2.0
        labels = scores - span * penalty[None, :, None, :]
        indices = labels.topk(self.topk, dim=-1).indices.squeeze(2)
        self.last_scores, self.last_labels, self.last_indices = scores, labels.detach(), indices.detach()
        b = q.shape[0]
        pk, pv = self.pk.unsqueeze(0).expand(b, -1, -1, -1), self.pv.unsqueeze(0).expand(b, -1, -1, -1)
        if not dense:
            gather = indices[..., None].expand(-1, -1, -1, pk.shape[-1])
            pk, pv = pk.gather(2, gather), pv.gather(2, gather)
        return mean_q.float() @ pk.float().transpose(-1, -2), pv

    def losses(self, dense=False):
        # Match the original sum over layers, heads and top-k targets.
        zero = self.pk.sum() * 0
        router, old = zero, zero
        if self.last_scores is not None and not dense:
            scores = self.last_scores.squeeze(2)
            targets = self.last_labels.squeeze(2).topk(self.topk, -1).indices
            selected = torch.zeros_like(scores).scatter(-1, targets, 1)
            span = scores.detach().amax(-1, True) - scores.detach().amin(-1, True)
            adjusted = scores + (1 - selected) * self.epsilon * span
            logp = F.log_softmax(adjusted, -1)
            router = -logp.gather(-1, targets).sum((1, 2)).mean()
        if bool(self.has_old):
            for h in range(self.pk.shape[0]):
                anchors = self.old_pk[h, self.used[h]].detach()
                if anchors.numel():
                    targets = (anchors @ self.old_pk[h].T).topk(self.topk, -1).indices
                    logp = F.log_softmax(anchors @ self.pk[h].T, -1)
                    old = old - logp.gather(-1, targets).sum(-1).mean()
        return router, old


def expert_attention(self, hidden_states, position_embeddings, attention_mask=None,
                     past_key_values=None, **kwargs):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv
    if not self.experts_enabled:
        return self.original_forward(hidden_states, position_embeddings, attention_mask,
                                     past_key_values=past_key_values, **kwargs)
    if past_key_values is not None:
        raise ValueError("SMoPE classification requires use_cache=False")
    b, n, _ = hidden_states.shape
    q, gate = self.q_proj(hidden_states).view(b, n, -1, self.head_dim * 2).chunk(2, -1)
    gate = gate.reshape(b, n, -1)
    q = self.q_norm(q).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(b, n, -1, self.head_dim)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(b, n, -1, self.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
    # Float32 scores/softmax prevent bf16 overflow; selected values keep the native dtype.
    ordinary = (q.float() @ k.float().transpose(-1, -2)) * self.scaling
    if attention_mask is not None:
        ordinary = ordinary + attention_mask[..., :n, :n]
    else:
        causal = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
        ordinary = ordinary.masked_fill(~causal, -torch.inf)
    ordinary = ordinary.masked_fill(~self.valid_tokens[:, None, None, :], -torch.inf)
    prompt_scores, pv = self.expert.select(q, self.valid_tokens, self.dense, self.training)
    scores = torch.cat(((prompt_scores * self.scaling).expand(-1, -1, n, -1), ordinary), -1)
    weights = scores.softmax(-1).to(v.dtype)
    values = torch.cat((pv.to(v.dtype), v), 2)
    out = (weights @ values).transpose(1, 2).reshape(b, n, -1)
    out = self.o_proj(out * torch.sigmoid(gate))
    return out, None


class QwenClassifier(nn.Module):
    def __init__(self, backbone, classes, method="smope", experts=25, topk=5,
                 epsilon=0.4, router_weight=1e-5, old_weight=1e-5):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.method, self.router_weight, self.old_weight = method, router_weight, old_weight
        self.classifier = nn.Linear(backbone.config.text_config.hidden_size, classes)
        self.layer_ids = []
        # Select by type/config, not hardcoded layer indices. Register experts only once.
        for name, module in backbone.named_modules():
            if module.__class__.__name__ == "Qwen3_5Attention" and "language_model" in name:
                if method == "smope":
                    module.expert = Experts(backbone.config.text_config.num_attention_heads,
                                            experts, module.head_dim, topk, epsilon)
                    module.original_forward = module.forward
                    module.forward = MethodType(expert_attention, module)
                    module.experts_enabled, module.dense = True, False
                    self.layer_ids.append(module.layer_idx)
        if method == "smope" and not self.layer_ids:
            raise ValueError("No supported Qwen3_5Attention modules found; check pinned Transformers")

    def expert_modules(self):
        return [m for m in self.backbone.modules() if isinstance(m, Experts)]

    def forward(self, inputs, dense=False):
        valid = inputs["attention_mask"].bool()
        for module in self.backbone.modules():
            if hasattr(module, "experts_enabled"):
                module.valid_tokens, module.dense = valid, dense
        outputs = self.backbone(**inputs, use_cache=False, return_dict=True)
        positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
        last = positions.masked_fill(~valid, -1).max(-1).values
        features = outputs.last_hidden_state[torch.arange(valid.shape[0], device=valid.device), last]
        logits = self.classifier(features.float())
        router, old = logits.sum() * 0, logits.sum() * 0
        for expert in self.expert_modules():
            r, o = expert.losses(dense)
            router, old = router + r, old + o
        return logits, router * self.router_weight, old * self.old_weight, features.float()

    def lightweight_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.startswith("classifier.") or ".expert." in k}

    def load_lightweight_state(self, state):
        expected = set(self.lightweight_state())
        if set(state) != expected:
            raise ValueError("Adapter keys differ: incompatible model/method/expert configuration")
        self.load_state_dict(state, strict=False)


def load_model(path, classes, device, **kwargs):
    import transformers
    from transformers import Qwen3_5Model
    if transformers.__version__ != "5.3.0":
        raise RuntimeError("This adapter is tested against transformers==5.3.0; install requirements-qwen.txt")
    backbone, info = Qwen3_5Model.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager", output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Incomplete/incompatible base weights: {info}")
    model = QwenClassifier(backbone, classes, **kwargs).to(device)
    if model.method == "smope":
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model
