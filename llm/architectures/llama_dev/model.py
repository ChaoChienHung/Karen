from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from llm.architectures.llama_dev.config import LlamaDevConfig


class LlamaConfig(PretrainedConfig):
    model_type = "llama_dev"

    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        rms_norm_eps: float,
        rope_theta: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


def _rope_cache(
    seq_len: int,
    dim: int,
    theta: float,
    device,
    dtype,
):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y = torch.stack((-x2, x1), dim=-1)
    return y.flatten(-2)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[None, None, :, :].repeat(q.size(0), q.size(1), 1, 1)
    sin = sin[None, None, :, :].repeat(q.size(0), q.size(1), 1, 1)
    q2 = (q * cos) + (_rotate_half(q) * sin)
    k2 = (k * cos) + (_rotate_half(k) * sin)
    return q2, k2


class _Attention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = _rope_cache(t, self.head_dim, self.theta, x.device, x.dtype)
        q, k = _apply_rope(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_head * self.head_dim)
        return self.o_proj(y)


class _MLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class _Block(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attn_norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = _Attention(config)
        self.mlp_norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = _MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LlamaForCausalLM(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([_Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **_) -> CausalLMOutputWithCrossAttentions:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)


def build_model(*, config: LlamaDevConfig, vocab_size: int):
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
        )
    )
