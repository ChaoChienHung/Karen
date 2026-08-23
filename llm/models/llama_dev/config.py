from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LlamaDevConfig:
    max_position_embeddings: int = 2048
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
