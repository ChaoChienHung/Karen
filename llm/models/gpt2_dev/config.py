from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2DevConfig:
    n_positions: int = 1024
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
