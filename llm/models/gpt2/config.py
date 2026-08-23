from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2PretrainedConfig:
    model_name_or_path: str = "gpt2"
    quantize_8bit: bool = False
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("c_attn", "c_proj")
