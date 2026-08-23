from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Llama2_7bConfig:
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    quantize_8bit: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
