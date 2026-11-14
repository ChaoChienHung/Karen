from __future__ import annotations

from typing import Any

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from llm.architectures.gpt2.config import GPT2PretrainedConfig


def load_model(
    *,
    config: GPT2PretrainedConfig,
    tokenizer,
    cache_dir: str,
    device_map: Any,
):
    bnb_config = None
    if config.quantize_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))

    if config.use_lora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=list(config.lora_target_modules),
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model
