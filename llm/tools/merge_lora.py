from __future__ import annotations

import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m llm.tools.merge_lora")
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="checkpoints/merged")
    args = p.parse_args(argv)

    adapter_dir = args.adapter_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config = PeftConfig.from_pretrained(adapter_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map=None,
    )

    adapter_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    adapter_weights = load_file(adapter_path) if os.path.exists(adapter_path) else None

    model = PeftModel.from_pretrained(base_model, adapter_dir, adapter_state_dict=adapter_weights)
    merged_model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
