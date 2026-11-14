#!/usr/bin/env python3
"""
llm/tools/inspect_checkpoint.py - Audit and inspect checkpoints & run directories.

Usage:
    python -m llm.tools.inspect_checkpoint --path checkpoints/
    python -m llm.tools.inspect_checkpoint --path runs/
    python -m llm.tools.inspect_checkpoint --path runs/my_run/adapter
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def get_dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(Path(entry.path))
    except Exception:
        pass
    return total


def format_size(bytes_val: int) -> str:
    val = float(bytes_val)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if val < 1024.0:
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} PB"


def inspect_checkpoint(target_path: Path):
    if not target_path.exists():
        print(f"Error: Path does not exist: {target_path}")
        return

    print(f"\n=======================================================")
    print(f" Inspecting Checkpoint / Run: {target_path}")
    print(f" Total Directory Size: {format_size(get_dir_size(target_path))}")
    print(f"=======================================================")

    adapter_config_path = target_path / "adapter_config.json"
    model_config_path = target_path / "config.json"
    args_json_path = target_path / "args.json"

    if adapter_config_path.exists():
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print("\n[LoRA Adapter Detected]")
            print(f" - Base Model: {cfg.get('base_model_name_or_path')}")
            print(f" - LoRA Rank (r): {cfg.get('r')}")
            print(f" - LoRA Alpha: {cfg.get('lora_alpha')}")
            print(f" - Target Modules: {cfg.get('target_modules')}")
            print(f" - Peft Type: {cfg.get('peft_type')}")
        except Exception as e:
            print(f" - Warning reading adapter_config.json: {e}")

    if model_config_path.exists():
        try:
            with open(model_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            print("\n[Base / Full Model Detected]")
            print(f" - Architectures: {cfg.get('architectures') or cfg.get('model_type')}")
            print(f" - Hidden Size: {cfg.get('hidden_size') or cfg.get('n_embd')}")
            print(f" - Num Layers: {cfg.get('num_hidden_layers') or cfg.get('n_layer')}")
            print(f" - Num Heads: {cfg.get('num_attention_heads') or cfg.get('n_head')}")
        except Exception as e:
            print(f" - Warning reading config.json: {e}")

    if args_json_path.exists():
        try:
            with open(args_json_path, "r", encoding="utf-8") as f:
                args = json.load(f)
            print("\n[Training Run Arguments Snapshot]")
            print(f" - Arch: {args.get('arch')}")
            print(f" - Engine: {args.get('engine')}")
            print(f" - Learning Rate: {args.get('learning_rate')}")
            print(f" - Epochs: {args.get('num_train_epochs')}")
        except Exception as e:
            pass

    # List weights files
    weight_files = list(target_path.glob("**/*.safetensors")) + list(target_path.glob("**/*.bin")) + list(target_path.glob("**/*.pt"))
    if weight_files:
        print("\n[Weight Files Found]")
        for wf in weight_files[:10]:
            rel = wf.relative_to(target_path)
            print(f" - {rel} ({format_size(wf.stat().st_size)})")
        if len(weight_files) > 10:
            print(f" ... and {len(weight_files) - 10} more files.")
    else:
        print("\n[No binary weight files (.safetensors, .bin, .pt) found in root.]")

    print("=======================================================\n")


def main():
    p = argparse.ArgumentParser(description="Audit and inspect checkpoints & run directories.")
    p.add_argument("--path", type=str, default="checkpoints", help="Path to checkpoint directory or runs/")
    args = p.parse_args()

    inspect_checkpoint(Path(args.path))


if __name__ == "__main__":
    main()
