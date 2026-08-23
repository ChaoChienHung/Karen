from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime

import torch
import transformers
from transformers import set_seed

from llm.config import DataConfig, TrainConfig
from llm.engines.loop import run_loop
from llm.engines.trainer import run_trainer
from llm.preprocessing.dolly import load_dolly_splits
from llm.logging import append_jsonl, is_wandb_enabled, maybe_finish_wandb, maybe_init_wandb, setup_run_logging, utc_now_iso, write_json
from llm.models.gpt2.config import GPT2PretrainedConfig
from llm.models.gpt2.model import load_model as load_gpt2
from llm.models.gpt2_dev.config import GPT2DevConfig
from llm.models.gpt2_dev.model import build_model as build_gpt2_dev
from llm.models.llama_2_7b.config import Llama2_7bConfig
from llm.models.llama_2_7b.model import load_model as load_llama_2_7b
from llm.models.llama_dev.config import LlamaDevConfig
from llm.models.llama_dev.model import build_model as build_llama_dev
from llm.models.registry import arches as model_arches
from llm.models.registry import default_tokenizer_name as default_tokenizer_name_for_arch


def _is_distributed() -> bool:
    return int(os.getenv("WORLD_SIZE", "1")) > 1


def _local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def _is_main_process() -> bool:
    if not _is_distributed():
        return True
    return int(os.getenv("RANK", "0")) == 0


def _build_arg_parser() -> argparse.ArgumentParser:
    data_defaults = DataConfig()
    train_defaults = TrainConfig()
    llama_2_7b_defaults = Llama2_7bConfig()
    gpt2_pretrained_defaults = GPT2PretrainedConfig()
    gpt2_dev_defaults = GPT2DevConfig()
    llama_dev_defaults = LlamaDevConfig()

    p = argparse.ArgumentParser(prog="python -m llm.train")
    p.add_argument("--arch", type=str, default="Llama-2-7b", choices=model_arches())
    p.add_argument("--engine", type=str, default=train_defaults.engine, choices=["trainer", "loop"])

    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--tokenizer_name_or_path", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default=data_defaults.dataset_name)
    p.add_argument("--dataset_path", type=str, default=data_defaults.dataset_path)

    p.add_argument("--max_length", type=int, default=data_defaults.max_length)
    p.add_argument("--dataset_cache_dir", type=str, default=data_defaults.dataset_cache_dir)
    p.add_argument("--tokenizer_cache_dir", type=str, default=data_defaults.tokenizer_cache_dir)
    p.add_argument("--model_cache_dir", type=str, default="model/hf_models")

    p.add_argument("--train_ratio", type=float, default=data_defaults.train_ratio)
    p.add_argument("--val_ratio", type=float, default=data_defaults.val_ratio)
    p.add_argument("--seed", type=int, default=data_defaults.seed)

    p.add_argument("--output_dir", type=str, default=train_defaults.output_dir)
    p.add_argument("--run_name", type=str, default=None)

    p.add_argument("--per_device_train_batch_size", type=int, default=train_defaults.per_device_train_batch_size)
    p.add_argument("--per_device_eval_batch_size", type=int, default=train_defaults.per_device_eval_batch_size)
    p.add_argument("--gradient_accumulation_steps", type=int, default=train_defaults.gradient_accumulation_steps)
    p.add_argument("--learning_rate", type=float, default=train_defaults.learning_rate)
    p.add_argument("--num_train_epochs", type=int, default=train_defaults.num_train_epochs)
    p.add_argument("--fp16", action="store_true", default=train_defaults.fp16)
    p.add_argument("--logging_steps", type=int, default=train_defaults.logging_steps)
    p.add_argument("--eval_steps", type=int, default=train_defaults.eval_steps)
    p.add_argument("--save_steps", type=int, default=train_defaults.save_steps)
    p.add_argument("--save_total_limit", type=int, default=train_defaults.save_total_limit)

    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="llm-lab")

    p.add_argument("--quantize_8bit", action="store_true", default=llama_2_7b_defaults.quantize_8bit)
    p.add_argument("--no_quantize_8bit", action="store_true", default=False)
    p.add_argument("--use_lora", action="store_true", default=llama_2_7b_defaults.use_lora)
    p.add_argument("--no_lora", action="store_true", default=False)
    p.add_argument("--lora_r", type=int, default=llama_2_7b_defaults.lora_r)
    p.add_argument("--lora_alpha", type=int, default=llama_2_7b_defaults.lora_alpha)
    p.add_argument("--lora_dropout", type=float, default=llama_2_7b_defaults.lora_dropout)
    p.add_argument("--lora_target_modules", type=str, default=",".join(llama_2_7b_defaults.lora_target_modules))

    p.add_argument("--scratch_n_layer", type=int, default=gpt2_dev_defaults.n_layer)
    p.add_argument("--scratch_n_head", type=int, default=gpt2_dev_defaults.n_head)
    p.add_argument("--scratch_d_model", type=int, default=gpt2_dev_defaults.n_embd)
    p.add_argument("--scratch_ffn_mult", type=int, default=4)
    p.add_argument("--llama_rope_theta", type=float, default=llama_dev_defaults.rope_theta)
    p.add_argument("--llama_rms_norm_eps", type=float, default=llama_dev_defaults.rms_norm_eps)

    return p


def _effective_model_name_or_path(args) -> str | None:
    if args.model_name_or_path:
        return args.model_name_or_path
    if args.arch == "Llama-2-7b":
        return Llama2_7bConfig().model_name_or_path
    if args.arch == "gpt2":
        return GPT2PretrainedConfig().model_name_or_path
    return None


def _resolve_tokenizer_name(args) -> str:
    if args.tokenizer_name_or_path:
        return args.tokenizer_name_or_path
    return default_tokenizer_name_for_arch(args.arch, _effective_model_name_or_path(args))


def _build_model_config(args, max_length: int):
    if args.arch == "Llama-2-7b":
        quantize_8bit = bool(args.quantize_8bit) and not bool(args.no_quantize_8bit)
        use_lora = bool(args.use_lora) and not bool(args.no_lora)
        return Llama2_7bConfig(
            model_name_or_path=_effective_model_name_or_path(args) or Llama2_7bConfig().model_name_or_path,
            quantize_8bit=quantize_8bit,
            use_lora=use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=tuple([s for s in args.lora_target_modules.split(",") if s]),
        )

    if args.arch == "gpt2":
        quantize_8bit = bool(args.quantize_8bit) and not bool(args.no_quantize_8bit)
        use_lora = bool(args.use_lora) and not bool(args.no_lora)
        default_cfg = GPT2PretrainedConfig()
        return GPT2PretrainedConfig(
            model_name_or_path=_effective_model_name_or_path(args) or default_cfg.model_name_or_path,
            quantize_8bit=quantize_8bit,
            use_lora=use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=tuple([s for s in args.lora_target_modules.split(",") if s]) or default_cfg.lora_target_modules,
        )

    if args.arch == "gpt2-dev":
        return GPT2DevConfig(
            n_positions=max_length,
            n_embd=args.scratch_d_model,
            n_layer=args.scratch_n_layer,
            n_head=args.scratch_n_head,
        )

    if args.arch == "llama-dev":
        return LlamaDevConfig(
            max_position_embeddings=max_length,
            hidden_size=args.scratch_d_model,
            intermediate_size=args.scratch_d_model * args.scratch_ffn_mult,
            num_hidden_layers=args.scratch_n_layer,
            num_attention_heads=args.scratch_n_head,
            rms_norm_eps=args.llama_rms_norm_eps,
            rope_theta=args.llama_rope_theta,
        )

    raise ValueError(f"unknown arch: {args.arch}")


def _build_model(args, tokenizer):
    config = _build_model_config(args, args.max_length)

    if args.arch == "Llama-2-7b":
        device_map = "auto"
        if _is_distributed():
            device_map = {"": _local_rank()}
        return load_llama_2_7b(
            config=config,
            tokenizer=tokenizer,
            cache_dir=args.model_cache_dir,
            device_map=device_map,
        )

    if args.arch == "gpt2":
        device_map = "auto"
        if _is_distributed():
            device_map = {"": _local_rank()}
        return load_gpt2(
            config=config,
            tokenizer=tokenizer,
            cache_dir=args.model_cache_dir,
            device_map=device_map,
        )

    if args.arch == "gpt2-dev":
        return build_gpt2_dev(config=config, vocab_size=len(tokenizer))

    if args.arch == "llama-dev":
        return build_llama_dev(config=config, vocab_size=len(tokenizer))

    raise ValueError(f"unknown arch: {args.arch}")


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode("utf-8").strip()
        return out or None
    except Exception:
        return None


def _write_run_artifacts(*, args, out_dir: str, run_name: str, is_main_process) -> None:
    os.makedirs(out_dir, exist_ok=True)

    setup_run_logging(
        os.path.join(out_dir, f"train.rank{int(os.getenv('RANK', '0'))}.log"),
        add_stream=is_main_process(),
    )

    logger = logging.getLogger(__name__)
    logger.info("run_name=%s out_dir=%s", run_name, out_dir)

    if not is_main_process():
        return

    write_json(os.path.join(out_dir, "args.json"), {**vars(args), "run_name": run_name, "out_dir": out_dir})

    env = {
        "timestamp_utc": utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "world_size": int(os.getenv("WORLD_SIZE", "1")),
        "rank": int(os.getenv("RANK", "0")),
        "local_rank": int(os.getenv("LOCAL_RANK", "0")),
        "git_commit": _git_commit(),
    }
    write_json(os.path.join(out_dir, "env.json"), env)
    append_jsonl(os.path.join(out_dir, "metrics.jsonl"), {"ts": utc_now_iso(), "event": "run_start", "run_name": run_name})


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    set_seed(args.seed)

    run_name = args.run_name or f"{args.arch}_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = os.path.join(args.output_dir, run_name)
    _write_run_artifacts(args=args, out_dir=out_dir, run_name=run_name, is_main_process=_is_main_process)
    wandb_enabled = is_wandb_enabled(bool(args.wandb))
    maybe_init_wandb(
        wandb_enabled and _is_main_process(),
        project=args.wandb_project,
        name=run_name,
        config={**vars(args)},
    )

    tokenizer_name = _resolve_tokenizer_name(args)
    splits = load_dolly_splits(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        tokenizer_name_or_path=tokenizer_name,
        max_length=args.max_length,
        dataset_cache_dir=args.dataset_cache_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        use_fast=args.arch in {"gpt2", "gpt2-dev"},
    )

    model = _build_model(args, splits.tokenizer)

    if args.engine == "trainer":
        run_trainer(
            args=args,
            model=model,
            splits=splits,
            wandb_enabled=wandb_enabled,
            run_name=run_name,
            is_main_process=_is_main_process,
        )
    else:
        run_loop(
            args=args,
            model=model,
            splits=splits,
            run_name=run_name,
            local_rank=_local_rank(),
            is_main_process=_is_main_process,
            wandb_enabled=wandb_enabled,
        )

    if _is_main_process():
        append_jsonl(os.path.join(out_dir, "metrics.jsonl"), {"ts": utc_now_iso(), "event": "run_end", "run_name": run_name})
    maybe_finish_wandb(wandb_enabled and _is_main_process())


if __name__ == "__main__":
    main()
