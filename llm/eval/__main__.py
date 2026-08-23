from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from llm.preprocessing.dolly import load_dolly_splits


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m llm.eval")
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--arch", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dataset_name", type=str, default="databricks/databricks-dolly-15k")
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--dataset_cache_dir", type=str, default="data/hf_datasets")
    p.add_argument("--tokenizer_cache_dir", type=str, default="model/hf_tokenizers")
    p.add_argument("--model_cache_dir", type=str, default="model/hf_models")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--max_eval_batches", type=int, default=None)

    p.add_argument("--format_eval_set", type=str, default="evaluation/eval_sets/json_format.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=256)

    p.add_argument("--exact_match_eval_set", type=str, default="evaluation/eval_sets/exact_match.jsonl")
    p.add_argument("--no_exact_match", action="store_true", default=False)

    p.add_argument("--smoke_eval_set", type=str, default="evaluation/eval_sets/smoke_prompts.jsonl")
    p.add_argument("--smoke_max_new_tokens", type=int, default=128)
    p.add_argument("--no_smoke", action="store_true", default=False)

    p.add_argument("--out_dir", type=str, default=None)
    return p


def _ensure_tokens(tokenizer):
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def _device() -> torch.device:
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    chunk = text[start : end + 1]
    chunk = re.sub(r",\s*}", "}", chunk)
    chunk = re.sub(r",\s*]", "]", chunk)
    try:
        return json.loads(chunk)
    except Exception:
        return None


def _greedy_generate(model, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None) -> torch.Tensor:
    out = input_ids
    for _ in range(max_new_tokens):
        logits = model(input_ids=out).logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=-1)
        if eos_token_id is not None and int(next_id.item()) == int(eos_token_id):
            break
    return out


def _generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    _ensure_tokens(tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(_device())
    input_len = input_ids.size(-1)

    try:
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    except Exception:
        out_ids = _greedy_generate(model, input_ids, max_new_tokens, tokenizer.eos_token_id)

    gen_ids = out_ids[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _load_model_and_tokenizer(args):
    device = _device()

    if args.run_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.run_dir, cache_dir=args.tokenizer_cache_dir)
        _ensure_tokens(tokenizer)

        adapter_dir = os.path.join(args.run_dir, "adapter")
        model_dir = os.path.join(args.run_dir, "model")

        if os.path.isdir(adapter_dir):
            from peft import PeftConfig, PeftModel

            peft_cfg = PeftConfig.from_pretrained(adapter_dir)
            base = AutoModelForCausalLM.from_pretrained(
                peft_cfg.base_model_name_or_path,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=args.model_cache_dir,
            )
            model = PeftModel.from_pretrained(base, adapter_dir)
            model.eval()
            return model, tokenizer

        if os.path.isdir(model_dir):
            cfg_path = os.path.join(model_dir, "config.json")
            model_type = None
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    model_type = json.load(f).get("model_type")

            if model_type == "gpt2_dev":
                from llm.models.gpt2_dev.model import GPT2LMHeadModel

                model = GPT2LMHeadModel.from_pretrained(model_dir)
                model.to(device)
                model.eval()
                return model, tokenizer

            if model_type == "llama_dev":
                from llm.models.llama_dev.model import LlamaForCausalLM

                model = LlamaForCausalLM.from_pretrained(model_dir)
                model.to(device)
                model.eval()
                return model, tokenizer

            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=args.model_cache_dir,
            )
            model.eval()
            return model, tokenizer

        raise ValueError("run_dir must contain adapter/ or model/")

    if not args.arch:
        raise ValueError("--arch is required when --run_dir is not set")

    from llm.models.registry import default_tokenizer_name as default_tokenizer_name_for_arch

    tokenizer_name = default_tokenizer_name_for_arch(args.arch, None)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=args.tokenizer_cache_dir, use_fast=args.arch.startswith("gpt2"))
    _ensure_tokens(tokenizer)

    if args.arch == "gpt2":
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=args.model_cache_dir,
        )
        model.eval()
        return model, tokenizer

    raise ValueError("for base eval without run_dir, only --arch gpt2 is supported for now")


def _compute_val_metrics(args, model, tokenizer) -> dict:
    from llm.models.registry import default_tokenizer_name as default_tokenizer_name_for_arch

    tokenizer_name_or_path = getattr(tokenizer, "name_or_path", None)
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = default_tokenizer_name_for_arch(args.arch, None) if args.arch else args.dataset_name
    splits = load_dolly_splits(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_length=args.max_length,
        dataset_cache_dir=args.dataset_cache_dir,
        tokenizer_cache_dir=args.tokenizer_cache_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
        use_fast=False,
    )

    ds = splits.val.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False)

    loss_sum = 0.0
    token_count = 0
    batch_count = 0

    device = _device()
    model.eval()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss.detach().float().item()
            n = int((labels != -100).sum().item())
            loss_sum += loss * max(n, 1)
            token_count += n
            batch_count += 1
            if args.max_eval_batches is not None and batch_count >= args.max_eval_batches:
                break

    val_loss = loss_sum / max(token_count, 1)
    val_ppl = float(math.exp(val_loss)) if val_loss < 50 else float("inf")
    return {
        "val_loss_masked": val_loss,
        "val_ppl_masked": val_ppl,
        "eval_tokens": token_count,
        "eval_batches": batch_count,
    }


def _load_format_eval_set(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _format_eval(args, model, tokenizer, out_dir: str) -> dict:
    items = _load_format_eval_set(args.format_eval_set)
    eval_path = os.path.join(out_dir, "instruction_eval.jsonl")

    ok = 0
    total = 0

    with open(eval_path, "w", encoding="utf-8") as wf:
        for item in items:
            prompt = item["prompt"]
            required = item.get("required_keys", [])
            gen = _generate_text(model, tokenizer, prompt, args.max_new_tokens)
            obj = _extract_json(gen)
            success = obj is not None and all(k in obj for k in required)
            ok += int(success)
            total += 1
            wf.write(
                json.dumps(
                    {
                        "id": item.get("id"),
                        "prompt": prompt,
                        "required_keys": required,
                        "output": gen,
                        "parsed": obj,
                        "success": success,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {"format_success_rate": ok / max(total, 1), "format_total": total}


def _normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _load_exact_match_eval_set(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _exact_match_eval(args, model, tokenizer, out_dir: str) -> dict:
    items = _load_exact_match_eval_set(args.exact_match_eval_set)
    eval_path = os.path.join(out_dir, "exact_match_eval.jsonl")

    ok = 0
    total = 0

    with open(eval_path, "w", encoding="utf-8") as wf:
        for item in items:
            prompt = item["prompt"]
            expected = item["answer"]
            gen = _generate_text(model, tokenizer, prompt, args.max_new_tokens)
            pred = gen.splitlines()[0].strip()
            success = _normalize_text(pred) == _normalize_text(expected)
            ok += int(success)
            total += 1
            wf.write(
                json.dumps(
                    {
                        "id": item.get("id"),
                        "prompt": prompt,
                        "expected": expected,
                        "output": gen,
                        "prediction": pred,
                        "success": success,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {"exact_match": ok / max(total, 1), "exact_match_total": total}


def _load_smoke_eval_set(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _smoke_eval(args, model, tokenizer, out_dir: str) -> dict:
    items = _load_smoke_eval_set(args.smoke_eval_set)
    out_path = os.path.join(out_dir, "smoke_generations.jsonl")
    total = 0

    with open(out_path, "w", encoding="utf-8") as wf:
        for item in items:
            prompt = item["prompt"]
            gen = _generate_text(model, tokenizer, prompt, args.smoke_max_new_tokens)
            total += 1
            wf.write(
                json.dumps(
                    {
                        "id": item.get("id"),
                        "prompt": prompt,
                        "output": gen,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {"smoke_total": total}


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    set_seed(args.seed)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.run_dir or os.path.join("runs", f"eval_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}")
    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    model, tokenizer = _load_model_and_tokenizer(args)

    metrics = {}
    metrics.update(_compute_val_metrics(args, model, tokenizer))
    metrics.update(_format_eval(args, model, tokenizer, eval_dir))
    if not args.no_exact_match:
        metrics.update(_exact_match_eval(args, model, tokenizer, eval_dir))
    if not args.no_smoke:
        metrics.update(_smoke_eval(args, model, tokenizer, eval_dir))

    with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(eval_dir)


if __name__ == "__main__":
    main()
