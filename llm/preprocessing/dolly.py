from __future__ import annotations

import os
from dataclasses import dataclass

import datasets
from transformers import AutoTokenizer


@dataclass(frozen=True)
class TokenizedSplits:
    train: datasets.Dataset
    val: datasets.Dataset
    test: datasets.Dataset
    tokenizer: object


def _format_dolly_example(x: dict) -> dict:
    context_str = f"\n\n### Context:\n{x['context']}" if x.get("context") else ""
    prompt = f"### Instruction:\n{x['instruction']}{context_str}\n\n### Response:\n"
    return {"prompt": prompt, "response": x["response"]}


def _tokenize_supervised(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> dict:
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = prompt + response + tokenizer.eos_token
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]
    prompt_len = min(len(prompt_ids), max_length)

    labels = list(enc["input_ids"])
    for i in range(prompt_len):
        labels[i] = -100

    pad_id = tokenizer.pad_token_id
    for i, token_id in enumerate(enc["input_ids"]):
        if token_id == pad_id:
            labels[i] = -100

    enc["labels"] = labels
    return enc


def _load_dolly_dataset(
    *,
    dataset_name: str,
    dataset_path: str | None,
    cache_dir: str,
) -> datasets.Dataset:
    if dataset_path and os.path.exists(dataset_path):
        return datasets.load_from_disk(dataset_path)
    return datasets.load_dataset(dataset_name, cache_dir=cache_dir)["train"]


def load_dolly_splits(
    *,
    dataset_name: str,
    dataset_path: str | None,
    tokenizer_name_or_path: str,
    max_length: int,
    dataset_cache_dir: str,
    tokenizer_cache_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_fast: bool = False,
) -> TokenizedSplits:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=use_fast,
        cache_dir=tokenizer_cache_dir,
    )

    # Re-map directories to data/preprocessed/<prefix>_<tokenizer_safe>_<max_length>_<train_ratio>_<val_ratio>_<seed>
    tokenizer_safe = tokenizer_name_or_path.replace("/", "_").replace("\\", "_")
    dataset_suffix = dataset_name.split("/")[-1]
    
    preprocessed_dir = os.path.join(
        "data", 
        "preprocessed", 
        f"{dataset_suffix}_{tokenizer_safe}_{max_length}_{train_ratio}_{val_ratio}_{seed}"
    )
    
    train_path = os.path.join(preprocessed_dir, "train")
    val_path = os.path.join(preprocessed_dir, "val")
    test_path = os.path.join(preprocessed_dir, "test")

    # If already preprocessed on disk, load and return immediately
    if os.path.isdir(train_path) and os.path.isdir(val_path) and os.path.isdir(test_path):
        print(f"Loading preprocessed tokenized splits from disk cache: {preprocessed_dir}")
        train_tok = datasets.load_from_disk(train_path)
        val_tok = datasets.load_from_disk(val_path)
        test_tok = datasets.load_from_disk(test_path)
        return TokenizedSplits(train=train_tok, val=val_tok, test=test_tok, tokenizer=tokenizer)

    print(f"Preprocessed splits not found at '{preprocessed_dir}'. Tokenizing and splitting dataset...")
    ds = _load_dolly_dataset(dataset_name=dataset_name, dataset_path=dataset_path, cache_dir=dataset_cache_dir)
    ds = ds.filter(lambda x: bool(x["response"] and x["response"].strip()))
    ds = ds.map(_format_dolly_example, remove_columns=ds.column_names)

    rest_ratio = 1.0 - train_ratio
    if rest_ratio <= 0:
        raise ValueError("train_ratio must be < 1.0")
    if val_ratio <= 0 or val_ratio >= rest_ratio:
        raise ValueError("val_ratio must be in (0, 1-train_ratio)")

    ds_split = ds.train_test_split(test_size=rest_ratio, seed=seed)
    train_ds = ds_split["train"]
    rest = ds_split["test"]
    val_fraction_of_rest = val_ratio / rest_ratio
    val_test = rest.train_test_split(test_size=1.0 - val_fraction_of_rest, seed=seed)
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    def tokenize_row(x: dict) -> dict:
        return _tokenize_supervised(tokenizer, x["prompt"], x["response"], max_length)

    train_tok = train_ds.map(tokenize_row, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize_row, remove_columns=val_ds.column_names)
    test_tok = test_ds.map(tokenize_row, remove_columns=test_ds.column_names)

    # Save splits to local disk cache
    print(f"Saving preprocessed tokenized splits to disk cache: {preprocessed_dir}")
    os.makedirs(preprocessed_dir, exist_ok=True)
    train_tok.save_to_disk(train_path)
    val_tok.save_to_disk(val_path)
    test_tok.save_to_disk(test_path)

    return TokenizedSplits(train=train_tok, val=val_tok, test=test_tok, tokenizer=tokenizer)
