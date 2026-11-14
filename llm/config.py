from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "databricks/databricks-dolly-15k"
    dataset_path: str | None = None
    max_length: int = 2048
    dataset_cache_dir: str = "llm/data/hf_datasets"
    tokenizer_cache_dir: str = "checkpoints/hf_tokenizers"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class TrainConfig:
    engine: str = "trainer"
    output_dir: str = "runs"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    fp16: bool = True
    logging_steps: int = 10
    eval_steps: int | None = None
    save_steps: int | None = None
    save_total_limit: int = 2
