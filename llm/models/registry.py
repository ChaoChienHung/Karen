from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm.models.gpt2.config import GPT2PretrainedConfig
from llm.models.gpt2.model import load_model as load_gpt2
from llm.models.gpt2_dev.config import GPT2DevConfig
from llm.models.gpt2_dev.model import build_model as build_gpt2_dev
from llm.models.llama_2_7b.config import Llama2_7bConfig
from llm.models.llama_2_7b.model import load_model as load_llama_2_7b
from llm.models.llama_dev.config import LlamaDevConfig
from llm.models.llama_dev.model import build_model as build_llama_dev


@dataclass(frozen=True)
class ModelSpec:
    config_cls: type
    default_tokenizer_name: str | None
    build: Any


SPECS: dict[str, ModelSpec] = {
    "Llama-2-7b": ModelSpec(config_cls=Llama2_7bConfig, default_tokenizer_name=None, build=load_llama_2_7b),
    "gpt2": ModelSpec(config_cls=GPT2PretrainedConfig, default_tokenizer_name="gpt2", build=load_gpt2),
    "gpt2-dev": ModelSpec(config_cls=GPT2DevConfig, default_tokenizer_name="gpt2", build=build_gpt2_dev),
    "llama-dev": ModelSpec(config_cls=LlamaDevConfig, default_tokenizer_name="meta-llama/Llama-2-7b-hf", build=build_llama_dev),
}


def arches() -> list[str]:
    return list(SPECS.keys())


def default_tokenizer_name(arch: str, model_name_or_path: str | None) -> str:
    spec = SPECS[arch]
    if spec.default_tokenizer_name:
        return spec.default_tokenizer_name
    if model_name_or_path:
        return model_name_or_path
    raise ValueError("tokenizer_name_or_path is required")
