from __future__ import annotations

import logging
import os

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, default_data_collator

from llm.logging import append_jsonl, utc_now_iso


def _jsonable_metrics(d: dict) -> dict:
    out: dict = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
            continue
        try:
            out[k] = float(v)
        except Exception:
            out[k] = str(v)
    return out


class _RunMetricsCallback(TrainerCallback):
    def __init__(self, *, metrics_path: str, wandb_enabled: bool, is_main_process) -> None:
        self.metrics_path = metrics_path
        self.wandb_enabled = wandb_enabled
        self.is_main_process = is_main_process
        self.logger = logging.getLogger(__name__)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.is_main_process():
            append_jsonl(self.metrics_path, {"ts": utc_now_iso(), "event": "train_begin"})
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not self.is_main_process():
            return control
        payload = {"ts": utc_now_iso(), "step": state.global_step, **_jsonable_metrics(logs)}
        append_jsonl(self.metrics_path, payload)
        if self.wandb_enabled:
            import wandb

            wandb.log(_jsonable_metrics(logs), step=state.global_step)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or not self.is_main_process():
            return control
        payload = {"ts": utc_now_iso(), "step": state.global_step, "event": "evaluate", **_jsonable_metrics(metrics)}
        append_jsonl(self.metrics_path, payload)
        if self.wandb_enabled:
            import wandb

            wandb.log(_jsonable_metrics(metrics), step=state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.is_main_process():
            append_jsonl(self.metrics_path, {"ts": utc_now_iso(), "event": "train_end", "step": state.global_step})
        return control


def run_trainer(
    *,
    args,
    model,
    splits,
    wandb_enabled: bool,
    run_name: str,
    is_main_process,
):
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")

    eval_strategy = "epoch" if args.eval_steps is None else "steps"
    save_strategy = "epoch" if args.save_steps is None else "steps"
    report_to = []

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        fp16=bool(args.fp16) and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        evaluation_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=report_to,
        run_name=run_name,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits.train,
        eval_dataset=splits.val,
        tokenizer=splits.tokenizer,
        data_collator=default_data_collator,
        callbacks=[
            _RunMetricsCallback(metrics_path=metrics_path, wandb_enabled=wandb_enabled, is_main_process=is_main_process)
        ],
    )
    trainer.train()

    if is_main_process():
        save_name = "adapter" if getattr(model, "peft_config", None) is not None else "model"
        model.save_pretrained(os.path.join(out_dir, save_name))
        splits.tokenizer.save_pretrained(out_dir)
