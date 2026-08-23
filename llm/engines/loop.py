from __future__ import annotations

import logging
import os

import torch
from torch.utils.data import DataLoader

from llm.logging import append_jsonl, utc_now_iso


def run_loop(
    *,
    args,
    model,
    splits,
    run_name: str,
    local_rank: int,
    is_main_process,
    wandb_enabled: bool = False,
):
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    logger = logging.getLogger(__name__)

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()

    train_ds = splits.train.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds = splits.val.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.per_device_eval_batch_size, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    use_amp = bool(args.fp16) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    step = 0
    for epoch in range(int(args.num_train_epochs)):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(input_ids=input_ids, labels=labels).loss
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            if is_main_process() and (step + 1) % args.logging_steps == 0:
                train_loss = loss.item() * args.gradient_accumulation_steps
                logger.info("step=%s train_loss=%.6f", step + 1, train_loss)
                append_jsonl(
                    metrics_path,
                    {"ts": utc_now_iso(), "step": step + 1, "epoch": epoch + 1, "train_loss": train_loss},
                )
                if wandb_enabled:
                    import wandb

                    wandb.log({"train_loss": train_loss, "epoch": epoch + 1}, step=step + 1)

            step += 1

        model.eval()
        if is_main_process():
            losses = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        loss = model(input_ids=input_ids, labels=labels).loss
                    losses.append(loss.detach().float().cpu())
            if losses:
                val_loss = torch.stack(losses).mean().item()
                logger.info("epoch=%s val_loss=%.6f", epoch + 1, val_loss)
                append_jsonl(metrics_path, {"ts": utc_now_iso(), "epoch": epoch + 1, "val_loss": val_loss})
                if wandb_enabled:
                    import wandb

                    wandb.log({"val_loss": val_loss, "epoch": epoch + 1}, step=step)
        model.train()

    if is_main_process():
        model.save_pretrained(os.path.join(out_dir, "model"))
        splits.tokenizer.save_pretrained(out_dir)
