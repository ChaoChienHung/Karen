from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


def setup_run_logging(log_file: str, *, add_stream: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if add_stream and not has_stream:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(sh)

    abs_log_file = os.path.abspath(log_file)
    has_file = any(
        isinstance(h, logging.FileHandler) and os.path.abspath(getattr(h, "baseFilename", "")) == abs_log_file
        for h in root.handlers
    )
    if not has_file:
        os.makedirs(os.path.dirname(abs_log_file), exist_ok=True)
        fh = logging.FileHandler(abs_log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(fh)


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_wandb_enabled(flag: bool) -> bool:
    if flag:
        return True
    v = os.getenv("WANDB_ENABLED", "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def maybe_init_wandb(enabled: bool, project: str, name: str | None, config: dict | None) -> None:
    if not enabled:
        return
    import wandb

    wandb.init(project=project, name=name, config=config)


def maybe_finish_wandb(enabled: bool) -> None:
    if not enabled:
        return
    import wandb

    wandb.finish()
