#!/usr/bin/env python3
"""
scripts/serve_local.py - Local Model Bridge for Karen Web Dashboard.

Provides an SSE (Server-Sent Events) streaming HTTP API for the Karen React/Vite Sandbox.
Allows live interactive testing of Karen checkpoints (runs/<run_name>) or baseline models.

Usage:
    # Run in simulation/demo mode (no GPU/model weights required)
    python scripts/serve_local.py --port 8000

    # Serve a specific Karen training run checkpoint
    python scripts/serve_local.py --run_dir runs/my_run --port 8000

    # Serve a registered Karen architecture
    python scripts/serve_local.py --arch gpt2 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

app = FastAPI(title="Karen Local Model Bridge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer instances (loaded lazily or on startup)
_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None
_GLOBAL_CONFIG = {
    "run_dir": None,
    "arch": None,
    "mock": False,
}


def _load_karen_model(run_dir: str | None, arch: str | None):
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER
    if run_dir or arch:
        try:
            from llm.eval.__main__ import _build_arg_parser, _load_model_and_tokenizer
            args = _build_arg_parser().parse_args([])
            args.run_dir = run_dir
            args.arch = arch
            print(f"[Karen Bridge] Loading model (run_dir={run_dir}, arch={arch})...")
            _GLOBAL_MODEL, _GLOBAL_TOKENIZER = _load_model_and_tokenizer(args)
            print("[Karen Bridge] Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"[Karen Bridge] Warning: Failed to load model ({e}). Falling back to simulation mode.")


async def _mock_stream(prompt: str, system_prompt: str, model_id: str) -> AsyncGenerator[str, None]:
    header = f"**[Karen Local Runner]** Response from `{model_id or 'karen-local'}`\n\n"
    if system_prompt:
        header += f"> *System Prompt Applied:* {system_prompt}\n\n"

    body = (
        f"Thank you for your prompt: *\"{prompt}\"*\n\n"
        "This response is streamed directly through Karen's local SSE streaming bridge. "
        "Karen's unified pipeline seamlessly connects local training runs (`runs/<run_name>/`) "
        "and offline benchmarks with the interactive React Sandbox.\n\n"
        "### Key Local Model Stats\n"
        "- **Pipeline:** Unified Pretrained + Scratch\n"
        "- **Checkpoint Protocol:** PEFT / LoRA & PyTorch State Dict\n"
        "- **Streaming Latency:** < 50ms time-to-first-chunk via SSE\n\n"
        "```python\n"
        "# Connected via scripts/serve_local.py\n"
        "print(\"Hello from Karen Local Model Runner!\")\n"
        "```\n"
    )

    full_text = header + body
    tokens = full_text.split(" ")
    for token in tokens:
        chunk = json.dumps({"text": token + " "})
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.04)

    yield "data: [DONE]\n\n"


async def _generate_stream(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 512,
    model_id: str = "",
) -> AsyncGenerator[str, None]:
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER
    if _GLOBAL_MODEL is None or _GLOBAL_TOKENIZER is None or _GLOBAL_CONFIG["mock"]:
        async for chunk in _mock_stream(prompt, system_prompt, model_id):
            yield chunk
        return

    from transformers import TextIteratorStreamer

    tokenizer = _GLOBAL_TOKENIZER
    model = _GLOBAL_MODEL

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"### System:\n{system_prompt}\n\n### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(full_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_tokens,
        temperature=max(temperature, 1e-4),
        do_sample=temperature > 0.0,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        if new_text:
            chunk = json.dumps({"text": new_text})
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.005)

    yield "data: [DONE]\n\n"


@app.get("/api/local/generate")
@app.get("/stream")
async def local_generate(
    request: Request,
    prompt: str = Query("", description="User prompt"),
    systemPrompt: str = Query("", description="Optional system instruction"),
    temperature: float = Query(0.7, description="Sampling temperature"),
    maxTokens: int = Query(512, description="Max generated tokens"),
    modelId: str = Query("local-karen", description="Target model identifier"),
):
    """
    SSE stream endpoint compatible with the Karen web dashboard.
    """
    generator = _generate_stream(
        prompt=prompt,
        system_prompt=systemPrompt,
        temperature=temperature,
        max_tokens=maxTokens,
        model_id=modelId,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
@app.get("/api/local/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _GLOBAL_MODEL is not None,
        "run_dir": _GLOBAL_CONFIG["run_dir"],
        "arch": _GLOBAL_CONFIG["arch"],
        "mock": _GLOBAL_CONFIG["mock"],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Karen Local Model Bridge for Web Dashboard")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    p.add_argument("--port", type=int, default=8000, help="Port to listen on")
    p.add_argument("--run_dir", type=str, default=None, help="Path to runs/<run_name> checkpoint")
    p.add_argument("--arch", type=str, default=None, help="Registered arch name (e.g. gpt2, gpt2-dev)")
    p.add_argument("--mock", action="store_true", help="Force simulated streaming mode")
    return p.parse_args()


def main():
    args = parse_args()
    _GLOBAL_CONFIG["run_dir"] = args.run_dir
    _GLOBAL_CONFIG["arch"] = args.arch
    _GLOBAL_CONFIG["mock"] = args.mock

    if not args.mock and (args.run_dir or args.arch):
        _load_karen_model(args.run_dir, args.arch)
    else:
        print("[Karen Bridge] Running in simulation/mock streaming mode.")
        print("[Karen Bridge] To connect a real checkpoint, provide: --run_dir runs/<run_name>")

    print(f"[Karen Bridge] Starting SSE API server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
