from __future__ import annotations

import argparse
import os
import textwrap


def _default_answers_path(run_name: str) -> str:
    return os.path.join("evaluation", "results", "model_answers", run_name, "mt-bench.jsonl")


def _default_log_path(run_name: str) -> str:
    return os.path.join("evaluation", "logs", f"mt-bench_{run_name}.log")


def _print_cmds(*, run_name: str, model_path: str, model_id: str | None) -> None:
    model_id = model_id or run_name
    answers_path = _default_answers_path(run_name)
    log_path = _default_log_path(run_name)
    os.makedirs(os.path.dirname(answers_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    msg = f"""
1) Install FastChat (recommended in a separate venv)

   git clone https://github.com/lm-sys/FastChat.git
   cd FastChat
   pip install -e ".[model_worker,llm_judge]"

2) Generate model answers (MT-Bench)

   cd FastChat/fastchat/llm_judge
   python gen_model_answer.py --model-path {model_path} --model-id {model_id}

   # FastChat default output:
   # data/mt_bench/model_answer/{model_id}.jsonl

3) Copy answers into this repo for tracking

   cd <this-repo-root>
   cp FastChat/fastchat/llm_judge/data/mt_bench/model_answer/{model_id}.jsonl {answers_path}

4) Generate judgments (LLM-as-a-judge)

   cd FastChat/fastchat/llm_judge
   export OPENAI_API_KEY=...
   python gen_judgment.py --model-list {model_id} --parallel 2

5) Log outputs

   cd <this-repo-root>
   python -c "import shutil; shutil.copyfile('FastChat/fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_single.jsonl','{log_path}')"
"""
    print(textwrap.dedent(msg).strip())


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_print = sub.add_parser("print_cmds")
    p_print.add_argument("--run_name", type=str, required=True)
    p_print.add_argument("--model_path", type=str, required=True)
    p_print.add_argument("--model_id", type=str, default=None)

    args = p.parse_args()

    if args.cmd == "print_cmds":
        _print_cmds(run_name=args.run_name, model_path=args.model_path, model_id=args.model_id)
        return
    raise RuntimeError("unknown command")


if __name__ == "__main__":
    main()
