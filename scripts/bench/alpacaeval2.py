from __future__ import annotations

import argparse
import os
import textwrap


def _default_judge_dir(judge_name: str) -> str:
    return os.path.join("evaluation", "results", "judge", judge_name, "alpacaeval2")


def _default_log_path(run_name: str) -> str:
    return os.path.join("evaluation", "logs", f"alpacaeval2_{run_name}.log")


def _print_cmds(*, run_name: str, model_outputs: str, judge_name: str, annotators_config: str) -> None:
    judge_dir = _default_judge_dir(judge_name)
    log_path = _default_log_path(run_name)
    os.makedirs(judge_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    msg = f"""
1) Install AlpacaEval (Python >= 3.10 recommended)

   pip install alpaca-eval

2) Prepare model outputs JSON

   Path: {model_outputs}
   Format: a JSON list of objects with keys: instruction, output

3) Run AlpacaEval 2 (evaluates outputs and writes annotations + leaderboard)

   export OPENAI_API_KEY=...
   alpaca_eval --model_outputs "{model_outputs}" --annotators_config "{annotators_config}" --output_path "{judge_dir}"

4) Save a run log

   alpaca_eval --model_outputs "{model_outputs}" --annotators_config "{annotators_config}" --output_path "{judge_dir}" > "{log_path}" 2>&1
"""
    print(textwrap.dedent(msg).strip())


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_print = sub.add_parser("print_cmds")
    p_print.add_argument("--run_name", type=str, required=True)
    p_print.add_argument(
        "--model_outputs",
        type=str,
        default=None,
        help='Path to JSON file containing [{"instruction": ..., "output": ...}, ...]',
    )
    p_print.add_argument("--judge_name", type=str, default="gpt-4o-mini")
    p_print.add_argument("--annotators_config", type=str, default="weighted_alpaca_eval_gpt4_turbo")

    args = p.parse_args()

    if args.cmd == "print_cmds":
        model_outputs = args.model_outputs or os.path.join(
            "evaluation", "results", "model_answers", args.run_name, "alpacaEval2.json"
        )
        _print_cmds(
            run_name=args.run_name,
            model_outputs=model_outputs,
            judge_name=args.judge_name,
            annotators_config=args.annotators_config,
        )
        return
    raise RuntimeError("unknown command")


if __name__ == "__main__":
    main()
