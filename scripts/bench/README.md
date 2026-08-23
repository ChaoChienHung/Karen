# External Bench Recipes (Tier 3)

These benchmarks are intentionally kept outside `python -m llm.eval` to keep the core evaluation entrypoint stable and cheap (Tier 0/1/2).

All outputs should be stored under:

- `evaluation/results/model_answers/<run_name>/`
- `evaluation/results/judge/<judge_name>/`
- `evaluation/logs/`

## MT-Bench (FastChat)

FastChat is used as the MT-Bench runner and LLM-judge harness.

Install (recommended in a separate venv):

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
```

Print a run-specific command recipe:

```bash
python scripts/bench/mtbench_fastchat.py print_cmds --run_name <run_name> --model_path <hf_id_or_local_path>
```

## AlpacaEval 2

Install:

```bash
pip install alpaca-eval
```

Run using model outputs JSON (must contain `instruction` + `output` per example):

```bash
python scripts/bench/alpacaeval2.py print_cmds --run_name <run_name> --model_outputs evaluation/results/model_answers/<run_name>/alpacaEval2.json
```

AlpacaEval uses API-based judges by default. Only provide keys via environment variables (do not commit keys).
