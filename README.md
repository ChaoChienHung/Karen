# Karen

This repository is a personal LLM lab. It supports both pretrained fine‑tuning (HuggingFace + LoRA) and scratch mini‑models (GPT2/Llama) under a unified training entrypoint, sharing the same dataset pipeline.

---

## 🚀 Highlights

- Pretrained path: HuggingFace `from_pretrained` + LoRA (optionally 8-bit).  
- Scratch path: mini GPT2/Llama implemented in PyTorch, trainable with Trainer or a custom loop.  
- Single training entry: `python -m llm.train` with `--arch/--engine` switching.  
- Same dataset pipeline shared across architectures.  
- Wandb is optional (off by default).
- Every run is traceable under `runs/<run_name>/` (args/env/logs/metrics/checkpoints).
- Benchmarking plan and experiment logs: see `docs/dev-notes.md`.

## 🧩 Models (Architectures)

| `--arch` | Type | Size / Default config | Default tokenizer | Notes |
|---|---|---|---|---|
| `Llama-2-7b` | Pretrained (HF) | 7B | from `model_name_or_path` | LoRA + optional 8-bit supported (requires model access + extra deps) |
| `gpt2` | Pretrained (HF) | ~124M | `gpt2` | Good small baseline; supports LoRA if PEFT is installed |
| `gpt2-dev` | Scratch (PyTorch) | `n_layer=8, n_embd=512, n_head=8` | `gpt2` | Fast iterations; useful for pipeline / ablation experiments |
| `llama-dev` | Scratch (PyTorch) | `layers=8, hidden=512, heads=8` | `meta-llama/Llama-2-7b-hf` | Llama-style toy model for architecture experiments |

## 📌 Project Rules

Immutable constraints (non-regression) and collaboration guardrails live in `AGENTS.md`. README focuses on how to run the project.

## 📚 Documentation Map

- `AGENTS.md` — non-negotiables + collaboration contract (what must always remain true)
- `docs/doc-map.md` — single index for all documents under `docs/`
- `docs/` — workflows / architecture / guardrails / checklists / dev notes for team collaboration
- `TODO.md` — task list and prioritization

---

## 📁 Repository Structure

- `llm/` – Main package (clean core for reuse as subrepository)
  - `llm/train.py` – Unified training entry (`python -m llm.train`)
  - `llm/eval/` – Evaluation entry (`python -m llm.eval`)
  - `llm/preprocessing/` – Dataset preprocessing + tokenization
  - `llm/models/` – Models (each model in its own folder, with its own config)
  - `llm/logging.py` – Run logging utilities + optional wandb
  - `llm/tools/` – Utilities (e.g. merge LoRA)
- `evaluation/` – Evaluation datasets and benchmark results (tracked by Git)
  - `evaluation/eval_sets/` – Local offline evaluation question sets (JSONL)
  - `evaluation/results/` – Model outputs and judge annotations (AlpacaEval, MT-Bench)
  - `evaluation/logs/` – Stdout/stderr logs from running external benchmark CLI commands
- `data/` – Local datasets (ignored by git, except `.gitkeep`)
  - `data/raw_data/` – Downloaded raw datasets (e.g., Dolly-15k)
  - `data/preprocessed/` – Cached tokenized dataset splits (generated automatically on first run)
- `scripts/` – Helper scripts (e.g. download datasets, run cluster jobs)
  - `scripts/download/` – Scripts for downloading raw datasets
  - `scripts/bench/` – External benchmark recipes (MT-Bench / AlpacaEval2)
  - `scripts/train/` – Cluster run templates (`sbatch`, `srun`)
- `frontend/` – React/Vite web application (ModelHub, Leaderboard, Articles, Sandbox, Express server, etc.)
- `requirements.txt` – Python dependencies for the project.  

---

## 🔧 Setup & Usage

### Prerequisites

- Python 3.8+ (or compatible version).  
- GPUs with sufficient memory (fine‑tuning LLaMA‑2‑7B typically requires > 20 GB VRAM per GPU, depending on batch size).  
- Access to the base model weights for LLaMA‑2‑7B from Meta (ensure you comply with licensing).  
- The Dolly‑15K dataset (via HuggingFace or Databricks).  

### Installation

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/ChaoChienHung/Fine-TunedLLaMa.git  
   cd Fine-TunedLLaMa  
   ```

2. Create and activate a Python virtual environment (`venv`):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Upgrade pip and install the core dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. *(Optional)* If you plan to fine-tune or evaluate the **LLaMA-2-7B** model (which requires additional PEFT, quantization, and evaluation packages), install the model-specific requirements:
   ```bash
   pip install -r llm/models/llama_2_7b/requirements.txt
   ```
### Configuration

Most settings are passed via CLI flags. The recommended interface is `python -m llm.train --help`.

### Running Training

Download dataset to local disk:
```bash
python scripts/download/download_dolly.py --dataset_name databricks/databricks-dolly-15k
```

Use local dataset (optional):
```bash
python -m llm.train --dataset_path data/raw_data/dolly
```

Pretrained LLaMA (default):
```bash
python -m llm.train --arch Llama-2-7b --engine trainer
```

Pretrained GPT2:
```bash
python -m llm.train --arch gpt2 --engine trainer
```

Dev GPT2 (from scratch):
```bash
python -m llm.train --arch gpt2-dev --engine trainer --scratch_d_model 256 --scratch_n_layer 6 --scratch_n_head 8
```

Dev Llama (from scratch):
```bash
python -m llm.train --arch llama-dev --engine loop --scratch_d_model 256 --scratch_n_layer 6 --scratch_n_head 8
```

Multi‑GPU (DDP, Trainer):
```bash
torchrun --nproc_per_node 2 -m llm.train --arch Llama-2-7b --engine trainer
```

### Outputs / Checkpoints

By default, training outputs are written under `runs/` (override with `--output_dir`, and control folder name with `--run_name`).

- Common run artifacts (both engines)
  - Run config: `runs/<run_name>/args.json`
  - Environment snapshot: `runs/<run_name>/env.json`
  - Logs: `runs/<run_name>/train.rank<RANK>.log` (DDP will write one per rank; rank 0 also prints to stdout)
  - Metrics stream: `runs/<run_name>/metrics.jsonl`
- Trainer engine (`--engine trainer`)
  - Checkpoints: `runs/<run_name>/checkpoint-*`
  - Final weights:
    - LoRA: `runs/<run_name>/adapter/`
    - Non‑LoRA: `runs/<run_name>/model/`
  - Tokenizer: `runs/<run_name>/`
- Custom loop (`--engine loop`)
  - Final weights: `runs/<run_name>/model/`
  - Tokenizer: `runs/<run_name>/`

### Wandb

Wandb is optional and off by default. Enable it with either:

```bash
python -m llm.train ... --wandb --wandb_project llm-lab
```

or:

```bash
WANDB_ENABLED=1 python -m llm.train ...
```

Login / Auth:

```bash
wandb login
```

If you are running on a cluster/non-interactive environment, you can also set the API key via environment variable:

```bash
export WANDB_API_KEY=...
```

Do not commit API keys into this repository.

### Running Evaluation

Evaluate a training run directory:
```bash
python -m llm.eval --run_dir runs/<run_name>
```

Outputs:
- Metrics: `runs/<run_name>/eval/metrics.json`
- Per-example format eval: `runs/<run_name>/eval/instruction_eval.jsonl`
- Per-example exact match eval: `runs/<run_name>/eval/exact_match_eval.jsonl`
- Smoke generations: `runs/<run_name>/eval/smoke_generations.jsonl`
### After Training

You can merge the LoRA weights into the base model for inference using:
```bash
python -m llm.tools.merge_lora --adapter_dir <path-to-adapter> --output_dir model/merged
```

Then use the merged model for inference or deployment.

### Running on a GPU Cluster (Optional / Slurm)

For larger experiments, you can execute training and evaluations on a GPU cluster (e.g. using Slurm and DDP). We provide templates inside `scripts/train/` based on the `skills/cluster_skill.md` guide.

#### 1. Interactive Resource Allocation & Debugging (`srun` / `salloc`)
You can use `salloc` to allocate resources and then run your script, or execute `run_srun.sh` directly for foreground/interactive jobs:
```bash
# Method 1: Foreground execution (requests resources and runs python directly)
./scripts/train/run_srun.sh

# Method 2: Interactive shell allocation
salloc -N1 -n1 --gres=gpu:1 -c 8 --mem=32G -p gpu -t 01:00:00 bash
# Inside the allocated shell:
source venv/bin/activate
python -m llm.train --arch Llama-2-7b --engine trainer --run_name llama_interactive_run
exit
```
See `scripts/train/run_srun.sh` for details on modifying GPU counts or changing hyperparameters.

#### 2. Submitting Batch Jobs (`sbatch`)
To submit a non-interactive background training job to the queue, use:
```bash
sbatch scripts/train/submit_sbatch.sh
```
See `scripts/train/submit_sbatch.sh` for inline guidelines on:
- How to request multi-GPU DDP training (switching to `torchrun`).
- How to tweak hardware configuration like CPU cores (`--cpus-per-task`) and memory (`--mem`).

#### 3. Job Profiling using HDF5
To collect fine-grained hardware metrics (energy, memory, interconnects) during cluster execution, you can pass profiling flags to `srun` or configure your job profile:
```bash
# Profile all task metrics and sample every 10 seconds
srun --profile=all --acctg-freq task=10 ./my_program
```
Once the job completes, combine the profiling step files into a single HDF5 file:
```bash
sh5util -j $SLURM_JOB_ID
```

---

## 📊 Results & Benchmarking

Benchmarking plan and experiment logs are tracked in `docs/dev-notes.md`.

External benchmarks (Tier 3) are run via external tools and saved under `evaluation/` for traceability:

- Recipes: `scripts/bench/`
- Model answers: `evaluation/results/model_answers/<run_name>/`
- Judge outputs: `evaluation/results/judge/<judge_name>/`
- Logs: `evaluation/logs/`

---

## 🖥️ Web Dashboard & Interactive Sandbox (`frontend/`)

Karen includes a modern React + Vite + Tailwind dashboard with:
- **ModelHub**: Explore models, architecture specs, and strengths/weaknesses.
- **Leaderboard**: Side-by-side benchmark score rankings across MMLU, HumanEval, MATH, and GPQA.
- **Articles**: Technical deep dives on LLM scaling, architecture trade-offs, and fine-tuning.
- **Interactive Sandbox**: Multi-model split-view playground with real-time SSE streaming. Compare cloud models (Gemini, Claude, GPT-4o, DeepSeek) with your local Karen trained checkpoints!

### Quick Start (Web Dashboard)

```bash
cd frontend
npm install

# (Optional) If you want to use Gemini API comparisons in Sandbox:
# cp .env.example .env.local and add GEMINI_API_KEY=...

# Start dev server (Express + Vite)
npm run dev
# The dashboard opens at http://localhost:3000
```

### Connecting Local Karen Checkpoints to Sandbox

You can stream generations from your local Karen training runs (`runs/<run_name>/`) or baseline architectures directly into the Sandbox:

```bash
# 1. Start the local streaming runner (FastAPI SSE on port 8000)
python scripts/serve_local.py --run_dir runs/<run_name>

# Or start in simulation/demo mode without weights:
python scripts/serve_local.py --mock

# 2. In the Sandbox, select "Karen Llama-2-7B (LoRA/SFT)" or "Karen GPT-2 Dev (Scratch)"
# The Express server automatically proxies the SSE stream to your local Python runner!
```

---

## 🧠 Why This Matters

Fine‐tuning large language models from scratch is expensive. Using PEFT approaches like LoRA enables:

- Much smaller number of trainable parameters (faster training, less memory).  
- Retaining the majority of the base model while adapting to a new dataset (like Dolly‑15K).  
- A practical workflow to bring base‑model capabilities into a more specialized setting with fewer resources.

---

## ✅ Contribution & Usage Notes

- This is a **practice** project: meant for experimentation / learning.  
- If you adapt or extend this work, please ensure license compliance for LLaMA‑2 and Dolly‑15K.  
- Contributions, issues, enhancements welcome (e.g., alternative datasets, hyperparameter tuning, inference pipelines).  
- For deployment, further steps may be needed (quantization, optimization, serving infrastructure).

---

## 📚 References

- Meta’s LLaMA‑2 model (7B variant) – check licensing.  
- Databricks Dolly‑15K dataset.  
- PEFT / LoRA methods for efficient fine‑tuning.  
- Benchmarks: AlpacaEval2, MT‑Bench.

---

## 📄 License

Specify license here (e.g., MIT, Apache 2.0) depending on your preference.  
If no license file included, consider adding one to clarify usage rights.

---

Thank you for exploring this repository! If you have questions or suggestions, feel free to open an issue or pull request.
# Karen
