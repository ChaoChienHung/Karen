# Dev Notes

## 文件分工（Contract）

這個 repo 的文件分工採「單一事實來源」原則，避免內容重複與漂移：

- `AGENTS.md`：不可變更約束（Non-negotiables / Guardrails）
  - 放永遠要成立的規則：train/eval 單一入口、`runs/<run_name>/` 追溯規範、安全/隱私、資料資產放置、向後相容等
  - 不放教學、benchmark 設計、實驗流水帳
- `README.md`：對外入口（How to use / What is this repo）
  - 放快速上手、常用指令、模型列表、輸出路徑、wandb 啟用與 login
  - 不放長期研究計畫與實驗紀錄
- `TODO.md`：任務清單與優先級（工作規劃入口）
- `docs/dev-notes.md`：設計決策 + 評估/benchmark + 實驗紀錄（Living doc）
  - 放「為什麼這樣設計」、評估/benchmark 規格、實驗模板/紀錄、坑與已知問題
  - 不複製貼上 `AGENTS.md` 的 non-regression 條款；需要引用時只指向 `AGENTS.md`

小規則（寫文件時用來決策要放哪裡）：
- 任何「必須永遠成立」的句子 → 放 `AGENTS.md`
- 任何「第一次使用者要怎麼跑」 → 放 `README.md`
- 任何「為什麼這樣設計／這次實驗怎麼做、結果如何」 → 放 `docs/dev-notes.md`
- `docs/dev-notes.md`/`README.md` 需要提到約束時，只引用 `AGENTS.md`，不要複製貼上

---

## Training Observability（Logging / Metrics）

Paths and usage are documented in `README.md`. This section records the intent/semantics for experiment traceability.

每次訓練會在 `runs/<run_name>/` 產出可追溯的觀測檔，方便長跑與事後比較（兩個 engine 行為一致）：

- `args.json`：當次 run 的 CLI 參數快照
- `env.json`：環境快照（python/torch/transformers/cuda、DDP rank/world size、git commit）
- `train.rank<RANK>.log`：訓練 log（DDP 每個 rank 一份；rank 0 另外會輸出到 stdout）
- `metrics.jsonl`：訓練/評估的逐筆 metrics（JSON Lines，適合 grep / 後處理 / 繪圖）

Wandb：
- 預設關閉；用 `--wandb` 或 `WANDB_ENABLED=1` 啟用
- 啟用時會同步把 `train_loss/val_loss` 等指標上報（仍以 `runs/<run_name>/metrics.jsonl` 作為本地真實來源）

## 評估與基準測試架構重構（Reorganization of Evaluation & Benchmark Assets）

### 1. 動機與目的 (Motivation & Rationale)
為了提升專案結構的工程化品質，我們將所有評估與基準測試相關的資源進行了集中重構。主要考量如下：
* **程式與資料解耦 (Decoupling Code and Data)**：原本的測試集存放在 `llm/eval_sets/` 中。然而，`llm/` 是此專案的核心 Python 程式套件（Package），若夾帶靜態測試資料（JSONL），會使套件結構不夠純粹。因此我們將資料集移出套件，僅保留評估執行代碼於 `llm/eval/` 內。
* **根目錄整潔度 (Root Cleanliness)**：先前執行測試產生的暫時性資源（如 `model_answer/`、`gpt-4o-mini/`、`benchmark/` 的 stdout 重導向日誌）散落在根目錄下，結構混亂。將其整合能大幅提升專案外觀的專業度。
* **評估資產可追溯性 (Traceability of Evaluation Assets)**：微調大模型是一項實驗導向的工作。將「評估測資」、「各模型生成答案 (Model Answers)」、「裁判模型評分 (Judge Annotations/Leaderboards)」與「運行日誌 (Logs)」統一收納，能讓實驗結果一目了然，便於 Git 版本控制與團隊協作。

### 2. 重構後目錄架構 (Directory Layout)
重構後的資源統一收納於專案根目錄的 `evaluation/` 底下：
* **`evaluation/eval_sets/`**：存放本地離線評估的 Prompt 與預期答案（如 JSON 格式、精確匹配及 Smoke Test 測資）。
* **`evaluation/results/model_answers/`**：存放各模型對基準測試（AlpacaEval 2、MT-Bench）的原始生成文字，用於分析模型答題品質。
* **`evaluation/results/judge/gpt-4o-mini/`**：存放使用 `gpt-4o-mini` 作為 Judge 評分後的詳細 Annotations（含理由）與 Leaderboard 排行榜。
* **`evaluation/logs/`**：存放執行評估指令時產生的標準輸出日誌（如 `alpacaEval2.log`、`mt-bench.log`），便於排查執行過程中的 API 錯誤或 Gateway 問題。
* **`data/raw_data/`**（排除於 Git）：存放下載後的原始資料集檔案（如 Dolly-15k）。
* **`data/preprocessed/`**（排除於 Git）：存放經 Tokenize 與切分後的預處理資料集快取，自動按 Tokenizer 及長度命名，實現秒級重載。
* **`data/hf_datasets/`**（排除於 Git）：存放 Hugging Face 原始資料集快取。
* **`model/`**（除 `.gitkeep` 外排除於 Git）：存放預訓練模型與 Tokenizer 快取，以及合併後的權重檔。
  - `model/hf_models/`：預訓練模型快取。
  - `model/hf_tokenizers/`：Tokenizer 快取。
  - `model/merged/`：合併後的 LoRA 完整權重。
* **`scripts/download/`**：存放資料集下載腳本（如 `download_dolly.py`）。
* **`scripts/train/`**：存放用於 GPU 叢集排程的 Slurm 作業腳本範本（如 `submit_sbatch.sh` 與 `run_srun.sh`）。

---

## Benchmark Plan (Draft)

Goal: a single evaluation entrypoint that outputs both Tier 1 (loss/ppl) and Tier 2 (instruction-follow offline tests) results, so experiments are comparable across pretrained and dev (from-scratch) models.

### Tiers

- Tier 0: Smoke test (prompt → generation snapshot)
  - Purpose: ensure generation works, formatting doesn’t collapse, sanity-check responses after training.
  - Output: JSONL of prompts + generations for quick manual inspection.
- Tier 1: Validation loss / perplexity
  - Purpose: stable, cheap, reproducible metric.
  - Note: for SFT-style data, compute loss/ppl only on `labels != -100` tokens (response tokens).
  - Output: `runs/<run_name>/eval/metrics.json`
- Tier 2: Instruction-follow offline eval set
  - Purpose: approximate “instruction-follow” quality without external judges.
  - Track two kinds of tasks:
    - Format/constraint tasks: JSON validity, required keys, regex match rate.
    - Answerable tasks: simple extraction/classification with deterministic scoring (exact match / normalized match).
  - Output:
    - `runs/<run_name>/eval/instruction_eval.jsonl` (per-example outputs + scores)
    - `runs/<run_name>/eval/metrics.json` (aggregate metrics)
- Tier 3: External benchmarks / LLM-as-a-judge (future)
  - Examples: MT-Bench / AlpacaEval2 style comparisons.
  - Run as external recipes (`scripts/bench/`) and store outputs under `evaluation/`.
  - Requires: judge model + prompt templates + higher cost; do later after Tier 1/2 are stable.

### Default Artifacts Layout

- Training outputs: `runs/<run_name>/`
- Evaluation outputs: `runs/<run_name>/eval/`
  - `metrics.json`
  - `instruction_eval.jsonl`
  - `smoke_generations.jsonl`

### Running Evaluation

```bash
python -m llm.eval --run_dir runs/<run_name>
```

Notes:
- Smoke test is enabled by default and writes `smoke_generations.jsonl` (disable with `--no_smoke`).
- Exact match is enabled by default and writes `exact_match_eval.jsonl` (disable with `--no_exact_match`).

## Experiment Log (Fill as you go)

### Template

- Date:
- Run name:
- Arch:
- Engine:
- Dataset:
- Key params:
  - lr:
  - batch:
  - epochs:
  - lora/quant:
- Results:
  - val_loss_masked:
  - val_ppl_masked:
  - format_success_rate:
  - exact_match:
- Notes:
