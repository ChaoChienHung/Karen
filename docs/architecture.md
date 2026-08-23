# Architecture

本文件描述本專案的核心模組邊界、資料流與產物治理，目標是讓新增功能/重構時有一致的對齊語言。

## Core Package：`llm/`

- `llm/train.py`：單一訓練入口（`python -m llm.train`）
  - 以 `--arch` 選擇模型架構，以 `--engine` 選擇訓練引擎
  - 將「可追溯性」落在 `runs/<run_name>/`（args/env/log/metrics/checkpoints）
- `llm/engines/`
  - `trainer`：基於 HuggingFace Trainer（可選 LoRA/8-bit）
  - `loop`：自寫 PyTorch loop（用於 scratch 小模型與快速迭代）
- `llm/models/`
  - 各模型以獨立資料夾封裝（HF pretrained 或 scratch build）
  - `registry.py` 提供 `arches()` 等註冊表，讓 train/eval 兩端以相同鍵值選擇模型
- `llm/preprocessing/`
  - Dataset preprocessing + tokenization
  - 核心原則：同一份資料管線同時服務 pretrained fine-tune 與 scratch 小模型
  - tokenized cache 統一落在 `data/preprocessed/`（可重用、可辨識）
- `llm/eval/`：單一評估入口（`python -m llm.eval --run_dir runs/<run_name>`）
  - Tier 0/1/2 的離線評估與輸出落在 `runs/<run_name>/eval/`
- `llm/logging.py`：run logging utilities（`metrics.jsonl` 為本地 SSOT；wandb 選配且預設關閉）
- `llm/tools/`：工具型命令（例如 merge LoRA）

## Artifacts & Data

- `runs/<run_name>/`：訓練與評估的統一追溯根目錄（每次 run 的 SSOT）
- `evaluation/`：離線評估集與 benchmark 產物（可納入版本控制，用於比較/回放）
  - `evaluation/eval_sets/`：離線測資（JSONL）
  - `evaluation/results/`：外部 benchmark 的 model answers / judge annotations
  - `evaluation/logs/`：執行外部 recipes 的 stdout/stderr logs
- `data/`：本地資料與快取（預設不進 Git）
  - `data/raw_data/`：下載後的原始資料
  - `data/preprocessed/`：tokenized dataset 快取（由 preprocessing 自動生成）
- `model/`：本地模型/權重快取與輸出（預設不進 Git）
- `scripts/`：下載、叢集、benchmark recipes 與輔助腳本

## Boundary Rules（必須一直成立）

- `llm/` 保持「純程式」：不得混入大型靜態資料資產（離線測資/bench 產物 → `evaluation/`）
- train/eval 的所有落地產物必須以 `runs/<run_name>/` 為中心，避免散落根目錄
- 任何新增 arch/engine/dataset/metric 都必須能被 `python -m llm.train` / `python -m llm.eval` 覆蓋（不要新增第二套入口）
