# Architecture

本文件描述本專案的核心模組邊界、資料流與產物治理，目標是讓新增功能/重構時有一致的對齊語言。

## Core Package：`llm/`

- `llm/train/`：單一訓練入口套件（`python -m llm.train`）
  - 以 `--arch` 選擇模型架構，以 `--engine` 選擇訓練引擎
  - **Jobs 抽象**：每次訓練為一個獨立 Job，產物以 `runs/<run_name>/` 為 SSOT，並鏡像鏡射至 `llm/train/jobs/<job_name>/`（包含 `runs/`、`logs/`、`args.json`、`env.json`、`metrics.jsonl`）
- `llm/eval/`：單一評估入口套件（`python -m llm.eval --run_dir runs/<run_name>`）
  - Tier 0/1/2 評估產物落地於 `runs/<run_name>/eval/`，並以 Job 命名鏡像儲存於 `llm/eval/results/<eval_job_name>/`（包含 `predictions.jsonl`、`metrics.json`、`logs/`、`runs/`）
  - `calibration.py`：模型置信度與校準度評估（Brier 分數、ECE、Risk-Coverage 曲線）
- `llm/data/`：模組專屬資料層（大型快取預設不進 Git，用 `.gitkeep` 保留）
  - `llm/data/raw_data/`：原始資料集
  - `llm/data/preprocessed/`：tokenized dataset 快取（由 preprocessing 自動生成）
  - `llm/data/eval_sets/`：快速本機評估集（與 `evaluation/eval_sets/` 互為 fallback）
- `llm/docs/`：LLM 模組內部專屬設計文檔
  - `jobs.md`：Train 與 Eval 統一之 Jobs 抽象規範
  - `data.md`：資料管線與快取策略契約
  - `plan.md`：研究與實驗路線圖
- `llm/engines/`
  - `trainer`：基於 HuggingFace Trainer（可選 LoRA/8-bit）
  - `loop`：自寫 PyTorch loop（用於 scratch 小模型與快速迭代）
- `llm/architectures/`
  - 各模型架構以獨立資料夾封裝（HF pretrained 或 scratch build）
  - `registry.py` 提供 `arches()` 等註冊表，讓 train/eval 兩端以相同鍵值選擇模型
- `llm/preprocessing/`
  - Dataset preprocessing + tokenization
  - 核心原則：同一份資料管線同時服務 pretrained fine-tune 與 scratch 小模型
  - tokenized cache 統一落在 `llm/data/preprocessed/`（可重用、可辨識）
- `llm/rewards/`：可驗證獎勵工程函數庫（RLVR / DPO / GRPO）
  - 提供正確率（含非對稱危害矩陣）、證據引用 F1、校準度獎勵等純規則判分
- `llm/logging.py`：run logging utilities（`metrics.jsonl` 為本地 SSOT；wandb 選配且預設關閉）
- `llm/tools/`：工具型命令（例如 merge LoRA、inspect checkpoints）

## Jobs 抽象（Unified Jobs Paradigm）

專案中 `train/` 與 `eval/` 本質上合構為一個完整的 **Jobs 概念**：
- 每個訓練或評估任務都是一個帶有唯一名稱（Job Name）的 Job
- **Train Job**：`llm/train/jobs/<job_name>/` 擁有其對應之程式碼快照、`runs/`（權重/checkpoints）、`logs/`（如 `train.rank0.log`）、`args.json`、`env.json` 與 `metrics.jsonl`
- **Eval Job**：`llm/eval/results/<eval_job_name>/` 擁有其 `predictions.jsonl`、`metrics.json`、`logs/` 與 `runs/`
- 透過單一入口（CLI flags）驅動，維持各 Job 執行結果的可追溯性與可比較性

## Artifacts & Data

- `runs/<run_name>/`：訓練與評估的統一追溯根目錄（每次 run 的 SSOT）
- `evaluation/`：離線評估集與 benchmark 產物（可納入版本控制，用於比較/回放）
  - `evaluation/eval_sets/`：離線測資（JSONL）
  - `evaluation/results/`：外部 benchmark 的 model answers / judge annotations
  - `evaluation/logs/`：執行外部 recipes 的 stdout/stderr logs
- `llm/data/`：本地資料與快取（預設不進 Git）
- `checkpoints/`：本地模型權重快取與輸出（`checkpoints/hf_models/`、`checkpoints/hf_tokenizers/`，預設不進 Git）
- `scripts/`：下載、叢集、benchmark recipes 與輔助腳本

## Web Dashboard & Local Bridge：`frontend/` & `scripts/`

- `frontend/`：React 19 + Vite + Express + Tailwind Web 儀表板
  - `src/components/ModelHub.tsx`：模型庫與架構規格檢視
  - `src/components/Leaderboard.tsx`：基準測試跑分排行榜
  - `src/components/Articles.tsx`：技術手冊與 Scaling 教學
  - `src/components/Sandbox.tsx`：多模型即時串流比較遊樂場（SSE）
  - `server.ts`：Express 後端代理，提供 `/api/models` 與 `/api/compare/stream`（自動代理至本地 Python runner 或 Gemini API）
- `scripts/serve_local.py`：本地模型推論橋接伺服器（FastAPI + SSE）
  - 支援載入 `runs/<run_name>/` 訓練產物（PEFT / PyTorch）
  - 提供即時 token 串流（相容 Karen Sandbox SSE 格式）

## Boundary Rules（必須一直成立）

- `llm/` 保持「純程式」：不得混入大型靜態資料資產（離線測資/bench 產物 → `evaluation/`）
- train/eval 的所有落地產物必須以 `runs/<run_name>/` 為中心，避免散落根目錄
- 任何新增 arch/engine/dataset/metric 都必須能被 `python -m llm.train` / `python -m llm.eval` 覆蓋（不要新增第二套入口）
- `frontend/` 為觀測與互動層，不干涉訓練/評估核心管線；本地模型推論透過 `scripts/serve_local.py` 進行解耦橋接
