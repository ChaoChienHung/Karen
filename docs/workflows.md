# Workflows

本文件把日常協作需要「可執行」的流程固定下來，讓不同人/不同時間的變更能一致落地。

## 訓練（Training）

- 唯一入口：`python -m llm.train`
- 典型決策點
  - 選擇模型：`--arch <arch>`
  - 選擇引擎：`--engine trainer|loop`
  - 選擇 run / job 名稱：`--run_name <name>`（產物以 `runs/<name>/` 為 SSOT，並鏡像於 `llm/train/jobs/<name>/`）
  - dataset 來源：`--dataset_path ...` 或下載到 `llm/data/raw_data/` 後再指向本地路徑

### Run / Job 產物（必須落地）

`runs/<run_name>/`（與 `llm/train/jobs/<job_name>/`）至少包含：

- `args.json`
- `env.json`
- `train.rank<RANK>.log`（位於 `logs/` 或根目錄）
- `metrics.jsonl`

Trainer/loop 的 checkpoint/最終權重可以不同，但「可追溯的觀測檔」必須一致，且 `metrics.jsonl` 為本地 SSOT。

## 評估（Evaluation）

- 唯一入口：`python -m llm.eval --run_dir runs/<run_name>`
- 輸出位置：`runs/<run_name>/eval/`，並自動鏡射記錄至 `llm/eval/results/<eval_job_name>/`
  - `metrics.json`
  - `instruction_eval.jsonl`
  - `exact_match_eval.jsonl`
  - `smoke_generations.jsonl`

### 指標語意（避免跨 engine 不可比）

- SFT 類資料的 loss/ppl 預期只計算 `labels != -100` 的 tokens（response tokens）
- 若在 training loop 內新增/調整 val_loss 定義，必須在此文件更新：
  - 聚合方式（token-weighted / batch mean / micro/macro）
  - masking 規則（哪些 tokens 會被排除）
  - 與 `llm.eval` 的 Tier 1 是否一致

## 新增模組的協作流程

### 新增/修改 `--arch`（模型）

- 必須滿足
  - `python -m llm.train --arch <new>` 可跑通（trainer 與/或 loop）
  - `python -m llm.eval --run_dir runs/<run_name>` 能載入並產生 eval 產物
- 必須同步更新
  - `README.md`：Models 表與使用範例（新增 arch 的定位與注意事項）
  - `docs/architecture.md`：模組邊界（新增模型屬於 pretrained 或 scratch）
  - `docs/checklist.md`：新增驗收項（若有特殊依賴/權重治理）

### 新增/修改 `--engine`（訓練引擎）

- 必須滿足
  - 仍由 `python -m llm.train --engine <new>` 觸發（不新增第二入口）
  - 產物仍落在 `runs/<run_name>/`，且 observability 檔案齊全
  - 指標語意可對齊（至少能對應到 Tier 1 masked loss/ppl）
- 必須同步更新
  - `docs/architecture.md`：engine 邊界與責任
  - `docs/guardrails.md`：回歸檢查與常見踩雷點

### 新增 dataset / 調整 preprocessing

- 必須滿足
  - 同一份 preprocessing contract 同時支援 pretrained 與 scratch 路徑
  - tokenized cache 命名可辨識、可重用、可避免不同設定互相污染
- 必須同步更新
  - `docs/workflows.md`：dataset contract（欄位、模板、labels masking）
  - `docs/dev-notes.md`：設計取捨與實驗紀錄模板（若涉及語意變更）

### 新增離線評估集 / benchmark recipe

- 離線 eval set：新增到 `evaluation/eval_sets/`，並在 `docs/dev-notes.md` 記錄目的與評分方式
- 外部 benchmark：在 `scripts/bench/` 增加 recipe，所有產物（answers/judge/logs）集中到 `evaluation/`

## Web 儀表板與本地模型遊樂場（Web Dashboard & Sandbox）

### 啟動前端開發伺服器

```bash
cd frontend
npm install
npm run dev # 啟動 Express + Vite，服務於 http://localhost:3000
```

### 連接本地模型至 Sandbox

1. 啟動本地 SSE 串流服務：
   ```bash
   # 連接訓練產物
   python scripts/serve_local.py --run_dir runs/<run_name> --port 8000
   # 或以模擬模式測試前端串流
   python scripts/serve_local.py --mock --port 8000
   ```
2. 在 Sandbox 介面左上角選擇 Karen 本地模型（例如 `Karen Llama-2-7B` 或 `Karen GPT-2 Dev`）。
3. 前端後端 Express（Port 3000）會自動偵測 `local-` 模型前綴，並將 SSE 串流透明代理至 `http://127.0.0.1:8000`。

