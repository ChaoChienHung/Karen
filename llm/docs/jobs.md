# Jobs Architecture (Train & Eval Jobs)

本文件定義 Karen 專案中 `llm/train/` 與 `llm/eval/` 的核心 **Jobs（作業）** 抽象設計與產物規範。

---

## 1. 核心設計哲學：Jobs 概念

在 Karen 中，所有的訓練（Training）與評估（Evaluation）任務合起來統稱為 **Jobs**：
- 每個獨立的實驗或評估任務都是一個具名的 **Job**，由唯一的 `<job_name>` 標識。
- 每個 Job 都是一個高度自治、可自包含回放的單元，其專屬目錄下收納該次作業的所有代碼快照、引數設定、輸出日誌、評估指標與模型檢查點。

```
llm/
├── train/                        # 訓練工作（Training Jobs）
│   ├── __main__.py               # 單一訓練入口 (python -m llm.train)
│   └── <job_name>/               # 依 job name 命名的獨立訓練作業
│       ├── runs/                 # 當次訓練產物（checkpoints、LoRA adapter、tokenizer）
│       ├── logs/                 # 當次訓練的 rank 日誌 (train.rank0.log)
│       ├── args.json             # CLI 參數快照
│       ├── env.json              # 執行環境快照 (Python/Torch/CUDA/Git commit)
│       └── metrics.jsonl         # 串流訓練/驗證 loss 指標 (本地 SSOT)
│
└── eval/                         # 評估工作（Evaluation Jobs）
    ├── __main__.py               # 單一評估入口 (python -m llm.eval)
    ├── calibration.py            # 置信度與校準度評估 (Brier, ECE, Risk-Coverage)
    └── results/
        └── <eval_job_name>/      # 依 eval job name 命名的評估結果
            ├── predictions.jsonl # 逐筆生成文字與標籤結果
            ├── metrics.json      # 彙整評估指標 (Format rate, Exact match, PPL)
            ├── logs/             # 評估執行過程日誌
            └── runs/             # 對應的評估產物快照
```

---

## 2. Training Job 結構規範

### 啟動方式
```bash
python -m llm.train --arch gpt2-dev --engine loop --run_name gpt2_dev_baseline
```

### 產物佈局
在訓練執行時，系統會為該 Job 建立專屬目錄（預設位於 `llm/train/<job_name>/` 或與根目錄 `runs/<job_name>/` 鏡像對齊）：
1. **`runs/`**：
   - 包含保存的 PyTorch state dict 或 Hugging Face 檢查點。
   - 若使用 LoRA，存放 `adapter_config.json` 與 `adapter_model.safetensors`。
2. **`logs/`**：
   - 存放各 DDP Rank 的標準輸出與警告資訊（如 `train.rank0.log`）。
3. **可追溯觀測檔**：
   - `args.json`：訓練參數完整記錄。
   - `env.json`：硬體與軟體環境指紋。
   - `metrics.jsonl`：包含每一步的 `loss`、`val_loss`、`learning_rate` 等數值串流。

---

## 3. Evaluation Job 結構規範

### 啟動方式
```bash
python -m llm.eval --run_dir runs/gpt2_dev_baseline --out_dir llm/eval/results/gpt2_dev_eval
```

### 產物佈局
位於 `llm/eval/results/<eval_job_name>/`：
1. **`predictions.jsonl`**（或 `instruction_eval.jsonl` / `exact_match_eval.jsonl`）：
   - 模型對各題目的原始回答、預期答案、解析結果與正確性判斷。
2. **`metrics.json`**：
   - 包含格式準確率（Format Success Rate）、精確匹配率（Exact Match）、驗證困惑度（Perplexity）及校準誤差（ECE）。
3. **`logs/`**：
   - 評估過程日誌。
4. **`runs/`**：
   - 與評估關聯之測試設定快照。

---

## 4. 命名慣例（Job Naming Conventions）

所有 Job 名稱建議遵循下述命名規則：
`<arch>-<engine>-<dataset>-<description>-<date/tag>`

* 範例：
  - `llama7b-trainer-dolly-lora_r16-20261114`
  - `gpt2dev-loop-dolly-baseline`
  - `eval-llama7b-exactmatch-n900`
