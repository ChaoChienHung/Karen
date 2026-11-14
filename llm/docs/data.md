# Data & Preprocessing Contract

本文件定義 `llm/data/` 的資料組織、預處理管線合約與快取治理規範。

---

## 1. 目錄結構

```
llm/data/
├── raw_data/               # 下載之原始資料集（如 Dolly-15k，被 .gitignore 排除）
├── preprocessed/           # Tokenized 預處理分割集快取（被 .gitignore 排除）
└── eval_sets/              # 離線基準評測題庫（JSONL，納入 Git 版本控制）
    ├── exact_match.jsonl   # 精確匹配題庫
    ├── json_format.jsonl   # 結構化輸出約束題庫
    └── smoke_prompts.jsonl # 快速煙霧測試題庫
```

---

## 2. 共享前處理合約（Preprocessing Contract）

- **單一資料管線**：預訓練模型微調（HF + LoRA）與從頭自研小模型（Scratch PyTorch）**必須共用同一套資料處理管線**。
- **快取目錄命名規範**：
  Tokenized 快取儲存於 `llm/data/preprocessed/<prefix>_<tokenizer>_<max_length>_<train_ratio>_<val_ratio>_<seed>/`。
- **Masking 語意**：
  在指令微調（SFT）資料集中，只針對模型的回應（Response）部分計算 Loss，非回應部分（Instruction / Context / Prompt）的 Token 標籤必須遮蔽為 `-100`。
