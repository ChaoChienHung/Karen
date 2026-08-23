# Guardrails

本文件補足 `AGENTS.md` 的「如何驗證」與常見踩雷點，作為日常改動時的回歸指南。

## 入口與產物

- 不要新增第二套入口
  - 訓練：只能是 `python -m llm.train`
  - 評估：只能是 `python -m llm.eval --run_dir runs/<run_name>`
- 不要讓產物散落根目錄
  - 任何訓練/評估落地，都以 `runs/<run_name>/` 為中心
  - 外部 benchmark 產物集中到 `evaluation/`

## Metrics 語意一致性

- `metrics.jsonl` 是本地 SSOT：任何新指標都應可在此串流中被追溯
- 修改 loss/ppl 定義時，必須能回答：
  - 是否使用 `labels != -100` 的 masking 規則
  - 聚合方式是 token-weighted 還是 batch mean
  - 與 `llm.eval` 的 Tier 1 是否可比；若不可比，文件需明確寫出差異與原因

## 資料與快取治理

- `llm/` 不放靜態測資或大型資料檔
- `data/`、`model/` 的大型快取/權重預設不進 Git
- preprocessing cache 的命名需足以區分 tokenizer/max_length/seed/split ratios，避免快取互相污染

## Wandb / Secrets

- wandb 預設關閉；啟用必須是 opt-in（flag 或 env var）
- 不提交任何 token/API key；若 logs/benchmark outputs 出現敏感資訊，需先清理再納入版本控制
