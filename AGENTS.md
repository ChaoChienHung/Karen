# AGENT

本文件列出此專案的不可變更約束（non-negotiables）與協作契約。後續新增功能或重構時，請以「不讓這些特性退化」為第一優先。

本文件只放「必須永遠成立」的規則；使用教學請看 `README.md`，設計決策與實驗紀錄請看 `docs/dev-notes.md`，其餘協作手冊與驗收清單集中在 `docs/`。

## 文件分工（Contract）

- `AGENTS.md`：不可退化約束 + 協作契約（本文件）
- `README.md`：如何安裝/訓練/評估（面向第一次使用者）
- `TODO.md`：任務清單與優先級（工作規劃入口）
- `docs/doc-map.md`：`docs/` 文件索引（入口頁）
- `docs/dev-notes.md`：設計取捨、benchmark 規格、實驗紀錄（living doc）
- `docs/`：協作手冊（workflow/規範/驗收清單/架構圖），避免把所有內容擠在 README

## 不退化原則（Non-regression）

- 必須維持單一訓練入口：`python -m llm.train`（以 `--arch/--engine` 切換）
- 必須維持單一評估入口：`python -m llm.eval --run_dir runs/<run_name>`
- dataset pipeline 必須能同時服務 pretrained fine-tune 與 scratch 小模型
- `runs/<run_name>/` 必須是統一的輸出與追溯根目錄（訓練產物、評估產物、tokenizer、設定等）
- 每次訓練必須在 `runs/<run_name>/` 落地可追溯觀測檔（`args.json`、`env.json`、`train.rank<RANK>.log`、`metrics.jsonl`）
- wandb 必須維持可選且預設關閉

## 角色與權責（Roles & Ownership）

本專案的角色是「責任邊界」而不是「人」。同一個人可以同時扮演多個角色，但每個變更都必須能對應到至少一個 owner，避免灰色地帶。

- 契約守門人（Contract Owner）
  - 維護本文件的不可退化約束與協作契約
  - 任何會影響 CLI 入口語意、`runs/` 產物布局、資料/產物治理策略的變更，必須先更新本文件再改程式
- 訓練系統 Owner（Training Owner）
  - 維護 `python -m llm.train` 行為一致性：args → 落地檔案 → 指標輸出
  - 新增/修改 engine（Trainer/loop）時，確保 run artifact 規格與 metrics 定義可對齊
- 評估系統 Owner（Evaluation Owner）
  - 維護 `python -m llm.eval --run_dir ...` 的輸出格式與評估 tiers
  - 新增指標時，需明確定義其語意（例如 masked token-weighted vs batch mean）並寫入 docs
- 資料管線 Owner（Data Pipeline Owner）
  - 維護 dataset 介面，確保 pretrained fine-tune 與 scratch 小模型共用同一份 preprocessing contract
  - 維護 `data/preprocessed/` cache 命名與再利用策略，避免快取污染與不可重現
- Bench/產物治理 Owner（Benchmark & Assets Owner）
  - 維護 `evaluation/` 的離線測資、bench 產物、logs 的結構與可追溯性
  - 任何引入外部 judge/線上 API 的 recipe，需確保可重跑且產物能被審查（不含敏感資訊）
- 文件一致性維護者（Docs Steward）
  - 維護文件分工與 doc map，避免 README/TODO/AGENTS/docs 漂移與重複

## 協同機制（Collaboration Mechanics）

- 單一真相來源（SSOT）
  - 每次訓練的本地真實來源是 `runs/<run_name>/` 下的落地檔（尤其是 `metrics.jsonl`）
  - benchmark/離線評估資產的真實來源集中在 `evaluation/`，避免散落根目錄或混入 `llm/`
- 變更類型 → 必須同步更新的文件
  - CLI flags/語意變更：`README.md` + `docs/workflows.md`（必要時加 migration/替代路徑）
  - 新增 arch/engine：`README.md`（models/usage）+ `docs/architecture.md`（模組邊界）+ `docs/checklist.md`
  - dataset pipeline 調整：`docs/workflows.md`（dataset contract）+ `docs/dev-notes.md`（設計取捨/實驗）
  - 評估指標/tiers 調整：`docs/workflows.md`（指標語意）+ `docs/dev-notes.md`（benchmark 設計）
- 驗收語言統一
  - 以 `docs/checklist.md` 作為變更驗收清單，讓 review 不依賴個人記憶
  - 以 `docs/guardrails.md` 補充「如何驗證」與常見踩雷點

## 資料與產物管理

- `llm/` 作為核心 Python package，避免混入大型靜態資料資產（例如離線測資 JSONL）
- `evaluation/` 用於存放離線評估集與 benchmark 產物（model answers / judge annotations / logs），以利追溯與比較
- `data/` 與 `model/` 下的大型快取/權重預設不進 Git（用 `.gitkeep` 保留結構即可）

## 可追溯性（Traceability）

- `runs/<run_name>/` 必須能獨立支撐回放與比對
  - `args.json`：CLI 參數快照
  - `env.json`：環境快照（python/torch/transformers/cuda、DDP rank/world size、git commit）
  - `train.rank<RANK>.log`：訓練 log（DDP 每個 rank 一份）
  - `metrics.jsonl`：訓練/評估 metrics 串流（本地真實來源）
- 新增任何 metrics 欄位時，必須在 `docs/workflows.md` 定義語意與聚合方式（避免不同 engine 不可比）

## 介面穩定性

- 優先採取向後相容：新增參數可以，但避免移除既有 CLI flags 或更改其語意
- 若必須破壞相容，先提供替代路徑或轉換方式（例如 checkpoint 轉換腳本），並在 README/docs/dev-notes.md 說明

## doc-map

本索引模組需與 `docs/doc-map.md` 保持同步。

### Root Core Docs

- `AGENTS.md`：不可退化約束 + 協作契約（永遠要成立）
- `README.md`：快速上手（安裝 / 訓練 / 評估 / 產物）
- `TODO.md`：任務清單與優先級（工作規劃入口）

### docs/ Index

- `docs/doc-map.md`：本索引頁（文件地圖）
- `docs/readme.md`：`docs/` 概覽與導覽入口（導向 doc-map）
- `docs/architecture.md`：模組邊界、資料流、產物治理
- `docs/workflows.md`：日常工作流（train/eval、新增 arch/engine/dataset/metric 的同步更新點）
- `docs/guardrails.md`：回歸檢查策略與常見踩雷點（如何驗證）
- `docs/checklist.md`：變更驗收清單（PR/自我檢查）
- `docs/dev-notes.md`：設計取捨、benchmark 規格、實驗紀錄（living doc）

## 文檔更新指令處理模塊

本模塊的目標是降低協作成本：在完成 feature/refactor/任何 repo 變更之後，不必每次把所有文檔全量重讀，而是以 `docs/doc-map.md` 作為索引，僅更新「受影響」的文檔並保持一致性。

### 觸發時機

- 當完成任何 repo 變更（feature/refactor/bugfix/文檔調整）後，需主動評估是否要同步更新文檔
- 當使用者提出「請協助更新項目的相關文檔」類需求時，需按下列流程處理

### 流程

- 變更盤點：先看此次變更影響到哪些面向（CLI/訓練輸出/評估輸出/dataset pipeline/benchmark recipe/目錄結構/檔名）
- 以 `docs/doc-map.md` 作為入口：根據 doc-map 的職責邊界，定位需要更新的最小文件集合
- 優先更新高影響文件
  - 介面與協作契約：`AGENTS.md`、`README.md`
  - 工作流與驗收：`docs/workflows.md`、`docs/guardrails.md`、`docs/checklist.md`
  - 架構與背景：`docs/architecture.md`、`docs/dev-notes.md`
- 分段交付：允許分多輪完成，不要求一次性調整完所有文檔
- 索引同步：每次有文檔新增/遷移/更名/職責變更，必須同步更新 `docs/doc-map.md` 與本文件的 `doc-map` 模組

### 最小驗證

- 受影響文檔的路徑與引用連結已更新，且無殘留舊檔名/舊路徑
- 根目錄僅保留 `AGENTS.md`、`README.md`、`TODO.md`；`docs/` 內文檔皆為 kebab-case

## 安全與隱私

- 不在 repo 內提交任何 token、API key、權重下載連結中的敏感參數
- 執行評估/benchmark 的 logs 若包含敏感資訊，需先清理再納入版本控制
