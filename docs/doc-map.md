# doc-map

本文件是 `docs/` 的單一索引頁，用於梳理所有協作文件的職責邊界與入口路徑。不可退化約束以根目錄 `AGENTS.md` 為準。

## Root Core Docs

- `AGENTS.md`：不可退化約束 + 協作契約（永遠要成立）
- `README.md`：快速上手（安裝 / 訓練 / 評估 / 產物）
- `TODO.md`：任務清單與優先級（工作規劃入口）
- `INSIGHT.md`：研究洞見、實驗事實、失敗分析與假設驗證（實驗結論與洞見記錄）

## docs/ Index

- `docs/doc-map.md`：本索引頁（文件地圖）
- `docs/readme.md`：`docs/` 概覽與導覽入口（導向 doc-map）
- `docs/architecture.md`：模組邊界、資料流、產物治理
- `docs/workflows.md`：日常工作流（train/eval、新增 arch/engine/dataset/metric 的同步更新點）
- `docs/guardrails.md`：回歸檢查策略與常見踩雷點（如何驗證）
- `docs/checklist.md`：變更驗收清單（PR/自我檢查）
- `docs/dev-notes.md`：設計取捨、benchmark 規格、實驗紀錄（living doc）
- `docs/research-notes/`：研究手冊與實證紀錄（支援 `INSIGHT.md` 引用）
