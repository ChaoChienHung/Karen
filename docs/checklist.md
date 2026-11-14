# Checklist

用於 feature / refactor 的自我驗收與 code review 對齊。

## 通用

- 變更類型已被清楚描述（新增 arch/engine/dataset/metric/recipe/重構）
- 變更沒有引入第二套訓練/評估入口（仍走 `python -m llm.train` / `python -m llm.eval`）
- `AGENTS.md` 的不可退化原則沒有被破壞；若需要擴充契約，先更新文件再改程式

## 可追溯性

- 新跑一次 training 後，`runs/<run_name>/` 仍會產生：
  - `args.json`、`env.json`、`train.rank<RANK>.log`、`metrics.jsonl`
- 新跑一次 eval 後，`runs/<run_name>/eval/` 仍會產生：
  - `metrics.json` 與 per-example JSONL

## 指標語意

- 若變更 training loop / eval metrics：
  - masking 規則、聚合方式、對齊策略已寫入 `docs/workflows.md`
  - `metrics.jsonl` 欄位的語意清楚且可追溯

## 資料與產物

- `llm/` 未混入靜態測資/大檔案
- `evaluation/` 的新增內容不含敏感資訊，且可被審查（model answers/judge/logs）
- `data/`、`model/` 仍維持大檔案不進 Git
- `frontend/` 之 `node_modules/` 與 `dist/` 維持不進 Git，且 `npm run lint` 與 `npm run build` 通過

## 文件同步

- `README.md` 的使用方式/範例與實作一致
- `docs/dev-notes.md` 的設計取捨/實驗紀錄（若受影響）已更新
- `docs/architecture.md` / `docs/workflows.md` / `docs/guardrails.md`（依變更範圍）已更新
