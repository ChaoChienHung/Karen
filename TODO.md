# TODO

## 現有優點（請保持不退化）

- 單一訓練入口：`python -m llm.train`（以 `--arch/--engine` 切換）
- 同一套 dataset pipeline 共享於 pretrained fine-tune 與 scratch 小模型
- 單一評估入口：`python -m llm.eval --run_dir runs/<run_name>`
- 產物集中在 `runs/<run_name>/`，wandb 可選（預設關閉）

## P0（先把訓練/評估流程跑順）

- [ ] 將資料集格式寫成明確規範（SFT prompt template、欄位定義、split 命名）並在 `data/` 放最小範例
  - [ ] preprocessing 階段篩除 empty response
  - [ ] prompt 樣板對齊 SFT（`### Instruction:` / `### Context:` / `### Response:`）
  - [ ] 固定 seed 進行 `train/val/test` 切分
- [ ] 增加「小型 smoke dataset」與 smoke 測試（確保 train/eval 端到端不會壞）
- [ ] 先把 SFT baseline 跑通（pretrained + LoRA）
  - [ ] `meta-llama/Llama-2-7b-hf` LoRA fine-tune
  - [ ] 記錄並輸出訓練/驗證 loss 曲線（可重跑、可比較）
- [ ] 先把 evaluation pipeline 跑通（先本地/小模型，再擴到大模型）
  - [ ] Tier 0：smoke generations（prompt → generation snapshot）
  - [ ] Tier 1：masked loss / perplexity（只算 `labels != -100`）
  - [ ] Tier 2：離線指令跟隨小測（格式/約束 + 可確定打分的題）
  - [ ] 固定輸出到 `runs/<run_name>/eval/`（`metrics.json`、`instruction_eval.jsonl`、`smoke_generations.jsonl`）
- [ ] 做一組「立刻可跑、可比較」的對照實驗（固定資料/訓練步數/評估集，只改一個變因）
  - [ ] `gpt2`：Base（不訓練） vs Full SFT（Dolly）
  - [ ] `Llama-2-7b`：Base（不訓練） vs LoRA SFT（Dolly）
  - [ ] `Llama-2-7b`：LoRA SFT（Dolly） vs Full SFT（Dolly，不用 LoRA；需要足夠 GPU/分散式資源才做）
  - [ ] 同一份 Dolly：比對 prompt template / masking 設計對 Tier 1 masked ppl 與 Tier 2 小測的影響
- [ ] `runs/<run_name>/` 產物結構固定化（例如 `hparams.json`、`metrics.json`、`samples/`、tokenizer）

## P1（研究擴展）

- [ ] 自建 Llama 7B（架構對齊）並做 Pre-norm vs Post-norm 對照實驗
- [ ] 基準測試與 Judge 評估（先能跑，再追求分數）
  - [ ] AlpacaEval 2（Tier 3，外部工具 recipe；輸出收斂到 `evaluation/`）
    - [ ] 產生 `evaluation/results/model_answers/<run_name>/alpacaEval2.json`
    - [ ] 跑 `scripts/bench/alpacaeval2.py print_cmds --run_name <run_name>` 並把 annotations/leaderboard 存到 `evaluation/results/judge/<judge_name>/alpacaeval2/`
  - [ ] MT-Bench（FastChat, Tier 3；輸出收斂到 `evaluation/`）
    - [ ] 跑 `scripts/bench/mtbench_fastchat.py print_cmds --run_name <run_name> --model_path <...>`
    - [ ] 產生 `evaluation/results/model_answers/<run_name>/mt-bench.jsonl` 與對應 log（`evaluation/logs/`）
  - [ ] LLM-as-a-judge（先定義輸出格式與成本控制，產出 `annotations.json`、`leaderboard.csv`）
- [ ] 支援更多資料集與格式
  - [ ] 支援 JSONL / ShareGPT / 自訂 instruction 資料集
  - [ ] `--dataset <name>` 切換（或 dataset registry）
- [ ] 擴充基準測試（挑 2–3 個先做通）
  - [ ] MMLU / GSM8K / BBH / ARC / TruthfulQA（依你想走的方向挑）
- [ ] 自訂推論/生成腳本（可比較、可重現）
  - [ ] greedy / sampling / beam
  - [ ] decoding sweep（temperature/top-p/num_beams/max_new_tokens）與報表
- [ ] 多 GPU 分散式訓練（先 DDP）
  - [ ] `torchrun` DDP 跑通（sampler 正確、log/ckpt 只在 rank0）
  - [ ] 再評估是否要 FSDP / ZeRO

## P2（工程化與可維護）

- [ ] 自製 tokenizer（BPE / SentencePiece）並提供與現有資料管線可切換介面
- [ ] 新增 HuggingFace GPT2 pretrained 路徑（與 scratch GPT2 並行）
- [ ] 新增自寫 training loop 的更多功能（lr scheduler、gradient clipping、ckpt resume、early stopping）
- [ ] 讓 custom loop engine 支援 AMP（fp16/bf16）與 grad accumulation，並對齊 Trainer 的常用行為
- [ ] 統一 checkpoint 格式（trainer/loop 可互讀或至少有轉換腳本）
  - [ ] 支援「只存 LoRA / 存 full weights」切換
- [ ] 增加可重現性（seed、torch.backends 設定、記錄 git commit hash + 重要超參數到 `runs/<run_name>/`）
- [ ] 增加 config 機制（YAML/JSON）與 CLI override，方便保存實驗設定與重跑
- [ ] 新增單元測試（模型 forward shape、RoPE 正確性、loss ignore_index 行為）
- [ ] 建立 lint/format（例如 ruff/black/isort 或現有偏好）與 pre-commit
- [ ] 建立 CI（lint + unit tests + 簡單 smoke train/eval）
- [ ] 整理並補齊文件：常見工作流（下載資料→訓練→評估→推論）、範例指令、FAQ（GPU 記憶體、常見錯誤）
- [ ] 補上 License（選定 MIT/Apache-2.0 等）與第三方模型/資料集使用注意事項（LLaMA、Dolly）

## 實驗想法（可直接做）

- [ ] Baseline compare：同一份小 eval set，比較 `Llama-2-7b` vs `gpt2`（或 dev 小模型）的格式成功率與困惑度
- [ ] Data fraction curve：抽樣 1%/5%/10%/100% 的資料做 SFT，畫 masked ppl / 指令小測分數曲線
- [ ] Decode sweep：temperature × top_p × max_new_tokens，比較品質 vs latency
- [ ] Prompt template ablation：不同 prompt 結構對 loss/format success 的影響
- [ ] Error bucket：把錯誤按類型（格式錯誤/拒答/幻覺/答非所問）分 bucket，定位瓶頸

## 值得測試的模型（先用 pretrained 當 compare 範例）

### LLaMA

- [ ] `meta-llama/Llama-2-7b-hf`

### GPT2

- [ ] `gpt2`

### Dev（from scratch）

- [ ] `gpt2-dev`
- [ ] `llama-dev`

## 開放問題（需要你決定）

- [ ] Tier 2 離線指令小測要先選哪一組題型（format/constraint vs 可確定答案的任務）？
- [ ] Judge 模型要用哪個（成本/一致性/可重現性）？要不要先只做「pairwise + 固定 prompt」？
- [ ] 主要目標是追 benchmark 分數，還是先把工程流程（run/eval/report）做穩？
