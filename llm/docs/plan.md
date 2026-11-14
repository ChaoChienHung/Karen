# Research & Experiment Plan

本文件記錄 Karen 專案近期之核心訓練、評估與研究實驗規劃。

---

## 1. 近期優先事項 (P0)

1. **SFT 基準線跑通**：
   - 使用 `meta-llama/Llama-2-7b-hf` 搭配 LoRA 於 Dolly-15k 進行指令微調。
   - 驗證訓練 loss 與驗證集 perplexity 下降趨勢。
2. **多階層評估指標落地**：
   - **Tier 0**：Smoke Generations（輸出快照）。
   - **Tier 1**：Masked Perplexity / Loss（只計算 response tokens）。
   - **Tier 2**：離線指令小測（JSON 格式率與精確匹配率）。
   - **Tier 2.5**：置信度校準（Brier Score、ECE、Risk-at-Coverage）。

---

## 2. 實驗隊列 (Experiment Queue)

- [ ] `gpt2-base` vs `gpt2-sft` (Dolly-15k) 對照實驗。
- [ ] `llama-dev` scratch 模型在不同訓練 step 下的梯度穩定度分析。
- [ ] 比較 LoRA Rank ($r=8$ vs $r=16$) 對評估指標與顯存佔用的影響。
