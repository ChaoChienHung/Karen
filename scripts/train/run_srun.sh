#!/usr/bin/env bash
# ==============================================================================
# Slurm srun 互動式/直接任務執行範本 (run_srun.sh)
# ==============================================================================
# 使用說明:
#   `srun` 指令會直接在終端機前台執行任務，輸出會直接導向至您当前的終端螢幕。
#   這常用於互動式偵錯 (Debugging) 或是短時間的測試運行。
#
# 常見用法有兩種：
#   用法【一】：直接呼叫此腳本，Slurm 會自動在背景向排程器申請資源，獲得後直接在計算節點執行。
#             命令: ./scripts/train/run_srun.sh
#
#   用法【二】：先透過 `salloc` 申請互動式 Shell，然後在該分配環境中執行 `srun` 或直接執行 python。
#             1. 申請 1 台節點、1 張 GPU、8 核心 CPU、32G 記憶體，限時 1 小時：
#                salloc -N1 -n1 --gres=gpu:1 -c 8 --mem=32G -p gpu -t 01:00:00 bash
#             2. 進入配發的節點 Shell 後，執行此腳本或直接跑 python 命令：
#                source venv/bin/activate
#                python -m llm.train --arch Llama-2-7b --engine trainer
#             3. 偵錯完畢後輸入 `exit` 釋放資源。
# ==============================================================================

# 確保腳本在專案根目錄執行
cd "$(dirname "$0")/../.." || exit 1

# ------------------------------------------------------------------------------
# 1. 資源申請參數 (僅在直接執行時有效，若已在 salloc 中，以下參數將被忽略)
# ------------------------------------------------------------------------------
# 資源超參數設定：
#   -p gpu              : 指定分區
#   --gres=gpu:1        : 申請 1 張 GPU (多卡改為 gpu:2, gpu:4 等)
#   -c 4                : 每個任務 4 個 CPU 核心
#   --mem=16G           : 申請 16G 記憶體 (視模型與 Batch Size 調整)
#   -t 02:00:00         : 最長限時 2 小時
#   --unbuffered (-u)   : 使 Python 輸出不進緩衝區，即時印在螢幕上方便觀察
#   --label (-l)        : 在輸出前方加上 Task Rank 標籤，多卡時特別實用

# ------------------------------------------------------------------------------
# 2. 載入虛擬環境
# ------------------------------------------------------------------------------
if [ -d "venv" ]; then
    echo "[srun] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[srun] WARNING: venv not found! Using system python."
fi

# ------------------------------------------------------------------------------
# 3. 執行命令 (多卡或單卡)
# ------------------------------------------------------------------------------

# 【模式 A: 使用 srun 直接在計算節點執行單卡訓練】
echo "[srun] Starting training on allocated node..."
srun -p gpu --gres=gpu:1 -c 4 --mem=16G -t 02:00:00 -u \
    python -m llm.train \
        --arch Llama-2-7b \
        --engine trainer \
        --run_name llama_srun_run \
        --dataset_path data/raw_data/dolly

# 【模式 B: 多卡分散式訓練 (DDP)】
# 若要使用多卡 (例如 2 張 GPU) 並使用 torchrun 分散式框架，請註解上面模式 A，改用以下命令：
# 註：此時 srun 可以用來啟動單個 torchrun 指令，torchrun 會在被配發的節點上管理 GPU 通訊。
# srun -p gpu --gres=gpu:2 -c 8 --mem=32G -t 02:00:00 -u \
#     torchrun --nproc_per_node=2 -m llm.train \
#         --arch Llama-2-7b \
#         --engine trainer \
#         --run_name llama_srun_ddp_run \
#         --dataset_path data/raw_data/dolly
