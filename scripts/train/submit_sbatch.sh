#!/usr/bin/env bash
# ==============================================================================
# Slurm sbatch 訓練作業提交範本 (submit_sbatch.sh)
# ==============================================================================
# 使用方式:
#   sbatch scripts/train/submit_sbatch.sh
#
# 查詢狀態:
#   squeue -u $USER
#
# 取消作業:
#   scancel <JOB_ID>
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Slurm 資源排程設定 (#SBATCH 指令)
# ------------------------------------------------------------------------------
#SBATCH --job-name=karen-train        # 設定此作業在佇列中顯示的名稱
#SBATCH --partition=gpu              # 指定使用的分區 (Partition/Queue)，通常為 gpu 或 batch
#SBATCH --nodes=1                    # 請求分配的計算節點數量 (一般單機訓練設 1 即可)
#SBATCH --ntasks-per-node=1          # 每個節點執行的主任務數量 (使用 torchrun 時設 1 即可，由 torchrun 分發程序)
#SBATCH --cpus-per-task=8            # 每個任務分配的 CPU 核心數 (請根據 DataLoader 的 num_workers 或硬體規格調整)
#SBATCH --gres=gpu:1                 # 請求通用資源 (GRES)。格式為 gpu:<數量> 或 gpu:<型號>:<數量>
#SBATCH --mem=32G                    # 請求該節點的記憶體限制，例如 32G、64G (設為 0 表示使用該節點的所有記憶體)
#SBATCH --time=12:00:00              # 設定最長執行時間限值 (格式: 天-時:分:秒 或 時:分:秒)
#SBATCH --output=runs/train-%j.out   # 標準輸出重新導向路徑，%j 會自動替換為 Slurm Job ID
#SBATCH --error=runs/train-%j.err    # 標準錯誤輸出重新導向路徑，%j 會自動替換為 Job ID

# ------------------------------------------------------------------------------
# 2. 說明：如何調整硬體超參數與資源分配
# ------------------------------------------------------------------------------
# 【GPU 數量修改說明】
#   - 預設我們申請了 1 張 GPU (#SBATCH --gres=gpu:1)。
#   - 如果要改成 4 張 GPU，請將上述設定修改為:
#       #SBATCH --gres=gpu:4
#     並在下方「4. 執行訓練命令」中，將單卡 python 執行改為 torchrun 分散式執行。
#
# 【CPU & 記憶體修改說明】
#   - 如果發生 OOM (Out of Memory) 或 DataLoader 載入資料緩慢:
#     - 修改 #SBATCH --mem=64G 提高記憶體配額。
#     - 修改 #SBATCH --cpus-per-task=16 提高 CPU 核心分配，以利多執行緒加速。
#
# 【分區變更】
#   - 依照您叢集的設定，若有不同 GPU 卡型 (如 A100, V100)，可能對應不同的 partition。
#   - 例如：#SBATCH --partition=a100 或 #SBATCH --gres=gpu:a100:2
# ------------------------------------------------------------------------------

# 載入環境變數或模組 (若叢集有需要，例如 module load cuda/11.8)
# module load cuda/11.8

# 切換到作業提交的目錄 (預設 Slurm 就會在提交腳本的目錄下執行，但保險起見可確保在專案根目錄)
cd "${SLURM_SUBMIT_DIR:-.}"

# 確保輸出目錄存在，以防 %j 的 log 無法寫入
mkdir -p runs

# ------------------------------------------------------------------------------
# 3. 載入虛擬環境 (Python venv)
# ------------------------------------------------------------------------------
# 啟動已安裝套件的虛擬環境 (venv)
if [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "WARNING: venv directory not found! Running with system python."
fi

# ------------------------------------------------------------------------------
# 4. 執行訓練命令
# ------------------------------------------------------------------------------
# 【模式 A: 單卡 (Single GPU) 訓練】
# 適合偵錯與小模型，或是 LoRA 微調 (8-bit)
python -m llm.train \
    --arch Llama-2-7b \
    --engine trainer \
    --run_name llama_sft_run \
    --dataset_path data/raw_data/dolly

# 【模式 B: 多卡分散式 (Multi-GPU DDP) 訓練】
# 如果您在上方 #SBATCH --gres=gpu:X 申請了複數張 GPU，請註解掉上面的模式 A，並啟用以下命令。
# 這裡以 2 張 GPU 為例 (--nproc_per_node 需與 --gres=gpu 數量一致)：
# torchrun \
#     --nproc_per_node=2 \
#     -m llm.train \
#     --arch Llama-2-7b \
#     --engine trainer \
#     --run_name llama_ddp_run \
#     --dataset_path data/raw_data/dolly

echo "Training job completed at $(date)"
