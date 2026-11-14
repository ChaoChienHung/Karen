#!/usr/bin/env bash
#SBATCH --job-name=karen-grpo
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=runs/%x-%j.out

# Karen GRPO (Group Relative Policy Optimization) Slurm Template
# Run with: sbatch scripts/train/grpo_train.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from Karen repository root}"

test -f README.md && test -x venv/bin/python || { echo "Missing repo root or venv" >&2; exit 2; }
source venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export RUN_NAME="${RUN_NAME:-grpo-run-${SLURM_JOB_ID:?}}"

echo "Starting GRPO Training for Karen: $RUN_NAME"

python -m llm.train \
  --arch Llama-2-7b \
  --engine loop \
  --run_name "$RUN_NAME" \
  --num_train_epochs 2 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --fp16

echo "GRPO Job Completed: runs/$RUN_NAME"
