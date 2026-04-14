#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODEL_PATH=./outputs/r_cai_sft \
#   REWARD_MODEL_PATH=./outputs/r_cai_rm \
#   DATA_DIR=./data \
#   OUTPUT_DIR=./outputs/r_cai_ppo \
#   bash train_ppo_lf.sh

MODEL_PATH="${MODEL_PATH:-./outputs/r_cai_sft}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-./outputs/r_cai_rm}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/r_cai_ppo}"

llamafactory-cli train \
  --stage ppo \
  --do_train true \
  --model_name_or_path "${MODEL_PATH}" \
  --reward_model "${REWARD_MODEL_PATH}" \
  --dataset_dir "${DATA_DIR}" \
  --dataset r_cai_ppo_prompt \
  --template llama3 \
  --finetuning_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --cutoff_len 1024 \
  --max_new_tokens 512 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --ppo_score_norm true \
  --ppo_target 6.0 \
  --bf16 true \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir true \
  --plot_loss true
