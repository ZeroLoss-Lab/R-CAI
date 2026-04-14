#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODEL_PATH=/path/to/Llama-3-8B-Instruct \
#   DATA_DIR=./data \
#   OUTPUT_DIR=./outputs/r_cai_sft \
#   bash train_sft_lf.sh

MODEL_PATH="${MODEL_PATH:-./models/Llama-3-8B-Instruct}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/r_cai_sft}"

llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_dir "${DATA_DIR}" \
  --dataset r_cai_sft \
  --template llama3 \
  --finetuning_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --cutoff_len 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir true \
  --plot_loss true
