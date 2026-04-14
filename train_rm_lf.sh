#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODEL_PATH=./outputs/r_cai_sft \
#   DATA_DIR=./data \
#   OUTPUT_DIR=./outputs/r_cai_rm \
#   bash train_rm_lf.sh

MODEL_PATH="${MODEL_PATH:-./outputs/r_cai_sft}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/r_cai_rm}"

llamafactory-cli train \
  --stage rm \
  --do_train true \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_dir "${DATA_DIR}" \
  --dataset r_cai_rm \
  --template llama3 \
  --finetuning_type lora \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --cutoff_len 4096 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --bf16 true \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir true \
  --plot_loss true
