#!/usr/bin/env bash
# 4-bit QLoRA. BitsAndBytes and a CUDA GPU are required.
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py \
  --gpu_ids "${GPU_IDS:-0}" \
  --model_hf_id ai4bharat/indictrans2-en-indic-dist-200M \
  --train_mode qlora \
  --train_json data/sample/train.jsonl \
  --val_json data/sample/dev.jsonl \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --output_dir it2_out/qlora \
  --final_model_dir it2_out/qlora/final_model \
  --epochs 2 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --optim paged_adamw_32bit \
  --bnb_compute_dtype bfloat16 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant \
  --bf16 \
  --gradient_checkpointing \
  --eval_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model eval_loss \
  --top_n_checkpoints 3
