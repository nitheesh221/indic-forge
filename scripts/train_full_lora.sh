#!/usr/bin/env bash
# LoRA on attention and FFN modules. The final directory is a merged full model.
set -euo pipefail
cd "$(dirname "$0")/.."
python train.py \
  --gpu_ids "${GPU_IDS:-0}" \
  --model_hf_id ai4bharat/indictrans2-en-indic-dist-200M \
  --train_mode full_lora \
  --train_json data/sample/train.jsonl \
  --val_json data/sample/dev.jsonl \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --output_dir it2_out/full_lora \
  --final_model_dir it2_out/full_lora/final_model \
  --epochs 2 \
  --batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --optim adamw_bnb_8bit \
  --bf16 \
  --gradient_checkpointing \
  --eval_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model eval_loss \
  --top_n_checkpoints 3 \
  --save_only_model
