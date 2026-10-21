#!/usr/bin/env bash
# Full LoRA: all attention/FFN adapters are trained and merged into a full model.
# Every train.py argument is represented below and can be overridden via the environment.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_HF_ID="${MODEL_HF_ID:-ai4bharat/indictrans2-en-indic-dist-200M}"
MODEL_ID="${MODEL_ID:-$MODEL_HF_ID}"             # default: MODEL_HF_ID
ADAPTER_PATH="${ADAPTER_PATH:-}"                 # default: empty; set to continue an adapter
TRAIN_JSON="${TRAIN_JSON:-data/sample/train.jsonl}"
VAL_JSON="${VAL_JSON:-data/sample/dev.jsonl}"
DATA_DIR="${DATA_DIR:-}"                         # alternative to TRAIN_JSON + VAL_JSON
# Empty means every pair in the JSON whose direction matches MODEL_HF_ID.
SRC_LANG="${SRC_LANG:-}"
TGT_LANG="${TGT_LANG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-it2_out/full_lora}"
FINAL_MODEL_DIR="${FINAL_MODEL_DIR:-$OUTPUT_DIR/final_model}"
GPU_IDS="${GPU_IDS:-0}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

ARGS=(
  --train_mode full_lora
  --model_hf_id "$MODEL_HF_ID"
  --model_id "$MODEL_ID"
  --output_dir "$OUTPUT_DIR"
  --final_model_dir "$FINAL_MODEL_DIR"
  --epochs "${EPOCHS:-2}"
  --batch_size "${BATCH_SIZE:-1}"
  --eval_batch_size "${EVAL_BATCH_SIZE:-2}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --learning_rate "${LEARNING_RATE:-2e-5}"
  --weight_decay "${WEIGHT_DECAY:-0.01}"
  --warmup_ratio "${WARMUP_RATIO:-0.03}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}"
  --optim "${OPTIM:-adamw_bnb_8bit}"
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}"
  --logging_steps "${LOGGING_STEPS:-50}"
  --seed "${SEED:-42}"
  --max_length "${MAX_LENGTH:-256}"
  --eval_strategy "${EVAL_STRATEGY:-epoch}"
  --eval_steps "${EVAL_STEPS:-50}"
  --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS:-256}"
  --eval_accumulation_steps "${EVAL_ACCUMULATION_STEPS:-1}"
  --generation_num_beams "${GENERATION_NUM_BEAMS:-5}"
  --save_strategy "${SAVE_STRATEGY:-epoch}"
  --save_steps "${SAVE_STEPS:-500}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
  --metric_for_best_model "${METRIC_FOR_BEST_MODEL:-eval_loss}"
  --top_n_checkpoints "${TOP_N_CHECKPOINTS:-3}"
  --eval_metrics ${EVAL_METRICS:-chrf2pp comet bleu bertscore meteor ter exact_match}
  --eval_metrics_csv "${EVAL_METRICS_CSV:-eval_metrics.csv}"
  --eval_samples_csv "${EVAL_SAMPLES_CSV:-eval_samples.csv}"
  --training_logs_csv "${TRAINING_LOGS_CSV:-training_logs.csv}"
  --lora_r "${LORA_R:-8}"
  --lora_alpha "${LORA_ALPHA:-16}"
  --lora_dropout "${LORA_DROPOUT:-0.05}"
  --lora_target_modules "${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,out_proj,fc1,fc2}"
  --bnb_compute_dtype "${BNB_COMPUTE_DTYPE:-bfloat16}"
  --bnb_4bit_quant_type "${BNB_4BIT_QUANT_TYPE:-nf4}"
  --gpu_ids "$GPU_IDS"
  --num_workers "${NUM_WORKERS:-4}"
  --report_to "${REPORT_TO:-none}"
)

if [[ -n "$DATA_DIR" ]]; then
  ARGS+=(--data_dir "$DATA_DIR")
else
  ARGS+=(--train_json "$TRAIN_JSON" --val_json "$VAL_JSON")
fi
[[ -n "$SRC_LANG" ]] && ARGS+=(--src_lang "$SRC_LANG")
[[ -n "$TGT_LANG" ]] && ARGS+=(--tgt_lang "$TGT_LANG")
[[ -n "$ADAPTER_PATH" ]] && ARGS+=(--adapter_path "$ADAPTER_PATH")
[[ -n "$RESUME_FROM_CHECKPOINT" ]] && ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
[[ "${SAVE_ONLY_MODEL:-1}" == 1 ]] && ARGS+=(--save_only_model)
[[ "${BNB_4BIT_USE_DOUBLE_QUANT:-0}" == 1 ]] && ARGS+=(--bnb_4bit_use_double_quant)
[[ "${GRADIENT_CHECKPOINTING:-1}" == 1 ]] && ARGS+=(--gradient_checkpointing)
[[ "${BF16:-1}" == 1 ]] && ARGS+=(--bf16)
[[ "${FP16:-0}" == 1 ]] && ARGS+=(--fp16) # Do not enable together with BF16.

python train.py "${ARGS[@]}"
