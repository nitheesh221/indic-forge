#!/usr/bin/env bash
# JSON inference + metrics for every saved mode. Every `translate` argument is
# represented here and values can be overridden for Docker/Kubeflow.
#
# Input arguments are mutually exclusive:
#   --json-file: active below; expects src_lang, tgt_lang, src_text, tgt_text
#   --text:      one sentence; prints translation and does not calculate metrics
#   --file:      one source sentence per line; no references or metrics
# Example single text:
#   PYTHONPATH=src python -m indictrans2_finetune.cli translate \
#     --text "..." --src-lang asm_Beng --tgt-lang eng_Latn \
#     --model "$BASE_MODEL" --max-length 256 --num-beams 5 --batch-size 8
set -euo pipefail
cd "$(dirname "$0")/.."

JSON_FILE="${JSON_FILE:-/shared/nithish/INDICTRANS2_train_custom/data/valid.json}"
SRC_LANG="${SRC_LANG:-asm_Beng}"
TGT_LANG="${TGT_LANG:-eng_Latn}"
BASE_MODEL="${BASE_MODEL:-ai4bharat/indictrans2-indic-en-dist-200M}"
EVAL_METRICS="${EVAL_METRICS:-chrf2pp comet bleu bertscore meteor ter exact_match}"
MAX_LENGTH="${MAX_LENGTH:-256}"       # CLI default: 256
NUM_BEAMS="${NUM_BEAMS:-5}"           # CLI default: 5
BATCH_SIZE="${BATCH_SIZE:-8}"         # CLI default: 8 sentences per generate call
# VISUALIZE=1 writes inference_report.html. REPORT_HTML overrides that path.
VISUALIZE="${VISUALIZE:-0}"
REPORT_HTML="${REPORT_HTML:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-it2_out}"
# Space-separated subset of: lora qlora lora_continue full_lora full
MODES="${MODES:-lora qlora lora_continue full_lora full}"

run_inference() {
  local mode="$1"
  local model="$2"
  local adapter="$3"                  # empty for merged full/full_lora models
  local output_dir="$OUTPUT_ROOT/$mode"
  local args=(
    translate
    --json-file "$JSON_FILE"
    --src-lang "$SRC_LANG"
    --tgt-lang "$TGT_LANG"
    --model "$model"
    --eval-metrics $EVAL_METRICS
    --predictions-csv "$output_dir/inference_predictions.csv"
    --metrics-csv "$output_dir/inference_metrics.csv"
    --max-length "$MAX_LENGTH"
    --num-beams "$NUM_BEAMS"
    --batch-size "$BATCH_SIZE"
  )
  [[ "$VISUALIZE" == 1 ]] && args+=(--visualize)
  [[ -n "$REPORT_HTML" ]] && args+=(--report-html "$REPORT_HTML")
  [[ -n "$adapter" ]] && args+=(--adapter "$adapter")
  PYTHONPATH=src python -m indictrans2_finetune.cli "${args[@]}"
}

for mode in $MODES; do
  case "$mode" in
    lora)
      run_inference lora "$BASE_MODEL" "$OUTPUT_ROOT/lora/final_model"
      ;;
    qlora)
      # QLoRA adapters load on the normal base model during inference.
      run_inference qlora "$BASE_MODEL" "$OUTPUT_ROOT/qlora/final_model"
      ;;
    lora_continue)
      run_inference lora_continue "$BASE_MODEL" "$OUTPUT_ROOT/lora_continue/final_model"
      ;;
    full_lora)
      run_inference full_lora "$OUTPUT_ROOT/full_lora/final_model" ""
      ;;
    full)
      run_inference full "$OUTPUT_ROOT/full/final_model" ""
      ;;
    *)
      echo "Unknown mode '$mode'. Choose: lora qlora lora_continue full_lora full" >&2
      exit 2
      ;;
  esac
done
