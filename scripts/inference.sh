#!/usr/bin/env bash
# Score valid.json with every saved IndicTrans2 mode.
# JSON rows use the same fields as training: src_lang, tgt_lang, src_text, tgt_text.
# tgt_text is the reference. One metric failure is recorded; the others are still written.
set -euo pipefail
cd "$(dirname "$0")/.."

JSON_FILE="/shared/nithish/INDICTRANS2_train_custom/data/valid.json"
SRC_LANG="asm_Beng"
TGT_LANG="eng_Latn"
BASE_MODEL="ai4bharat/indictrans2-indic-en-dist-200M"

# LoRA adapter
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --json-file "$JSON_FILE" \
  --src-lang "$SRC_LANG" \
  --tgt-lang "$TGT_LANG" \
  --model "$BASE_MODEL" \
  --adapter it2_out/lora/final_model \
  --eval-metrics chrf2pp comet bleu bertscore meteor ter exact_match \
  --predictions-csv it2_out/lora/inference_predictions.csv \
  --metrics-csv it2_out/lora/inference_metrics.csv \
  --max-length 256 \
  --num-beams 5

# QLoRA adapter, loaded on the normal base model
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --json-file "$JSON_FILE" \
  --src-lang "$SRC_LANG" \
  --tgt-lang "$TGT_LANG" \
  --model "$BASE_MODEL" \
  --adapter it2_out/qlora/final_model \
  --eval-metrics chrf2pp comet bleu bertscore meteor ter exact_match \
  --predictions-csv it2_out/qlora/inference_predictions.csv \
  --metrics-csv it2_out/qlora/inference_metrics.csv \
  --max-length 256 \
  --num-beams 5

# Continued LoRA adapter
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --json-file "$JSON_FILE" \
  --src-lang "$SRC_LANG" \
  --tgt-lang "$TGT_LANG" \
  --model "$BASE_MODEL" \
  --adapter it2_out/lora_continue/final_model \
  --eval-metrics chrf2pp comet bleu bertscore meteor ter exact_match \
  --predictions-csv it2_out/lora_continue/inference_predictions.csv \
  --metrics-csv it2_out/lora_continue/inference_metrics.csv \
  --max-length 256 \
  --num-beams 5

# full_lora merged model
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --json-file "$JSON_FILE" \
  --src-lang "$SRC_LANG" \
  --tgt-lang "$TGT_LANG" \
  --model it2_out/full_lora/final_model \
  --eval-metrics chrf2pp comet bleu bertscore meteor ter exact_match \
  --predictions-csv it2_out/full_lora/inference_predictions.csv \
  --metrics-csv it2_out/full_lora/inference_metrics.csv \
  --max-length 256 \
  --num-beams 5

# full model
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --json-file "$JSON_FILE" \
  --src-lang "$SRC_LANG" \
  --tgt-lang "$TGT_LANG" \
  --model it2_out/full/final_model \
  --eval-metrics chrf2pp comet bleu bertscore meteor ter exact_match \
  --predictions-csv it2_out/full/inference_predictions.csv \
  --metrics-csv it2_out/full/inference_metrics.csv \
  --max-length 256 \
  --num-beams 5
