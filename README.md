# IndicTrans2 fine-tuning

Fine-tune [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) on your own parallel text. Training supports full fine-tuning, wide LoRA, LoRA, and QLoRA on the Hugging Face checkpoints, with `IndicTransToolkit` preprocessing.

This repo follows the official data layout from [`huggingface_interface/train_lora.py`](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface): one folder per `src-tgt` pair, sentence-aligned `train.*` / `dev.*` files, FLORES-200 tags such as `eng_Latn` and `hin_Deva`.

## What you get

- Convert JSONL or CSV into that layout, or let `train.py` convert it
- Pick the right checkpoint (`en-indic`, `indic-en`, or `indic-indic`; distilled or 1B)
- Train with `full`, `full_lora`, `lora`, or `qlora`
- Score chrF++, COMET, SacreBLEU, BERTScore, METEOR, TER, and exact match
- A small web UI for the recipe, data conversion, and a translation box
- A 40/8 English–Hindi **public-services** sample (Aadhaar, UPI, PM-Kisan, GST, …) so you can dry-run the pipeline

GPU is strongly recommended for training. Distilled models can run inference on CPU, slowly. The playground UI does **not** need a GPU until you press Translate.

## Setup

Use the Python already in `.venv`. No separate Python version is required. `IndicTransToolkit` is not supported on Windows. This repo pins **transformers 4.x**; IndicTransToolkit currently imports tokenizer types that moved in transformers 5.

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

PyTorch is installed from PyPI. If you have an NVIDIA GPU, install the CUDA wheel from [pytorch.org](https://pytorch.org) **before** `pip install -e .`.

## 1. Prepare data

JSONL rows may look like this:

```json
{"src": "PAN and Aadhaar have been successfully linked.", "tgt": "पैन और आधार सफलतापूर्वक लिंक हो गए हैं।", "src_lang": "eng_Latn", "tgt_lang": "hin_Deva"}
```

CSV needs a header with `src,tgt` (and optional `src_lang,tgt_lang`). If the language columns are missing, pass `--src-lang` / `--tgt-lang`.

```bash
indictrans2-ft prepare \
  --train data/sample/train.jsonl \
  --dev data/sample/dev.jsonl \
  --src-lang eng_Latn \
  --tgt-lang hin_Deva \
  --output-dir data/prepared
```

That writes:

```
data/prepared/
  train/eng_Latn-hin_Deva/train.eng_Latn
  train/eng_Latn-hin_Deva/train.hin_Deva
  dev/eng_Latn-hin_Deva/dev.eng_Latn
  dev/eng_Latn-hin_Deva/dev.hin_Deva
```

Use `indictrans2-ft langs` for every FLORES tag the models accept.

## 2. Fine-tune

`train.py` is the training entry point. It accepts `full`, `full_lora`, `lora`, and `qlora`. `full_qlora` is rejected.

| Mode | What trains | `--final_model_dir` |
| --- | --- | --- |
| `lora` | `q_proj,k_proj` | Adapter |
| `qlora` | 4-bit base + same LoRA | Adapter |
| `full_lora` | LoRA on attention and FFN, then merged | Full model |
| `full` | Every parameter | Full model |

Pass `--adapter_path` to continue an adapter. With `--train_mode full`, that adapter is merged before full fine-tuning.

One GPU:

```bash
python train.py \
  --gpu_ids 0 \
  --model_hf_id ai4bharat/indictrans2-en-indic-dist-200M \
  --train_mode lora \
  --train_json data/sample/train.jsonl \
  --val_json data/sample/dev.jsonl \
  --src_lang eng_Latn \
  --tgt_lang hin_Deva \
  --output_dir it2_out/lora \
  --final_model_dir it2_out/lora/final_model \
  --epochs 2 \
  --learning_rate 2e-4 \
  --bf16 \
  --gradient_checkpointing \
  --eval_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model eval_loss \
  --top_n_checkpoints 3
```

Several GPUs — do not also pass `--gpu_ids`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --model_hf_id ai4bharat/indictrans2-en-indic-dist-200M \
  --train_mode lora \
  --train_json data/sample/train.jsonl \
  --val_json data/sample/dev.jsonl \
  --src_lang eng_Latn --tgt_lang hin_Deva \
  --output_dir it2_out/lora \
  --bf16 --eval_strategy epoch --save_strategy epoch
```

Ready-made wrappers: `scripts/train_lora.sh`, `scripts/train_qlora.sh`, `scripts/train_full_lora.sh`, `scripts/train_full.sh`. Each wrapper lists every `train.py` argument, including parser defaults, mode-specific values, optional boolean flags, and mutually exclusive data inputs. Values are environment-overridable for Docker/Kubeflow:

```bash
TRAIN_JSON=/data/train.json \
VAL_JSON=/data/valid.json \
SRC_LANG=asm_Beng \
TGT_LANG=eng_Latn \
MODEL_HF_ID=ai4bharat/indictrans2-indic-en-dist-200M \
GPU_IDS=0 \
bash scripts/train_lora.sh
```

Leave `SRC_LANG` and `TGT_LANG` unset to train every pair in the JSON whose direction matches the checkpoint. With `indictrans2-indic-en-dist-200M`, Bengali, Kannada, and the other scheduled Indic languages can share one file and one run, and each row keeps its own `src_lang`. Rows whose target is `eng_Latn` are trained. A row aimed the other way is skipped and counted. The same command with `--src_lang asm_Beng --tgt_lang eng_Latn` still trains only Assamese. Validation chrF++, COMET, BLEU, BERTScore, METEOR, TER, and exact match stay one combined score.

If you omit `--learning_rate`, LoRA and QLoRA use `2e-4`; full and full_lora use `2e-5`. QLoRA defaults to `--optim paged_adamw_32bit`, NF4, and bfloat16 compute. Checkpoints are ranked only by `eval_loss`, `eval_entropy`, or `eval_mean_token_accuracy`.

| Direction | Distilled | Base |
| --- | --- | --- |
| English → Indic | `ai4bharat/indictrans2-en-indic-dist-200M` | `ai4bharat/indictrans2-en-indic-1B` |
| Indic → English | `ai4bharat/indictrans2-indic-en-dist-200M` | `ai4bharat/indictrans2-indic-en-1B` |
| Indic → Indic | `ai4bharat/indictrans2-indic-indic-dist-320M` | `ai4bharat/indictrans2-indic-indic-1B` |

The 40-line sample only proves the pipeline. Use thousands of clean in-domain pairs for a real run.

## 3. Translate and evaluate

```bash
indictrans2-ft translate \
  --text "Your Aadhaar OTP is valid for 10 minutes and should not be shared." \
  --src-lang eng_Latn \
  --tgt-lang hin_Deva \
  --adapter it2_out/lora/final_model

indictrans2-ft eval \
  --data-dir data/prepared \
  --src-lang eng_Latn \
  --tgt-lang hin_Deva \
  --split dev \
  --adapter it2_out/lora/final_model
```

`--text` and `--file` only print the translation. To score a file with the same fields as `train.json` / `valid.json` (`src_lang`, `tgt_lang`, `src_text`, `tgt_text`), pass `--json-file`. `tgt_text` is the reference. Choose one or more metrics with `--eval-metrics`; omit it to calculate all seven.

```bash
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --json-file /shared/nithish/INDICTRANS2_train_custom/data/valid.json \
  --src-lang asm_Beng \
  --tgt-lang eng_Latn \
  --model ai4bharat/indictrans2-indic-en-dist-200M \
  --adapter it2_out/lora/final_model \
  --eval-metrics chrf2pp comet bleu bertscore meteor ter exact_match \
  --predictions-csv it2_out/lora/inference_predictions.csv \
  --metrics-csv it2_out/lora/inference_metrics.csv \
  --max-length 256 \
  --num-beams 5 \
  --batch-size 8
```

That writes one row per sentence to `inference_predictions.csv` (`src`, `reference`, `prediction`) and one corpus row to `inference_metrics.csv`. A metric that cannot run, such as COMET when `pkg_resources` is missing, is recorded in `metric_errors`. The other scores are still written.

`scripts/inference.sh` has the full command for every saved mode: LoRA, QLoRA, continued LoRA, merged `full_lora`, and `full`. Adapter modes pass `--adapter`. `full_lora` and `full` pass only `--model` pointing at `it2_out/<mode>/final_model`.

```bash
bash scripts/inference.sh
```

Training writes `eval_metrics.csv`, `eval_samples.csv`, and `training_logs.csv` under `--output_dir`. Reported translation metrics, in priority order:

1. chrF++
2. COMET (`Unbabel/wmt22-comet-da`)
3. SacreBLEU (`tokenize=flores200`)
4. BERTScore (`xlm-roberta-large`)
5. METEOR
6. TER
7. Exact match

COMET and BERTScore download models on first use. A failure in either one is recorded, and the other metrics are still written. None of these seven choose the best checkpoint.

## Playground UI

```bash
indictrans2-ft ui --port 43147
```

Open the printed URL. Convert files, copy a train command, or try a sentence. The first Translate click downloads the Hugging Face model.

## Tests

```bash
pytest -q
```

These cover language routing and data conversion. They do not download the 200M/1B weights.

## Practical notes

- **Direction must match the checkpoint.** English→Hindi needs `en-indic`, not `indic-en`.
- **Preprocessing is required.** Training and inference run `IndicProcessor` the same way as the official LoRA script (including `is_target=True` on the target side).
- **fp16 is used only on CUDA.** CPU training is possible but not useful for the 1B models.
- **LoRA defaults are r=8, alpha=16, dropout=0.05.** `lora` and `qlora` target `q_proj,k_proj`. `full_lora` targets `q_proj,k_proj,v_proj,out_proj,fc1,fc2` and the saved result is a merged model.
- RoPE long-context variants live under [prajdabre's IndicTrans2-RoPE collection](https://huggingface.co/collections/prajdabre/indictrans2-rope-6742ddac669a05db0804db35) if you need 2k tokens.

## License

Sample data is original short administrative copy for demos. IndicTrans2 weights are used under the licenses on their Hugging Face model cards (CC-BY-4.0 for the public models). This training wrapper is MIT.
