# IndicTrans2 LoRA fine-tune

Fine-tune [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) on your own parallel text with LoRA, using the Hugging Face checkpoints and `IndicTransToolkit` preprocessing.

This repo follows the official data layout from [`huggingface_interface/train_lora.py`](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface): one folder per `src-tgt` pair, sentence-aligned `train.*` / `dev.*` files, FLORES-200 tags such as `eng_Latn` and `hin_Deva`.

## What you get

- Convert JSONL or CSV into that layout
- Pick the right checkpoint (`en-indic`, `indic-en`, or `indic-indic`; distilled 200M or 1B)
- LoRA train with the same target modules as the official script (`q_proj,k_proj`)
- Translate and score BLEU / chrF with or without an adapter
- A small web UI for the recipe, data conversion, and a translation box
- A 40/8 English–Hindi **public-services** sample (Aadhaar, UPI, PM-Kisan, GST, …) so you can dry-run the pipeline

GPU is strongly recommended for training. Distilled models can run inference on CPU, slowly. The playground UI does **not** need a GPU until you press Translate.

## Setup

Python 3.10+ on Linux or macOS (`IndicTransToolkit` is not supported on Windows). Pin **transformers 4.x** (this repo does that for you); IndicTransToolkit currently imports tokenizer types that moved in transformers 5.

```bash
python3 -m venv .venv
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

## 2. Fine-tune LoRA

```bash
indictrans2-ft train \
  --data-dir data/prepared \
  --src-lang eng_Latn \
  --tgt-lang hin_Deva \
  --output-dir outputs/lora \
  --model-size distilled \
  --max-steps 400 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --learning-rate 2e-4
```

Defaults match the official recipe: inverse-sqrt schedule, AdamW β2=0.98, LoRA r=16 / α=32, early stopping on `eval_loss`. Switch `--model-size base` for the 1B models, or pass `--model ai4bharat/indictrans2-en-indic-1B`.

For a real domain, plan on thousands of clean pairs, not the 40-line sample. Keep a held-out `dev` set in the **same domain**.

| Direction | Distilled (faster) | Base (better) |
| --- | --- | --- |
| English → Indic | `ai4bharat/indictrans2-en-indic-dist-200M` | `ai4bharat/indictrans2-en-indic-1B` |
| Indic → English | `ai4bharat/indictrans2-indic-en-dist-200M` | `ai4bharat/indictrans2-indic-en-1B` |
| Indic → Indic | `ai4bharat/indictrans2-indic-indic-dist-320M` | `ai4bharat/indictrans2-indic-indic-1B` |

Only adapter weights are saved under `--output-dir` (plus tokenizer files). Load them on top of the same base checkpoint.

## 3. Translate and evaluate

```bash
indictrans2-ft translate \
  --text "Your Aadhaar OTP is valid for 10 minutes and should not be shared." \
  --src-lang eng_Latn \
  --tgt-lang hin_Deva \
  --adapter outputs/lora

indictrans2-ft eval \
  --data-dir data/prepared \
  --src-lang eng_Latn \
  --tgt-lang hin_Deva \
  --split dev \
  --adapter outputs/lora
```

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
- **Do not expect the 40-sentence sample to beat the base model.** It is there to prove the pipeline, not to ship a production adapter.
- RoPE long-context variants live under [prajdabre's IndicTrans2-RoPE collection](https://huggingface.co/collections/prajdabre/indictrans2-rope-6742ddac669a05db0804db35) if you need 2k tokens.

## License

Sample data is original short administrative copy for demos. IndicTrans2 weights are used under the licenses on their Hugging Face model cards (CC-BY-4.0 for the public models). This training wrapper is MIT.
