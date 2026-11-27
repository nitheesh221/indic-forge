# IndicTrans2 Studio runbook

This file is the contract for ML Studio: the commands that work today, the package freeze from the L4 machine, the data layout, the hardware limits, the files training writes, and every flag. It does not change the trainer.

The run that this page treats as proven is a full fine-tune of `ai4bharat/indictrans2-indic-en-1B` on two NVIDIA L4 GPUs (23,034 MiB each), Python 3.13.5, with DDP, 8-bit Adam, and bf16. LoRA, QLoRA, and full-LoRA of the public distilled checkpoints use the same entry point.

Leave `--distribution` off. The default `auto` is one process when `--gpu_ids` has one id, and DDP when it has two or more.

## 1. How to run

Run every command from the repository root, inside the virtualenv that produced the freeze below.

```bash
cd /shared/nithish/INDICTRANS2_train_custom/indic-forge
source .venv/bin/activate
export PYTHONPATH=src
```

`PYTHONPATH=src` is required when the checkout is not installed. The freeze also has an editable install of this repo (`indictrans2-finetune==0.1.0`). Either path works. Do not create a second virtualenv.

### Python, CUDA, and the freeze

| Item | Value on the L4 machine |
| --- | --- |
| Python | 3.13.5 (CPython). Package metadata still accepts `>=3.10`. |
| PyTorch | `2.5.1+cu121` (the CUDA 12.1 wheel, not a source build) |
| Driver | `535.309.01` |
| `nvidia-smi` CUDA line | `12.2` (the newest CUDA this driver can run, not the toolkit PyTorch was built with) |
| cuDNN used by that PyTorch wheel | `nvidia-cudnn-cu12==9.1.0.70` |
| NCCL used by that PyTorch wheel | `nvidia-nccl-cu12==2.21.5` |
| cuBLAS used by that PyTorch wheel | `nvidia-cublas-cu12==12.1.3.1` |
| setuptools | `80.10.2`. Keep `setuptools>=69,<81`. COMET imports `pkg_resources`, which setuptools 82 removed. |

The freeze also contains CUDA 13 wheels (`nvidia-cudnn-cu13==9.24.0.43`, `cuda-toolkit==13.0.3.0`, and the other `*-cu13` lines). Those came in with other packages. The training process uses the cu12 set, because `torch==2.5.1+cu121` is the installed PyTorch.

`pyproject.toml` only lists minimum ranges. Installing from it will not reproduce this freeze. Install this exact set.

Nothing in the trainer is compiled by the user:

| Package | What to do |
| --- | --- |
| flash-attn | Not installed and not used. The model is loaded with `attn_implementation="eager"`. Do not add a flash-attn build to the image. |
| bitsandbytes `0.50.2` | Installed as a wheel. Required for `--train_mode qlora` and for `--optim adamw_bnb_8bit`. No local CUDA extension was compiled. |
| sentencepiece `0.2.2` | Wheel. IndicTrans tokenizers need it. |
| triton `3.8.0` and ninja `1.13.2` | Triton compiles small kernels at runtime. No separate build step. |
| deepspeed `0.20.1` | Present in the freeze. The trainer does not call it. `zero2`, `zero3`, `offload`, `fsdp`, and `deepspeed` are rejected. Do not compile DeepSpeed ops. |
| Cython `3.3.0` | Present in the freeze. This repo has no extension to compile. |
| IndicTransToolkit | `indictranstoolkit==1.1.1` and `indic-nlp-library-itt==0.1.1`. Not supported on Windows. |

A system CUDA toolkit is not required when the cu121 PyTorch wheel and the `nvidia-*-cu12` wheels are installed. A C compiler is required only if a wheel is missing and pip falls back to a source build. That did not happen on this machine.

First-time downloads, all public except the gated 1B checkpoints:

- COMET downloads `Unbabel/wmt22-comet-da` the first time `comet` is scored.
- BERTScore downloads `xlm-roberta-large` the first time `bertscore` is scored.
- METEOR downloads the NLTK `wordnet` corpus the first time `meteor` is scored.

### Training

One GPU, public distilled Indic→English model, LoRA. This is the small command. It writes an adapter, not a full model.

```bash
python train.py \
  --gpu_ids 0 \
  --model_hf_id ai4bharat/indictrans2-indic-en-dist-200M \
  --train_mode lora \
  --train_json /shared/nithish/INDICTRANS2_train_custom/data/train.json \
  --val_json /shared/nithish/INDICTRANS2_train_custom/data/valid.json \
  --output_dir it2_out/lora \
  --final_model_dir it2_out/lora/final_model \
  --epochs 2 \
  --learning_rate 2e-4 \
  --bf16 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model eval_loss \
  --top_n_checkpoints 3
```

QLoRA is the same command with `--train_mode qlora`, `--optim paged_adamw_32bit`, `--bnb_compute_dtype bfloat16`, `--bnb_4bit_quant_type nf4`, and `--bnb_4bit_use_double_quant`. The saved folder is still an adapter. Inference loads that adapter on the normal (not 4-bit) base model.

`full_lora` trains a wider adapter and merges it into a full model at the end. Use `--learning_rate 2e-5`, `--batch_size 1`, `--gradient_accumulation_steps 8`, `--optim adamw_bnb_8bit`, `--gradient_checkpointing`, and `--save_only_model`.

Full fine-tune of the distilled 200M model, one GPU:

```bash
python train.py \
  --gpu_ids 0 \
  --model_hf_id ai4bharat/indictrans2-indic-en-dist-200M \
  --train_mode full \
  --train_json /shared/nithish/INDICTRANS2_train_custom/data/train.json \
  --val_json /shared/nithish/INDICTRANS2_train_custom/data/valid.json \
  --output_dir it2_out/full \
  --final_model_dir it2_out/full/final_model \
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
```

The 1B Indic→English command that trained on the two L4s. Effective batch is `8 × 4 × 2 = 64`. `comet` and `bertscore` are left out of the in-training metrics because they download extra models and dominate the epoch. Score them afterwards with the inference command.

```bash
python train.py \
  --gpu_ids 0,1 \
  --model_hf_id ai4bharat/indictrans2-indic-en-1B \
  --train_mode full \
  --train_json /shared/nithish/INDICTRANS2_train_custom/indic-forge/split_dataset/train.json \
  --val_json /shared/nithish/INDICTRANS2_train_custom/indic-forge/split_dataset/valid.json \
  --output_dir OUT2/indic-en-1b \
  --final_model_dir OUT2/indic-en-1b/final_model \
  --epochs 8 \
  --learning_rate 2e-5 \
  --bf16 \
  --gradient_checkpointing \
  --optim adamw_bnb_8bit \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --eval_batch_size 8 \
  --generation_num_beams 1 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model eval_loss \
  --early_stopping_patience 0 \
  --top_n_checkpoints 2 \
  --eval_metrics chrf2pp bleu meteor ter exact_match \
  --seed 42
```

`--distribution` is omitted on purpose. `--gpu_ids 0,1` makes `auto` relaunch under `torchrun` as DDP. If validation generation runs out of memory while the training step fits, keep `--batch_size 8` and set `--eval_batch_size 4`.

The shell wrappers `scripts/train_lora.sh`, `scripts/train_qlora.sh`, `scripts/train_full_lora.sh`, and `scripts/train_full.sh` pass every flag. Override them with environment variables of the same name in upper case, for example `GPU_IDS=0,1 BATCH_SIZE=8 bash scripts/train_full.sh`. Their default model is the English→Indic distilled 200M checkpoint, not the 1B Indic→English checkpoint above.

Continue a saved LoRA adapter by passing `--adapter_path it2_out/lora/final_model` with `--train_mode lora` or `qlora`. `--train_mode full` with `--adapter_path` merges that adapter into the base model and then trains every weight.

Resume a stopped run with `--resume_from_checkpoint <output_dir>/checkpoint-<step>`.

### Evaluation

Training already scores the validation file at each epoch (or each `--eval_steps` when `--eval_strategy steps`). That writes `eval_metrics.csv` and `eval_samples.csv`. There is no separate test split inside `train.py`.

A held-out JSON file is scored with the translate command. `tgt_text` is the reference. This is the evaluation command that works today, for the merged 1B model:

```bash
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --json-file /shared/nithish/INDICTRANS2_train_custom/indic-forge/split_dataset/valid.json \
  --model OUT2/indic-en-1b/final_model \
  --eval-metrics chrf2pp comet bleu bertscore meteor ter exact_match \
  --predictions-csv OUT2/indic-en-1b/inference_predictions.csv \
  --metrics-csv OUT2/indic-en-1b/inference_metrics.csv \
  --max-length 256 \
  --num-beams 5 \
  --batch-size 8
```

LoRA and QLoRA pass the base hub id and the adapter directory. The adapter directory must contain `adapter_config.json`.

```bash
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --json-file /shared/nithish/INDICTRANS2_train_custom/data/valid.json \
  --model ai4bharat/indictrans2-indic-en-dist-200M \
  --adapter it2_out/lora/final_model \
  --eval-metrics chrf2pp comet bleu bertscore meteor ter exact_match \
  --predictions-csv it2_out/lora/inference_predictions.csv \
  --metrics-csv it2_out/lora/inference_metrics.csv \
  --max-length 256 \
  --num-beams 5 \
  --batch-size 8
```

`--gpu-ids 0,1` splits that file across both GPUs. Rank 0 scores and writes the CSVs. One id, or omitting the flag, stays on one GPU.

`scripts/inference.sh` runs the same command for `lora`, `qlora`, `lora_continue`, `full_lora`, and `full`. Override `JSON_FILE`, `BASE_MODEL`, `EVAL_METRICS`, `GPU_IDS`, `MODES`, `MAX_LENGTH`, `NUM_BEAMS`, and `BATCH_SIZE`.

Add `--visualize` to also write `inference_report.html` next to the metrics CSV.

### Inference without a reference

One sentence. This prints the translation and does not write metrics.

```bash
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --text "নৱাগত ৰিয়া চক্ৰৱৰ্তীক এক গুৰুত্বপূৰ্ণ ভূমিকা পালন কৰিবলৈ নিয়োগ কৰা হৈছে।" \
  --src-lang asm_Beng \
  --tgt-lang eng_Latn \
  --model OUT2/indic-en-1b/final_model \
  --max-length 256 \
  --num-beams 5 \
  --batch-size 8
```

A text file is one source sentence per line. It also needs `--src-lang` and `--tgt-lang`, and it does not score metrics, because there is no reference.

```bash
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --file sentences.txt \
  --src-lang hin_Deva \
  --tgt-lang eng_Latn \
  --model OUT2/indic-en-1b/final_model
```

`--text`, `--file`, and `--json-file` are mutually exclusive.

After training, the offline report is:

```bash
PYTHONPATH=src python -m indictrans2_finetune.cli analyze --run-dir OUT2/indic-en-1b
```

That writes `rca_report.html` in the run directory. Training writes it automatically unless `--no-write_rca` is set.

### Installed packages

Versions from the L4 virtualenv (`latest_requirements_python3_13_5`). The editable line is this repository.

```text
accelerate==1.15.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.3
aiosignal==1.4.0
alabaster==1.0.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
attrs==26.1.0
babel==2.18.0
bert-score==0.3.13
bitsandbytes==0.50.2
certifi==2026.7.22
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.3
cuda-bindings==13.4.3
cuda-pathfinder==1.8.3
cuda-toolkit==13.0.3.0
cycler==0.12.1
Cython==3.3.0
datasets==5.0.1
deepspeed==0.20.1
defusedxml==0.7.1
dill==0.4.1
docutils==0.22.4
einops==0.8.2
entmax==1.3
evaluate==0.4.6
fastapi==0.142.2
filelock==4.0.9
fonttools==4.66.1
frozenlist==1.8.0
fsspec==2026.6.0
h11==0.16.0
hf-xet==1.6.0
hjson==3.1.0
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==0.36.2
idna==3.20
imagesize==2.0.1
indic-nlp-library-itt==0.1.1
indictrans2-finetune==0.1.0
indictranstoolkit==1.1.1
Jinja2==3.1.6
joblib==1.6.0
jsonargparse==3.13.1
kiwisolver==1.5.1
lightning-utilities==0.15.3
lxml==6.1.3
markdown-it-py==4.2.0
MarkupSafe==3.0.4
matplotlib==3.11.2
mdurl==0.1.2
Morfessor==2.0.6
mpmath==1.3.0
msgpack==1.2.4
multidict==6.9.1
multiprocess==0.70.19
networkx==3.7
ninja==1.13.2
nltk==3.10.3
numpy==1.26.4
nvidia-cublas==13.1.1.3
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti==13.0.85
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime==13.0.96
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.0.0.61
nvidia-cufft-cu12==11.0.2.54
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver==12.0.4.66
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse==12.6.3.3
nvidia-cusparse-cu12==12.1.0.106
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu12==2.21.5
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
nvidia-nvtx-cu12==12.1.105
opentelemetry-api==1.45.0
packaging==26.3
pandas==3.0.6
peft==0.21.2
pillow==12.3.0
portalocker==4.4.0
propcache==0.5.4
protobuf==4.25.9
psutil==7.2.2
py-cpuinfo==9.0.0
pyarrow==25.0.1
pydantic==2.13.5
pydantic_core==2.46.5
Pygments==2.21.0
pyparsing==3.3.3
python-dateutil==2.9.0.post0
python-multipart==0.0.32
pytorch-lightning==2.6.6
PyYAML==6.0.3
regex==2026.9.29
requests==2.34.2
rich==15.0.0
roman-numerals==4.1.0
sacrebleu==2.6.0
sacremoses==0.2.0
safetensors==0.8.0
scipy==1.17.1
sentencepiece==0.2.2
setuptools==80.10.2
shellingham==1.5.4
six==1.17.0
snowballstemmer==3.1.1
Sphinx==9.1.0
sphinx-argparse==0.6.1
sphinx_rtd_theme==3.1.0
sphinxcontrib-applehelp==2.0.0
sphinxcontrib-devhelp==2.0.0
sphinxcontrib-htmlhelp==2.1.0
sphinxcontrib-jquery==4.1
sphinxcontrib-jsmath==1.0.1
sphinxcontrib-qthelp==2.0.0
sphinxcontrib-serializinghtml==2.0.0
starlette==1.7.0
sympy==1.13.1
tabulate==0.10.0
tokenizers==0.22.2
torch==2.5.1+cu121
torchmetrics==0.10.3
tqdm==4.70.1
transformers==4.57.6
triton==3.8.0
typer==0.27.2
typing-inspection==0.4.4
typing_extensions==4.16.0
unbabel-comet==2.2.7
urllib3==2.8.0
uvicorn==0.54.0
xxhash==4.0.1
yarl==1.25.1
```

## 2. Data

You supply the split. `train.py` does not cut one file into train, validation, and test. Pass a train file and a validation file. A third file, if you have one, is scored only by the translate command in section 1. It is never read during training.

Accepted input files: `.json`, `.jsonl`, `.csv`, `.tsv`, `.tab`.

A JSON file may be an array of objects, one JSON object per line, or one object with a list under a key. The sample file uses a `translation` list. Language tags are read from each row. Leave `--src_lang` and `--tgt_lang` off when the file already has them. Set those flags only to keep one pair, or when a CSV has text columns and no language columns.

Field names, first match wins:

| Role | Accepted names |
| --- | --- |
| Source language | `src_lang`, `source_lang`, `src_code` |
| Target language | `tgt_lang`, `target_lang`, `tgt_code` |
| Source text | `src`, `source`, `src_text`, `source_text`, `src_sentence`, `source_sentence`, `sentence_src`, `original` |
| Target text | `tgt`, `target`, `tgt_text`, `target_text`, `tgt_sentence`, `target_sentence`, `sentence_tgt`, `translation`, `translated` |

Tags are FLORES-200 tags (`hin_Deva`, `eng_Latn`) or a short alias (`hi`, `en`). One example row:

```json
{
  "src_lang": "ben_Beng",
  "tgt_lang": "eng_Latn",
  "src_text": "আমার নাম রাম।",
  "tgt_text": "My name is Ram."
}
```

`sample_train_data/indic_en_22.json` has one Indic→English row for each scheduled language. The only language strings the code accepts are these FLORES tags:

`asm_Beng`, `awa_Deva`, `ben_Beng`, `bho_Deva`, `brx_Deva`, `doi_Deva`, `eng_Latn`, `gom_Deva`, `guj_Gujr`, `hin_Deva`, `hne_Deva`, `kan_Knda`, `kas_Arab`, `kas_Deva`, `kha_Latn`, `lus_Latn`, `mag_Deva`, `mai_Deva`, `mal_Mlym`, `mar_Deva`, `mni_Beng`, `mni_Mtei`, `npi_Deva`, `ory_Orya`, `pan_Guru`, `san_Deva`, `sat_Olck`, `snd_Arab`, `snd_Deva`, `tam_Taml`, `tel_Telu`, `urd_Arab`.

Aliases that map onto those tags: `en` / `english`, `hi` / `hindi`, `bn` / `bengali`, `ta` / `tamil`, `te` / `telugu`, `ml` / `malayalam`, `kn` / `kannada`, `mr` / `marathi`, `gu` / `gujarati`, `pa` / `punjabi`, `or` / `odia`, `as` / `assamese`, `ur` / `urdu`, `ne` / `nepali`, `sa` / `sanskrit`. Anything else is rejected. `PYTHONPATH=src python -m indictrans2_finetune.cli langs` prints the same tags.

What training does with the file, before the first step:

1. Reads the rows.
2. Collapses each sentence to one line of whitespace.
3. Skips a row with no language, no text, or the same language on both sides.
4. Writes the official layout under `<output_dir>/prepared/`.
5. Keeps only pairs whose direction matches the checkpoint (`indic-en`, `en-indic`, or `indic-indic`).
6. Requires the validation file to contain every pair that training kept.

The written layout, for a Bengali→English row, is:

```text
<output_dir>/prepared/train/ben_Beng-eng_Latn/train.ben_Beng
<output_dir>/prepared/train/ben_Beng-eng_Latn/train.eng_Latn
<output_dir>/prepared/dev/ben_Beng-eng_Latn/dev.ben_Beng
<output_dir>/prepared/dev/ben_Beng-eng_Latn/dev.eng_Latn
```

Each line in the source file aligns with the same line in the target file. IndicTransToolkit tokenization happens inside the training loop. There is no separate tokenize step to run.

If the files are already in that layout, pass `--data_dir` and omit `--train_json` and `--val_json`. The validation split folder is `dev`, not `valid`.

The optional converter, if you want the folders before training:

```bash
PYTHONPATH=src python -m indictrans2_finetune.cli prepare \
  --train train.json \
  --dev valid.json \
  --output-dir data/prepared
```

## 3. Model and hardware

### Checkpoints

Weights come from the Hugging Face hub. `--model_hf_id` must be one of these six. `--model_id` may then point at a local directory of the same checkpoint. The default is the hub id.

| Direction | Distilled, public | 1B, gated |
| --- | --- | --- |
| English → Indic | `ai4bharat/indictrans2-en-indic-dist-200M` | `ai4bharat/indictrans2-en-indic-1B` |
| Indic → English | `ai4bharat/indictrans2-indic-en-dist-200M` | `ai4bharat/indictrans2-indic-en-1B` |
| Indic → Indic | `ai4bharat/indictrans2-indic-indic-dist-320M` | `ai4bharat/indictrans2-indic-indic-1B` |

Weight size in bf16 is about 0.4 GB (200M), 0.6 GB (320M), and 2 GB (1B). The vocabulary is about 256k tokens. A full Adam state for the 1B model is on the order of 18 GB before activations. That is why the 1B full run uses 8-bit Adam.

A pair is trained only on the matching direction. Bengali→English rows are rejected by an `en-indic` checkpoint.

### GPU compute capability

The code does not check the compute capability. These are the limits of the precision it actually runs.

| Precision | Where | Minimum capability | Why |
| --- | --- | --- | --- |
| bf16 | Training, when `--bf16` is set. This is the L4 recipe. | 8.0 | bf16 tensor cores start with Ampere. |
| fp16 | Training, when `--fp16` is set instead. Inference always loads fp16 on CUDA. | 7.0 | fp16 tensor cores exist on T4 (7.5). |
| eager attention | Training and inference | none beyond the precision above | `attn_implementation="eager"`. flash-attn is not used, so its 8.0 requirement does not apply. |

| GPU | Capability | This recipe |
| --- | --- | --- |
| T4 | 7.5 | No bf16. `--fp16` is the training switch. 16 GB is below the 1B full run that fit on 24 GB. |
| A100 | 8.0 | bf16 is allowed. |
| A10 | 8.6 | bf16 is allowed. A 24 GB card matches the memory this run used. |
| L4 | 8.9 | Tested. Two cards, 23,034 MiB each. |
| H100 | 9.0 | bf16 is allowed. Not part of this run. |
| B200 | 10.0 | bf16 is allowed. Not part of this run. |
| RTX PRO 6000 | 12.0 | bf16 is allowed. Not part of this run. |

Studio image for the tested recipe: Python 3.13.5, driver `535.309.01` or another driver whose `nvidia-smi` CUDA line is at least 12.1, PyTorch `2.5.1+cu121`, cuDNN `9.1.0.70`, and the freeze above. Do not add flash-attn. Do not make DeepSpeed a required build.

### Memory and precision

| Run | What fit |
| --- | --- |
| LoRA or QLoRA, distilled | One GPU. The adapter is small. The shell default batch is 2. |
| Full 200M | One 24 GB L4 at batch 32. Batch 64 ran out of memory. The shell default batch is 1, with gradient accumulation 8. |
| Full 1B | One 24 GB L4 with `--optim adamw_bnb_8bit`, `--bf16`, `--gradient_checkpointing`, and `--batch_size 8`. A second L4 is a second copy, for speed, not extra room. |

DDP does not pool memory. A model that does not fit on one GPU does not fit because a second GPU was added.

Training precision is bf16 on the L4 command. `--bf16` and `--fp16` cannot both be set. Inference on CUDA loads the model in fp16 even when training used bf16. CPU inference uses fp32 and is only practical for a distilled model.

### Multi-GPU

Supported: DDP, by passing two or more ids in `--gpu_ids`. The default distribution `auto` does that relaunch. `--distribution ddp` is the same path.

Not supported: DeepSpeed ZeRO, ZeRO offload, FSDP, DataParallel, tensor parallelism, and pipeline parallelism. Those flags were removed after they failed to train this checkpoint. The record is `docs/deepspeed.md`.

Generation uses `use_cache=False`. IndicTrans2 beam search crashes if the KV cache is left on. That is not a flag.

### Credentials

Store these outside the code and outside the image.

| Name | When |
| --- | --- |
| `HF_TOKEN` | Required for the three 1B checkpoints. The Hugging Face account that owns the token must accept the model card. Set `HUGGING_FACE_HUB_TOKEN` to the same value in the shell that launches training. |
| `HUGGING_FACE_HUB_TOKEN` | Same token. Set both. A token exported in a different shell is invisible to the process. |

The distilled 200M and 320M checkpoints are public. COMET, BERTScore, and the NLTK wordnet download do not need a token. No other API key is read.

## 4. Outputs and metrics

### Files training writes

All of these go under `--output_dir` unless the flag already contains a directory.

| Path | What it is |
| --- | --- |
| `prepared/` | Converted train and dev pair files. |
| `checkpoint-<step>/` | A Trainer checkpoint. Top-3 and last-2 are kept by default. |
| `checkpoint_manifest.json` | Which checkpoint is in the top list, the recent list, or both. |
| `final_model/` | The weights to load. Default path is `<output_dir>/final_model`. |
| `final_model/finetune_meta.json` | Mode, checkpoint id, pairs, learning rate, best checkpoint, best metric, epochs completed. |
| `training_logs.csv` | One row each time the trainer logs. |
| `eval_metrics.csv` | One row per evaluation. |
| `eval_samples.csv` | `step`, `src_lang`, `tgt_lang`, `src`, `ref`, `hyp` for the validation generation. |
| `rca_report.html` | Written at the end when `--write_rca` is on (the default). |

With `--eval_strategy` and `--save_strategy` both set to `epoch` or both set to `steps`, `final_model` is the best checkpoint by `--metric_for_best_model`, not the last epoch. The last epoch remains as a `checkpoint-<step>` directory when it is inside the last-K list.

`--save_only_model` stores weights without the optimizer. The full and full-LoRA shell scripts turn that on. The 1B command above did not, so those checkpoints can resume the optimizer. A checkpoint saved with `--save_only_model` resumes the weights only.

### How to load the result

| Mode | What `final_model` contains | Translate command |
| --- | --- | --- |
| `full` | A full IndicTrans2 model | `--model <final_model>` |
| `full_lora` | The adapter merged into a full model | `--model <final_model>` |
| `lora`, `qlora` | A PEFT adapter (`adapter_config.json`, adapter weights) | `--model <base hub id> --adapter <final_model>` |

In Python, a merged model is `AutoModelForSeq2SeqLM.from_pretrained(final_model_dir, trust_remote_code=True)`. An adapter is the base model plus `PeftModel.from_pretrained(model, adapter_dir)`.

### What to show live

Every `--logging_steps` (default 50), the process prints and appends to `training_logs.csv`:

- `loss` (training loss)
- `learning_rate`
- `epoch`
- `step`

At each evaluation, the same log gains:

- `eval_loss`
- `eval_entropy`
- `eval_mean_token_accuracy`
- the translation metrics named in `--eval_metrics`

`--report_to tensorboard` also writes a TensorBoard event file. The default is `none`, so the live surface without that flag is stdout plus the three CSVs.

`eval_loss` is the metric that picks the best checkpoint and early stopping. Lower is better. `eval_mean_token_accuracy` is the one allowed metric where higher is better. chrF++, COMET, BLEU, BERTScore, METEOR, TER, and exact match are recorded and never choose the checkpoint.

### Final evaluation metrics

These are the scores in `eval_metrics.csv` and, for a held-out file, in `inference_metrics.csv`.

| Name | Meaning | Direction |
| --- | --- | --- |
| `eval_loss` | Teacher-forced validation loss | lower is better |
| `eval_entropy` | Token entropy on the validation batch | lower is better |
| `eval_mean_token_accuracy` | Mean token accuracy | higher is better |
| `chrf2pp` | chrF++ (sacrebleu, word order 2) | higher is better |
| `bleu` | SacreBLEU, `flores200` tokenization | higher is better |
| `ter` | Translation edit rate | lower is better |
| `meteor` | METEOR, after the NLTK wordnet download | higher is better |
| `comet` | `Unbabel/wmt22-comet-da`, scaled when the raw score is at most 1.5 | higher is better |
| `bertscore` | BERTScore F1 with `xlm-roberta-large`, times 100 | higher is better |
| `exact_match` | Percent of lines that match the reference after Unicode NFC and whitespace collapse | higher is better |

`metric_errors` on the same row records a metric that failed without failing the run. chrF++, BLEU, TER, and exact match finish first. METEOR, BERTScore, and COMET are the slow ones and do not print a progress bar.

`inference_predictions.csv` columns: `src_lang`, `tgt_lang`, `src`, `reference`, `prediction`.

`inference_metrics.csv` columns: `src_lang`, `tgt_lang`, `n`, `model`, `adapter`, one column per requested metric, `metric_errors`.

## 5. Parameters

Levels: **Basic** is shown on every run, **Advanced** is behind “Show advanced options”, **Expert** is rarely changed.

A parenthetical list is the only set the parser accepts. A numeric range is the range to offer in Studio. The parser still accepts any number of that type outside the range, except where the cell says the code rejects it.

`train.py` defaults are below. The shell scripts override some of them: LoRA and QLoRA use batch 2 and accumulation 4; full and full-LoRA use batch 1, accumulation 8, learning rate `2e-5`, and 8-bit Adam.

### Training

| Name | What it does | Default | Allowed range / values | Level | Depends on |
| --- | --- | --- | --- | --- | --- |
| `--model_hf_id` | Hub id of the base checkpoint. Also selects the translation direction. | required | (`ai4bharat/indictrans2-en-indic-dist-200M`, `ai4bharat/indictrans2-en-indic-1B`, `ai4bharat/indictrans2-indic-en-dist-200M`, `ai4bharat/indictrans2-indic-en-1B`, `ai4bharat/indictrans2-indic-indic-dist-320M`, `ai4bharat/indictrans2-indic-indic-1B`) | Basic | Language pair |
| `--train_mode` | Which weights move. | `lora` | (`full`, `full_lora`, `lora`, `qlora`) | Basic | GPU memory |
| `--train_json` | Train file. | none | A `.json`, `.jsonl`, `.csv`, `.tsv`, or `.tab` path | Basic | Data split |
| `--val_json` | Validation file. | none | Same file types as `--train_json` | Basic | `--train_json` |
| `--output_dir` | Checkpoints, CSVs, and the report. | `./it2_out/exp` | Any path | Basic | Run name |
| `--epochs` | Passes over the training file. | `2` | 1–10 (usual: 1, 2, 3, 5, 8, 10) | Basic | Dataset size, early stopping |
| `--learning_rate` | Adam step size. | `2e-4` for `lora` and `qlora`; `2e-5` for `full` and `full_lora` | LoRA/QLoRA 1e-4–5e-4 (usual: 1e-4, 2e-4, 3e-4, 5e-4). Full and full-LoRA 1e-5–5e-5 (usual: 1e-5, 2e-5, 3e-5, 5e-5). | Basic | `--train_mode`, batch size |
| `--batch_size` | Sentences on one GPU per forward. | `2` for LoRA and QLoRA; `1` for full and full-LoRA. The 1B run used `8`. | 1–32 (usual: 1, 2, 4, 8, 16, 32) | Basic | GPU memory, `--max_length`, `--train_mode` |
| `--gpu_ids` | Which GPUs to use. Two or more ids turn `auto` into DDP. | none | (`0`, `0,1`) on these two L4s. Code: comma-separated digits. | Basic | Visible GPUs. Omit if already launched with torchrun. |
| `--bf16` | Train in bfloat16. | off in `train.py`; on in the shell scripts | (on, off) | Basic | Compute capability 8.0+. Not with `--fp16`. |
| `--gradient_accumulation_steps` | Forwards added before one optimizer step. | `4`; full scripts use `8` | 1–16 (usual: 1, 2, 4, 8, 16) | Advanced | `--batch_size`, GPU count |
| `--eval_batch_size` | Sentences per GPU during validation generation. | `2`. The 1B run used `8`. | 1–8 (usual: 1, 2, 4, 8) | Advanced | GPU memory, `--generation_num_beams` |
| `--optim` | Optimizer implementation. | `adamw_torch`; `paged_adamw_32bit` for `qlora`; full scripts use `adamw_bnb_8bit` | (`adamw_torch`, `adamw_torch_fused`, `adamw_hf`, `adamw_bnb_8bit`, `paged_adamw_8bit`, `paged_adamw_32bit`, `adafactor`) | Advanced | `--train_mode`, GPU memory |
| `--gradient_checkpointing` | Recompute activations to save memory. | off in `train.py`; on in the shell scripts | (on, off) | Advanced | GPU memory. Needed for full 1B on 24 GB. |
| `--fp16` | Train in float16. | off | (on, off) | Advanced | Compute capability below 8.0. Not with `--bf16`. |
| `--max_length` | Tokens kept on the source and on the target. | `256` | 64–256 (usual: 128, 256) | Advanced | GPU memory. 256 is the IndicTrans2 window. |
| `--eval_strategy` | When validation runs. | `epoch` | (`epoch`, `steps`, `no`) | Advanced | `--save_strategy` |
| `--save_strategy` | When a checkpoint is written. | `epoch` | (`epoch`, `steps`, `no`) | Advanced | `--eval_strategy` |
| `--eval_metrics` | Translation scores written at evaluation. They do not pick the checkpoint. | all seven | One or more of (`chrf2pp`, `comet`, `bleu`, `bertscore`, `meteor`, `ter`, `exact_match`) | Advanced | `--eval_strategy` |
| `--metric_for_best_model` | Score that ranks checkpoints and early stopping. | `eval_loss` | (`eval_loss`, `eval_entropy`, `eval_mean_token_accuracy`) | Advanced | `--eval_strategy`, `--save_strategy` |
| `--early_stopping_patience` | Stop after this many evaluations with no improvement. | `3`. The 1B run used `0`. | 0–10 (usual: 0, 1, 2, 3, 5). Code rejects negatives. | Advanced | `--metric_for_best_model`. Eval and save strategies must match and must not be `no`. |
| `--src_lang` | Keep only this source language. | none | The FLORES tags and aliases in section 2 | Advanced | `--model_hf_id` direction |
| `--tgt_lang` | Keep only this target language. | none | The same closed list as `--src_lang` | Advanced | `--src_lang`, `--model_hf_id` direction |
| `--final_model_dir` | Where the weights to keep are written. | `<output_dir>/final_model` | Any path | Advanced | `--output_dir` |
| `--lora_r` | LoRA rank. | `8` | 4–64 (usual: 4, 8, 16, 32, 64) | Advanced | `--train_mode` is `lora`, `qlora`, or `full_lora` |
| `--lora_alpha` | LoRA scale. | `16` | 8–128 (usual: 8, 16, 32, 64, 128). Keep it at twice `--lora_r`. | Advanced | `--lora_r` |
| `--lora_dropout` | Dropout on the adapter. | `0.05` | 0.0–0.2 (usual: 0, 0.05, 0.1) | Advanced | `--train_mode` is not `full` |
| `--lora_target_modules` | Modules that receive an adapter. | `q_proj,k_proj`. `full_lora` uses all six. | (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`), comma-separated | Advanced | `--train_mode` is not `full` |
| `--generation_num_beams` | Beam width while scoring the validation file during training. | `5`. The 1B run used `1`. | 1–5 (usual: 1, 2, 4, 5) | Advanced | `--eval_strategy`, GPU memory |
| `--eval_steps` | Evaluations every this many steps. | `50` | 10–1000 (usual: 50, 100, 200, 500) | Advanced | `--eval_strategy steps` |
| `--save_steps` | Checkpoints every this many steps. | `500` | 50–2000 (usual: 100, 500, 1000) | Advanced | `--save_strategy steps` |
| `--eval_max_new_tokens` | Maximum generated length during validation. | `256` | 16–256 (usual: 64, 128, 256) | Advanced | `--max_length` |
| `--top_n_checkpoints` | How many best checkpoints to keep. | `3` | 1–10 (usual: 1, 2, 3, 5) | Advanced | `--save_strategy` |
| `--last_k_checkpoints` | How many newest checkpoints to keep as well. | `2` | 0–5 (usual: 1, 2, 3) | Advanced | `--save_strategy`, `--top_n_checkpoints` |
| `--save_only_model` | Checkpoint stores weights and not the optimizer. | off in `train.py`; on in the full scripts | (on, off) | Advanced | Resume |
| `--data_dir` | Official layout already on disk. Skips JSON conversion. | none | A path with `train/` and `dev/` pair folders | Advanced | Replaces `--train_json` and `--val_json` |
| `--adapter_path` | Start from this saved adapter. | none | A directory with `adapter_config.json`, or omitted | Advanced | `--train_mode` |
| `--resume_from_checkpoint` | Continue a stopped run. | none | A `checkpoint-<step>` directory, or omitted | Advanced | A previous checkpoint, `--save_only_model` |
| `--seed` | Random seed. | `42` | Any integer (usual: 1, 42, 1234) | Expert | |
| `--weight_decay` | AdamW weight decay. | `0.01` | 0.0–0.2 (usual: 0, 0.001, 0.01, 0.05, 0.1) | Expert | Optimizer |
| `--warmup_ratio` | Fraction of steps used for LR warmup. | `0.03` | 0.0–0.20 (usual: 0.03, 0.05, 0.06, 0.1) | Expert | Total training steps |
| `--lr_scheduler_type` | Learning-rate schedule. | `cosine` | (`linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`) | Expert | Learning rate, warmup |
| `--max_grad_norm` | Gradient clipping. | `1.0` | 0.0–5.0 (usual: 0.5, 1.0) | Expert | Optimizer |
| `--label_smoothing_factor` | Label smoothing on the translation loss. | `0.1` | 0.0–0.2 (usual: 0, 0.1). Code rejects values outside `[0, 1)`. | Expert | Training loss |
| `--logging_steps` | How often a loss line is written. | `50` | 1–500 (usual: 10, 25, 50, 100) | Expert | |
| `--distribution` | `auto` follows `--gpu_ids`. `ddp` is the same multi-GPU path. | `auto` | (`auto`, `ddp`) | Expert | `--gpu_ids` |
| `--ddp_timeout` | Seconds a rank waits for the others. | `14400` | 60–86400 (usual: 1800, 3600, 7200, 14400). Code rejects values below 1. | Expert | Multi-GPU |
| `--num_workers` | Data-loader worker processes. | `4` | 0–16 (usual: 0, 2, 4, 8) | Expert | CPU cores |
| `--eval_accumulation_steps` | Evaluation batches gathered before metric tensors move. | `1` | 1–16 (usual: 1, 2, 4, 8) | Expert | `--eval_batch_size`, GPU memory |
| `--early_stopping_threshold` | Smallest metric move that counts as an improvement. | `0.0` | 0.0–1.0 (usual: 0, 0.001, 0.01). Code rejects negatives. | Expert | `--early_stopping_patience` |
| `--save_total_limit` | Extra cap on the top-K plus last-K union. `0` means no extra cap. | `0` | 0–10 (usual: 0, 2, 3, 5) | Expert | `--top_n_checkpoints`, `--last_k_checkpoints` |
| `--bnb_compute_dtype` | Compute dtype inside 4-bit QLoRA. | `bfloat16` | (`float16`, `bfloat16`) | Expert | `--train_mode qlora` |
| `--bnb_4bit_quant_type` | 4-bit storage type. | `nf4` | (`nf4`, `fp4`) | Expert | `--train_mode qlora` |
| `--bnb_4bit_use_double_quant` | Also quantize the quantization constants. | off | (on, off) | Expert | `--train_mode qlora` |
| `--model_id` | Local weights or a hub id to load. Direction still comes from `--model_hf_id`. | the value of `--model_hf_id` | A hub id or a local directory | Expert | `--model_hf_id` |
| `--report_to` | Live logger besides the CSVs. | `none` | (`none`, `tensorboard`) | Expert | |
| `--write_rca` / `--no-write_rca` | Write `rca_report.html` after training. | on | (on, off) | Expert | The training CSVs |
| `--eval_metrics_csv` | Validation metric file name. | `eval_metrics.csv` | A file name or a path | Expert | `--output_dir` |
| `--eval_samples_csv` | Validation sentence file name. | `eval_samples.csv` | A file name or a path | Expert | `--output_dir` |
| `--training_logs_csv` | Training log file name. | `training_logs.csv` | A file name or a path | Expert | `--output_dir` |

`full_qlora` is rejected. There is no `--train_method` flag. Training is supervised translation only.

Shell environment names are the flag names in upper case: `MODEL_HF_ID`, `TRAIN_JSON`, `VAL_JSON`, `GPU_IDS`, `EPOCHS`, `BATCH_SIZE`, `EVAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `LEARNING_RATE`, `OPTIM`, `BF16`, `FP16`, `GRADIENT_CHECKPOINTING`, `SAVE_ONLY_MODEL`, `EVAL_METRICS`, `DISTRIBUTION`, `DDP_TIMEOUT`, and the rest of the flags above. `BF16=0` turns bf16 off. `WRITE_RCA=0` passes `--no-write_rca`. `BNB_4BIT_USE_DOUBLE_QUANT=1` passes `--bnb_4bit_use_double_quant`.

### Inference and held-out evaluation

These are flags of `python -m indictrans2_finetune.cli translate`. Hyphens, not underscores.

| Name | What it does | Default | Allowed range / values | Level | Depends on |
| --- | --- | --- | --- | --- | --- |
| `--json-file` | Score a train-style JSON file. `tgt_text` is the reference. | none | A train-style JSON path | Basic | Not together with `--text` or `--file` |
| `--text` | Translate one sentence and print it. | none | Any string | Basic | `--src-lang`, `--tgt-lang` |
| `--file` | Translate a text file, one sentence per line. | none | A text path | Basic | `--src-lang`, `--tgt-lang` |
| `--model` | Merged model directory, or the base hub id when an adapter is used. | the hub id implied by the language pair | A hub id or a directory | Basic | `--adapter` for LoRA and QLoRA |
| `--adapter` | LoRA or QLoRA directory. | none | A directory with `adapter_config.json`, or omitted | Basic | `--train_mode` was `lora` or `qlora` |
| `--src-lang` | Source language for `--text` and `--file`. Optional filter for `--json-file`. | none | The FLORES tags and aliases in section 2 | Basic | `--text` or `--file` |
| `--tgt-lang` | Target language for `--text` and `--file`. Optional filter for `--json-file`. | none | The same closed list as `--src-lang` | Basic | `--src-lang` |
| `--eval-metrics` | Metrics to write for a JSON file. | none | One or more of (`chrf2pp`, `comet`, `bleu`, `bertscore`, `meteor`, `ter`, `exact_match`) | Basic | `--json-file` |
| `--predictions-csv` | Per-sentence output. | `inference_predictions.csv` | A file name or a path | Advanced | `--json-file` |
| `--metrics-csv` | Corpus scores. | `inference_metrics.csv` | A file name or a path | Advanced | `--json-file`, `--eval-metrics` |
| `--max-length` | Maximum generated length. | `256` | 64–256 (usual: 128, 256) | Advanced | GPU memory |
| `--num-beams` | Beam width. | `5` | 1–5 (usual: 1, 2, 4, 5) | Advanced | GPU memory |
| `--batch-size` | Sentences per generate call, per GPU. | `8` | 1–32 (usual: 1, 2, 4, 8, 16, 32) | Advanced | GPU memory, `--num-beams` |
| `--gpu-ids` | Split the file across these GPUs. | none | (`0`, `0,1`) on these two L4s | Advanced | Visible GPUs |
| `--visualize` | Write an HTML chart page. | off | (on, off) | Advanced | `--json-file` |
| `--report-html` | Path for that chart page. | `inference_report.html` next to the metrics CSV | Any path | Expert | `--visualize` |

`--local-rank` is set by `torchrun`. Do not set it in Studio.
