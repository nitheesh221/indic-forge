# IndicTrans2 fine-tuning, explained from the start

This page is the map of the whole project. Read it once before you change flags. Each section says what the code actually does, why, and the value it starts from. Deeper notes live in the linked pages. Those pages do not replace this one.

The project fine-tunes AI4Bharat IndicTrans2 on your parallel sentences. A parallel sentence is one source line and the translation you want for it.

## Architecture

IndicTrans2 is an encoder–decoder translation model.

- The **encoder** reads the source sentence and builds a numeric summary of it.
- The **decoder** writes the translation one token at a time. A token is a piece of a word, not always a whole word.
- During training the decoder is shown the correct translation so far and asked for the next token. That is called teacher forcing. The loss is cross-entropy: how surprised the model was by the correct next token. Label smoothing of `0.1` softens that target a little so the model does not put all of its confidence on one token.

The model does not handle every language pair with one checkpoint. There are three families:

| Direction | Example | Distilled checkpoint | 1B checkpoint |
| --- | --- | --- | --- |
| English → Indic | English to Hindi | `ai4bharat/indictrans2-en-indic-dist-200M` | `ai4bharat/indictrans2-en-indic-1B` |
| Indic → English | Hindi to English | `ai4bharat/indictrans2-indic-en-dist-200M` | `ai4bharat/indictrans2-indic-en-1B` |
| Indic → Indic | Hindi to Tamil | `ai4bharat/indictrans2-indic-indic-dist-320M` | `ai4bharat/indictrans2-indic-indic-1B` |

Distilled means a smaller student model. It is the right starting point. The 1B model is the larger one. Use it when the distilled model is not accurate enough and you have the GPU memory.

IndicTrans2 was trained with a maximum length of 256 tokens. This repo keeps that window. A longer sentence is cut. The loss never sees the cut-off ending.

The tags on each sentence are FLORES-200 tags, such as `eng_Latn` and `hin_Deva`. The tag names the language and the script. `hi` is accepted as an alias and stored as `hin_Deva`. Run `indictrans2-ft langs` for the full list.

## Preprocessing of data

Training never reads your JSON directly as model input. It converts the file, tags the sentences, then tokenizes them.

**Step 1. Your file.** JSON, JSONL, CSV, or TSV. Each row needs a source sentence, a target sentence, and the two language tags. The usual keys are `src` or `src_text`, `tgt` or `tgt_text`, `src_lang`, and `tgt_lang`. A headerless mess is skipped and counted. Empty lines and rows where source and target are the same language are skipped.

**Step 2. Official folder layout.** The converter writes:

```text
prepared/train/eng_Latn-hin_Deva/train.eng_Latn
prepared/train/eng_Latn-hin_Deva/train.hin_Deva
prepared/dev/eng_Latn-hin_Deva/dev.eng_Latn
prepared/dev/eng_Latn-hin_Deva/dev.hin_Deva
```

Line 10 in the source file is the pair of line 10 in the target file. If you already have this layout, pass `--data_dir` and the JSON conversion is skipped.

**Step 3. Which pairs are kept.** The checkpoint direction decides. An English→Indic model will not train a Hindi→English pair. If you leave `--src_lang` and `--tgt_lang` empty, every pair in the file that matches the checkpoint is trained together. Set both flags only when you want one pair. Every trained pair must also exist in the validation split.

**Step 4. IndicTransToolkit tags.** Before tokenization, `IndicProcessor` rewrites each sentence into the form the model was trained on. Source sentences are tagged as inputs. Target sentences are tagged as outputs. The target call swaps the language arguments with `is_target=True`. That swap is the same pattern as the official IndicTrans2 LoRA script. Do not "fix" it.

**Step 5. Tokens.** The source is tokenized up to 256 tokens. The target is tokenized with the target side of the tokenizer, also up to 256, and stored as `labels`. Padding is not applied yet.

**Step 6. A batch.** At training time the collator left-pads the source and right-pads the labels. Padded label positions are `-100`, so they do not count in the loss. The decoder input is built from those labels.

On more than one GPU, only rank 0 writes the `prepared/` folder. The other ranks wait, then read the same files.

A small English–Hindi public-service sample ships with the repo so you can check that the pipeline runs. It is not a training set. A real full fine-tune needs a large, clean, multi-pair file. Balance the languages in that file yourself. The trainer gives every sentence the same weight, so a language with 100 times more lines gets 100 times more of the update.

## GPU selection

One process must use one GPU. The flag is `--gpu_ids`.

- `--gpu_ids 0` uses the first visible NVIDIA GPU.
- `--gpu_ids 1` uses the second one, when you want to leave the first card free.
- `--gpu_ids 0,1` uses both. The script relaunches itself under `torchrun`. You do not type `torchrun` yourself.

The shell scripts read `GPU_IDS`. The default is `0`.

```bash
GPU_IDS=0 bash scripts/train_lora.sh
GPU_IDS=0,1 bash scripts/train_lora.sh
```

If you start `torchrun` yourself, do not also pass `--gpu_ids`. The process group is already started, and the extra flag is rejected.

What this does not do: it does not look at free memory and it does not pick a GPU for you. You choose the ids. `CUDA_VISIBLE_DEVICES` is set from those ids before PyTorch loads, so the process cannot see the other cards.

QLoRA on one id stays off DataParallel. DataParallel would put the 4-bit model on every visible GPU inside one process, and that path fails.

## Multi-GPU support

Two GPUs means two processes. Each process holds its own copy of the model when you use `auto` or `ddp`, sees different sentences, and averages gradients once per update.

The latest multi-GPU behavior, which is what you should expect on a 2× GPU box:

1. Training arguments are built before the model loads. That pins process 0 to the first id and process 1 to the second, and it opens the process group with `--ddp_timeout` (default 14400 seconds, which is 4 hours).
2. Only rank 0 converts JSON and only rank 0 writes `final_model`, `finetune_meta.json`, and the RCA report.
3. Each GPU translates its own slice of the validation set. Rank 0 gathers the translations, scores them, and writes the CSVs. If one slice fails, every rank still meets at the gather, and the error is stored in `metric_errors`.
4. Length-grouped batches are on for one GPU and off for two or more. Grouping by length can give the two processes a different number of steps, and the shorter one then waits until the timeout.
5. Before a checkpoint is deleted, both ranks wait. Rank 0 then deletes. This covers both Top-K and Last-K.

DDP does not make a model fit that did not fit on one GPU. It copies the model. A full 1B update that runs out of memory on one card still runs out of memory as two copies. That case is ZeRO, in the distribution section below.

Adding a GPU multiplies the effective batch:

`batch size × number of GPUs × gradient accumulation steps`

If the one-GPU run used accumulation 8, the two-GPU run should use accumulation 4 when you want the same number of sentences per update. The learning rate stays where it was.

## Base model selection

Pick the checkpoint in this order.

1. **Direction.** English source and Indic target → an `en-indic` model. Indic source and English target → `indic-en`. Both sides Indic → `indic-indic`. The code rejects a pair that does not match `--model_hf_id`.
2. **Size.** Start with the distilled model (`200M`, or `320M` for Indic→Indic). Move to the `1B` model only when you need the accuracy and you have the memory.
3. **Local weights.** `--model_hf_id` is always one of the six official ids, because that is how the direction is checked. `--model_id` is the folder or hub id that is actually loaded. It defaults to the same value. Point `--model_id` at a previous `final_model` when you are continuing from your own full fine-tune. For an adapter, keep the base id in `--model_id` and pass the adapter with `--adapter_path`.

`--train_mode full` plus `--adapter_path` merges that adapter into the base model first, then trains every weight. The other modes continue the adapter.

## Training method

This repo trains with supervised fine-tuning only. There is no `--train_method` flag, because PPO, GRPO, DPO, and KTO are not implemented. The choice you pass is `--train_mode`, which only picks which weights move.

SFT fits this data because each row already has the correct translation. The model is not asked to invent a sentence and then get a score from a reward model. PPO, GRPO, DPO, and KTO are for that second situation. The IndicTrans2 paper's own quality gain on human bitext came from another supervised stage, not from those methods.

The four training modes do not change the lesson. They change which numbers inside the model are allowed to move:

| Mode | What moves | Learning rate when you do not set one | Saved `final_model` |
| --- | --- | --- | --- |
| `lora` | A small adapter on `q_proj` and `k_proj` | `2e-4` | The adapter |
| `qlora` | The same adapter, base weights stored in 4-bit | `2e-4` | The adapter |
| `full_lora` | A wider adapter, on attention and the feed-forward layers | `2e-5` | The merged full model |
| `full` | Every weight | `2e-5` | The full model |

`2e-4` on a full update is too high and will usually damage the pretrained model. `2e-5` on a small LoRA adapter is too low and the adapter learns slowly. Leave each mode on its own rate unless you are running a deliberate experiment.

The full explanation, with the papers, is [why-sft.md](why-sft.md).

## Hyperparameter tuning

Change one knob at a time, and keep a note of the chrF++ and the validation loss. The defaults below are the ones in `train.py` and the shell scripts. They are a starting point, not a searched optimum.

| Flag | Starts at | What it does | When you touch it |
| --- | --- | --- | --- |
| `--epochs` | 2 | How many passes over the training file | Validation loss rising after epoch 1: stop at 1. Still falling at the end, on a large set: try 3. |
| `--learning_rate` | `2e-4` LoRA / QLoRA, `2e-5` full / full LoRA | Size of each update | Loss explodes or the output becomes nonsense: divide by 10. Loss barely moves on LoRA: you are probably already at `2e-4`. |
| `--batch_size` | 2 in `train.py`, 1 in `scripts/train_full.sh` | Sentences on the GPU at once | Out of memory: set 1. Lots of free memory on LoRA: try 4 or 8. |
| `--gradient_accumulation_steps` | 4 in `train.py`, 8 in `scripts/train_full.sh` | How many micro-batches are added before one update | Effective batch = batch × accumulation × GPUs. Raise it when the loss jitters and memory is already tight. |
| `--warmup_ratio` | 0.03 | Fraction of steps spent raising the learning rate from zero | Leave it. A huge warmup on a short run wastes the only steps you have. |
| `--lr_scheduler_type` | cosine | How the rate falls after warmup | Cosine is the default. `inverse_sqrt` is the older machine-translation schedule. Change it only in a comparison run. |
| `--weight_decay` | 0.01 | A small penalty on large weights | Leave it for the first runs. |
| `--max_grad_norm` | 1.0 | Clips a huge gradient | Leave it. It is the safety brake. |
| `--label_smoothing_factor` | 0.1 | Softens the target token | Leave it for translation. The loss will not fall to zero. That plateau is normal. |
| `--optim` | `adamw_torch`, except QLoRA when you call `train.py` with no optimizer (then `paged_adamw_32bit`). `scripts/train_full.sh` passes `adamw_bnb_8bit` | Which Adam implementation | Full 1B that does not fit: `adamw_bnb_8bit`. Distilled and LoRA: `adamw_torch`. |
| `--max_length` | 256 | Tokens kept per side | Keep 256. That is the IndicTrans2 window. |
| `--seed` | 42 | Shuffle and initialization | Change it only when you want a second run of the same settings. |

`--logging_steps` (50) only controls how often a line is written. It does not change the update.

There is no automatic search over these values in the repo. A useful first comparison, after the run works, is three LoRA learning rates (`1e-4`, `2e-4`, `3e-4`) with everything else fixed, judged by validation loss and chrF++.

## Framework used

| Piece | Role |
| --- | --- |
| PyTorch | Tensors, GPU math, and the process group for multi-GPU |
| Hugging Face Transformers | Loads IndicTrans2 and runs `Seq2SeqTrainer` |
| PEFT | LoRA and QLoRA adapters |
| bitsandbytes | 4-bit QLoRA, and 8-bit Adam when you ask for it |
| Accelerate / `torchrun` | Starts one process per GPU. This repo relaunches `torchrun` for you |
| DeepSpeed | Optional. Only for `zero2`, `zero3`, and `offload` |
| IndicTransToolkit | Language-tag preprocessing and postprocessing |
| `datasets` | Holds the tokenized sentences |
| sacrebleu | chrF++, BLEU, TER |
| unbabel-comet, bert-score, NLTK | COMET, BERTScore, METEOR. Used when you ask for those metrics |
| TensorBoard | Optional live curves, `--report_to tensorboard` |

The Python package is the interpreter in `.venv`. IndicTransToolkit is not supported on Windows. This repo stays on Transformers 4.x, because the toolkit still imports tokenizer types that moved in Transformers 5.

## PEFT training

PEFT here means a LoRA adapter. The frozen base model keeps the general translation ability. The adapter is a small pair of matrices on chosen layers. Four approaches are wired. They are not a grid that was searched. They are the four modes the code knows how to build.

**`lora`.** Rank `8`, alpha `16`, dropout `0.05`, modules `q_proj,k_proj`. Learning rate `2e-4`. Optimizer `adamw_torch`. This is the default experiment. Rank is the width of the adapter. Alpha scales the adapter update. The usual ratio in this repo is alpha = 2 × rank. Dropout `0.05` randomly zeros adapter activations so the adapter does not memorize a small file.

**`qlora`.** The same adapter numbers. The base weights are loaded in 4-bit (`nf4`, compute dtype `bfloat16`). Double quantization is off unless you pass `--bnb_4bit_use_double_quant`. Calling `train.py` without `--optim` selects `paged_adamw_32bit`, which is the optimizer that can page to CPU when the GPU is tight. The shell script `scripts/train_qlora.sh` passes `adamw_torch` unless you set `OPTIM`. QLoRA runs with `auto` or `ddp` only. It refuses ZeRO and FSDP.

**`full_lora`.** Same rank, alpha, and dropout. The modules widen to `q_proj,k_proj,v_proj,out_proj,fc1,fc2`, so the feed-forward layers get an adapter too. The learning rate drops to `2e-5` because more of the network is moving. At the end the adapter is merged into a full model, except under ZeRO or FSDP, where a merge would see only a shard and the adapter is saved unmerged.

**`full`.** No adapter. Every weight trains at `2e-5`. `scripts/train_full.sh` uses batch 1, accumulation 8, 8-bit Adam, gradient checkpointing, and bf16, because a full update stores an Adam state for every parameter. The LoRA flags are still accepted by that script so the command shape stays the same. They are not used.

Knobs you can combine, and what a change does:

| Knob | Safe first change | Effect |
| --- | --- | --- |
| `--lora_r` 4, 8, or 16 | Stay at 8 until 8 underfits | Higher rank can fit a more specific domain and overfit a small file faster. If you raise rank, set alpha to twice the rank (`8→16`, `16→32`). |
| `--lora_alpha` | Keep it at twice the rank | Alpha / rank is the scale of the adapter. Raising alpha without raising rank makes each step larger. |
| `--lora_dropout` 0, 0.05, 0.1 | Stay at 0.05 | Higher dropout is for a small dataset that memorizes. `0` is for a very large set. |
| `--lora_target_modules` | Use the mode default | Adding `v_proj` and `out_proj` is the step from `lora` toward `full_lora`. You do not need to type the modules if you pick `--train_mode full_lora`. |

One combination at a time is enough to read the result. A full factorial of rank × alpha × dropout × modules × learning rate is a later experiment, after one setting already beats the untouched checkpoint on your validation file.

Gradient checkpointing is on in the shell scripts. For LoRA it needs the embedding outputs to require grad, which the code sets. Without that, the first backward step fails. It slows each step and lets a larger model fit.

## Checkpoints: Top-K and Last-K

Two lists are kept, and a folder that is on both lists is stored once.

- **Top-K** (`--top_n_checkpoints`, default 3) keeps the best scores of `--metric_for_best_model`. The default metric is `eval_loss`. Lower is better. Token accuracy is the one metric where higher is better. chrF++ is not allowed as this metric. The code would treat a higher chrF++ as worse.
- **Last-K** (`--last_k_checkpoints`, default 2) keeps the newest saves, so you can still see the end of training when the best score was earlier.

`checkpoint_manifest.json` records which folder is in which list. With eval and save both set to `epoch`, the weights in memory at the end are the best validation loss, and those weights are what `final_model` saves. The last epoch remains on disk as a `checkpoint-<step>` folder.

Early stopping uses that same score. `--early_stopping_patience` defaults to `3`, which means three evaluations in a row with no improvement. The first evaluation only records the starting score, so a 2-epoch run still finishes both epochs. Set `--epochs` higher than the patience when you want a long run to be able to stop. `--early_stopping_patience 0` turns it off. `--early_stopping_threshold` is how much the score must move to count as an improvement; `0.0` means any strict improvement resets the count. The patience counts evaluations, so with `--eval_strategy steps` it is not the same as three epochs. chrF++ does not stop the run. On more than one GPU, rank 0 decides and the other ranks stop with it. After the stop, the best checkpoint is still what `final_model` saves. Resuming a run keeps the previous best score and starts the stall count again from zero.

`finetune_meta.json` next to `final_model` records `best_model_checkpoint`, `best_metric`, and `epochs_completed`.

`--save_total_limit 0` means no extra cap. A positive number trims the union down to that many folders, best scores first.

On multi-GPU, both ranks sync before a delete, then only rank 0 deletes. ZeRO and FSDP must enter the final save on every rank. Only rank 0 writes the files.

The longer version is [checkpoints.md](checkpoints.md).

## Evaluation of the trained model

There are two different evaluations. Do not mix them up.

**During training.** Every epoch (the default) the trainer computes validation loss, token entropy, and token accuracy on the dev set. That is teacher-forced: the correct translation is shown, and the model predicts the next token. This is what selects the best checkpoint. After that, the code also generates a translation for each dev sentence and, if you asked, scores it. Generation uses the beam width in `--generation_num_beams` (default 5) and runs with the KV cache off, because IndicTrans2 beam search breaks when the cache is left on. That generation does not choose the checkpoint. It is there so you can see translation quality. On a long dev set it can take longer than the training epoch. For a first experiment, `GENERATION_NUM_BEAMS=1` and a short metric list (`chrf2pp` or `chrf2pp bleu ter exact_match`) keep the epoch moving. You choose the batch, the accumulation, and the metric list for each experiment.

**After training.** When the GPU is free, score a saved model on a JSON file. This is the run where beam 5, COMET, and BERTScore belong.

```bash
PYTHONPATH=src python -m indictrans2_finetune.cli translate \
  --json-file /path/to/valid.json \
  --model it2_out/full/final_model \
  --eval-metrics chrf2pp comet bleu bertscore meteor ter exact_match \
  --predictions-csv it2_out/full/inference_predictions.csv \
  --metrics-csv it2_out/full/inference_metrics.csv \
  --max-length 256 --num-beams 5 --batch-size 8
```

A LoRA or QLoRA model needs `--model` set to the base checkpoint and `--adapter` set to `final_model`. A full model or a merged `full_lora` model is passed only as `--model`.

`inference_predictions.csv` has one row per sentence: source, reference, prediction. `inference_metrics.csv` has one corpus row. A metric that fails is named in `metric_errors`. The other scores are still written.

`indictrans2-ft eval` scores a prepared `data_dir` split the same way.

## Evaluation metrics

None of these seven choose the best checkpoint. `eval_loss` does.

| Metric | What a student should picture | Higher or lower | Needs a download |
| --- | --- | --- | --- |
| chrF++ | Overlap of character groups, with a little word order. The main translation score in this project. | Higher | No |
| BLEU | Overlap of word groups, FLORES tokenization | Higher | No |
| TER | How much editing the hypothesis still needs | Lower | No |
| Exact match | The hypothesis equals the reference after whitespace is tidied | Higher | No. Usually near zero on real sentences. |
| METEOR | Word overlap with a synonym list from NLTK WordNet | Higher | WordNet, once |
| COMET | A neural judge (`Unbabel/wmt22-comet-da`) scores source, hypothesis, and reference together | Higher | Yes. The model is about a gigabyte. |
| BERTScore | Token similarity with `xlm-roberta-large` | Higher | Yes. Another large model. |

chrF++, BLEU, TER, and exact match are the ones to watch while the GPU is training. They run inside the process and do not load a second model.

COMET and BERTScore are for the offline command above. COMET imports `pkg_resources`, so install `setuptools>=69,<81` before `unbabel-comet`. The first call downloads the COMET model and then reuses the cache. COMET is run on the CPU on purpose, so it does not fight IndicTrans2 for the GPU. It is slow. BERTScore wants the GPU. Run it after training has exited, when the translation model is no longer loaded. If either one throws, the error string is stored and the rest of the metrics are kept.

The training CSV is `eval_metrics.csv`. A bare file name is written under `--output_dir`. A path that already contains a directory is used as given, so it is not nested twice. If the first row is a failure and a later row adds chrF++, the header is rewritten so the new columns are not dropped. `eval_samples.csv` stores `src_lang`, `tgt_lang`, `src`, `ref`, and `hyp` for each sentence.

## Distribution strategies

`--distribution` decides where the weights and the Adam state sit. The default is `auto`.

| Value | Use it when |
| --- | --- |
| `auto` | Always, until something does not fit. One GPU is a normal run. Two or more ids become DDP. |
| `ddp` | You want the command to say DDP. Same path as `auto` with two or more GPUs. |
| `zero2` | Full fine-tune of the 1B model, and the Adam state does not fit. Set `SAVE_ONLY_MODEL=0`. Install DeepSpeed. |
| `zero3` | The weights themselves do not fit. Rare for IndicTrans2. The 1B weights are about 2 GB. |
| `offload` | Last resort. Weights and Adam sit in CPU RAM. It is slow. |
| `fsdp` | Not the usual choice. DDP is faster when the model fits. `zero2` is the memory tool for a full 1B run. |
| `deepspeed` | Old name. It is treated as `zero2`. |

QLoRA allows only `auto` and `ddp`.

The full decision list, including what this project deliberately does not use (DataParallel, ZeRO-1, tensor and pipeline parallelism), is [distribution.md](distribution.md).

## Visualization for analysis (RCA)

After training, `rca_report.html` is written in the output directory unless you passed `--no-write_rca` or set `WRITE_RCA=0`. On multi-GPU, rank 0 writes it. Rebuild it any time:

```bash
indictrans2-ft analyze --run-dir it2_out/lora
```

Open the file in a browser. The charts are drawn in the file. No network is required.

Read it in this order:

1. The box at the top. It names the best validation-loss step and the last step. `final_model` follows the best loss when eval and save use the same schedule.
2. The findings. They compare the best step with the last step. A run that peaked early and then got worse is named, with the step.
3. The loss chart. Training loss and validation loss share one chart. A growing gap is overfitting.
4. The chrF++ chart. If its best step is not the best loss step, the page says so. Trust this when you care about translation quality, and still load `final_model` from the loss unless you deliberately pick another checkpoint folder.
5. The language table, worst chrF++ first. This is which pair failed.
6. The sentence table. Each source shows the latest hypothesis and the best earlier hypothesis, so you can see a sentence that got worse.
7. The checkpoint table. Same Top-K and Last-K list as the manifest.

Live curves during the run, if you want them, are `--report_to tensorboard`. The HTML report is the offline record.

How to read each block is written again, a little shorter, in [rca.md](rca.md).
