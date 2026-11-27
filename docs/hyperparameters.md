# Training hyperparameters

Every knob below is a `train.py` flag. The shell scripts in `scripts/` pass the same flags and fill in a default when you do not set the matching environment variable.

Two defaults exist for many flags:

| Source | What it is |
| --- | --- |
| `train.py` | The value used when you call `python train.py` and omit the flag. |
| Shell script | The value `scripts/train_*.sh` writes onto the command when the environment variable is unset. |

If a flag has a `choices` list, those strings are the only values `train.py` accepts. Anything else is rejected before training starts. A number with no list is accepted as that type. The Studio table in [README_STUDIO.md](../README_STUDIO.md) writes each number as a range plus the usual values, for example `0.0–0.2 (usual: 0, 0.001, 0.01, 0.05, 0.1)`. The parser does not reject a number outside that range unless the row says the code checks it.

The 1B Indic-to-English run that trained on two 24 GB L4s used DDP, `--batch_size 8`, `--eval_batch_size 8`, `--gradient_accumulation_steps 4`, `--optim adamw_bnb_8bit`, `--learning_rate 2e-5`, `--epochs 8`, `--generation_num_beams 1`, and `--early_stopping_patience 0`. Those are the values that fit, not a coded range.

`zero2`, `zero3`, `offload`, `fsdp`, and `deepspeed` are not accepted. The failed runs are in [deepspeed.md](deepspeed.md). What `auto` and `ddp` do is in [distribution.md](distribution.md).

## How a step is sized

One optimizer update sees this many sentences:

`batch_size × gradient_accumulation_steps × number of GPUs`

`--batch_size` is the sentences on one GPU in one forward pass. Raising it uses more memory and usually smooths the gradient. `--gradient_accumulation_steps` adds that many forwards before one update, so the effective batch grows without a bigger micro-batch. A second GPU under DDP multiplies the count again, because each GPU holds its own copy and sees different sentences.

The learning rate is not scaled when you add a GPU. If you want the same update size as a one-GPU run, cut accumulation as you add GPUs.

## What is trained

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--train_mode` | Which weights are allowed to move. The objective stays supervised translation. | `lora` and `qlora` train a small adapter. `full_lora` trains a wider adapter and merges it into a full model at the end. `full` updates every weight. | `full`, `full_lora`, `lora`, `qlora`. `train.py` default: `lora`. Each shell script fixes one mode. `full_qlora` is rejected. |
| `--model_hf_id` | Official checkpoint id. Also the direction check. | An English→Indic id refuses an Indic→English pair, and the other way around. It does not by itself load a different folder if `--model_id` is set. | The six ids: `ai4bharat/indictrans2-en-indic-dist-200M`, `ai4bharat/indictrans2-en-indic-1B`, `ai4bharat/indictrans2-indic-en-dist-200M`, `ai4bharat/indictrans2-indic-en-1B`, `ai4bharat/indictrans2-indic-indic-dist-320M`, `ai4bharat/indictrans2-indic-indic-1B`. Required. |
| `--model_id` | Folder or Hub id that is actually loaded. | Point it at a previous `final_model` to continue a full fine-tune. For an adapter, keep the base id here and pass `--adapter_path`. | Any string. Default: the same value as `--model_hf_id`. |
| `--adapter_path` | A saved LoRA directory. | `full` merges that adapter, then trains every weight. `lora`, `qlora`, and `full_lora` continue the adapter. | A path, or omitted. |
| `--src_lang`, `--tgt_lang` | Optional pair filter. | Empty: every JSON pair whose direction matches `--model_hf_id` is trained together. Both set: only that pair. One without the other is not a filter by itself. | Only the tags in `langs.py`, or their aliases. Tags: `asm_Beng`, `awa_Deva`, `ben_Beng`, `bho_Deva`, `brx_Deva`, `doi_Deva`, `eng_Latn`, `gom_Deva`, `guj_Gujr`, `hin_Deva`, `hne_Deva`, `kan_Knda`, `kas_Arab`, `kas_Deva`, `kha_Latn`, `lus_Latn`, `mag_Deva`, `mai_Deva`, `mal_Mlym`, `mar_Deva`, `mni_Beng`, `mni_Mtei`, `npi_Deva`, `ory_Orya`, `pan_Guru`, `san_Deva`, `sat_Olck`, `snd_Arab`, `snd_Deva`, `tam_Taml`, `tel_Telu`, `urd_Arab`. Aliases: `en`, `hi`, `bn`, `ta`, `te`, `ml`, `kn`, `mr`, `gu`, `pa`, `or`, `as`, `ur`, `ne`, `sa`, and the English names `english`, `hindi`, `bengali`, `tamil`, `telugu`, `malayalam`, `kannada`, `marathi`, `gujarati`, `punjabi`, `odia`, `assamese`, `urdu`, `nepali`, `sanskrit`. Any other string is rejected. |
| `--train_json`, `--val_json` | Sentence files with `src_lang`, `tgt_lang`, `src_text`, `tgt_text`. | The training file is what the model learns. The validation file is what `eval_loss` and the metrics see. Both are required unless `--data_dir` is set. | Paths. |
| `--data_dir` | Already prepared official layout. | Skips JSON conversion. | A path, or omitted. |
| `--max_length` | Tokens kept on the source side and on the target side. | Longer sentences are truncated. Shorter ones are padded. 256 is the IndicTrans2 window. A larger value uses more memory and attends past the length the pretrained model was built for. | Code: any integer. Usual: `128` or `256`. Default: `256`. |
| `--seed` | Shuffle and initialization seed. | Same seed and same data give the same order. Change it only for a second run of the same settings. | Code: any integer. Usual: `42`. Other runs often use `1` or `1234`. |

## Step size and schedule

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--epochs` | Passes over the training file. | More epochs keep moving the weights after the file has been seen. Validation loss that rises is the sign to stop. The shell default of 2 is a short run. The 1B L4 job used 8, with early stopping off, so all 8 ran unless you kill it. | Code: any integer. Usual: `1`, `2`, `3`, `5`, `8`, `10`. Default: `2`. The 1B run used `8`. |
| `--learning_rate` | Size of each Adam update after warmup. | Too large: the loss spikes or the translation becomes nonsense. Too small: the loss barely moves. `2e-4` is the adapter rate. `2e-5` is the full-model rate. Using `2e-4` on `full` is large enough to damage the pretrained weights. | Code: any float. Usual for `lora` and `qlora`: `1e-4`, `2e-4`, `3e-4`, `5e-4`. Usual for `full` and `full_lora`: `1e-5`, `2e-5`, `3e-5`, `5e-5`. Defaults: `2e-4` and `2e-5`. |
| `--warmup_ratio` | Fraction of updates spent raising the learning rate from zero up to `--learning_rate`. | A short warmup avoids a huge first step. A warmup that covers most of a short run wastes the only updates you have. | 0.0–0.20 (usual: 0.03, 0.05, 0.06, 0.1). Default: `0.03`. The parser still accepts any float. |
| `--lr_scheduler_type` | Shape of the learning rate after warmup. | `cosine` falls smoothly toward zero. `linear` falls in a straight line. `inverse_sqrt` is the older machine-translation schedule. `constant` stays at the peak. `constant_with_warmup` stays at the peak after warmup. `cosine_with_restarts` repeats the cosine drop. `polynomial` decays as a polynomial. | (`linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`). Default: `cosine`. Any other string is rejected. |
| `--weight_decay` | Penalty on large weights, applied by AdamW. | A small penalty reduces memorizing a small file. `0` turns that penalty off. Very large values fight the update. | 0.0–0.2 (usual: 0, 0.001, 0.01, 0.05, 0.1). Default: `0.01`. The parser still accepts any float. |
| `--max_grad_norm` | Gradient clipping. | If the gradient vector is longer than this, it is rescaled to this length. It stops one bad batch from taking a huge step. | Code: any float. Usual: `0.5`, `1.0`. Default: `1.0`. |
| `--label_smoothing_factor` | Softens the target-token label. | `0.1` means the correct token is not a hard 1. The loss then levels off above zero, which is normal for translation. `0` is ordinary cross-entropy. | Code: float in `[0, 1)`. `1` and negatives are rejected. Usual: `0` or `0.1`. Default: `0.1`. |

## Batch, memory, and precision

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--batch_size` | Sentences per GPU per forward. | The main memory knob. The 200M full run used 32 and ran out of memory at 64 on a 24 GB L4. The 1B full run with 8-bit Adam fit at 8. | Code: any integer. Usual: `1`, `2`, `4`, `8`, `16`, `32`. Defaults: `2` for LoRA and QLoRA, `1` for full and full-LoRA. The 1B run used `8`. `64` did not fit a full 200M update on one 24 GB L4. |
| `--eval_batch_size` | Sentences per GPU when the validation file is scored. | Generation holds the decoder cache and the vocabulary logits, so this can run out of memory even when the training batch fits. It does not change the training update. | Code: any integer. Usual: `1`, `2`, `4`, `8`. Scripts: `2`. The 1B run used `8`, and `4` is the fallback when only evaluation runs out of memory. |
| `--gradient_accumulation_steps` | Forwards added before one update. | Raises the effective batch without raising `--batch_size`. | Code: any integer. Usual: `1`, `2`, `4`, `8`, `16`. Default: `4`. Full scripts: `8`. The 1B run used `4`. |
| `--gradient_checkpointing` | Recompute activations during the backward pass instead of storing them. | Uses less memory and makes each step slower. Needed for a full 1B update on a 24 GB card. On by default in the shell scripts. Off if you call `train.py` and do not pass the flag. IndicTrans still uses the old checkpointing hook, so a warning about `_set_gradient_checkpointing` can appear. Training continues. | Present or absent. Shell scripts turn it on unless `GRADIENT_CHECKPOINTING=0`. |
| `--bf16` | bfloat16 mixed precision. | The L4 tensor-core type. Half the weight memory of fp32, with bf16’s exponent range. | Present or absent. Shell scripts turn it on unless `BF16=0`. Cannot be combined with `--fp16`. |
| `--fp16` | float16 mixed precision. | Smaller memory than fp32, with a smaller exponent range than bf16. | Present or absent. Shell scripts leave it off (`FP16=0`). Cannot be combined with `--bf16`. |
| `--optim` | Which optimizer implementation runs the update. | `adamw_torch` is ordinary AdamW. `adamw_torch_fused` is the fused CUDA kernel of the same update. `adamw_hf` is the older Hugging Face AdamW. `adamw_bnb_8bit` stores Adam state in 8 bits so a full update fits on one 24 GB GPU. `paged_adamw_8bit` and `paged_adamw_32bit` can move optimizer state to CPU when the GPU is tight. `adafactor` uses less optimizer memory and is a different update rule. | Exactly one of: `adamw_torch`, `adamw_torch_fused`, `adamw_hf`, `adamw_bnb_8bit`, `paged_adamw_8bit`, `paged_adamw_32bit`, `adafactor`. If omitted, `train.py` uses `paged_adamw_32bit` for `qlora` and `adamw_torch` otherwise. Shell scripts pass `adamw_torch` (LoRA), `paged_adamw_32bit` (QLoRA), or `adamw_bnb_8bit` (full and full LoRA). |

## LoRA and QLoRA

These flags change the adapter. In `full` mode the shell script still passes them, and `train.py` accepts them, but a full update does not build an adapter from them.

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--lora_r` | Rank, the width of each adapter matrix. | Higher rank can fit a more specific domain and memorizes a small file faster. | Code: any integer. Usual: `4`, `8`, `16`, `32`, `64`. Default: `8`. Stay at `4`, `8`, or `16` until `8` underfits. |
| `--lora_alpha` | Scale of the adapter update. | The step size of the adapter grows with alpha / rank. The shipped default is alpha = 2 × rank. | Code: any integer. Usual: twice the rank, so `8`, `16`, `32`, `64`, `128` for ranks `4`, `8`, `16`, `32`, `64`. Default: `16`. |
| `--lora_dropout` | Dropout on the adapter. | Randomly zeros adapter activations. Higher values resist memorizing a small file. `0` turns it off. | Code: any float. Usual: `0`, `0.05`, `0.1`. Default: `0.05`. |
| `--lora_target_modules` | Which linear layers get an adapter. | More modules means more of the network moves. | Code: any comma-separated names. The layers this repo attaches to are `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`. `lora` and `qlora` default to `q_proj,k_proj`. `full_lora` defaults to all six. |
| `--bnb_compute_dtype` | Compute dtype for 4-bit QLoRA. | The frozen base stays 4-bit. The matmul runs in this dtype. | `float16` or `bfloat16`. Default: `bfloat16`. |
| `--bnb_4bit_quant_type` | 4-bit storage type. | `nf4` is the type QLoRA was built around. `fp4` is the other 4-bit code. | `nf4` or `fp4`. Default: `nf4`. |
| `--bnb_4bit_use_double_quant` | Quantize the quantization constants too. | Saves a little more memory. Off unless you pass it or set `BNB_4BIT_USE_DOUBLE_QUANT=1`. | Present or absent. |

## Evaluation during training

These flags do not change the training loss. They change what is measured, and how expensive the end of an epoch is.

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--eval_strategy` | When validation runs. | `epoch` scores the validation file at the end of each pass. `steps` scores every `--eval_steps` updates. `no` never scores, so early stopping cannot run. | `epoch`, `steps`, `no`. Default: `epoch`. |
| `--eval_steps` | How often, when the strategy is `steps`. | Ignored when the strategy is `epoch`. A small number spends most of the job in evaluation. | Code: any integer. Usual: `50`, `100`, `200`, `500`. Default: `50`. Used only when `--eval_strategy steps`. |
| `--generation_num_beams` | Beam width for validation translations. | The training loss does not use beams. Beam 5 repeats the decode. Beam 1 is one decode and is what made the 1B validation pass practical. | Code: any integer. Usual: `1`, `2`, `4`, `5`. Default: `5`. The 1B run used `1`. |
| `--eval_max_new_tokens` | Maximum tokens generated for a validation sentence. | Caps the decode length. It does not change `--max_length` for the training batch. | Code: any integer. Usual: `64`, `128`, `256`. Default: `256`. Keep it at or below `--max_length`. |
| `--eval_accumulation_steps` | How many evaluation batches are gathered before the metric tensors are moved. | Lowers the memory peak of metric computation. Does not change the score. | Code: any integer. Usual: `1`, `2`, `4`, `8`. Default: `1`. |
| `--eval_metrics` | Which translation scores are computed on the validation generations. | chrF++, BLEU, TER, and exact match are cheap. METEOR is slow on a large validation file. COMET and BERTScore each load another neural model onto the GPU. None of these scores chooses the best checkpoint. | One or more of: `chrf2pp`, `comet`, `bleu`, `bertscore`, `meteor`, `ter`, `exact_match`. Default: all seven. Pass the flag once with several names. |

## Checkpoints and early stopping

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--save_strategy` | When a checkpoint is written. | `epoch` saves at the end of each pass. `steps` saves every `--save_steps` updates. `no` does not save, so early stopping cannot reload a best checkpoint. | `epoch`, `steps`, `no`. Default: `epoch`. Should match `--eval_strategy` when you rank checkpoints. |
| `--save_steps` | How often, when the strategy is `steps`. | Ignored for `epoch`. | Code: any integer. Usual: `100`, `500`, `1000`. Default: `500`. |
| `--metric_for_best_model` | The score that ranks checkpoints and drives early stopping. | `eval_loss` and `eval_entropy`: lower is better. `eval_mean_token_accuracy`: higher is better. chrF++ is not in this list. The trainer would treat a higher chrF++ as worse. | `eval_loss`, `eval_entropy`, `eval_mean_token_accuracy`. Default: `eval_loss`. |
| `--top_n_checkpoints` | How many best checkpoints to keep. | A folder that is both a best score and a recent save is stored once. | Code: any integer. Usual: `1`, `2`, `3`, `5`. Default: `3`. |
| `--last_k_checkpoints` | How many newest checkpoints to keep, in addition to the best ones. | Lets you still see the end of training when the best score was earlier. | Code: any integer. Usual: `1`, `2`, `3`. Default: `2`. |
| `--save_total_limit` | Extra cap on the union of those two lists. | `0` means no extra cap. A positive number keeps the best scores first, then fills with newer ones. | Code: any integer. `0` means no extra cap. Usual: `0`, `2`, `3`, `5`. Default: `0`. |
| `--save_only_model` | Checkpoint contains weights and not the optimizer. | Smaller files. A resume from that checkpoint reloads weights only. The full and full-LoRA shell scripts turn this on unless `SAVE_ONLY_MODEL=0`. `train.py` leaves it off unless you pass the flag. The working 1B DDP command did not pass it. | Present or absent. |
| `--early_stopping_patience` | Stop after this many evaluations in a row with no improvement. | The first evaluation only records the starting score. Patience counts evaluations, so with `--eval_strategy steps` it is not a count of epochs. `0` turns it off and the run finishes `--epochs`. After a stop, `final_model` is still the best checkpoint, not the last one. | Code: integer `≥ 0`. Negatives are rejected. Usual: `0`, `1`, `2`, `3`, `5`. Default: `3`. The 1B run used `0`. Requires `--eval_strategy` and `--save_strategy` to match, and neither may be `no`. |
| `--early_stopping_threshold` | How much the metric must move to count as an improvement. | `0.0` means any strict improvement resets the patience count. A positive number ignores tiny moves. | Code: float `≥ 0`. Negatives are rejected. Usual: `0`, `0.001`, `0.01`. Default: `0.0`. |
| `--resume_from_checkpoint` | A checkpoint directory to continue. | Loads weights and, when the checkpoint has it, the optimizer. The patience count starts again. The previous best score is kept. | A path, or omitted. |
| `--output_dir` | Where checkpoints, CSVs, and `rca_report.html` are written. | A second run with the same directory mixes with the first. | A path. `train.py` default: `./it2_out/exp`. Shell scripts: `it2_out/<mode>`. |
| `--final_model_dir` | Where the weights to keep are written at the end. | This is the best checkpoint when evaluation and saving are on and they match. Otherwise it is whatever is in memory at the end. | A path. Default: `<output_dir>/final_model`. |

## Hardware and process

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--distribution` | Where the weights sit. | `auto` with one GPU is a single process. `auto` or `ddp` with two or more ids copies the model onto each GPU. That is the path the 1B L4 run used. | `auto` or `ddp`. Default: `auto`. |
| `--gpu_ids` | Which visible NVIDIA devices to use. | `0` is the first. `0,1` uses both and relaunches under `torchrun`. Do not pass this if you already started `torchrun` yourself. | Code: comma-separated digits, or omitted. On the two L4s the values are `0` and `0,1`. |
| `--ddp_timeout` | Seconds a rank waits for the others. | A long validation generation needs a long timeout or the process group is killed. | Code: integer `≥ 1`. Usual: `1800`, `3600`, `7200`, `14400`. Default: `14400` (4 hours). |
| `--num_workers` | Data-loader worker processes. | More workers overlap disk reads with the GPU step. `0` loads batches in the training process. | Code: any integer. Usual: `0`, `2`, `4`, `8`. Default: `4`. |
| `--logging_steps` | How often a loss line is written. | Does not change the update. | Code: any integer. Usual: `10`, `25`, `50`, `100`. Default: `50`. |
| `--report_to` | Live logger. | `none` keeps the CSV files only. `tensorboard` also writes a TensorBoard event file. | `none` or `tensorboard`. Default: `none`. |
| `--write_rca` / `--no-write_rca` | Write `rca_report.html` after training. | The report reads the CSVs. It does not change the weights. | On by default. Shell scripts honor `WRITE_RCA=0` as off. |

## Shell environment names

`scripts/train_full.sh`, `scripts/train_full_lora.sh`, `scripts/train_lora.sh`, and `scripts/train_qlora.sh` map these variables onto the flags above. Unset means the default in that script.

`EPOCHS`, `BATCH_SIZE`, `EVAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `WARMUP_RATIO`, `LR_SCHEDULER_TYPE`, `OPTIM`, `MAX_GRAD_NORM`, `LOGGING_STEPS`, `SEED`, `MAX_LENGTH`, `EVAL_STRATEGY`, `EVAL_STEPS`, `EVAL_MAX_NEW_TOKENS`, `EVAL_ACCUMULATION_STEPS`, `GENERATION_NUM_BEAMS`, `SAVE_STRATEGY`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`, `METRIC_FOR_BEST_MODEL`, `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_THRESHOLD`, `TOP_N_CHECKPOINTS`, `LAST_K_CHECKPOINTS`, `LABEL_SMOOTHING_FACTOR`, `DISTRIBUTION`, `EVAL_METRICS`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES`, `BNB_COMPUTE_DTYPE`, `BNB_4BIT_QUANT_TYPE`, `BNB_4BIT_USE_DOUBLE_QUANT`, `GPU_IDS`, `DDP_TIMEOUT`, `NUM_WORKERS`, `GRADIENT_CHECKPOINTING`, `BF16`, `FP16`, `SAVE_ONLY_MODEL`, `WRITE_RCA`, `REPORT_TO`.

`FSDP` and `DEEPSPEED` are no longer read. `--distribution` is only `auto` or `ddp`.
