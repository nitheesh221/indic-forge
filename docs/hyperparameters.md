# Training hyperparameters

Every knob below is a `train.py` flag. The shell scripts in `scripts/` pass the same flags and fill in a default when you do not set the matching environment variable.

Two defaults exist for many flags:

| Source | What it is |
| --- | --- |
| `train.py` | The value used when you call `python train.py` and omit the flag. |
| Shell script | The value `scripts/train_*.sh` writes onto the command when the environment variable is unset. |

If a flag has a `choices` list, those strings are the only values `train.py` accepts. Anything else is rejected before training starts. If a flag is a number with no list, the parser accepts any number of that type, then applies the limits named in the row. This page does not add a search range on top of that. A number that is not in the script is not a supported setting just because it is plausible.

The 1B Indic-to-English run that trained on two 24 GB L4s used DDP, `--batch_size 8`, `--eval_batch_size 8`, `--gradient_accumulation_steps 4`, `--optim adamw_bnb_8bit`, `--learning_rate 2e-5`, `--epochs 8`, `--generation_num_beams 1`, and `--early_stopping_patience 0`. Those are the values that fit, not a coded range.

Distribution failures (`zero2`, `zero3`, `offload`, `fsdp` on this checkpoint) are in [deepspeed.md](deepspeed.md). What each distribution flag means is in [distribution.md](distribution.md).

## How a step is sized

One optimizer update sees this many sentences:

`batch_size × gradient_accumulation_steps × number of GPUs`

`--batch_size` is the sentences on one GPU in one forward pass. Raising it uses more memory and usually smooths the gradient. `--gradient_accumulation_steps` adds that many forwards before one update, so the effective batch grows without a bigger micro-batch. A second GPU under DDP multiplies the count again, because each GPU holds its own copy and sees different sentences.

The learning rate is not scaled when you add a GPU. If you want the same update size as a one-GPU run, cut accumulation as you add GPUs.

## What is trained

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--train_mode` | Which weights are allowed to move. The objective stays supervised translation. | `lora` and `qlora` train a small adapter. `full_lora` trains a wider adapter and, except under ZeRO or FSDP, merges it into a full model at the end. `full` updates every weight. | `full`, `full_lora`, `lora`, `qlora`. `train.py` default: `lora`. Each shell script fixes one mode. `full_qlora` is rejected. |
| `--model_hf_id` | Official checkpoint id. Also the direction check. | An English→Indic id refuses an Indic→English pair, and the other way around. It does not by itself load a different folder if `--model_id` is set. | The six ids: `ai4bharat/indictrans2-en-indic-dist-200M`, `ai4bharat/indictrans2-en-indic-1B`, `ai4bharat/indictrans2-indic-en-dist-200M`, `ai4bharat/indictrans2-indic-en-1B`, `ai4bharat/indictrans2-indic-indic-dist-320M`, `ai4bharat/indictrans2-indic-indic-1B`. Required. |
| `--model_id` | Folder or Hub id that is actually loaded. | Point it at a previous `final_model` to continue a full fine-tune. For an adapter, keep the base id here and pass `--adapter_path`. | Any string. Default: the same value as `--model_hf_id`. |
| `--adapter_path` | A saved LoRA directory. | `full` merges that adapter, then trains every weight. `lora`, `qlora`, and `full_lora` continue the adapter. | A path, or omitted. |
| `--src_lang`, `--tgt_lang` | Optional pair filter. | Empty: every JSON pair whose direction matches `--model_hf_id` is trained together. Both set: only that pair. One without the other is not a filter by itself. | FLORES-style tags such as `hin_Deva`, `eng_Latn`. The code also accepts the short forms it already normalizes. The pair must match the checkpoint direction. |
| `--train_json`, `--val_json` | Sentence files with `src_lang`, `tgt_lang`, `src_text`, `tgt_text`. | The training file is what the model learns. The validation file is what `eval_loss` and the metrics see. Both are required unless `--data_dir` is set. | Paths. |
| `--data_dir` | Already prepared official layout. | Skips JSON conversion. | A path, or omitted. |
| `--max_length` | Tokens kept on the source side and on the target side. | Longer sentences are truncated. Shorter ones are padded. 256 is the IndicTrans2 window. A larger value uses more memory and attends past the length the pretrained model was built for. | Any integer. `train.py` and every shell script: `256`. |
| `--seed` | Shuffle and initialization seed. | Same seed and same data give the same order. Change it only for a second run of the same settings. | Any integer. Default: `42`. |

## Step size and schedule

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--epochs` | Passes over the training file. | More epochs keep moving the weights after the file has been seen. Validation loss that rises is the sign to stop. The shell default of 2 is a short run. The 1B L4 job used 8, with early stopping off, so all 8 ran unless you kill it. | Any integer. `train.py` and every shell script: `2`. |
| `--learning_rate` | Size of each Adam update after warmup. | Too large: the loss spikes or the translation becomes nonsense. Too small: the loss barely moves. `2e-4` is the adapter rate. `2e-5` is the full-model rate. Using `2e-4` on `full` is large enough to damage the pretrained weights. | Any float. If you omit it, `train.py` uses `2e-4` for `lora` and `qlora`, and `2e-5` for `full` and `full_lora`. Shell scripts pass `2e-4` (`train_lora.sh`, `train_qlora.sh`) or `2e-5` (`train_full.sh`, `train_full_lora.sh`). |
| `--warmup_ratio` | Fraction of updates spent raising the learning rate from zero up to `--learning_rate`. | A short warmup avoids a huge first step. A warmup that covers most of a short run wastes the only updates you have. | Any float. Default everywhere: `0.03`. |
| `--lr_scheduler_type` | Shape of the learning rate after warmup. | `cosine` falls smoothly toward zero. `linear` falls in a straight line. `inverse_sqrt` is the older machine-translation schedule. `constant` stays at the peak. `constant_with_warmup` stays at the peak after warmup. `cosine_with_restarts` repeats the cosine drop. `polynomial` decays as a polynomial. | Exactly one of: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`, `inverse_sqrt`. Default: `cosine`. |
| `--weight_decay` | Penalty on large weights, applied by AdamW. | A small penalty reduces memorizing a small file. `0` turns that penalty off. Very large values fight the update. | Any float. Default: `0.01`. |
| `--max_grad_norm` | Gradient clipping. | If the gradient vector is longer than this, it is rescaled to this length. It stops one bad batch from taking a huge step. | Any float. Default: `1.0`. |
| `--label_smoothing_factor` | Softens the target-token label. | `0.1` means the correct token is not a hard 1. The loss then levels off above zero, which is normal for translation. `0` is ordinary cross-entropy. | Any float in `[0, 1)`. `1` is rejected. Default: `0.1`. |

## Batch, memory, and precision

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--batch_size` | Sentences per GPU per forward. | The main memory knob. The 200M full run used 32 and ran out of memory at 64 on a 24 GB L4. The 1B full run with 8-bit Adam fit at 8. | Any integer. `train.py`, `train_lora.sh`, `train_qlora.sh`: `2`. `train_full.sh`, `train_full_lora.sh`: `1`. |
| `--eval_batch_size` | Sentences per GPU when the validation file is scored. | Generation holds the decoder cache and the vocabulary logits, so this can run out of memory even when the training batch fits. It does not change the training update. | Any integer. Shell scripts: `2`. The 1B DDP run used `8`. |
| `--gradient_accumulation_steps` | Forwards added before one update. | Raises the effective batch without raising `--batch_size`. | Any integer. `train.py`, LoRA, QLoRA scripts: `4`. Full and full-LoRA scripts: `8`. |
| `--gradient_checkpointing` | Recompute activations during the backward pass instead of storing them. | Uses less memory and makes each step slower. Needed for a full 1B update on a 24 GB card. On by default in the shell scripts. Off if you call `train.py` and do not pass the flag. IndicTrans still uses the old checkpointing hook, so a warning about `_set_gradient_checkpointing` can appear. Training continues. | Present or absent. Shell scripts turn it on unless `GRADIENT_CHECKPOINTING=0`. |
| `--bf16` | bfloat16 mixed precision. | The L4 tensor-core type. Half the weight memory of fp32, with bf16’s exponent range. | Present or absent. Shell scripts turn it on unless `BF16=0`. Cannot be combined with `--fp16`. |
| `--fp16` | float16 mixed precision. | Smaller memory than fp32, with a smaller exponent range than bf16. | Present or absent. Shell scripts leave it off (`FP16=0`). Cannot be combined with `--bf16`. |
| `--optim` | Which optimizer implementation runs the update. | `adamw_torch` is ordinary AdamW. `adamw_torch_fused` is the fused CUDA kernel of the same update. `adamw_hf` is the older Hugging Face AdamW. `adamw_bnb_8bit` stores Adam state in 8 bits so a full update fits on one 24 GB GPU. `paged_adamw_8bit` and `paged_adamw_32bit` can move optimizer state to CPU when the GPU is tight. `adafactor` uses less optimizer memory and is a different update rule. ZeRO rewrites the 8-bit and paged names to `adamw_torch`, because ZeRO only shards a normal Adam. | Exactly one of: `adamw_torch`, `adamw_torch_fused`, `adamw_hf`, `adamw_bnb_8bit`, `paged_adamw_8bit`, `paged_adamw_32bit`, `adafactor`. If omitted, `train.py` uses `paged_adamw_32bit` for `qlora` and `adamw_torch` otherwise. Shell scripts pass `adamw_torch` (LoRA), `paged_adamw_32bit` (QLoRA), or `adamw_bnb_8bit` (full and full LoRA). |

## LoRA and QLoRA

These flags change the adapter. In `full` mode the shell script still passes them, and `train.py` accepts them, but a full update does not build an adapter from them.

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--lora_r` | Rank, the width of each adapter matrix. | Higher rank can fit a more specific domain and memorizes a small file faster. | Any integer. Default: `8`. |
| `--lora_alpha` | Scale of the adapter update. | The step size of the adapter grows with alpha / rank. The shipped default is alpha = 2 × rank. | Any integer. Default: `16`. |
| `--lora_dropout` | Dropout on the adapter. | Randomly zeros adapter activations. Higher values resist memorizing a small file. `0` turns it off. | Any float. Default: `0.05`. |
| `--lora_target_modules` | Which linear layers get an adapter. | More modules means more of the network moves. | Comma-separated names. If omitted, `lora` and `qlora` use `q_proj,k_proj`. `full_lora` uses `q_proj,k_proj,v_proj,out_proj,fc1,fc2`. The shell scripts pass `q_proj,k_proj` unless `LORA_TARGET_MODULES` is set. |
| `--bnb_compute_dtype` | Compute dtype for 4-bit QLoRA. | The frozen base stays 4-bit. The matmul runs in this dtype. | `float16` or `bfloat16`. Default: `bfloat16`. |
| `--bnb_4bit_quant_type` | 4-bit storage type. | `nf4` is the type QLoRA was built around. `fp4` is the other 4-bit code. | `nf4` or `fp4`. Default: `nf4`. |
| `--bnb_4bit_use_double_quant` | Quantize the quantization constants too. | Saves a little more memory. Off unless you pass it or set `BNB_4BIT_USE_DOUBLE_QUANT=1`. | Present or absent. |

## Evaluation during training

These flags do not change the training loss. They change what is measured, and how expensive the end of an epoch is.

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--eval_strategy` | When validation runs. | `epoch` scores the validation file at the end of each pass. `steps` scores every `--eval_steps` updates. `no` never scores, so early stopping cannot run. | `epoch`, `steps`, `no`. Default: `epoch`. |
| `--eval_steps` | How often, when the strategy is `steps`. | Ignored when the strategy is `epoch`. A small number spends most of the job in evaluation. | Any integer. Default: `50`. |
| `--generation_num_beams` | Beam width for validation translations. | The training loss does not use beams. Beam 5 repeats the decode. Beam 1 is one decode and is what made the 1B validation pass practical. | Any integer. Shell scripts: `5`. |
| `--eval_max_new_tokens` | Maximum tokens generated for a validation sentence. | Caps the decode length. It does not change `--max_length` for the training batch. | Any integer. Default: `256`. |
| `--eval_accumulation_steps` | How many evaluation batches are gathered before the metric tensors are moved. | Lowers the memory peak of metric computation. Does not change the score. | Any integer. Default: `1`. |
| `--eval_metrics` | Which translation scores are computed on the validation generations. | chrF++, BLEU, TER, and exact match are cheap. METEOR is slow on a large validation file. COMET and BERTScore each load another neural model onto the GPU. None of these scores chooses the best checkpoint. | One or more of: `chrf2pp`, `comet`, `bleu`, `bertscore`, `meteor`, `ter`, `exact_match`. Default: all seven. Pass the flag once with several names. |

## Checkpoints and early stopping

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--save_strategy` | When a checkpoint is written. | `epoch` saves at the end of each pass. `steps` saves every `--save_steps` updates. `no` does not save, so early stopping cannot reload a best checkpoint. | `epoch`, `steps`, `no`. Default: `epoch`. Should match `--eval_strategy` when you rank checkpoints. |
| `--save_steps` | How often, when the strategy is `steps`. | Ignored for `epoch`. | Any integer. Default: `500`. |
| `--metric_for_best_model` | The score that ranks checkpoints and drives early stopping. | `eval_loss` and `eval_entropy`: lower is better. `eval_mean_token_accuracy`: higher is better. chrF++ is not in this list. The trainer would treat a higher chrF++ as worse. | `eval_loss`, `eval_entropy`, `eval_mean_token_accuracy`. Default: `eval_loss`. |
| `--top_n_checkpoints` | How many best checkpoints to keep. | A folder that is both a best score and a recent save is stored once. | Any integer. `train.py` default: `3`. |
| `--last_k_checkpoints` | How many newest checkpoints to keep, in addition to the best ones. | Lets you still see the end of training when the best score was earlier. | Any integer. Default: `2`. |
| `--save_total_limit` | Extra cap on the union of those two lists. | `0` means no extra cap. A positive number keeps the best scores first, then fills with newer ones. | Any integer. Default: `0`. |
| `--save_only_model` | Checkpoint contains weights and not the optimizer. | Smaller files. ZeRO and FSDP cannot reload the best epoch from a checkpoint that has no optimizer state. The full and full-LoRA shell scripts turn this on unless `SAVE_ONLY_MODEL=0`. `train.py` leaves it off unless you pass the flag. The working 1B DDP command did not pass it. | Present or absent. |
| `--early_stopping_patience` | Stop after this many evaluations in a row with no improvement. | The first evaluation only records the starting score. Patience counts evaluations, so with `--eval_strategy steps` it is not a count of epochs. `0` turns it off and the run finishes `--epochs`. After a stop, `final_model` is still the best checkpoint, not the last one. | Any integer `≥ 0`. Default: `3`. Requires `--eval_strategy` and `--save_strategy` to be the same, and neither may be `no`. |
| `--early_stopping_threshold` | How much the metric must move to count as an improvement. | `0.0` means any strict improvement resets the patience count. A positive number ignores tiny moves. | Any float `≥ 0`. Default: `0.0`. |
| `--resume_from_checkpoint` | A checkpoint directory to continue. | Loads weights and, when the checkpoint has it, the optimizer. The patience count starts again. The previous best score is kept. | A path, or omitted. |
| `--output_dir` | Where checkpoints, CSVs, and `rca_report.html` are written. | A second run with the same directory mixes with the first. | A path. `train.py` default: `./it2_out/exp`. Shell scripts: `it2_out/<mode>`. |
| `--final_model_dir` | Where the weights to keep are written at the end. | This is the best checkpoint when evaluation and saving are on and they match. Otherwise it is whatever is in memory at the end. | A path. Default: `<output_dir>/final_model`. |

## Hardware and process

| Flag | What it is | Effect on training | Values the script accepts |
| --- | --- | --- | --- |
| `--distribution` | Where the weights and the Adam state sit. | `auto` with one GPU is a single process. `auto` or `ddp` with two or more ids copies the model onto each GPU. `zero2`, `zero3`, `offload`, and the old name `deepspeed` go through DeepSpeed. `fsdp` slices the model with PyTorch FSDP. On the two-L4 1B job, only DDP trained. The others failed before a step. See [deepspeed.md](deepspeed.md). | `auto`, `ddp`, `zero2`, `zero3`, `offload`, `fsdp`, `deepspeed`. Default: `auto`. |
| `--gpu_ids` | Which visible NVIDIA devices to use. | `0` is the first. `0,1` uses both and relaunches under `torchrun`. Do not pass this if you already started `torchrun` yourself. | Comma-separated device ids, or omitted. |
| `--ddp_timeout` | Seconds a rank waits for the others. | A long validation generation needs a long timeout or the process group is killed. Covers DDP and DeepSpeed, because the training arguments open the group. | Any integer `≥ 1`. Default: `14400` (4 hours). |
| `--num_workers` | Data-loader worker processes. | More workers overlap disk reads with the GPU step. `0` loads batches in the training process. | Any integer. Default: `4`. |
| `--deepspeed` | Replace the shipped ZeRO JSON. | Used only with `zero2`, `zero3`, or `offload`. Those flags already select `configs/ds_zero2.json`, `configs/ds_zero3.json`, or `configs/ds_offload.json`. | A path to a JSON file, or omitted. |
| `--fsdp` | FSDP strategy string. | Read only when `--distribution fsdp`. The trainer also sets `min_num_params` to `1000000`. | Any string. Default: `full_shard auto_wrap`. |
| `--logging_steps` | How often a loss line is written. | Does not change the update. | Any integer. Default: `50`. |
| `--report_to` | Live logger. | `none` keeps the CSV files only. `tensorboard` also writes a TensorBoard event file. | `none` or `tensorboard`. Default: `none`. |
| `--write_rca` / `--no-write_rca` | Write `rca_report.html` after training. | The report reads the CSVs. It does not change the weights. | On by default. Shell scripts honor `WRITE_RCA=0` as off. |

## Shell environment names

`scripts/train_full.sh`, `scripts/train_full_lora.sh`, `scripts/train_lora.sh`, and `scripts/train_qlora.sh` map these variables onto the flags above. Unset means the default in that script.

`EPOCHS`, `BATCH_SIZE`, `EVAL_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `WARMUP_RATIO`, `LR_SCHEDULER_TYPE`, `OPTIM`, `MAX_GRAD_NORM`, `LOGGING_STEPS`, `SEED`, `MAX_LENGTH`, `EVAL_STRATEGY`, `EVAL_STEPS`, `EVAL_MAX_NEW_TOKENS`, `EVAL_ACCUMULATION_STEPS`, `GENERATION_NUM_BEAMS`, `SAVE_STRATEGY`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`, `METRIC_FOR_BEST_MODEL`, `EARLY_STOPPING_PATIENCE`, `EARLY_STOPPING_THRESHOLD`, `TOP_N_CHECKPOINTS`, `LAST_K_CHECKPOINTS`, `LABEL_SMOOTHING_FACTOR`, `DISTRIBUTION`, `FSDP`, `EVAL_METRICS`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_TARGET_MODULES`, `BNB_COMPUTE_DTYPE`, `BNB_4BIT_QUANT_TYPE`, `BNB_4BIT_USE_DOUBLE_QUANT`, `GPU_IDS`, `DDP_TIMEOUT`, `NUM_WORKERS`, `GRADIENT_CHECKPOINTING`, `BF16`, `FP16`, `SAVE_ONLY_MODEL`, `WRITE_RCA`, `REPORT_TO`.
