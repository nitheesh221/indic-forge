# How IndicTrans2 training uses the GPU

This page is for someone new to multi-GPU training. You do not need to memorize it. Start with the default. Change one flag only when the run fails because the GPU is out of memory, or when you have extra GPUs and want the same job to go faster.

The flag is `--distribution`. The default is `auto`.

## What this flag does, and what it does not do

It decides where the model copy, the gradients, and the optimizer sit during supervised fine-tuning.

It does **not** look at the checkpoint name, and it does **not** measure free GPU memory. `auto` will not silently switch to ZeRO because you picked the 1B model. You choose the next step yourself, using the list below.

IndicTrans2 is small compared with chat models:

| Checkpoint | Weight size in bf16 (about) |
| --- | --- |
| distilled 200M | 0.4 GB |
| distilled 320M | 0.6 GB |
| 1B | 2 GB |

LoRA trains a tiny adapter on top of those weights, so every IndicTrans2 LoRA run fits on one normal GPU. Full fine-tuning of the 1B model also stores Adam optimizer state, which is much larger than the 2 GB of weights (on the order of 18 GB before activations). That is the only common case that needs a memory strategy.

## Start here

1. Use `--distribution auto`.
2. If you have one GPU, that is a normal single-process run.
3. If the model already fits and you have several GPUs, pass all of them. `auto` then uses DDP so the job is faster.
4. Change the flag only for the cases in the next section.

```bash
# One GPU. This is the usual command.
python train.py --gpu_ids 0 --distribution auto --train_mode lora ...

# Two GPUs, model already fits. auto becomes DDP.
python train.py --gpu_ids 0,1 --distribution auto --train_mode lora ...
```

`--distribution ddp` is the same multi-GPU path, written out explicitly. Use it when you want the command to say “DDP” instead of “auto”.

## The options

### `auto` — default

**What.** One GPU: one process, one copy of the model. Several ids in `--gpu_ids`: DDP (explained next).

**Why.** One copy is the simplest correct setup, and these models fit that way for LoRA and for the distilled full fine-tunes.

**When.** Every new run. Also every 1B LoRA run. Add GPUs only when this already works and you want more sentences per second.

A bigger batch on one GPU is `--gradient_accumulation_steps` (default 4). That is not a second GPU.

### `ddp` — faster when the model already fits

**What.** Each GPU holds a full copy of the model and sees different sentences. Once per update, the GPUs average their gradients. The result is the same training step, done on more data at once.

**Why.** Splitting the model would add communication you do not need when the copy already fits.

**When.** LoRA, QLoRA, full LoRA, or full fine-tuning of the 200M / 320M model, and you have 2 or more GPUs.

DDP does not save memory. It also multiplies the effective batch:

`batch_size × number of GPUs × gradient_accumulation_steps`

If you want the same batch as the one-GPU run, lower `--gradient_accumulation_steps` as you add GPUs.

One GPU with `--distribution ddp` behaves like a normal single-process run.

On two or more processes the run also does four things that a one-GPU run does not need:

- Each GPU translates its own slice of the validation set. Rank 0 gathers those translations, scores them, and writes the CSVs. One GPU no longer generates the whole dev set while the other waits out the process-group timeout.
- `--ddp_timeout` defaults to 14400 seconds (4 hours). The training arguments are built before the model loads, and that is what opens the process group, so the same timeout covers DDP and DeepSpeed.
- Each rank is pinned to its own GPU before `from_pretrained`. Rank 0 alone converts the JSON into `prepared/` and alone writes `final_model` and `finetune_meta.json`. ZeRO and FSDP still enter the save on every rank so the shards can be gathered.
- Length-grouped batches stay on for one process and turn off when `WORLD_SIZE` is greater than 1. Grouping can give the ranks a different number of steps, and the shorter rank then waits until the timeout.

### `zero2` — full 1B, optimizer does not fit

**What.** DeepSpeed ZeRO stage 2. Each GPU still holds the full weights. The Adam optimizer state and the gradients are split across the GPUs. The config is `configs/ds_zero2.json`.

**Why.** The 1B weights are about 2 GB. The optimizer is what fills a 16 GB card, or a tight 24 GB card, during full fine-tuning. Splitting the optimizer fixes that and still writes a normal checkpoint.

**When.** `--train_mode full` on a 1B checkpoint, and only after DDP with `--optim adamw_bnb_8bit` and `--gradient_checkpointing` is still out of memory. Install DeepSpeed first (`pip install deepspeed`). The same install covers `zero3` and `offload`.

On Python 3.13 with DeepSpeed 0.20.1, ZeRO does not start: importing an unused Muon kernel calls `torch.compile`, and that raises `Dynamo is not supported on Python 3.13+` before Adam is checked. `zero3` and `offload` hit the same line. A two-L4 full 1B run that already fits on DDP at batch 8 should stay on DDP. The error text, the failed patches, and the working command are in [deepspeed.md](deepspeed.md).

Do this first:

```bash
python train.py \
  --gpu_ids 0 \
  --distribution auto \
  --train_mode full \
  --optim adamw_bnb_8bit \
  --gradient_checkpointing \
  --batch_size 1 \
  ...
```

If that still runs out of memory, then:

```bash
SAVE_ONLY_MODEL=0 DISTRIBUTION=zero2 GPU_IDS=0,1 bash scripts/train_full.sh
```

`scripts/train_full.sh` turns on `--save_only_model` unless you set `SAVE_ONLY_MODEL=0`. ZeRO and FSDP refuse to start when that flag is combined with reloading the best checkpoint, which this repo does by default. `SAVE_ONLY_MODEL=0` keeps the optimizer in the checkpoint so ZeRO can reload the best weights and so a stopped run can resume. Those checkpoints are large.

ZeRO-2 can also run on a single GPU (`--gpu_ids 0`). It still needs the DeepSpeed launcher, which this repo starts for you. On one GPU the optimizer is not split across cards, so two GPUs are the useful case.

LoRA does not need `zero2`. The command will warn you and still run.

### `zero3` — the weights themselves do not fit

**What.** DeepSpeed ZeRO stage 3. Weights, gradients, and optimizer state are all split. Each layer is gathered from the other GPUs only while it runs. The config is `configs/ds_zero3.json`.

**Why.** This saves more memory than ZeRO-2 and costs more communication. For a 2 GB model that extra split is usually wasted work.

**When.** The bf16 weights themselves do not fit on the GPU. That is rare for IndicTrans2 (the 1B weights are about 2 GB). If the weights fit and only Adam does not, stay on `zero2`.

### `offload` — last resort

**What.** ZeRO-3 with the weights and the optimizer parked in CPU RAM. The config is `configs/ds_offload.json`. Training still runs, but each step moves data between CPU and GPU, so it is slow.

**Why.** A very small GPU can still full-fine-tune the 1B model when you cannot use QLoRA and ZeRO across GPUs is not available.

**When.** Nothing above fits, and you still need a full (not LoRA) update.

To offload to disk instead of CPU RAM, copy `configs/ds_offload.json`, change both `"device"` values from `"cpu"` to `"nvme"`, and add an `"nvme_path"` folder that exists on that machine. Then pass the file:

```bash
python train.py --gpu_ids 0 --distribution offload --deepspeed /path/to/my_nvme.json --train_mode full ...
```

NVMe is slower than CPU offload. Use it only when CPU RAM is also too small.

### QLoRA is a training mode, not a distribution

`--train_mode qlora` stores the base model in 4-bit and trains the adapter. Use it when a full 1B update does not fit and you are fine training an adapter instead of every weight.

QLoRA is allowed with `auto` and `ddp` only. It refuses `zero2`, `zero3`, `offload`, and `fsdp`, because 4-bit weights are not split safely here. Several GPUs means one 4-bit copy on each GPU (DDP), not one model sliced across them.

### `fsdp` — not supported for this IndicTrans checkpoint

FSDP slices the model across GPUs. It does not load DeepSpeed, so it skips the Python 3.13 Muon crash. A full 1B launch on two L4s still died in Accelerate before the first step: `ensure_weights_retied` looks up a tied weight named `decoder` on `IndicTransForConditionalGeneration`, and that class has no such attribute. The warning about `_set_gradient_checkpointing` printed first and is not the failure. Changing batch size or the FSDP strategy string does not reach those settings. The traceback and the command that was run are in [deepspeed.md](deepspeed.md). QLoRA is refused with FSDP. Use DDP when the copy fits.

`--distribution deepspeed` is the old name for `zero2`. New commands should say `zero2`.

## What we do not use

| Left out | Why |
| --- | --- |
| DataParallel | Older one-process copy. GPU 0 does extra work, and it breaks QLoRA. `auto` stays off this path. |
| ZeRO-1 | Splits only a small part of the optimizer. Not enough to be its own choice. |
| Tensor parallelism | Cuts a single layer across GPUs. Built for models of tens of billions of parameters. |
| Pipeline parallelism | Puts stacks of layers on different GPUs. Same reason: IndicTrans2 is far smaller than that. |

## A short decision list

1. `auto` on one GPU.
2. `auto` or `ddp` with `--gpu_ids 0,1` when more GPUs already fit the model and you want speed.
3. For a full 1B out-of-memory error: one GPU with `--gradient_checkpointing` and `--optim adamw_bnb_8bit`.
4. Still out of memory on that full 1B run: `zero2`. On Python 3.13 with DeepSpeed 0.20.1, read [deepspeed.md](deepspeed.md) first. ZeRO dies at import, and DDP with 8-bit Adam is the path that fits two 24 GB L4s.
5. Prefer an adapter instead of updating every weight: `--train_mode qlora` with `auto` or `ddp`.
6. The weights themselves do not fit: `zero3`.
7. Last: `offload` (CPU, or NVMe if you edit the config).
