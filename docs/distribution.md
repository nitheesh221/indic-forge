# How IndicTrans2 training uses the GPU

This page is for someone new to multi-GPU training. You do not need to memorize it. Start with the default. Change one flag only when the run fails because the GPU is out of memory, or when you have extra GPUs and want the same job to go faster.

The flag is `--distribution`. The default is `auto`.

## What this flag does, and what it does not do

It decides where the model copy, the gradients, and the optimizer sit during supervised fine-tuning.

It does **not** look at the checkpoint name, and it does **not** measure free GPU memory. `auto` stays a single process on one GPU, and DDP when you pass two or more GPU ids.

IndicTrans2 is small compared with chat models:

| Checkpoint | Weight size in bf16 (about) |
| --- | --- |
| distilled 200M | 0.4 GB |
| distilled 320M | 0.6 GB |
| 1B | 2 GB |

LoRA trains a tiny adapter on top of those weights, so every IndicTrans2 LoRA run fits on one normal GPU. Full fine-tuning of the 1B model also stores Adam optimizer state, which is much larger than the 2 GB of weights (on the order of 18 GB before activations). 8-bit Adam plus gradient checkpointing is how that update fits on a 24 GB GPU. DDP then copies that fitted run onto a second GPU.

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

**When.** The copy already fits and you have 2 or more GPUs. That covers LoRA, QLoRA, full LoRA, the distilled full fine-tunes, and the 1B full fine-tune with 8-bit Adam and gradient checkpointing.

DDP does not save memory. It also multiplies the effective batch:

`batch_size × number of GPUs × gradient_accumulation_steps`

If you want the same batch as the one-GPU run, lower `--gradient_accumulation_steps` as you add GPUs.

One GPU with `--distribution ddp` behaves like a normal single-process run.

On two or more processes the run also does four things that a one-GPU run does not need:

- Each GPU translates its own slice of the validation set. Rank 0 gathers those translations, scores them, and writes the CSVs. One GPU no longer generates the whole dev set while the other waits out the process-group timeout.
- `--ddp_timeout` defaults to 14400 seconds (4 hours). The training arguments are built before the model loads, and that is what opens the process group.
- Each rank is pinned to its own GPU before `from_pretrained`. Rank 0 alone converts the JSON into `prepared/` and alone writes `final_model` and `finetune_meta.json`.
- Length-grouped batches stay on for one process and turn off when `WORLD_SIZE` is greater than 1. Grouping can give the ranks a different number of steps, and the shorter rank then waits until the timeout.

### QLoRA is a training mode, not a distribution

`--train_mode qlora` stores the base model in 4-bit and trains the adapter. Use it when a full update does not fit and an adapter is enough. Several GPUs means one 4-bit copy on each GPU (DDP).

## What we do not use

| Left out | Why |
| --- | --- |
| `zero2`, `zero3`, `offload`, `deepspeed` | Removed. ZeRO did not start on Python 3.13, and the 1B model already trains with DDP. The record is [deepspeed.md](deepspeed.md). |
| `fsdp` | Removed. Accelerate died looking up `decoder` on `IndicTransForConditionalGeneration` before a step. The same record. |
| DataParallel | Older one-process copy. GPU 0 does extra work, and it breaks QLoRA. `auto` stays off this path. |
| ZeRO-1 | Splits only a small part of the optimizer. Not its own flag. |
| Tensor parallelism | Cuts a single layer across GPUs. Built for models of tens of billions of parameters. |
| Pipeline parallelism | Puts stacks of layers on different GPUs. IndicTrans2 is far smaller than that. |

## A short decision list

1. `auto` on one GPU.
2. `auto` or `ddp` with `--gpu_ids 0,1` when the model fits and you want both GPUs to train.
3. Out of memory: `--gradient_checkpointing`, `--optim adamw_bnb_8bit`, and a smaller `--batch_size`.
4. Prefer an adapter instead of updating every weight: `--train_mode qlora` with `auto` or `ddp`.
