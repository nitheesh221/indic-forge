# Why this 1B run uses DDP, not DeepSpeed

This page records the Indic-to-English 1B full fine-tune on two NVIDIA L4 GPUs (about 23–24 GB each), the errors that stopped ZeRO, what was tried, and why the other distribution flags are the wrong tool once DDP already fits.

The short version: DeepSpeed is installed and the `zero2` / `zero3` / `offload` flags are still in the trainer. On this machine they do not start. DDP with 8-bit Adam does, at batch 8. Stay on that run.

The flag guide is [distribution.md](distribution.md). This page is the incident record for the 1B L4 job.

## The run

| Item | Value |
| --- | --- |
| Model | `ai4bharat/indictrans2-indic-en-1B` |
| Mode | `full` (every weight, learning rate `2e-5`) |
| GPUs | 2× L4, about 23–24 GB each |
| Python | 3.13, virtualenv under the training tree |
| DeepSpeed | 0.20.1. Confirmed by `deepspeed/runtime/zero/utils.py`: the Muon import sits inside `is_zero_supported_optimizer` at line 81. That placement is 0.20.1. |
| PyTorch | The build that raises `Dynamo is not supported on Python 3.13+` from `torch/__init__.py` |
| Trainer checkout that actually ran | `a0c7bbd` (`Stop COMET from waiting for a GPU that already exited.`) |

`a0c7bbd` is real, and it is older than the ZeRO workaround on `main`. How that was proven is below.

## What each strategy would have done here

IndicTrans2 1B weights in bf16 are about 2 GB. A full update also stores a gradient and an Adam state for every parameter. Full Adam is on the order of 18 GB before the activation memory from a 256-token decoder and a vocabulary of about 256k tokens. That is why ZeRO was the first recommendation: DDP keeps a whole copy of that footprint on every GPU, and 18 GB plus activations looked too large for a 24 GB card.

8-bit Adam (`adamw_bnb_8bit`) stores those optimizer states in 8 bits. Together with `--gradient_checkpointing` and `--bf16`, the full 1B update then fit on one L4. DDP can copy that smaller footprint onto the second L4. Measured result: `--batch_size 8`, `--eval_batch_size 8`, `--gradient_accumulation_steps 4` trains on both GPUs. Effective batch is `8 × 4 × 2 = 64`.

ZeRO is a memory tool for the case that did not happen. It splits optimizer state (stage 2) or weights as well (stage 3) and pays for that with communication and a second launcher. Once the copy fits, that payment buys nothing.

## Error 1 — the 1B checkpoint is gated

The first launch died before any step, while both ranks loaded the tokenizer:

```text
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/ai4bharat/indictrans2-indic-en-1B.
401 Client Error.
Cannot access gated repo for url
https://huggingface.co/ai4bharat/indictrans2-indic-en-1B/resolve/main/config.json.
Access to model ai4bharat/indictrans2-indic-en-1B is restricted.
You must have access to it and be authenticated to access it. Please log in.
```

The distilled 200M checkpoint is public, which is why the earlier full run never asked for a token. The 1B checkpoint is gated. Access is per Hugging Face account: the account that created the token has to open the model page and accept the form. A token exported in a different terminal is invisible to this process. Set both of these in the same shell that launches `train.py`, then check them before training:

```bash
export HF_TOKEN=hf_...
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
python -c "from huggingface_hub import whoami, hf_hub_download; print(whoami()['name']); print(hf_hub_download('ai4bharat/indictrans2-indic-en-1B', 'config.json'))"
```

`whoami` must print the account that accepted the form, and the next line must be a local path to `config.json`. A 401 there is still the gate, not the GPUs.

A token that was pasted into a chat is burned. Revoke it at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a new Read token. Do not put the new value in a chat, a commit, or this document.

Later logs reached `trainer.train`, so this gate was cleared. It is not why ZeRO failed.

## Error 2 — ZeRO dies inside DeepSpeed, before Adam is checked

Every `zero2` launch died in the same place. Rank 0 and rank 1 both raised it. The NCCL line that follows (`process group has NOT been destroyed`) is the other rank noticing that this one already exited. It is not a second bug.

```text
File "transformers/trainer.py", line 2480, in _inner_training_loop
    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
File "accelerate/accelerator.py", line 2450, in _prepare_deepspeed
    engine, optimizer, _, lr_scheduler = ds_initialize(**kwargs)
File "deepspeed/runtime/engine.py", line 2424, in _do_optimizer_sanity_check
    if not is_zero_supported_optimizer(basic_optimizer):
File "deepspeed/runtime/zero/utils.py", line 81, in is_zero_supported_optimizer
    from deepspeed.runtime.zero.muon.muon_optimizer import MuonWithAuxAdam
File "deepspeed/runtime/zero/muon/original_muon.py", line 50, in <module>
    @compiler.compile()
File "torch/__init__.py", line 2416, in compile
    raise RuntimeError("Dynamo is not supported on Python 3.13+")
RuntimeError: Dynamo is not supported on Python 3.13+
```

The call chain, in order:

1. `Seq2SeqTrainer.train` asks Accelerate to prepare the model and the optimizer.
2. `--distribution zero2` makes Accelerate call `deepspeed.initialize`. `zero3` and `offload` enter this same function. DDP does not.
3. DeepSpeed builds the engine and checks that ZeRO knows the optimizer.
4. DeepSpeed 0.20.1 does that check by importing `MuonWithAuxAdam`, even when the run uses Adam. The import is inside the function, so a plain `import deepspeed` succeeds and the crash waits until `trainer.train`.
5. `original_muon.py` decorates its Newton–Schulz kernels with `@compiler.compile()`. In DeepSpeed 0.20.1 that helper is `deepspeed/runtime/compiler.py`, and `compile()` returns `torch.compile` whenever the attribute exists.
6. The decorator calls `torch.compile` while the file is importing. On Python 3.13 this PyTorch build raises immediately. Dynamo, the compiler backend, does not support 3.13. No environment variable skips that raise, because the raise is the first line of `torch.compile`, before Dynamo runs.
7. The Muon kernels are not used by this trainer. The import exists only so ZeRO can add Muon to its supported-optimizer list. Adam never gets inspected.

Changing `--optim` does not avoid the raise. The traceback is identical for `adamw_bnb_8bit` and for `adamw_torch`. The optimizer argument is read later, after this import.

`import deepspeed` working, and `nvidia-smi` showing two GPUs, do not mean ZeRO can start. The failure is one decorator in an optimizer this run does not call.

## The error that is waiting behind that import

If the Muon file imported cleanly, DeepSpeed 0.20.1 would still reject 8-bit Adam. `is_zero_supported_optimizer` allows an optimizer only when its exact class is in a fixed list: `torch.optim.Adam`, `torch.optim.AdamW`, and DeepSpeed’s own fused Adam, Adagrad, and Lion. `bitsandbytes` `AdamW8bit` is a different class. The engine then asserts:

```text
You are using an untested ZeRO Optimizer. Please add
<"zero_allow_untested_optimizer": true> in the configuration file to use it.
```

Setting that flag is the wrong fix. ZeRO shards every tensor in the optimizer state. 8-bit Adam also keeps small quantization tables. Sharding those tables makes the state unrestorable. The 8-bit optimizer is how DDP fits on one 24 GB card. ZeRO cannot use that optimizer. ZeRO’s supported optimizer is full `adamw_torch`, which is the large state DDP just avoided.

On `main` after `a0c7bbd`, a ZeRO run rewrites `adamw_bnb_8bit`, `paged_adamw_8bit`, and `paged_adamw_32bit` to `adamw_torch` and prints a note. That code never ran on this machine. See the next section.

## What was tried

### 1. A workaround in the trainer

`af7e372` and `e61332d` on `main` do two things before `trainer.train` when the distribution is `zero2`, `zero3`, or `offload`:

- On Python 3.13, replace `torch.compile` and `torch.compiler.compile` with a decorator that returns the original function. DeepSpeed looks up `torch.compile` at decoration time. Replacing only `torch.compile` is not enough, because `torch.compiler.compile` keeps its own reference. The log line for this path is `Python 3.13: ZeRO imports without torch.compile.`
- Force `adamw_torch` so ZeRO is handed a class it can shard.

The training directory never loaded those commits. `git log -1` printed:

```text
a0c7bbd (HEAD -> main) Stop COMET from waiting for a GPU that already exited.
```

And the file that crashed still had the old line numbers:

```text
runner.py line 43:  run(args)
runner.py line 136: trainer.train(...)
```

In the workaround, `run(args)` is no longer line 43 and `trainer.train` is no longer line 136. Python reads those lines from the file on disk, so the process that crashed was `a0c7bbd`. `git pull origin main` from that checkout did not move `HEAD`. An explicit fetch was the next step, and it was not what the crashing process used:

```bash
git fetch https://github.com/nitheesh221/indic-forge.git main
git merge --ff-only FETCH_HEAD
git log -1 --oneline
```

`git log` has to print `e61332d` or a later commit before that workaround is in the process. Until then, editing the training command cannot reach it.

### 2. Patch DeepSpeed’s `compiler.py` in the virtualenv

The proposed edit was in the installed package, not in this repo:

`.../site-packages/deepspeed/runtime/compiler.py`

`compile()` would return `dummy_decorator` on Python 3.13 instead of `torch.compile`. Every `@compiler.compile()` in DeepSpeed would then keep the function eager.

Later tracebacks still ended in `torch/__init__.py`, line 2416. That means `compile()` was still returning `torch.compile` when training started. The virtualenv file had not been saved, or training was launched again before the edit. A patched `compiler.py` cannot produce this traceback.

### 3. Delete the decorators in `original_muon.py`

The same virtualenv file that raises, `deepspeed/runtime/zero/muon/original_muon.py`, has `@compiler.compile()` on the kernels (line 50 is the first). Deleting lines whose stripped text starts with `@compiler.compile` lets that module import on Python 3.13. The kernels stay ordinary functions. This trainer never calls them.

This edit was not confirmed. The next launch still showed line 50 as `@compiler.compile()`, which is the unedited file. Training was started again before the file changed.

### 4. Leave DeepSpeed and use DDP

This is the attempt that ran.

```bash
python train.py \
  --gpu_ids 0,1 \
  --distribution ddp \
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

DDP never calls `deepspeed.initialize`, so it never imports Muon and never calls `torch.compile` from that decorator. 8-bit Adam is a legal Hugging Face optimizer on this path. The run proceeded with batch 8, evaluation batch 8, and gradient accumulation 4.

One watch point remains. Generation at the end of an epoch holds the decoder cache and the vocabulary logits. That peak can be higher than a checkpointed training step at the same batch. If epoch 1 dies with `CUDA out of memory` during evaluation only, keep `--batch_size 8` and set `--eval_batch_size 4`.

## Why the other strategies are not next

| Flag | Why it is not this run |
| --- | --- |
| `zero2` | Same DeepSpeed initialize path, so the same Python 3.13 import crash. After that crash is fixed it still cannot use the 8-bit Adam that made DDP fit. It would shard full Adam across the two L4s and add communication the fitted DDP run does not need. Checkpoints must keep the optimizer (`--save_only_model` has to stay off) or ZeRO cannot reload the best epoch. |
| `zero3` | Same import crash. It also slices the 2 GB of weights and gathers each layer from the other GPU while it runs. Those weights already fit, so the extra traffic is wasted. |
| `offload` | Same import crash, then ZeRO-3 with weights and Adam parked in CPU memory. Every step copies them back. It is the slow path for a GPU that cannot hold the model. These L4s can. |
| `deepspeed` | Old name. The trainer rewrites it to `zero2` and then hits the same crash. |
| `fsdp` | Does not import DeepSpeed’s Muon file, so it would not hit this particular `torch.compile` error. It still slices a 2 GB model across GPUs. DDP is faster when each GPU can hold the copy, which this run has shown. FSDP with LoRA is fragile in this repo, and QLoRA is refused. |
| `qlora` with ZeRO or FSDP | Refused on purpose. 4-bit weights stay on one device per GPU. QLoRA is allowed with `auto` and `ddp` only, and this job is a full update, not an adapter. |
| DataParallel | One process, with GPU 0 doing the extra gather. It is the older multi-GPU path and it breaks QLoRA. `auto` does not use it. |
| ZeRO-1 | Splits only a slice of the optimizer. Not a separate flag here. Stage 2 is the first ZeRO stage that would matter, and stage 2 does not start on this Python. |
| Tensor or pipeline parallelism | Those cut one layer, or a stack of layers, across GPUs. They are for models of tens of billions of parameters. A 2 GB encoder–decoder does not need them. |

## When ZeRO becomes the right flag again

Use `zero2` only if all of these are true:

1. DDP with `--optim adamw_bnb_8bit`, `--gradient_checkpointing`, `--bf16`, and `--batch_size 1` is out of memory on one L4. That is not the current run.
2. Python is 3.12 or older, or DeepSpeed no longer calls `torch.compile` while importing Muon. On Python 3.13 with DeepSpeed 0.20.1, ZeRO does not reach the first step.
3. `--optim` is `adamw_torch`. Do not pass `adamw_bnb_8bit` and expect ZeRO to shard it.
4. `--save_only_model` is off, so the checkpoint still contains the optimizer ZeRO has to reload.

`zero3` waits until the bf16 weights themselves do not fit. `offload` waits until ZeRO across the two GPUs is still out of memory. Neither of those is this machine.
