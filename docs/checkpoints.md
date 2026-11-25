# Checkpoints: best model, last model, and what is kept on disk

This page explains the files a training run leaves behind. There are two different things, and they are easy to mix up.

| Path | What it is |
| --- | --- |
| `checkpoint-<step>/` | A full trainer save from that step. Used to resume, and to pick an earlier or later model. |
| `final_model/` | One clean copy written after training ends. This is the model you normally run. |

`final_model` is **not** a copy of the newest `checkpoint-<step>` folder. With the default settings it is the **best** model. The **last** model stays in the newest checkpoint folder.

## What is saved during training

On each save (default: end of every epoch) the trainer writes:

```text
it2_out/lora/checkpoint-50/
it2_out/lora/checkpoint-100/
it2_out/lora/checkpoint-150/
```

The number is the training step. Each folder can resume a run (`--resume_from_checkpoint`) and also holds the weights from that moment.

Two rules decide which of those folders stay. The rest are deleted. On more than one GPU, every rank waits until the checkpoint write has finished, and only rank 0 deletes.

### Top-N: the best scores

`--top_n_checkpoints` (default `3`) keeps the best checkpoints according to `--metric_for_best_model`.

The default metric is `eval_loss`. Lower is better. So Top-3 means the three saves with the lowest validation loss.

You can rank by `eval_entropy` (lower is better) or `eval_mean_token_accuracy` (higher is better). Those are training metrics: how surprised the model is on the next token, and how often that token is correct.

chrF++, COMET, BLEU, BERTScore, METEOR, TER, and exact match are written to `eval_metrics.csv` for the report. They never choose which checkpoint is kept, and they never choose `final_model`.

### Last-N: the most recent saves

`--last_k_checkpoints` (default `2`) keeps the most recent checkpoint folders even when their score is worse than the best ones.

This is already implemented. There is no separate `--last_n_checkpoints` flag. Last-N **is** `--last_k_checkpoints`.

Why keep a model that is not the best:

- You can resume from the end of training, not only from an earlier good epoch.
- You can inspect late training in the RCA report when the model started to overfit.
- The best epoch and the last epoch are often different. Both should still be on disk.

### Both lists are kept together

A folder is kept if it is in Top-N **or** in Last-N. If one checkpoint is both a best score and a recent save, it is stored once.

Default `3 + 2` means up to 5 folders when the lists do not overlap, and fewer when they do.

After every save the run rewrites `checkpoint_manifest.json` in the output directory. Each entry has:

- `path` and `step`
- `score` for the ranking metric
- `in_top_k`: true when this folder is one of the best
- `in_last_k`: true when this folder is one of the most recent

The report from `indictrans2-ft analyze` shows the same table.

`--save_total_limit` is an optional hard cap. `0` (the default) means no extra cap: keep the full Top-N plus Last-N union. If you set it to a positive number and the union is larger, the best scores are kept first, then the newest remaining folders fill the rest.

## What `final_model` is

When training finishes, the code saves whatever weights are in memory to `--final_model_dir`. The default path is `<output_dir>/final_model`. For `full_lora`, the LoRA adapter is merged into the base model before that save, except under ZeRO or FSDP, where a local merge would see only a shard and the adapter is saved unmerged. `finetune_meta.json` is written next to those weights by rank 0. It names `best_model_checkpoint`, `best_metric`, and `epochs_completed`. On more than one GPU, ZeRO and FSDP still enter the save on every rank so the shards can be gathered. Only rank 0 writes the files.

Whether those weights are the best epoch or the last step depends on one switch inside the trainer: reload the best checkpoint at the end.

That reload is **on** when all of these are true:

- `--eval_strategy` and `--save_strategy` are the same (`epoch` and `epoch`, or `steps` and `steps`)
- evaluation is not turned off (`--eval_strategy` is not `no`)

Both strategies default to `epoch`, so a normal run reloads the best checkpoint, then saves it as `final_model`.

## Early stopping

`--early_stopping_patience` (default `3`) stops the run after that many evaluations fail to improve `--metric_for_best_model`. The first evaluation sets the baseline and does not count as a miss. `--early_stopping_patience 0` leaves the run going for the full `--epochs`.

Patience counts evaluations. With the default `--eval_strategy epoch`, three means three epochs without a better score. With `--eval_strategy steps`, three means three evaluation calls, which can fall inside one epoch.

`--early_stopping_threshold` (default `0.0`) is the smallest change that counts as an improvement. For `eval_loss`, the loss has to fall by more than this amount. For `eval_mean_token_accuracy`, the accuracy has to rise by more than this amount.

The stop uses the same metric as the best-checkpoint choice. chrF++, COMET, BLEU, BERTScore, METEOR, TER, and exact match do not stop training. Evaluation and saving must stay on the same schedule (`epoch`/`epoch` or `steps`/`steps`), because the reload of the best checkpoint depends on that. A run that stops early still writes `final_model` from the best checkpoint, and `finetune_meta.json` sets `stopped_early` to true.

On more than one GPU, rank 0 reads the score and the other ranks stop on that same decision. Resuming from a checkpoint keeps the previous best score and starts the stall count from zero.

**Normal run:** `final_model` is the best model (lowest `eval_loss`, unless you changed the metric). It can be an earlier epoch. The last epoch is still on disk as the newest `checkpoint-<step>` folder, because Last-N kept it.

**Reload turned off:** set `--eval_strategy` and `--save_strategy` to different values, or set `--eval_strategy no`. Then nothing is reloaded. `final_model` is the last model, the weights from the final training step.

Keep the two strategies the same if you want `final_model` to be the best model. The trainer also prints a warning when they differ, because Top-N ranking is reliable only when a save happens at the same time as an eval.

## How to run the best model and the last model

Best model, the usual inference path:

```bash
indictrans2-ft translate \
  --text "Your Aadhaar OTP is valid for 10 minutes." \
  --src-lang eng_Latn \
  --tgt-lang hin_Deva \
  --adapter it2_out/lora/final_model
```

For `--train_mode full` or `full_lora`, `final_model` is a full model. Pass it as `--model`, not `--adapter`.

Last model: open `checkpoint_manifest.json`, find the row with `"in_last_k": true` and the highest `step`, and pass that folder.

```text
it2_out/lora/checkpoint-150
```

That folder is a trainer checkpoint, not a second copy named `last_model`. It is enough to resume training or to load those weights. This repo does not write a separate clean `last_model/` directory.

## One setting to avoid

Leave `--top_n_checkpoints` at 1 or higher while the best-model reload is on.

If Top-N is `0`, only Last-N folders stay. The best epoch can be deleted for being too old. At the end the trainer still tries to reload that best folder, and the load fails because the files are gone.

Last-N can be `0` only in the sense that you asked for none. If both Top-N and Last-N are `0`, the run keeps 1 checkpoint anyway so the output directory is not empty.

## A short picture

Five epoch saves, `eval_loss`, Top-3, Last-2:

| Step | eval_loss | Kept? | Why |
| --- | --- | --- | --- |
| 50 | 1.20 | no | worse than the best three, and not recent |
| 100 | 0.80 | yes | best score (Top-N) |
| 150 | 0.90 | yes | second best (Top-N) |
| 200 | 1.05 | yes | third best, and also recent (Top-N and Last-N) |
| 250 | 1.40 | yes | most recent (Last-N), even though the loss got worse |

`final_model` on this run is the step 100 weights, because 0.80 is the lowest loss. Step 250 remains in `checkpoint-250` because Last-N kept it. Use step 100 (`final_model`) for inference unless you specifically want to compare the last epoch.
