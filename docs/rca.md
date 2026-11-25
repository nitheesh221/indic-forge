# Reading the training report

Training writes `rca_report.html` in the output directory. Rebuild it any time with:

```bash
indictrans2-ft analyze --run-dir it2_out/lora
```

Open the file in a browser. The charts are drawn inside the file, so it works on a machine with no internet.

## What to read first

The green box at the top answers the practical question: which step was best, and is the last step the same one?

- **Best validation loss** is the step `final_model` comes from when eval and save use the same schedule (the default).
- **Last eval** is the end of training. Those weights stay in the newest `checkpoint-<step>` folder.
- If chrF++ peaked on a different step, the page says so. chrF++ is there to read the run. It does not choose `final_model`.

## Findings

Warnings compare the **best** step with the **last** step. A run that got worse in the middle and then recovered is not called a failure just because the end is better than the start. A run that peaked early and then got worse is called out, with the step number.

Exact match is reported, then set aside. On real sentences it is usually near zero, so it does not show which language failed.

## Charts

Each metric has its own chart. Train loss and validation loss share one chart, because they use the same units and the gap between them is the overfitting picture. chrF++, COMET, BLEU, and the rest are separate, so a 0–1 score cannot flatten a 0–100 score.

The green dot is the best point on that chart. The red dot is the last point. “Best” follows the metric: lower for loss, entropy, and TER; higher for chrF++, COMET, BLEU, BERTScore, METEOR, token accuracy, and exact match.

## Languages and sentences

The language table is the latest eval, worst chrF++ first. That is the column that shows which pair failed.

The sentence table is also sorted by the latest chrF++. For each source it also shows the best chrF++ that sentence reached on any earlier eval, and the hypothesis from that step. A negative change means the latest output is worse than an earlier one.

## Checkpoints

The checkpoint table is the same Top-N and Last-N list as `checkpoint_manifest.json`. Use it to find the folder for the last model when you want to compare it with `final_model`.
