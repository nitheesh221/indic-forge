"""Checkpoint retention (Top-K ∪ Last-K) and CSV logs.

Translation metrics are written for analysis but never select checkpoints.
Checkpoint ranking stays on SFT metrics: eval_loss, eval_entropy,
eval_mean_token_accuracy.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import torch
from transformers import TrainerCallback

from indictrans2_finetune.metrics_mt import score_translations
from indictrans2_finetune.toolkit import get_processor


def retention_count(top_k: int, last_k: int, save_total_limit: int | None = None) -> int:
    """Maximum checkpoints kept when Top-K and Last-K do not overlap."""
    union = max(0, top_k) + max(0, last_k)
    if union == 0:
        union = 1
    if save_total_limit is None or save_total_limit <= 0:
        return union
    return max(1, min(union, save_total_limit))


class CsvLogCallback(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        row = {"step": state.global_step, **{key: _plain(value) for key, value in logs.items()}}
        _append_row(self.path, row)


class CheckpointRetentionCallback(TrainerCallback):
    """Keep the union of Top-K by metric and Last-K by recency.

    Top-K selects the best SFT metric values for final model choice.
    Last-K keeps the most recent saves so you can resume or inspect late
    training even if the metric drifted. Overlap is kept once.
    """

    def __init__(
        self,
        metric: str,
        greater_is_better: bool,
        top_k: int,
        last_k: int,
        save_total_limit: int | None = None,
        manifest_name: str = "checkpoint_manifest.json",
    ):
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.top_k = max(0, int(top_k))
        self.last_k = max(0, int(last_k))
        if self.top_k == 0 and self.last_k == 0:
            self.last_k = 1
        self.save_total_limit = save_total_limit
        self.manifest_name = manifest_name
        # (score, step, path)
        self.scored: list[tuple[float, int, str]] = []

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint.exists():
            return
        score = _latest_metric(state.log_history, self.metric)
        if score is None:
            # Save happened without a fresh eval (strategy mismatch). Still
            # track for Last-K using a sentinel that never wins Top-K.
            score = float("-inf") if self.greater_is_better else float("inf")
        self.scored = [
            (existing_score, step, path)
            for existing_score, step, path in self.scored
            if step != state.global_step
        ]
        self.scored.append((float(score), state.global_step, str(checkpoint)))
        keep = self._paths_to_keep()
        for _score, _step, path in list(self.scored):
            if path not in keep:
                target = Path(path)
                if target.exists():
                    shutil.rmtree(target)
        self.scored = [
            (score, step, path) for score, step, path in self.scored if path in keep
        ]
        self._write_manifest(Path(args.output_dir))

    def _paths_to_keep(self) -> set[str]:
        ranked = sorted(
            self.scored,
            key=lambda item: item[0],
            reverse=self.greater_is_better,
        )
        # Drop sentinels from Top-K contention when real scores exist.
        real = [
            item
            for item in ranked
            if item[0] not in (float("inf"), float("-inf"))
        ]
        top = {path for _score, _step, path in (real or ranked)[: self.top_k]}
        recent = sorted(self.scored, key=lambda item: item[1], reverse=True)
        last = {path for _score, _step, path in recent[: self.last_k]}
        keep = top | last
        if self.save_total_limit and self.save_total_limit > 0 and len(keep) > self.save_total_limit:
            # Prefer metric Top-K only, then fill with most recent until the hard cap.
            # Do not walk the full metric ranking here — that would pull mediocre
            # Last-K members ahead of the newest checkpoint.
            # Stable Top-K order: best metric first.
            ordered = [
                path
                for _score, _step, path in (real or ranked)
                if path in top
            ]
            for _score, _step, path in sorted(self.scored, key=lambda item: item[1], reverse=True):
                if path in keep and path not in ordered:
                    ordered.append(path)
            keep = set(ordered[: self.save_total_limit])
        return keep

    def _write_manifest(self, output_dir: Path) -> None:
        ranked = sorted(
            self.scored,
            key=lambda item: item[0],
            reverse=self.greater_is_better,
        )
        recent = sorted(self.scored, key=lambda item: item[1], reverse=True)
        top_paths = {path for _s, _st, path in ranked[: self.top_k]}
        last_paths = {path for _s, _st, path in recent[: self.last_k]}
        payload = {
            "metric": self.metric,
            "greater_is_better": self.greater_is_better,
            "top_k": self.top_k,
            "last_k": self.last_k,
            "save_total_limit": self.save_total_limit,
            "checkpoints": [
                {
                    "path": path,
                    "step": step,
                    "score": None if score in (float("inf"), float("-inf")) else score,
                    "in_top_k": path in top_paths,
                    "in_last_k": path in last_paths,
                }
                for score, step, path in sorted(self.scored, key=lambda item: item[1])
            ],
        }
        (output_dir / self.manifest_name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


# Backward-compatible name used by older tests and imports.
class TopNCheckpointCallback(CheckpointRetentionCallback):
    def __init__(self, metric: str, greater_is_better: bool, keep: int):
        super().__init__(
            metric=metric,
            greater_is_better=greater_is_better,
            top_k=keep,
            last_k=0,
            save_total_limit=None,
        )


class TranslationMetricsCallback(TrainerCallback):
    def __init__(
        self,
        trainer_box: dict,
        sources: list[str],
        references: list[str],
        args,
        src_langs: list[str] | None = None,
        tgt_langs: list[str] | None = None,
    ):
        self.trainer_box = trainer_box
        self.sources = sources
        self.references = references
        self.user_args = args
        self.src_langs = src_langs or [args.src_lang] * len(sources)
        self.tgt_langs = tgt_langs or [args.tgt_lang] * len(sources)
        self.processor = get_processor(inference=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return
        model = kwargs.get("model") or self.trainer_box.get("model")
        tokenizer = self.trainer_box["tokenizer"]
        try:
            hypotheses = generate_translations(
                model,
                tokenizer,
                self.processor,
                self.sources,
                self.src_langs[0],
                self.tgt_langs[0],
                self.user_args.eval_max_new_tokens,
                self.user_args.generation_num_beams,
                self.user_args.eval_batch_size,
                src_langs=self.src_langs,
                tgt_langs=self.tgt_langs,
            )
        except Exception as exc:  # noqa: BLE001 — translation scores must not abort training
            output = Path(self.user_args.output_dir)
            _append_row(
                output / self.user_args.eval_metrics_csv,
                {
                    "step": state.global_step,
                    "metric_errors": f"generation failed: {type(exc).__name__}: {exc}",
                },
            )
            return
        scores, errors = score_translations(
            self.sources,
            hypotheses,
            self.references,
            self.user_args.eval_metrics,
        )
        output = Path(self.user_args.output_dir)
        metric_row = {"step": state.global_step, **{key: _plain(value) for key, value in (metrics or {}).items()}}
        metric_row.update({key: "" if value is None else value for key, value in scores.items()})
        if errors:
            metric_row["metric_errors"] = " | ".join(f"{name}: {msg}" for name, msg in errors.items())
        _append_row(output / self.user_args.eval_metrics_csv, metric_row)
        sample_rows = [
            {
                "step": state.global_step,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src": src,
                "ref": ref,
                "hyp": hyp,
            }
            for src_lang, tgt_lang, src, ref, hyp in zip(
                self.src_langs, self.tgt_langs, self.sources, self.references, hypotheses
            )
        ]
        _append_rows(output / self.user_args.eval_samples_csv, sample_rows)
        if metrics is not None:
            metrics.update({key: value for key, value in scores.items() if value is not None})


def generate_translations(
    model,
    tokenizer,
    processor,
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    max_length: int,
    num_beams: int,
    batch_size: int,
    src_langs: list[str] | None = None,
    tgt_langs: list[str] | None = None,
) -> list[str]:
    if src_langs is None:
        src_langs = [src_lang] * len(sentences)
    if tgt_langs is None:
        tgt_langs = [tgt_lang] * len(sentences)
    if not (len(sentences) == len(src_langs) == len(tgt_langs)):
        raise ValueError("sentences, src_langs, and tgt_langs must have the same length")
    device = next(model.parameters()).device
    hypotheses = [""] * len(sentences)
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for index, (sentence, pair_src, pair_tgt) in enumerate(zip(sentences, src_langs, tgt_langs)):
        grouped.setdefault((pair_src, pair_tgt), []).append((index, sentence))
    model.eval()
    for (pair_src, pair_tgt), items in grouped.items():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            prepared = processor.preprocess_batch(
                [sentence for _index, sentence in chunk],
                src_lang=pair_src,
                tgt_lang=pair_tgt,
            )
            inputs = tokenizer(
                prepared,
                truncation=True,
                padding="longest",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    use_cache=False,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=0,
                )
            decoded = tokenizer.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for (index, _sentence), hypothesis in zip(
                chunk, processor.postprocess_batch(decoded, lang=pair_tgt)
            ):
                hypotheses[index] = hypothesis
    return hypotheses


def _latest_metric(history: list[dict], name: str):
    for row in reversed(history):
        if name in row:
            return float(row[name])
    return None


def _plain(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _append_row(path: Path, row: dict) -> None:
    _append_rows(path, [row])


def _append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    write_header = not path.exists()
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
