"""Score a fine-tuned adapter on a prepared split or a train/val JSON file."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from indictrans2_finetune.data import iter_rows, rows_to_pairs, validate_official_dir
from indictrans2_finetune.infer import (
    inference_rank_and_world,
    maybe_adapter_base,
    release_inference_model,
    shutdown_inference_group,
    translate_across_ranks,
    translate_batch,
)
from indictrans2_finetune.langs import default_model, normalize_lang
from indictrans2_finetune.metrics_mt import METRIC_ORDER, score_translations


def evaluate_split(
    data_dir: Path,
    split: str,
    src_lang: str,
    tgt_lang: str,
    model_name: str | None = None,
    adapter: str | None = None,
    max_examples: int | None = None,
    batch_size: int = 8,
) -> dict:
    src_lang = normalize_lang(src_lang)
    tgt_lang = normalize_lang(tgt_lang)
    counts = validate_official_dir(data_dir, split, [src_lang], [tgt_lang])
    src_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{src_lang}"
    tgt_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{tgt_lang}"
    sources = src_path.read_text(encoding="utf-8").splitlines()
    references = tgt_path.read_text(encoding="utf-8").splitlines()
    if max_examples:
        sources = sources[:max_examples]
        references = references[:max_examples]
    model_name = model_name or maybe_adapter_base(Path(adapter) if adapter else None)
    model_name = model_name or default_model(src_lang, tgt_lang)
    predictions = translate_batch(
        sources,
        src_lang,
        tgt_lang,
        model_name=model_name,
        adapter=adapter,
        batch_size=batch_size,
    )
    scores, errors = score_translations(sources, predictions, references, list(METRIC_ORDER))
    samples = [
        {"src": src, "ref": ref, "hyp": hyp}
        for src, ref, hyp in zip(sources[:5], references[:5], predictions[:5])
    ]
    return {
        "pair": f"{src_lang}-{tgt_lang}",
        "split": split,
        "n": counts[f"{src_lang}-{tgt_lang}"] if not max_examples else len(sources),
        "metrics": scores,
        "metric_errors": errors,
        "model": model_name,
        "adapter": adapter,
        "samples": samples,
    }


def resolve_metrics(names: list[str] | None) -> list[str]:
    if not names:
        return list(METRIC_ORDER)
    selected: list[str] = []
    for name in names:
        for piece in name.replace(",", " ").split():
            if piece not in METRIC_ORDER:
                allowed = ", ".join(METRIC_ORDER)
                raise SystemExit(f"Unknown metric {piece!r}. Choose from: {allowed}")
            if piece not in selected:
                selected.append(piece)
    return selected


def evaluate_json(
    json_path: Path,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
    model_name: str | None = None,
    adapter: str | None = None,
    metrics: list[str] | None = None,
    predictions_csv: Path | None = None,
    metrics_csv: Path | None = None,
    max_length: int = 256,
    num_beams: int = 5,
    batch_size: int = 8,
    visualize: bool = False,
    report_html: Path | None = None,
) -> dict:
    """Translate a train/val JSON file and score tgt_text as the reference."""
    selected = resolve_metrics(metrics)
    pairs, skipped = rows_to_pairs(iter_rows(Path(json_path)), src_lang, tgt_lang)
    if src_lang and tgt_lang:
        wanted = (normalize_lang(src_lang), normalize_lang(tgt_lang))
        kept = [pair for pair in pairs if pair[:2] == wanted]
        skipped += len(pairs) - len(kept)
        pairs = kept
    if not pairs:
        raise SystemExit(f"No parallel rows found in {json_path}")

    grouped: dict[tuple[str, str], tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    for pair_src, pair_tgt, source, reference in pairs:
        grouped[(pair_src, pair_tgt)][0].append(source)
        grouped[(pair_src, pair_tgt)][1].append(reference)

    rank, world = inference_rank_and_world()
    if rank == 0 and world > 1:
        print(f" | > inference gpus={world}. Each GPU translates its own share of the file.")
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    translated: list[tuple[str, str, str, list[str], list[str], list[str]]] = []
    resolved_model = model_name or maybe_adapter_base(Path(adapter) if adapter else None)
    for (pair_src, pair_tgt), (sources, references) in grouped.items():
        active_model = resolved_model or default_model(pair_src, pair_tgt)
        hypotheses = translate_across_ranks(
            sources,
            pair_src,
            pair_tgt,
            model_name=active_model,
            adapter=adapter,
            max_length=max_length,
            num_beams=num_beams,
            batch_size=batch_size,
        )
        if rank == 0:
            translated.append((pair_src, pair_tgt, active_model, sources, references, hypotheses))
    release_inference_model()
    shutdown_inference_group()
    if rank != 0:
        return {
            "n": 0,
            "skipped": skipped,
            "metrics": [],
            "predictions_csv": "",
            "metrics_csv": "",
            "report_html": None,
            "primary": False,
        }
    for pair_src, pair_tgt, active_model, sources, references, hypotheses in translated:
        scores, errors = score_translations(sources, hypotheses, references, selected)
        for source, reference, hypothesis in zip(sources, references, hypotheses):
            prediction_rows.append(
                {
                    "src_lang": pair_src,
                    "tgt_lang": pair_tgt,
                    "src": source,
                    "reference": reference,
                    "prediction": hypothesis,
                }
            )
        metric_rows.append(
            {
                "src_lang": pair_src,
                "tgt_lang": pair_tgt,
                "n": len(sources),
                "model": active_model,
                "adapter": adapter or "",
                **{name: "" if scores[name] is None else scores[name] for name in selected},
                "metric_errors": " | ".join(f"{name}: {msg}" for name, msg in errors.items()),
            }
        )

    predictions_path = Path(predictions_csv or "inference_predictions.csv")
    metrics_path = Path(metrics_csv or "inference_metrics.csv")
    _write_csv(predictions_path, prediction_rows)
    _write_csv(metrics_path, metric_rows)
    report_path = None
    if visualize or report_html is not None:
        from indictrans2_finetune.infer_report import default_report_path, write_inference_report

        report_path = Path(report_html) if report_html is not None else default_report_path(metrics_path)
        write_inference_report(
            prediction_rows,
            metric_rows,
            report_path,
            source_name=str(json_path),
        )
    return {
        "n": len(prediction_rows),
        "skipped": skipped,
        "metrics": metric_rows,
        "predictions_csv": str(predictions_path),
        "metrics_csv": str(metrics_path),
        "report_html": None if report_path is None else str(report_path),
        "primary": True,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
