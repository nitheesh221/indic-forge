"""Command-line interface for IndicTrans2 LoRA fine-tuning."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from indictrans2_finetune.data import prepare_file
from indictrans2_finetune.langs import LANGUAGES, default_model, normalize_lang

app = typer.Typer(help="Fine-tune IndicTrans2 with LoRA on your parallel data.", no_args_is_help=True)


@app.command("langs")
def langs_cmd() -> None:
    """Print FLORES language tags accepted by IndicTrans2."""
    for code, meta in LANGUAGES.items():
        typer.echo(f"{code:12}  {meta['name']:16}  {meta['script']}")


@app.command("prepare")
def prepare_cmd(
    train: Path = typer.Option(..., exists=True, help="Train JSONL/CSV/TSV"),
    dev: Path = typer.Option(..., exists=True, help="Dev JSONL/CSV/TSV"),
    output_dir: Path = typer.Option(Path("data/prepared"), help="Official layout root"),
    src_lang: Optional[str] = typer.Option(None, help="Default source FLORES tag"),
    tgt_lang: Optional[str] = typer.Option(None, help="Default target FLORES tag"),
) -> None:
    """Convert JSONL or CSV into the official train/dev pair folders."""
    train_report = prepare_file(train, output_dir, "train", src_lang, tgt_lang)
    dev_report = prepare_file(dev, output_dir, "dev", src_lang, tgt_lang)
    payload = {
        "output_dir": str(output_dir),
        "train": train_report.pairs["train"],
        "dev": dev_report.pairs["dev"],
        "skipped_train": train_report.skipped,
        "skipped_dev": dev_report.skipped,
    }
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@app.command(
    "train",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train_cmd(ctx: typer.Context) -> None:
    """Run train.py. Example: indictrans2-ft train --gpu_ids 0 --model_hf_id ..."""
    from indictrans2_finetune.runner import main

    raise typer.Exit(main(ctx.args))


@app.command("translate")
def translate_cmd(
    text: Optional[str] = typer.Option(None, help="Single sentence"),
    file: Optional[Path] = typer.Option(None, exists=True, help="One sentence per line"),
    json_file: Optional[Path] = typer.Option(
        None,
        "--json-file",
        exists=True,
        help="Train/val JSON. tgt_text is the reference.",
    ),
    src_lang: Optional[str] = typer.Option(None),
    tgt_lang: Optional[str] = typer.Option(None),
    model: Optional[str] = typer.Option(None),
    adapter: Optional[Path] = typer.Option(None, help="Directory with saved LoRA weights"),
    eval_metrics: Optional[list[str]] = typer.Option(
        None,
        "--eval-metrics",
        help="One or more of chrf2pp, comet, bleu, bertscore, meteor, ter, exact_match.",
    ),
    predictions_csv: Optional[Path] = typer.Option(
        None,
        "--predictions-csv",
        help="Per-sentence src, reference, and prediction.",
    ),
    metrics_csv: Optional[Path] = typer.Option(
        None,
        "--metrics-csv",
        help="Corpus scores for the selected metrics.",
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize",
        help="Write an offline HTML chart page for a --json-file run.",
    ),
    report_html: Optional[Path] = typer.Option(
        None,
        "--report-html",
        help="HTML path for --visualize. Default: inference_report.html next to the metrics CSV.",
    ),
    max_length: int = typer.Option(256, "--max-length"),
    num_beams: int = typer.Option(5, "--num-beams"),
    batch_size: int = typer.Option(8, "--batch-size", help="Sentences per generate call."),
    gpu_ids: Optional[str] = typer.Option(
        None,
        "--gpu-ids",
        help="Comma-separated GPU indices, for example 0,1. Two or more split the file.",
    ),
    local_rank: Optional[int] = typer.Option(
        None,
        "--local-rank",
        "--local_rank",
        help="Set by torchrun.",
    ),
) -> None:
    """Translate one sentence, a text file, or a train/val JSON file."""
    del local_rank
    from indictrans2_finetune.infer_launch import relaunch_translate_if_needed

    relaunch_translate_if_needed(gpu_ids)
    from indictrans2_finetune.evaluate import evaluate_json
    from indictrans2_finetune.infer import (
        maybe_adapter_base,
        release_inference_model,
        shutdown_inference_group,
        translate_across_ranks,
    )

    if json_file and (text or file):
        raise typer.BadParameter("Pass --json-file or --text/--file, not both.")
    if json_file:
        result = evaluate_json(
            json_file,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            model_name=model,
            adapter=str(adapter) if adapter else None,
            metrics=eval_metrics,
            predictions_csv=predictions_csv,
            metrics_csv=metrics_csv,
            max_length=max_length,
            num_beams=num_beams,
            batch_size=batch_size,
            visualize=visualize or report_html is not None,
            report_html=report_html,
        )
        if not result.get("primary", True):
            return
        lines = [
            f"scored {result['n']} rows, skipped {result['skipped']}",
            f"predictions: {result['predictions_csv']}",
            f"metrics: {result['metrics_csv']}",
        ]
        if result.get("report_html"):
            lines.append(f"report: {result['report_html']}")
        typer.echo("\n".join(lines))
        for row in result["metrics"]:
            typer.echo(json.dumps(row, ensure_ascii=False))
        return

    if not text and not file:
        raise typer.BadParameter("Provide --text, --file, or --json-file")
    if not src_lang or not tgt_lang:
        raise typer.BadParameter("--src-lang and --tgt-lang are required with --text or --file")
    sentences = [text] if text else []
    if file:
        sentences.extend(file.read_text(encoding="utf-8").splitlines())
    sentences = [sentence for sentence in sentences if sentence and sentence.strip()]
    if not sentences:
        raise typer.BadParameter("Provide --text or --file")
    src = normalize_lang(src_lang)
    tgt = normalize_lang(tgt_lang)
    model_name = model or maybe_adapter_base(adapter) or default_model(src, tgt)
    hypotheses = translate_across_ranks(
        sentences,
        src,
        tgt,
        model_name=model_name,
        adapter=str(adapter) if adapter else None,
        max_length=max_length,
        num_beams=num_beams,
        batch_size=batch_size,
    )
    release_inference_model()
    shutdown_inference_group()
    if int(os.environ.get("RANK", "0") or "0") != 0:
        return
    for src_sent, hyp in zip(sentences, hypotheses):
        typer.echo(f"{src}: {src_sent}")
        typer.echo(f"{tgt}: {hyp}")
        typer.echo("")


def collapse_eval_metrics(argv: list[str]) -> list[str]:
    """Accept `--eval-metrics chrf2pp bleu ter` as well as repeated flags."""
    collapsed: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token != "--eval-metrics" and not token.startswith("--eval-metrics="):
            collapsed.append(token)
            index += 1
            continue
        raw = token.split("=", 1)[1] if token.startswith("--eval-metrics=") else ""
        index += 1
        pieces = raw.replace(",", " ").split()
        while index < len(argv) and not argv[index].startswith("--"):
            pieces.extend(argv[index].replace(",", " ").split())
            index += 1
        for piece in pieces:
            collapsed.extend(["--eval-metrics", piece])
    return collapsed


@app.command("eval")
def eval_cmd(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False),
    src_lang: str = typer.Option(...),
    tgt_lang: str = typer.Option(...),
    split: str = typer.Option("dev"),
    model: Optional[str] = typer.Option(None),
    adapter: Optional[Path] = typer.Option(None),
    max_examples: Optional[int] = typer.Option(None),
    batch_size: int = typer.Option(8, "--batch-size"),
) -> None:
    """Compute BLEU and chrF on a prepared split."""
    from indictrans2_finetune.evaluate import evaluate_split

    result = evaluate_split(
        data_dir,
        split,
        src_lang,
        tgt_lang,
        model_name=model,
        adapter=str(adapter) if adapter else None,
        max_examples=max_examples,
        batch_size=batch_size,
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command("analyze")
def analyze_cmd(
    run_dir: Path = typer.Option(
        ...,
        "--run-dir",
        exists=True,
        file_okay=False,
        help="Training --output_dir with CSV logs.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        help="HTML path. Defaults to <run-dir>/rca_report.html.",
    ),
    worst_n: int = typer.Option(25, "--worst-n", help="Worst hypotheses to list."),
) -> None:
    """Build an offline RCA HTML report from training CSVs."""
    from indictrans2_finetune.analyze import analyze_run

    path = analyze_run(run_dir, output_html=output, worst_n=worst_n)
    typer.echo(str(path))


@app.command("ui")
def ui_cmd(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(43147),
) -> None:
    """Open the data-prep and translation playground."""
    from indictrans2_finetune.app import launch

    launch(host=host, port=port)


def main() -> None:
    sys.argv = [sys.argv[0], *collapse_eval_metrics(sys.argv[1:])]
    app()


if __name__ == "__main__":
    main()
