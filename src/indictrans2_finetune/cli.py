"""Command-line interface for IndicTrans2 LoRA fine-tuning."""

from __future__ import annotations

import json
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
    src_lang: str = typer.Option(...),
    tgt_lang: str = typer.Option(...),
    model: Optional[str] = typer.Option(None),
    adapter: Optional[Path] = typer.Option(None, help="Directory with saved LoRA weights"),
) -> None:
    """Translate with the base model or a fine-tuned adapter."""
    from indictrans2_finetune.infer import maybe_adapter_base, translate_batch

    sentences = [text] if text else []
    if file:
        sentences.extend(file.read_text(encoding="utf-8").splitlines())
    if not sentences:
        raise typer.BadParameter("Provide --text or --file")
    src = normalize_lang(src_lang)
    tgt = normalize_lang(tgt_lang)
    model_name = model or maybe_adapter_base(adapter) or default_model(src, tgt)
    for src_sent, hyp in zip(
        sentences,
        translate_batch(sentences, src, tgt, model_name=model_name, adapter=str(adapter) if adapter else None),
    ):
        typer.echo(f"{src}: {src_sent}")
        typer.echo(f"{tgt}: {hyp}")
        typer.echo("")


@app.command("eval")
def eval_cmd(
    data_dir: Path = typer.Option(..., exists=True, file_okay=False),
    src_lang: str = typer.Option(...),
    tgt_lang: str = typer.Option(...),
    split: str = typer.Option("dev"),
    model: Optional[str] = typer.Option(None),
    adapter: Optional[Path] = typer.Option(None),
    max_examples: Optional[int] = typer.Option(None),
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
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command("ui")
def ui_cmd(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(43147),
) -> None:
    """Open the data-prep and translation playground."""
    from indictrans2_finetune.app import launch

    launch(host=host, port=port)


if __name__ == "__main__":
    app()
