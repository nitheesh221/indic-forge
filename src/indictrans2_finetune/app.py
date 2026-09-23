"""Local playground: prepare data, inspect language tags, try translations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr

from indictrans2_finetune.data import prepare_file
from indictrans2_finetune.langs import LANGUAGES, default_model, normalize_lang

SAMPLE_EN = "Your Aadhaar enrolment is complete. You can download the e-Aadhaar from the UIDAI portal."
LANG_CHOICES = [f"{code} — {meta['name']}" for code, meta in LANGUAGES.items()]


def _code(label: str) -> str:
    return label.split(" — ", 1)[0]


def format_langs() -> str:
    rows = ["| Tag | Language | Script |", "| --- | --- | --- |"]
    for code, meta in LANGUAGES.items():
        rows.append(f"| `{code}` | {meta['name']} | {meta['script']} |")
    return "\n".join(rows)


def convert_files(train_file, dev_file, src_label, tgt_label):
    if train_file is None or dev_file is None:
        return "Upload both a train and a dev file.", ""
    src = _code(src_label)
    tgt = _code(tgt_label)
    out = Path(tempfile.mkdtemp(prefix="indictrans2-"))
    try:
        train_report = prepare_file(Path(train_file.name), out, "train", src, tgt)
        dev_report = prepare_file(Path(dev_file.name), out, "dev", src, tgt)
    except Exception as exc:  # noqa: BLE001 - surface conversion errors in the UI
        return f"Could not convert files: {exc}", ""
    command = (
        "indictrans2-ft train "
        f"--data-dir {out} "
        f"--src-lang {src} --tgt-lang {tgt} "
        "--output-dir outputs/lora --max-steps 400 --batch-size 4"
    )
    summary = json.dumps(
        {
            "output_dir": str(out),
            "train": train_report.pairs["train"],
            "dev": dev_report.pairs["dev"],
            "skipped_train": train_report.skipped,
            "skipped_dev": dev_report.skipped,
            "base_model": default_model(src, tgt),
        },
        indent=2,
        ensure_ascii=False,
    )
    return summary, command


def run_translate(text, src_label, tgt_label, adapter_dir):
    from indictrans2_finetune.infer import maybe_adapter_base, translate_batch

    if not text or not text.strip():
        return "Enter a sentence to translate."
    src = normalize_lang(_code(src_label))
    tgt = normalize_lang(_code(tgt_label))
    adapter = adapter_dir.strip() or None
    model_name = default_model(src, tgt)
    if adapter:
        model_name = maybe_adapter_base(Path(adapter)) or model_name
    try:
        hyps = translate_batch([text], src, tgt, model_name=model_name, adapter=adapter)
    except Exception as exc:  # noqa: BLE001
        return (
            f"Translation failed: {exc}\n\n"
            "The first run downloads the Hugging Face checkpoint. "
            "A GPU is recommended; CPU works for the distilled 200M models but is slow."
        )
    return hyps[0] if hyps else ""


def recipe(src_label, tgt_label, size, steps):
    src = _code(src_label)
    tgt = _code(tgt_label)
    model = default_model(src, tgt, size)
    return f"""# 1. Convert your parallel data
indictrans2-ft prepare \\
  --train data/sample/train.jsonl \\
  --dev data/sample/dev.jsonl \\
  --src-lang {src} --tgt-lang {tgt} \\
  --output-dir data/prepared

# 2. Fine-tune LoRA on {model}
indictrans2-ft train \\
  --data-dir data/prepared \\
  --src-lang {src} --tgt-lang {tgt} \\
  --model {model} \\
  --output-dir outputs/lora \\
  --max-steps {int(steps)} \\
  --batch-size 4 --grad-accum-steps 4

# 3. Translate with the adapter
indictrans2-ft translate \\
  --text "{SAMPLE_EN}" \\
  --src-lang {src} --tgt-lang {tgt} \\
  --adapter outputs/lora
"""


def build_ui() -> gr.Blocks:
    en = next(label for label in LANG_CHOICES if label.startswith("eng_Latn"))
    hi = next(label for label in LANG_CHOICES if label.startswith("hin_Deva"))
    with gr.Blocks(title="IndicTrans2 Fine-tune") as demo:
        gr.Markdown(
            """
            # IndicTrans2 LoRA fine-tune
            Prepare parallel data in the official AI4Bharat layout, pick the right
            `en-indic` / `indic-en` / `indic-indic` checkpoint, and LoRA-adapt it
            to your domain (admin, legal, healthcare, product copy, …).
            """
        )
        with gr.Tab("Recipe"):
            src = gr.Dropdown(LANG_CHOICES, value=en, label="Source")
            tgt = gr.Dropdown(LANG_CHOICES, value=hi, label="Target")
            size = gr.Radio(["distilled", "base"], value="distilled", label="Checkpoint size")
            steps = gr.Slider(50, 5000, value=400, step=50, label="Max training steps")
            recipe_box = gr.Code(label="Commands", language="bash")
            for widget in (src, tgt, size, steps):
                widget.change(recipe, [src, tgt, size, steps], recipe_box)
            demo.load(recipe, [src, tgt, size, steps], recipe_box)
        with gr.Tab("Prepare data"):
            gr.Markdown(
                "Upload JSONL (`src`/`tgt` or FLORES keys) or CSV with a header. "
                "Language columns can be omitted if you set the dropdowns."
            )
            train_file = gr.File(label="Train file", file_types=[".jsonl", ".csv", ".tsv"])
            dev_file = gr.File(label="Dev file", file_types=[".jsonl", ".csv", ".tsv"])
            p_src = gr.Dropdown(LANG_CHOICES, value=en, label="Default source")
            p_tgt = gr.Dropdown(LANG_CHOICES, value=hi, label="Default target")
            convert = gr.Button("Convert to official layout", variant="primary")
            summary = gr.Code(label="Summary", language="json")
            command = gr.Code(label="Train command for this folder", language="bash")
            convert.click(convert_files, [train_file, dev_file, p_src, p_tgt], [summary, command])
        with gr.Tab("Translate"):
            t_src = gr.Dropdown(LANG_CHOICES, value=en, label="Source")
            t_tgt = gr.Dropdown(LANG_CHOICES, value=hi, label="Target")
            adapter = gr.Textbox(label="LoRA adapter directory (optional)", placeholder="outputs/lora")
            sentence = gr.Textbox(label="Source text", value=SAMPLE_EN, lines=3)
            go = gr.Button("Translate", variant="primary")
            output = gr.Textbox(label="Translation", lines=3)
            go.click(run_translate, [sentence, t_src, t_tgt, adapter], output)
        with gr.Tab("Languages"):
            gr.Markdown(format_langs())
    return demo


def launch(host: str = "0.0.0.0", port: int = 43147) -> None:
    build_ui().launch(server_name=host, server_port=port, show_error=True)


if __name__ == "__main__":
    launch()
