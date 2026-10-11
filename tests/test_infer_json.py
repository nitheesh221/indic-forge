import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from indictrans2_finetune.cli import app, collapse_eval_metrics
from indictrans2_finetune.evaluate import evaluate_json, resolve_metrics


def test_eval_metrics_accept_spaces_and_commas():
    argv = collapse_eval_metrics(
        ["translate", "--eval-metrics", "chrf2pp", "bleu,ter", "--max-length", "256"]
    )
    assert argv == [
        "translate",
        "--eval-metrics",
        "chrf2pp",
        "--eval-metrics",
        "bleu",
        "--eval-metrics",
        "ter",
        "--max-length",
        "256",
    ]
    assert resolve_metrics(["chrf2pp", "bleu"]) == ["chrf2pp", "bleu"]


def test_translate_batch_splits_sentences_and_keeps_order(monkeypatch):
    import torch

    from indictrans2_finetune.infer import translate_batch

    seen: list[list[str]] = []

    class Tokenizer:
        def __call__(self, batch, **_kwargs):
            rows = list(batch)
            seen.append(rows)

            class Batch:
                def to(self, _device):
                    return {"input_ids": torch.zeros(len(rows), 1, dtype=torch.long)}

            return Batch()

        def batch_decode(self, generated, **_kwargs):
            return [f"hyp-{index}" for index in range(generated.shape[0])]

    class Model:
        def parameters(self):
            yield torch.zeros(1)

        def generate(self, **kwargs):
            return kwargs["input_ids"]

    class Processor:
        def preprocess_batch(self, chunk, **_kwargs):
            return list(chunk)

        def postprocess_batch(self, decoded, **_kwargs):
            return [f"{item}-out" for item in decoded]

    monkeypatch.setattr(
        "indictrans2_finetune.infer.load_pipeline",
        lambda *_args, **_kwargs: (Tokenizer(), Model(), Processor()),
    )
    hypotheses = translate_batch(
        ["a", "b", "c", "d", "e"],
        "asm_Beng",
        "eng_Latn",
        model_name="base",
        batch_size=2,
    )
    assert seen == [["a", "b"], ["c", "d"], ["e"]]
    assert hypotheses == ["hyp-0-out", "hyp-1-out", "hyp-0-out", "hyp-1-out", "hyp-0-out"]


def test_unknown_metric_is_rejected():
    with pytest.raises(SystemExit, match="Unknown metric"):
        resolve_metrics(["bleu", "wer"])


def test_json_inference_writes_prediction_and_metric_csvs(tmp_path: Path, monkeypatch):
    source = "নৱাগত ৰিয়া"
    reference = "Newcomer Rhea"
    payload = [
        {
            "src_lang": "asm_Beng",
            "tgt_lang": "eng_Latn",
            "src_text": source,
            "tgt_text": reference,
        },
        {
            "src_lang": "hin_Deva",
            "tgt_lang": "eng_Latn",
            "src_text": "ignore",
            "tgt_text": "ignore",
        },
    ]
    json_path = tmp_path / "valid.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    def fake_translate(sentences, src_lang, tgt_lang, **_kwargs):
        assert sentences == [source]
        assert src_lang == "asm_Beng"
        assert tgt_lang == "eng_Latn"
        return ["Newcomer Rhea"]

    monkeypatch.setattr("indictrans2_finetune.evaluate.translate_batch", fake_translate)
    result = evaluate_json(
        json_path,
        src_lang="asm_Beng",
        tgt_lang="eng_Latn",
        model_name="base",
        adapter="it2_out/lora/final_model",
        metrics=["exact_match", "chrf2pp"],
        predictions_csv=tmp_path / "predictions.csv",
        metrics_csv=tmp_path / "metrics.csv",
    )

    assert result["n"] == 1
    assert result["skipped"] == 1
    with (tmp_path / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "src_lang": "asm_Beng",
            "tgt_lang": "eng_Latn",
            "src": source,
            "reference": reference,
            "prediction": "Newcomer Rhea",
        }
    ]
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    assert metrics[0]["exact_match"] == "100.0"
    assert float(metrics[0]["chrf2pp"]) > 90
    assert metrics[0]["metric_errors"] == ""
    assert metrics[0]["n"] == "1"


def test_translate_cli_accepts_spaced_metrics(tmp_path: Path, monkeypatch):
    json_path = tmp_path / "valid.json"
    json_path.write_text("[]", encoding="utf-8")
    captured: dict = {}

    def fake_evaluate_json(*_args, **kwargs):
        captured.update(kwargs)
        return {"n": 0, "skipped": 0, "metrics": [], "predictions_csv": "p.csv", "metrics_csv": "m.csv"}

    monkeypatch.setattr("indictrans2_finetune.evaluate.evaluate_json", fake_evaluate_json)
    argv = collapse_eval_metrics(
        [
            "translate",
            "--json-file",
            str(json_path),
            "--src-lang",
            "asm_Beng",
            "--tgt-lang",
            "eng_Latn",
            "--model",
            "base",
            "--adapter",
            "it2_out/lora/final_model",
            "--eval-metrics",
            "chrf2pp",
            "bleu",
            "exact_match",
            "--predictions-csv",
            "predictions.csv",
            "--metrics-csv",
            "metrics.csv",
            "--max-length",
            "128",
            "--num-beams",
            "4",
        ]
    )
    result = CliRunner().invoke(app, argv)
    assert result.exit_code == 0, result.output
    assert captured["metrics"] == ["chrf2pp", "bleu", "exact_match"]
    assert captured["max_length"] == 128
    assert captured["num_beams"] == 4
    assert captured["adapter"] == "it2_out/lora/final_model"
