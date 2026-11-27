import json
from pathlib import Path

import pytest

from indictrans2_finetune.analyze import analyze_run, _token_edit_rate
from indictrans2_finetune.callbacks import (
    CheckpointRetentionCallback,
    EarlyStoppingCallback,
    advance_early_stopping,
    retention_count,
)
from indictrans2_finetune.config import parse_args
from indictrans2_finetune.methods import recommend_distribution

BASE = "ai4bharat/indictrans2-en-indic-dist-200M"


def _args(**extra):
    argv = [
        "--model_hf_id",
        BASE,
        "--train_json",
        "train.jsonl",
        "--val_json",
        "dev.jsonl",
        "--src_lang",
        "en",
        "--tgt_lang",
        "hi",
    ]
    for key, value in extra.items():
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
            continue
        argv.extend([f"--{key}", str(value)])
    return parse_args(argv)


def test_sft_defaults_have_no_method_flag():
    args = _args()
    assert not hasattr(args, "train_method")
    assert args.label_smoothing_factor == 0.1
    assert args.early_stopping_patience == 3
    assert args.early_stopping_threshold == 0.0
    assert args.top_n_checkpoints == 3
    assert args.last_k_checkpoints == 2
    assert args.distribution == "auto"
    assert args.keep_checkpoints == 5


def test_early_stopping_rejects_a_run_that_cannot_reload_the_best():
    with pytest.raises(SystemExit, match="0 or higher"):
        _args(early_stopping_patience=-1)
    with pytest.raises(SystemExit, match="evaluation or saving is off"):
        _args(early_stopping_patience=2, eval_strategy="no")
    with pytest.raises(SystemExit, match="to match"):
        _args(early_stopping_patience=2, eval_strategy="steps", save_strategy="epoch")
    off = _args(early_stopping_patience=0, eval_strategy="steps", save_strategy="epoch")
    assert off.early_stopping_patience == 0


def test_early_stopping_counts_misses_and_respects_direction():
    best, counter, stop, _message = advance_early_stopping(
        None, 0, 1.5, greater_is_better=False, patience=2, threshold=0.0, metric="eval_loss", step=10,
    )
    assert (best, counter, stop) == (1.5, 0, False)
    best, counter, stop, _message = advance_early_stopping(
        best, counter, 1.4, greater_is_better=False, patience=2, threshold=0.0, metric="eval_loss", step=20,
    )
    assert (best, counter, stop) == (1.4, 0, False)
    best, counter, stop, _message = advance_early_stopping(
        best, counter, 1.45, greater_is_better=False, patience=2, threshold=0.0, metric="eval_loss", step=30,
    )
    assert (best, counter, stop) == (1.4, 1, False)
    best, counter, stop, message = advance_early_stopping(
        best, counter, 1.5, greater_is_better=False, patience=2, threshold=0.0, metric="eval_loss", step=40,
    )
    assert (best, counter, stop) == (1.4, 2, True)
    assert "step 40" in message

    # A change smaller than the threshold is not an improvement.
    best, counter, stop, _message = advance_early_stopping(
        1.0, 1, 0.995, greater_is_better=False, patience=3, threshold=0.01, metric="eval_loss", step=5,
    )
    assert (best, counter, stop) == (1.0, 2, False)
    best, counter, stop, _message = advance_early_stopping(
        0.40, 2, 0.42, greater_is_better=True, patience=3, threshold=0.0, metric="eval_mean_token_accuracy", step=6,
    )
    assert (best, counter, stop) == (0.42, 0, False)


class _StopState:
    def __init__(self, step: int, best_metric):
        self.global_step = step
        self.best_metric = best_metric


class _StopControl:
    def __init__(self):
        self.should_training_stop = False


def test_early_stopping_callback_stops_and_keeps_a_restored_best():
    callback = EarlyStoppingCallback("eval_loss", greater_is_better=False, patience=1, threshold=0.0)
    control = _StopControl()
    callback.on_evaluate(None, _StopState(4, best_metric=1.0), control, metrics={"eval_loss": 1.2})
    assert control.should_training_stop
    assert callback.best == 1.0
    assert callback.counter == 1

    fresh = EarlyStoppingCallback("eval_loss", greater_is_better=False, patience=2, threshold=0.0)
    control = _StopControl()
    fresh.on_evaluate(None, _StopState(1, best_metric=None), control, metrics={})
    assert not control.should_training_stop
    assert fresh.counter == 0


def test_recommend_distribution():
    assert recommend_distribution("lora", 1) == "auto"
    assert recommend_distribution("qlora", 2) == "ddp"
    assert recommend_distribution("full", 2) == "ddp"
    assert recommend_distribution("full_lora", 4) == "ddp"


def test_write_rca_default_on():
    args = _args()
    assert args.write_rca is True
    off = parse_args(
        [
            "--model_hf_id",
            BASE,
            "--train_json",
            "train.jsonl",
            "--val_json",
            "dev.jsonl",
            "--src_lang",
            "en",
            "--tgt_lang",
            "hi",
            "--no-write_rca",
        ]
    )
    assert off.write_rca is False


def test_retention_union_without_hard_cap():
    assert retention_count(3, 2, None) == 5
    assert retention_count(3, 2, 0) == 5
    assert retention_count(3, 2, 4) == 4


def test_top_k_and_last_k_union(tmp_path: Path):
    for step in (1, 2, 3, 4):
        (tmp_path / f"checkpoint-{step}").mkdir()
    callback = CheckpointRetentionCallback(
        "eval_loss",
        greater_is_better=False,
        top_k=2,
        last_k=2,
    )
    history = []
    args = type("Args", (), {"output_dir": str(tmp_path)})()
    # Better scores first (low loss), then worse late steps that still win Last-K.
    for step, loss in ((1, 0.40), (2, 0.30), (3, 0.90), (4, 0.95)):
        history.append({"eval_loss": loss})
        state = type(
            "State",
            (),
            {
                "is_world_process_zero": True,
                "global_step": step,
                "log_history": list(history),
            },
        )()
        callback.on_save(args, state, None)

    assert (tmp_path / "checkpoint-1").exists()  # top-k (0.40)
    assert (tmp_path / "checkpoint-2").exists()  # top-k (0.30)
    assert (tmp_path / "checkpoint-3").exists()  # last-k
    assert (tmp_path / "checkpoint-4").exists()  # last-k
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["top_k"] == 2
    assert manifest["last_k"] == 2
    assert len(manifest["checkpoints"]) == 4


def test_hard_cap_prefers_top_then_recent(tmp_path: Path):
    for step in (1, 2, 3, 4):
        (tmp_path / f"checkpoint-{step}").mkdir()
    callback = CheckpointRetentionCallback(
        "eval_loss",
        greater_is_better=False,
        top_k=2,
        last_k=2,
        save_total_limit=3,
    )
    history = []
    args = type("Args", (), {"output_dir": str(tmp_path)})()
    for step, loss in ((1, 0.40), (2, 0.30), (3, 0.90), (4, 0.95)):
        history.append({"eval_loss": loss})
        state = type(
            "State",
            (),
            {
                "is_world_process_zero": True,
                "global_step": step,
                "log_history": list(history),
            },
        )()
        callback.on_save(args, state, None)
    kept = {path.name for path in tmp_path.iterdir() if path.is_dir()}
    assert "checkpoint-2" in kept  # best
    assert "checkpoint-1" in kept  # second best
    assert "checkpoint-4" in kept  # most recent fill
    assert "checkpoint-3" not in kept


def test_removed_distributions_are_rejected():
    for name in ("zero2", "zero3", "offload", "fsdp", "deepspeed"):
        with pytest.raises(SystemExit):
            _args(distribution=name)


def test_analyze_builds_html(tmp_path: Path):
    (tmp_path / "training_logs.csv").write_text(
        "step,loss,eval_loss,learning_rate,eval_entropy,eval_mean_token_accuracy\n"
        "1,1.2,1.1,0.0002,2.0,0.4\n"
        "2,0.9,1.3,0.0001,1.8,0.5\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_metrics.csv").write_text(
        "step,chrf2pp,bleu,exact_match\n1,45.0,20.0,10.0\n2,40.0,18.0,8.0\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_samples.csv").write_text(
        "step,src_lang,tgt_lang,src,ref,hyp\n"
        "2,eng_Latn,hin_Deva,hello,नमस्ते,नमस्ते\n"
        "2,eng_Latn,hin_Deva,card,कार्ड,गलत\n",
        encoding="utf-8",
    )
    (tmp_path / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "metric": "eval_loss",
                "greater_is_better": False,
                "top_k": 1,
                "last_k": 1,
                "checkpoints": [
                    {
                        "path": str(tmp_path / "checkpoint-1"),
                        "step": 1,
                        "score": 1.1,
                        "in_top_k": True,
                        "in_last_k": False,
                    },
                    {
                        "path": str(tmp_path / "checkpoint-2"),
                        "step": 2,
                        "score": 1.3,
                        "in_top_k": False,
                        "in_last_k": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    report = analyze_run(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "Possible overfitting" in text
    assert "step 1" in text
    assert "chrF++ regressed" in text
    assert "eng_Latn→hin_Deva" in text
    assert "<svg" in text
    assert "chart.js" not in text.lower()
    assert "cdn.jsdelivr.net" not in text
    assert "chrF++" in text


def test_analyze_names_the_peak_when_the_end_is_not_the_start(tmp_path: Path):
    (tmp_path / "training_logs.csv").write_text(
        "step,loss,eval_loss\n1,2.0,2.0\n2,1.0,0.5\n3,0.8,0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_metrics.csv").write_text(
        "step,chrf2pp\n1,10.0\n2,50.0\n3,40.0\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_samples.csv").write_text(
        "step,src_lang,tgt_lang,src,ref,hyp\n"
        "2,eng_Latn,hin_Deva,card,कार्ड,कार्ड\n"
        "3,eng_Latn,hin_Deva,card,कार्ड,गलत\n",
        encoding="utf-8",
    )
    text = analyze_run(tmp_path).read_text(encoding="utf-8")
    assert "step 2" in text
    assert "0.5000" in text
    assert "Best chrF++" in text or "best chrF++" in text.lower() or "chrF++ is highest at step 2" in text


def test_token_edit_rate_identity():
    assert _token_edit_rate("a b c", "a b c") == 0.0
    assert _token_edit_rate("a b c", "a x c") > 0
