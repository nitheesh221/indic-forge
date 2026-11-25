import json
from pathlib import Path

import pytest

from indictrans2_finetune.analyze import analyze_run, _token_edit_rate
from indictrans2_finetune.callbacks import CheckpointRetentionCallback, retention_count
from indictrans2_finetune.config import parse_args
from indictrans2_finetune.methods import recommend_distribution, validate_train_method

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


def test_sft_is_default_method():
    args = _args()
    assert args.train_method == "sft"
    assert args.label_smoothing_factor == 0.1
    assert args.top_n_checkpoints == 3
    assert args.last_k_checkpoints == 2
    assert args.distribution == "auto"
    assert args.keep_checkpoints == 5


def test_only_sft_is_accepted():
    with pytest.raises(SystemExit, match="SFT only"):
        validate_train_method("grpo")
    with pytest.raises(SystemExit):
        _args(train_method="grpo")


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


def test_zero_configs_are_selected():
    zero2 = _args(distribution="zero2", train_mode="full")
    assert zero2.distribution == "zero2"
    assert zero2.deepspeed.endswith("ds_zero2.json")
    assert Path(zero2.deepspeed).is_file()
    zero3 = _args(distribution="zero3", train_mode="full")
    assert zero3.deepspeed.endswith("ds_zero3.json")
    offload = _args(distribution="offload", train_mode="full")
    assert offload.deepspeed.endswith("ds_offload.json")
    assert Path(offload.deepspeed).is_file()
    legacy = _args(distribution="deepspeed", train_mode="full")
    assert legacy.distribution == "zero2"
    assert legacy.deepspeed.endswith("ds_zero2.json")


def test_qlora_rejects_shard_and_offload():
    for name in ("fsdp", "deepspeed", "zero2", "zero3", "offload"):
        with pytest.raises(SystemExit, match="QLoRA"):
            _args(train_mode="qlora", distribution=name)


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
