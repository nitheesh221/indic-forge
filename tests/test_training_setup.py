import os
from pathlib import Path

import pytest
import torch

from indictrans2_finetune.callbacks import TopNCheckpointCallback, retention_count
from indictrans2_finetune.runner import _prepare_gradient_checkpointing
from indictrans2_finetune.toolkit import IndicBatchCollator
from indictrans2_finetune.config import maybe_relaunch, parse_args
from indictrans2_finetune.metrics_mt import exact_match, normalize_exact, score_translations
from indictrans2_finetune.train import preprocess_logits_for_metrics, token_stats


BASE = "ai4bharat/indictrans2-en-indic-dist-200M"


def _args(mode="lora", **extra):
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
        "--train_mode",
        mode,
    ]
    for key, value in extra.items():
        argv.extend([f"--{key}", str(value)])
    return parse_args(argv)


def test_mode_defaults():
    lora = _args("lora")
    assert lora.learning_rate == 2e-4
    assert lora.optim == "adamw_torch"
    assert lora.lora_target_modules == "q_proj,k_proj"
    full = _args("full")
    assert full.learning_rate == 2e-5
    wide = _args("full_lora")
    assert "fc1" in wide.lora_target_modules and "out_proj" in wide.lora_target_modules
    qlora = _args("qlora")
    assert qlora.optim == "paged_adamw_32bit"
    assert qlora.keep_checkpoints == 1


def test_direction_must_match_checkpoint():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model_hf_id",
                "ai4bharat/indictrans2-indic-en-dist-200M",
                "--train_json",
                "train.jsonl",
                "--val_json",
                "dev.jsonl",
                "--src_lang",
                "eng_Latn",
                "--tgt_lang",
                "hin_Deva",
            ]
        )


def test_torchrun_rejects_gpu_ids(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(SystemExit, match="torchrun"):
        maybe_relaunch(["--gpu_ids", "0", "--model_hf_id", BASE])


def test_exact_match_normalizes_whitespace_only():
    assert normalize_exact("A  b\n") == "A b"
    assert exact_match(["A  b", "Hello"], ["A b", "hello"]) == 50.0


def test_overlap_metrics_without_downloads():
    sentence = ["Your Aadhaar enrolment is complete and the card can be downloaded."]
    scores, errors = score_translations(
        sentence,
        sentence,
        sentence,
        ["chrf2pp", "bleu", "ter", "exact_match"],
    )
    assert errors == {}
    assert scores["exact_match"] == 100.0
    assert scores["ter"] == 0.0
    assert scores["bleu"] > 90
    assert scores["chrf2pp"] > 90


def test_token_statistics():
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0]]])
    labels = torch.tensor([[0, -100]])
    stats = preprocess_logits_for_metrics(logits, labels)
    result = token_stats(type("Preds", (), {"predictions": stats.numpy()})())
    assert result["mean_token_accuracy"] == 1.0
    assert result["entropy"] < 0.1


def test_top_n_keeps_lower_loss(tmp_path: Path):
    first = tmp_path / "checkpoint-1"
    second = tmp_path / "checkpoint-2"
    first.mkdir()
    second.mkdir()
    callback = TopNCheckpointCallback("eval_loss", greater_is_better=False, keep=1)
    state = type(
        "State",
        (),
        {"is_world_process_zero": True, "global_step": 1, "log_history": [{"eval_loss": 0.4}]},
    )()
    args = type("Args", (), {"output_dir": str(tmp_path)})()
    callback.on_save(args, state, None)
    state.global_step = 2
    state.log_history.append({"eval_loss": 0.9})
    callback.on_save(args, state, None)
    assert first.exists()
    assert not second.exists()
    assert retention_count(3, 1) == 1


def test_batch_collator_pads_without_toolkit_helper():
    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0

        def pad(self, features, padding, pad_to_multiple_of, return_tensors):
            assert self.padding_side == "left"
            width = max(len(feature["input_ids"]) for feature in features)
            rows = [[0] * (width - len(feature["input_ids"])) + list(feature["input_ids"]) for feature in features]
            return {"input_ids": torch.tensor(rows)}

    batch = IndicBatchCollator(Tokenizer(), pad_to_multiple_of=None)(
        [
            {"input_ids": [1, 2], "labels": [3]},
            {"input_ids": [4], "labels": [5, 6]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [0, 4]]
    assert batch["labels"].tolist() == [[3, -100], [5, 6]]


def test_checkpointed_frozen_embedding_keeps_adapter_grad():
    class FrozenEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.proj = torch.nn.Linear(4, 4)
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        def get_input_embeddings(self):
            return self.embed_tokens

        def forward(self, input_ids):
            hidden = self.embed_tokens(input_ids)
            return torch.utils.checkpoint.checkpoint(self.proj, hidden, use_reentrant=False)

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = FrozenEncoder()
            self.lora = torch.nn.Linear(4, 4, bias=False)

        def get_base_model(self):
            return self.base

        def get_input_embeddings(self):
            return self.base.embed_tokens

        def forward(self, input_ids):
            hidden = self.base(input_ids)
            return (hidden + self.lora(hidden)).sum()

    model = Adapter()
    args = type("Args", (), {"gradient_checkpointing": True})()
    _prepare_gradient_checkpointing(model, args)
    loss = model(torch.tensor([[1, 2, 3]]))
    loss.backward()
    assert model.lora.weight.grad is not None
    assert model.lora.weight.grad.abs().sum() > 0


def test_multi_gpu_relaunch_is_not_taken_when_rank_is_absent(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    # A single GPU id must stay in-process and must not replace this test process.
    maybe_relaunch(["--gpu_ids", "0", "--model_hf_id", BASE])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
