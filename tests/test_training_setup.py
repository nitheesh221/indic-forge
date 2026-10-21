import json
import os
from pathlib import Path

import pytest
import torch

from indictrans2_finetune.callbacks import (
    TopNCheckpointCallback,
    generate_translations,
    retention_count,
)
from indictrans2_finetune.config import maybe_relaunch, parse_args, pin_visible_gpus
from indictrans2_finetune.infer import (
    overlay_checkpoint,
    sanitize_saved_tokenizer,
    tokenizer_source,
)
from indictrans2_finetune.metrics_mt import exact_match, normalize_exact, score_translations
from indictrans2_finetune.runner import _prepare_gradient_checkpointing
from indictrans2_finetune.toolkit import IndicBatchCollator
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


def test_one_gpu_id_disables_data_parallel():
    args = _args("qlora", gpu_ids=0)
    assert args.visible_gpu_count == 1
    from indictrans2_finetune.runner import _training_arguments

    training = _training_arguments(args)
    assert training._n_gpu == 1


def test_pin_visible_gpus_sets_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert pin_visible_gpus(["--gpu_ids", "0", "--train_mode", "qlora"]) == ["0"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_language_flags_are_optional():
    args = parse_args(
        [
            "--model_hf_id",
            "ai4bharat/indictrans2-indic-en-dist-200M",
            "--train_json",
            "train.json",
            "--val_json",
            "valid.json",
        ]
    )
    assert args.src_lang is None
    assert args.tgt_lang is None


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


def test_generate_translations_keeps_order_across_languages():
    class _Processor:
        def preprocess_batch(self, sentences, src_lang, tgt_lang):
            return [f"{src_lang}|{sentence}" for sentence in sentences]

        def postprocess_batch(self, decoded, lang):
            return [f"{lang}:{text}" for text in decoded]

    class _Parameter:
        device = "cpu"

    class _Batch(dict):
        def to(self, _device):
            return self

    class _Tokenizer:
        def __call__(self, batch, **_kwargs):
            return _Batch(input_ids=list(batch))

        def batch_decode(self, generated, **_kwargs):
            return list(generated)

    class _Model:
        def parameters(self):
            yield _Parameter()

        def eval(self):
            return None

        def generate(self, **inputs):
            return inputs["input_ids"]

    hypotheses = generate_translations(
        _Model(),
        _Tokenizer(),
        _Processor(),
        ["kannada", "bengali", "kannada-2"],
        "kan_Knda",
        "eng_Latn",
        max_length=32,
        num_beams=1,
        batch_size=8,
        src_langs=["kan_Knda", "ben_Beng", "kan_Knda"],
        tgt_langs=["eng_Latn", "eng_Latn", "eng_Latn"],
    )
    assert hypotheses == [
        "eng_Latn:kan_Knda|kannada",
        "eng_Latn:ben_Beng|bengali",
        "eng_Latn:kan_Knda|kannada-2",
    ]


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


def test_merged_model_uses_hub_tokenizer(tmp_path: Path):
    model_dir = tmp_path / "final_model"
    model_dir.mkdir()
    (model_dir / "finetune_meta.json").write_text(
        '{"model_hf_id": "ai4bharat/indictrans2-indic-en-dist-200M"}\n',
        encoding="utf-8",
    )
    assert tokenizer_source(str(model_dir)) == "ai4bharat/indictrans2-indic-en-dist-200M"
    assert tokenizer_source("ai4bharat/indictrans2-indic-en-dist-200M") == (
        "ai4bharat/indictrans2-indic-en-dist-200M"
    )


def test_saved_tokenizer_config_drops_duplicate_vocab_keys(tmp_path: Path):
    config = {
        "src_vocab_file": "dict.SRC.json",
        "tgt_vocab_file": "dict.TGT.json",
        "src_spm": "model.SRC",
        "bos_token": "<s>",
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    sanitize_saved_tokenizer(tmp_path)
    saved = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert "src_vocab_file" not in saved
    assert saved["bos_token"] == "<s>"


def test_merged_checkpoint_keeps_base_buffers_and_ties_output(tmp_path: Path):
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(3, 2)
            self.register_buffer("positions", torch.tensor([7.0, 8.0]), persistent=False)

    class Seq2Seq(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()
            self.lm_head = torch.nn.Linear(2, 3, bias=False)

        def get_decoder(self):
            return self.decoder

    trained = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    torch.save({"decoder.embed_tokens.weight": trained}, tmp_path / "pytorch_model.bin")
    model = Seq2Seq()
    report = overlay_checkpoint(model, tmp_path)
    assert torch.equal(model.decoder.embed_tokens.weight, trained)
    assert torch.equal(model.lm_head.weight, trained)
    assert torch.equal(model.decoder.positions, torch.tensor([7.0, 8.0]))
    assert "lm_head.weight" in report["missing"]


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
