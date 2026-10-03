"""Dataset encoding and teacher-forced token statistics for IndicTrans2."""

from __future__ import annotations

import torch
from datasets import Dataset

from indictrans2_finetune.data import validate_official_dir
from indictrans2_finetune.langs import normalize_lang


def load_and_process_translation_dataset(
    data_dir,
    split: str,
    tokenizer,
    processor,
    src_lang_list: list[str],
    tgt_lang_list: list[str],
    max_length: int = 256,
    num_proc: int = 1,
    seed: int = 42,
) -> tuple[Dataset, dict[str, list[str]]]:
    src_lang_list = [normalize_lang(code) for code in src_lang_list]
    tgt_lang_list = [normalize_lang(code) for code in tgt_lang_list]
    validate_official_dir(data_dir, split, src_lang_list, tgt_lang_list)

    complete = {"sentence_SRC": [], "sentence_TGT": [], "raw_src": [], "raw_tgt": []}
    for src_lang in src_lang_list:
        for tgt_lang in tgt_lang_list:
            if src_lang == tgt_lang:
                continue
            src_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{src_lang}"
            tgt_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{tgt_lang}"
            src_lines = [line.strip() for line in src_path.read_text(encoding="utf-8").splitlines()]
            tgt_lines = [line.strip() for line in tgt_path.read_text(encoding="utf-8").splitlines()]
            complete["raw_src"] += src_lines
            complete["raw_tgt"] += tgt_lines
            complete["sentence_SRC"] += processor.preprocess_batch(
                src_lines, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False
            )
            complete["sentence_TGT"] += processor.preprocess_batch(
                tgt_lines, src_lang=tgt_lang, tgt_lang=src_lang, is_target=True
            )

    dataset = Dataset.from_dict(complete).shuffle(seed=seed)
    raw = {"src": list(dataset["raw_src"]), "tgt": list(dataset["raw_tgt"])}

    def preprocess_fn(example):
        model_inputs = tokenizer(
            example["sentence_SRC"], truncation=True, padding=False, max_length=max_length
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["sentence_TGT"], truncation=True, padding=False, max_length=max_length
            )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["raw_src"] = example["raw_src"]
        model_inputs["raw_tgt"] = example["raw_tgt"]
        return model_inputs

    tokenized = dataset.map(preprocess_fn, batched=True, num_proc=max(num_proc, 1))
    return tokenized, raw


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    if isinstance(labels, tuple):
        labels = labels[0]
    labels = labels.to(logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    predictions = logits.argmax(dim=-1)
    mask = labels.ne(-100)
    token_count = mask.sum(dim=-1).clamp(min=1)
    mean_entropy = (entropy * mask).sum(dim=-1) / token_count
    accuracy = ((predictions.eq(labels) & mask).sum(dim=-1)) / token_count
    return torch.stack([mean_entropy, accuracy], dim=-1).detach().cpu()


def token_stats(eval_preds):
    stats = eval_preds.predictions
    if getattr(stats, "ndim", 0) != 2 or stats.shape[-1] != 2:
        return {}
    return {
        "entropy": float(stats[:, 0].mean()),
        "mean_token_accuracy": float(stats[:, 1].mean()),
    }
