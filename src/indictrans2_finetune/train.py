"""LoRA fine-tuning for Hugging Face IndicTrans2 checkpoints."""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from indictrans2_finetune.data import validate_official_dir
from indictrans2_finetune.langs import default_model, normalize_lang

try:
    from IndicTransToolkit import IndicDataCollator, IndicProcessor
except ImportError:  # pragma: no cover - import path varies by toolkit version
    from IndicTransToolkit.IndicTransToolkit import IndicDataCollator, IndicProcessor


@dataclass
class TrainConfig:
    data_dir: Path
    output_dir: Path
    src_langs: list[str]
    tgt_langs: list[str]
    model: str | None = None
    model_size: str = "distilled"
    max_steps: int = 200
    num_train_epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    save_steps: int = 50
    eval_steps: int = 50
    max_length: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: str = "q_proj,k_proj"
    patience: int = 5
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    print_samples: bool = True
    num_workers: int = 0
    seed: int = 42
    report_to: str = "none"


def _processor() -> Any:
    return IndicProcessor(inference=False)


def load_and_process_translation_dataset(
    data_dir: Path,
    split: str,
    tokenizer,
    processor,
    src_lang_list: list[str],
    tgt_lang_list: list[str],
    max_length: int = 256,
    num_proc: int = 1,
    seed: int = 42,
) -> Dataset:
    src_lang_list = [normalize_lang(code) for code in src_lang_list]
    tgt_lang_list = [normalize_lang(code) for code in tgt_lang_list]
    validate_official_dir(data_dir, split, src_lang_list, tgt_lang_list)

    complete: dict[str, list[str]] = {"sentence_SRC": [], "sentence_TGT": []}
    for src_lang in src_lang_list:
        for tgt_lang in tgt_lang_list:
            if src_lang == tgt_lang:
                continue
            src_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{src_lang}"
            tgt_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{tgt_lang}"
            src_lines = src_path.read_text(encoding="utf-8").splitlines()
            tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()
            complete["sentence_SRC"] += processor.preprocess_batch(
                src_lines, src_lang=src_lang, tgt_lang=tgt_lang, is_target=False
            )
            complete["sentence_TGT"] += processor.preprocess_batch(
                tgt_lines, src_lang=tgt_lang, tgt_lang=src_lang, is_target=True
            )

    dataset = Dataset.from_dict(complete).shuffle(seed=seed)

    def preprocess_fn(example):
        model_inputs = tokenizer(
            example["sentence_SRC"], truncation=True, padding=False, max_length=max_length
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                example["sentence_TGT"], truncation=True, padding=False, max_length=max_length
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_fn, batched=True, num_proc=max(num_proc, 1))


def compute_metrics_factory(
    tokenizer, print_samples: bool = False, n_samples: int = 8
) -> Callable:
    bleu_metric = BLEU()
    chrf_metric = CHRF()

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels[labels == -100] = tokenizer.pad_token_id
        preds[preds == -100] = tokenizer.pad_token_id
        with tokenizer.as_target_tokenizer():
            pred_text = [
                text.strip()
                for text in tokenizer.batch_decode(
                    preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            ]
            label_text = [
                text.strip()
                for text in tokenizer.batch_decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            ]
        if print_samples and pred_text:
            sample = pd.DataFrame({"Predictions": pred_text, "References": label_text}).sample(
                n=min(n_samples, len(pred_text))
            )
            for pred, label in zip(sample["Predictions"], sample["References"]):
                print(f" | > Prediction: {pred}")
                print(f" | > Reference:  {label}\n")
        return {
            "BLEU": bleu_metric.corpus_score(pred_text, [label_text]).score,
            "chrF": chrf_metric.corpus_score(pred_text, [label_text]).score,
        }

    return compute_metrics


def _training_args(config: TrainConfig, use_cuda: bool) -> Seq2SeqTrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "do_train": True,
        "do_eval": True,
        "fp16": use_cuda,
        "bf16": False,
        "logging_strategy": "steps",
        "save_strategy": "steps",
        "logging_steps": max(10, config.eval_steps // 5),
        "save_total_limit": 2,
        "predict_with_generate": True,
        "load_best_model_at_end": True,
        "max_steps": config.max_steps,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": max(1, config.batch_size),
        "gradient_accumulation_steps": config.grad_accum_steps,
        "eval_accumulation_steps": config.grad_accum_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "num_train_epochs": config.num_train_epochs,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "dataloader_num_workers": config.num_workers,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "report_to": config.report_to,
        "generation_max_length": config.max_length,
        "generation_num_beams": 5,
        "group_by_length": True,
        "remove_unused_columns": False,
        "seed": config.seed,
        "lr_scheduler_type": "inverse_sqrt",
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
    }
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    if config.num_workers > 0:
        kwargs["dataloader_prefetch_factor"] = 2
    return Seq2SeqTrainingArguments(**kwargs)


def resolve_model_name(config: TrainConfig) -> str:
    if config.model:
        return config.model
    return default_model(config.src_langs[0], config.tgt_langs[0], config.model_size)


def train(config: TrainConfig) -> Path:
    src_langs = [normalize_lang(code) for code in config.src_langs]
    tgt_langs = [normalize_lang(code) for code in config.tgt_langs]
    model_name = resolve_model_name(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()

    print(f" | > Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    processor = _processor()
    collator = IndicDataCollator(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        pad_to_multiple_of=8 if use_cuda else None,
        label_pad_token_id=-100,
    )

    train_dataset = load_and_process_translation_dataset(
        config.data_dir,
        "train",
        tokenizer,
        processor,
        src_langs,
        tgt_langs,
        max_length=config.max_length,
        num_proc=1,
        seed=config.seed,
    )
    eval_dataset = load_and_process_translation_dataset(
        config.data_dir,
        "dev",
        tokenizer,
        processor,
        src_langs,
        tgt_langs,
        max_length=config.max_length,
        num_proc=1,
        seed=config.seed,
    )
    print(f" | > Train size: {len(train_dataset)}  Dev size: {len(eval_dataset)}")

    lora = LoraConfig(
        r=config.lora_r,
        bias="none",
        inference_mode=False,
        task_type="SEQ_2_SEQ_LM",
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[item.strip() for item in config.lora_target_modules.split(",")],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": _training_args(config, use_cuda),
        "data_collator": collator,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "compute_metrics": compute_metrics_factory(tokenizer, print_samples=config.print_samples),
        "callbacks": [
            EarlyStoppingCallback(early_stopping_patience=config.patience, early_stopping_threshold=1e-3)
        ],
    }
    trainer_init = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in trainer_init.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)
    print(" | > Starting training")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print(" | > Training interrupted; saving adapter")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "finetune_meta.json").write_text(
        json.dumps(
            {"base_model": model_name, "src_langs": src_langs, "tgt_langs": tgt_langs},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f" | > Saved LoRA adapter to {output_dir}")
    return output_dir
