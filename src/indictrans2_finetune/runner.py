"""IndicTrans2 training entry point shared by train.py and the CLI."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from indictrans2_finetune.callbacks import (
    CheckpointRetentionCallback,
    CsvLogCallback,
    TranslationMetricsCallback,
    barrier,
    resolve_log_path,
)
from indictrans2_finetune.config import DEEPSPEED_DISTRIBUTIONS, maybe_relaunch, parse_args
from indictrans2_finetune.data import list_pair_dirs, prepare_file, select_training_pairs
from indictrans2_finetune.infer import sanitize_saved_tokenizer
from indictrans2_finetune.toolkit import get_collator, get_processor
from indictrans2_finetune.train import (
    load_and_process_translation_dataset,
    preprocess_logits_for_metrics,
    token_stats,
)


def main(argv: list[str] | None = None) -> int:
    maybe_relaunch(argv)
    args = parse_args(argv)
    if not torch.cuda.is_available() and os.environ.get("IT2_ALLOW_CPU") != "1":
        raise SystemExit("CUDA is required for training. Set IT2_ALLOW_CPU=1 only for a dry run.")
    run(args)
    return 0


def run(args) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Building the training arguments pins this rank to its own GPU and opens
    # the process group with --ddp_timeout. That has to happen before from_pretrained,
    # or every rank loads the model onto GPU 0.
    training_args = _training_arguments(args)
    main = training_args.process_index == 0
    data_dir = _prepare_data(args, output_dir, write=main)
    tokenizer = AutoTokenizer.from_pretrained(args.model_hf_id, trust_remote_code=True)
    model = _load_model(args)
    _prepare_gradient_checkpointing(model, args)
    processor = get_processor(inference=False)
    train_pairs = [(src, tgt) for src, tgt, _count in args.train_pairs]
    train_dataset, _train_raw = load_and_process_translation_dataset(
        data_dir, "train", tokenizer, processor, train_pairs,
        max_length=args.max_length, seed=args.seed,
    )
    eval_dataset, eval_raw = load_and_process_translation_dataset(
        data_dir, "dev", tokenizer, processor, train_pairs,
        max_length=args.max_length, seed=args.seed,
    )
    collator = _FilteringCollator(
        get_collator(
            tokenizer,
            model,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
            label_pad_token_id=-100,
        )
    )
    trainer_box: dict = {"tokenizer": tokenizer}
    limit = args.save_total_limit if args.save_total_limit and args.save_total_limit > 0 else None
    callbacks = [
        CsvLogCallback(resolve_log_path(output_dir, args.training_logs_csv)),
        CheckpointRetentionCallback(
            args.metric_for_best_model,
            args.greater_is_better,
            top_k=args.top_n_checkpoints,
            last_k=args.last_k_checkpoints,
            save_total_limit=limit,
        ),
    ]
    if args.eval_strategy != "no":
        callbacks.append(
            TranslationMetricsCallback(
                trainer_box,
                eval_raw["src"],
                eval_raw["tgt"],
                args,
                src_langs=eval_raw["src_lang"],
                tgt_langs=eval_raw["tgt_lang"],
            )
        )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset if args.eval_strategy != "no" else None,
        "data_collator": collator,
        "compute_metrics": token_stats,
        "callbacks": callbacks,
        "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
    }
    if _accepts_processing_class():
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    trainer_box["model"] = trainer.model
    print(
        f" | > method={args.train_method} mode={args.train_mode} "
        f"distribution={args.distribution} model={args.model_id} "
        f"train={len(train_dataset)} dev={len(eval_dataset)} "
        f"checkpoints=top{args.top_n_checkpoints}+last{args.last_k_checkpoints}"
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = _save_final(trainer, tokenizer, args)
    if trainer.is_world_process_zero():
        state = trainer.state
        meta = {
            "train_method": args.train_method,
            "train_mode": args.train_mode,
            "distribution": args.distribution,
            "model_hf_id": args.model_hf_id,
            "model_id": args.model_id,
            "src_lang": args.trained_src_lang,
            "tgt_lang": args.trained_tgt_lang,
            "pairs": [f"{src}-{tgt}" for src, tgt, _count in args.train_pairs],
            "learning_rate": args.learning_rate,
            "label_smoothing_factor": args.label_smoothing_factor,
            "lora_target_modules": args.lora_target_modules,
            "top_n_checkpoints": args.top_n_checkpoints,
            "last_k_checkpoints": args.last_k_checkpoints,
            "metric_for_best_model": args.metric_for_best_model,
            "eval_metrics": args.eval_metrics,
            "best_model_checkpoint": state.best_model_checkpoint,
            "best_metric": _jsonable(state.best_metric),
            "epochs_completed": _jsonable(state.epoch),
        }
        (final_dir / "finetune_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(f" | > Saved final model to {final_dir}")
        if getattr(args, "write_rca", True):
            try:
                from indictrans2_finetune.analyze import analyze_run

                report = analyze_run(output_dir)
                print(f" | > RCA report: {report}")
            except Exception as exc:  # noqa: BLE001 — RCA must not fail the train run
                print(f" | > RCA report skipped: {type(exc).__name__}: {exc}")
    # Rank 0 may still be writing the tokenizer and the report. Every rank waits.
    barrier()
    return final_dir


class _FilteringCollator:
    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        slim = []
        for feature in features:
            slim.append(
                {
                    key: feature[key]
                    for key in ("input_ids", "attention_mask", "labels")
                    if key in feature
                }
            )
        return self.base(slim)


def _prepare_data(args, output_dir: Path, *, write: bool) -> Path:
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = output_dir / "prepared"
        if write:
            # One writer. Other ranks wait on the barrier below, then read.
            train_report = prepare_file(Path(args.train_json), data_dir, "train", args.src_lang, args.tgt_lang)
            dev_report = prepare_file(Path(args.val_json), data_dir, "dev", args.src_lang, args.tgt_lang)
            if train_report.skipped or dev_report.skipped:
                print(
                    " | > rows skipped while reading JSON: "
                    f"train={train_report.skipped} dev={dev_report.skipped}"
                )
    barrier()
    pairs, skipped = select_training_pairs(
        data_dir, args.model_hf_id, args.src_lang, args.tgt_lang, split="train"
    )
    missing = [
        f"{src}-{tgt}"
        for src, tgt, _count in pairs
        if (src, tgt) not in {(item[0], item[1]) for item in list_pair_dirs(data_dir, "dev")}
    ]
    if missing:
        raise SystemExit(
            "Validation is missing pairs that training selected: " + ", ".join(missing)
        )
    args.train_pairs = pairs
    sources = {src for src, _tgt, _count in pairs}
    targets = {tgt for _src, tgt, _count in pairs}
    args.trained_src_lang = next(iter(sources)) if len(sources) == 1 else None
    args.trained_tgt_lang = next(iter(targets)) if len(targets) == 1 else None
    if write:
        summary = ", ".join(f"{src}->{tgt} ({count})" for src, tgt, count in pairs)
        print(f" | > training pairs: {summary}")
        if skipped:
            skipped_text = ", ".join(f"{item['pair']} ({item['n']}, {item['reason']})" for item in skipped)
            print(f" | > skipped pairs: {skipped_text}")
    return data_dir


def _load_model(args):
    kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
    if args.train_mode == "qlora":
        try:
            from importlib.metadata import version

            import bitsandbytes  # noqa: F401

            version("bitsandbytes")
        except Exception as exc:
            raise SystemExit(
                "QLoRA needs a complete bitsandbytes install in this virtual environment.\n"
                "  pip install -U bitsandbytes\n"
                f"Check failed: {exc}"
            ) from exc
        from transformers import BitsAndBytesConfig

        dtype = torch.bfloat16 if args.bnb_compute_dtype == "bfloat16" else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, **kwargs)
    if args.train_mode == "full" and args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        return model
    if args.train_mode == "full":
        return model
    if args.train_mode == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if args.adapter_path:
        return PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def _save_final(trainer, tokenizer, args) -> Path:
    """Save the final weights.

    DDP and a single GPU write from rank 0 only. ZeRO and FSDP must enter
    ``save_model`` on every rank so the shards can be gathered; the Trainer
    still writes the files from rank 0. ``full_lora`` is merged first, except
    under ZeRO or FSDP, where a local merge would see only a shard.
    """
    final_dir = Path(args.final_model_dir)
    sharded = trainer.is_deepspeed_enabled or trainer.is_fsdp_enabled
    if args.train_mode == "full_lora" and not sharded and trainer.is_world_process_zero():
        trainer.model = trainer.model.merge_and_unload()
    elif args.train_mode == "full_lora" and sharded and trainer.is_world_process_zero():
        print(" | > full_lora under ZeRO/FSDP is saved unmerged. Merge the adapter after training.")
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        final_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(final_dir)
        sanitize_saved_tokenizer(final_dir)
    return final_dir


def _world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _training_arguments(args) -> Seq2SeqTrainingArguments:
    report_to = [] if args.report_to == "none" else [args.report_to]
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "label_smoothing_factor": args.label_smoothing_factor,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "logging_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_total_limit": None,
        "save_only_model": args.save_only_model,
        "metric_for_best_model": args.metric_for_best_model,
        "greater_is_better": args.greater_is_better,
        "load_best_model_at_end": args.eval_strategy == args.save_strategy and args.eval_strategy != "no",
        "predict_with_generate": False,
        "generation_max_length": args.eval_max_new_tokens,
        "generation_num_beams": args.generation_num_beams,
        "eval_accumulation_steps": args.eval_accumulation_steps,
        "dataloader_num_workers": args.num_workers,
        "seed": args.seed,
        "report_to": report_to,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        # Length-grouped batches can give two DDP ranks a different step count,
        # and the shorter rank then waits until the process-group timeout.
        # One process keeps grouping. WORLD_SIZE is set by torchrun before this runs.
        "group_by_length": _world_size() <= 1,
        "ddp_timeout": args.ddp_timeout,
    }
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__)
    strategy_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[strategy_key] = args.eval_strategy
    kwargs["save_strategy"] = args.save_strategy
    if args.num_workers > 0 and "dataloader_prefetch_factor" in signature.parameters:
        kwargs["dataloader_prefetch_factor"] = 2
    if args.gradient_checkpointing and "gradient_checkpointing_kwargs" in signature.parameters:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if args.distribution == "fsdp" and "fsdp" in signature.parameters:
        kwargs["fsdp"] = args.fsdp
        if "fsdp_config" in signature.parameters:
            kwargs["fsdp_config"] = {"min_num_params": 1_000_000}
    if (
        args.distribution in DEEPSPEED_DISTRIBUTIONS
        and args.deepspeed
        and "deepspeed" in signature.parameters
    ):
        kwargs["deepspeed"] = args.deepspeed
    training = Seq2SeqTrainingArguments(**kwargs)
    if getattr(args, "visible_gpu_count", None) == 1:
        # Hugging Face wraps a 4-bit QLoRA model in DataParallel when it
        # sees two GPUs. BitsAndBytes then fails with CUBLAS_STATUS_NOT_SUPPORTED.
        training._n_gpu = 1
    return training


def _prepare_gradient_checkpointing(model, args) -> None:
    """Keep LoRA grads when the frozen base is checkpointed.

    Checkpointing only reconnects the backward graph if an activation requires
    grad. A frozen IndicTrans2 embedding does not, so full_lora / lora backward
    dies with "does not require grad and does not have a grad_fn".
    """
    if not args.gradient_checkpointing:
        return
    if getattr(model, "config", None) is not None:
        model.config.use_cache = False
    _mark_embedding_outputs_trainable(model)
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if enable is None:
        return
    params = inspect.signature(enable).parameters
    accepts_kwargs = "gradient_checkpointing_kwargs" in params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    if accepts_kwargs:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        enable()


def _mark_embedding_outputs_trainable(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    modules = []
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for owner in (model, base):
        getter = getattr(owner, "get_input_embeddings", None)
        if getter is not None:
            modules.append(getter())
        for part_name in ("get_encoder", "get_decoder"):
            part_getter = getattr(owner, part_name, None)
            if part_getter is None:
                continue
            part = part_getter()
            embed_getter = getattr(part, "get_input_embeddings", None)
            if embed_getter is not None:
                modules.append(embed_getter())
            modules.append(getattr(part, "embed_tokens", None))

    def _hook(_module, _inputs, output):
        if torch.is_tensor(output):
            output.requires_grad_(True)

    seen: set[int] = set()
    for module in modules:
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        module.register_forward_hook(_hook)


def _accepts_processing_class() -> bool:
    return "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters
