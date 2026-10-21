"""Argparse surface for IndicTrans2 training. Field names match training_requirements.txt."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from indictrans2_finetune.langs import direction_for, normalize_lang

TRAIN_MODES = ("full", "full_lora", "lora", "qlora")
HF_MODEL_IDS = (
    "ai4bharat/indictrans2-en-indic-dist-200M",
    "ai4bharat/indictrans2-en-indic-1B",
    "ai4bharat/indictrans2-indic-en-dist-200M",
    "ai4bharat/indictrans2-indic-en-1B",
    "ai4bharat/indictrans2-indic-indic-dist-320M",
    "ai4bharat/indictrans2-indic-indic-1B",
)
EVAL_METRICS = ("chrf2pp", "comet", "bleu", "bertscore", "meteor", "ter", "exact_match")
BEST_METRICS = ("eval_loss", "eval_entropy", "eval_mean_token_accuracy")
HIGHER_IS_BETTER = {"eval_mean_token_accuracy"}
LORA_TARGETS = "q_proj,k_proj"
FULL_LORA_TARGETS = "q_proj,k_proj,v_proj,out_proj,fc1,fc2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune IndicTrans2 (full, full_lora, lora, qlora)."
    )
    parser.add_argument("--train_mode", choices=TRAIN_MODES, default="lora")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--model_hf_id", required=True, choices=HF_MODEL_IDS)
    parser.add_argument("--model_id", default=None, help="Hub id or local weights. Defaults to --model_hf_id.")
    parser.add_argument(
        "--src_lang",
        default=None,
        help="Optional source filter. Omit to train every JSON pair that matches the checkpoint.",
    )
    parser.add_argument(
        "--tgt_lang",
        default=None,
        help="Optional target filter. Omit to train every JSON pair that matches the checkpoint.",
    )
    parser.add_argument("--output_dir", default="./it2_out/exp")
    parser.add_argument("--final_model_dir", default=None)
    parser.add_argument("--save_only_model", action="store_true")

    parser.add_argument("--train_json", default=None)
    parser.add_argument("--val_json", default=None)
    parser.add_argument("--data_dir", default=None, help="Prepared official layout. Skips JSON conversion.")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
        ],
    )
    parser.add_argument(
        "--optim",
        default=None,
        choices=[
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_hf",
            "adamw_bnb_8bit",
            "paged_adamw_8bit",
            "paged_adamw_32bit",
            "adafactor",
        ],
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--eval_strategy", choices=["epoch", "steps", "no"], default="epoch")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--eval_max_new_tokens", type=int, default=256)
    parser.add_argument("--eval_accumulation_steps", type=int, default=1)
    parser.add_argument("--generation_num_beams", type=int, default=5)

    parser.add_argument("--save_strategy", choices=["epoch", "steps", "no"], default="epoch")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--metric_for_best_model", choices=BEST_METRICS, default="eval_loss")
    parser.add_argument("--top_n_checkpoints", type=int, default=1)

    parser.add_argument("--eval_metrics", nargs="+", choices=EVAL_METRICS, default=list(EVAL_METRICS))
    parser.add_argument("--eval_metrics_csv", default="eval_metrics.csv")
    parser.add_argument("--eval_samples_csv", default="eval_samples.csv")
    parser.add_argument("--training_logs_csv", default="training_logs.csv")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default=None)

    parser.add_argument("--bnb_compute_dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--bnb_4bit_quant_type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true")

    parser.add_argument("--gpu_ids", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", choices=["none"], default="none")
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.train_mode == "full_qlora":
        raise SystemExit("full_qlora is not supported. Use full_lora or qlora.")
    if args.bf16 and args.fp16:
        raise SystemExit("Do not combine --bf16 and --fp16.")
    if args.data_dir is None and (not args.train_json or not args.val_json):
        raise SystemExit("Pass --train_json and --val_json, or pass --data_dir.")
    if args.learning_rate is None:
        args.learning_rate = 2e-4 if args.train_mode in {"lora", "qlora"} else 2e-5
    if args.optim is None:
        args.optim = "paged_adamw_32bit" if args.train_mode == "qlora" else "adamw_torch"
    if args.lora_target_modules is None:
        args.lora_target_modules = FULL_LORA_TARGETS if args.train_mode == "full_lora" else LORA_TARGETS
    if args.final_model_dir is None:
        args.final_model_dir = str(Path(args.output_dir) / "final_model")
    if args.model_id is None:
        args.model_id = args.model_hf_id
    if args.src_lang:
        args.src_lang = normalize_lang(args.src_lang)
    if args.tgt_lang:
        args.tgt_lang = normalize_lang(args.tgt_lang)
    if args.src_lang and args.tgt_lang:
        direction = direction_for(args.src_lang, args.tgt_lang)
        if direction not in args.model_hf_id:
            raise SystemExit(
                f"{args.src_lang} -> {args.tgt_lang} needs an {direction} checkpoint, "
                f"not {args.model_hf_id}."
            )
    if args.eval_strategy != args.save_strategy:
        print(
            "Warning: --eval_strategy and --save_strategy differ. "
            "Top-N ranking is reliable when they match.",
            file=sys.stderr,
        )
    args.keep_checkpoints = min(args.top_n_checkpoints, args.save_total_limit)
    args.greater_is_better = args.metric_for_best_model in HIGHER_IS_BETTER
    if args.gpu_ids:
        args.visible_gpu_count = len(
            [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
        )
    elif os.environ.get("RANK") is not None or os.environ.get("LOCAL_RANK") is not None:
        args.visible_gpu_count = 1
    else:
        args.visible_gpu_count = None
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return resolve_args(build_parser().parse_args(argv))


def pin_visible_gpus(argv: list[str] | None = None) -> list[str] | None:
    """Set CUDA_VISIBLE_DEVICES from --gpu_ids before torch is imported."""
    argv = list(sys.argv[1:] if argv is None else argv)
    gpu_ids = None
    skip = False
    for item in argv:
        if skip:
            gpu_ids = item
            skip = False
            continue
        if item == "--gpu_ids":
            skip = True
            continue
        if item.startswith("--gpu_ids="):
            gpu_ids = item.split("=", 1)[1]
    if gpu_ids is None:
        return None
    ids = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    if not ids or any(not item.isdigit() for item in ids):
        raise SystemExit("--gpu_ids must be a comma-separated list of GPU indices, for example 0,1.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    return ids


def maybe_relaunch(argv: list[str] | None = None) -> None:
    """One GPU stays in-process. Several --gpu_ids relaunch under torchrun without --gpu_ids."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    known, _unknown = parser.parse_known_args(argv)
    distributed = os.environ.get("RANK") is not None or os.environ.get("LOCAL_RANK") is not None
    if distributed and known.gpu_ids:
        raise SystemExit("Do not pass --gpu_ids when launching with torchrun.")
    if distributed or not known.gpu_ids:
        return
    ids = pin_visible_gpus(argv)
    if ids is None or len(ids) == 1:
        return
    cleaned = _strip_gpu_ids(argv)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(len(ids)),
        str(Path(sys.argv[0]).resolve()),
        *cleaned,
    ]
    os.execvpe(cmd[0], cmd, os.environ)


def _strip_gpu_ids(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip = False
    for item in argv:
        if skip:
            skip = False
            continue
        if item == "--gpu_ids":
            skip = True
            continue
        if item.startswith("--gpu_ids="):
            continue
        cleaned.append(item)
    return cleaned
