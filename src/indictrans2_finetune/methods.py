"""Training-method decisions for IndicTrans2 fine-tuning.

IndicTrans2 is an encoder–decoder MT model. With parallel text the right
objective is supervised fine-tuning (SFT): teacher-forced cross-entropy on
target tokens. That is what AI4Bharat’s official LoRA script uses and what
this toolkit runs.

Decision summary (2025–2026 literature):

* SFT — default for domain adaptation when you have `src`/`tgt` pairs.
  WMT26 IndicTrans2 LoRA recipes and the official HF interface both use it.
* GRPO — recent work shows GRPO can work on encoder–decoder MT (NLLB-scale)
  with K sampled hypotheses and a reference-free reward (e.g. LaBSE +
  COMET-Kiwi), especially when parallel data is scarce. It needs on-policy
  sampling, a reward model, and a KL anchor (often “LoRA off” as π_ref).
  Prefer a solid SFT adapter first; treat GRPO as a second stage.
* DPO / ORPO — need chosen/rejected pairs. Domain Indic FT almost never
  ships those; synthesize them only if you already have a reward model.
* KTO — MT Summit 2025 showed progressive-perturbation KTO beating SFT on
  low-resource Indic *LLM* translation. Not yet the standard path for
  IndicTrans2 encoder–decoder adapters here.
* Continued pre-training / denoising — for monolingual adaptation, not for
  parallel domain transfer.

Parameter-efficiency modes (`lora`, `qlora`, `full_lora`, `full`) are
orthogonal to the training *method*: they change what is updated under SFT.
"""

from __future__ import annotations

TRAIN_METHODS = ("sft",)
EXPERIMENTAL_METHODS = ("grpo", "dpo", "kto")
ALL_METHODS = TRAIN_METHODS + EXPERIMENTAL_METHODS

METHOD_HELP = {
    "sft": "Supervised fine-tuning (teacher-forced CE). Default and recommended.",
    "grpo": (
        "Group Relative Policy Optimization. Emerging for seq2seq MT with a "
        "reference-free reward and K sampled hypotheses. Not implemented here — "
        "run SFT first; use GRPO only as a second stage when parallel data is thin."
    ),
    "dpo": (
        "Direct Preference Optimization. Needs chosen/rejected pairs. "
        "Not implemented in this release."
    ),
    "kto": (
        "Kahneman-Tversky Optimization. Preference-style alignment without "
        "paired rejects; shown on Indic LLM MT via progressive perturbation. "
        "Not implemented for IndicTrans2 in this release."
    ),
}


def validate_train_method(method: str) -> str:
    method = (method or "sft").lower().strip()
    if method in TRAIN_METHODS:
        return method
    if method in EXPERIMENTAL_METHODS:
        raise SystemExit(
            f"--train_method {method} is not available yet.\n"
            f"{METHOD_HELP[method]}\n"
            "Use --train_method sft (default). See README § Training method."
        )
    raise SystemExit(
        f"Unknown --train_method {method!r}. "
        f"Supported: {', '.join(TRAIN_METHODS)}. "
        f"Reserved: {', '.join(EXPERIMENTAL_METHODS)}."
    )


def recommend_distribution(train_mode: str, n_gpus: int | None) -> str:
    """Pick a sensible --distribution given mode and GPU count.

    LoRA / QLoRA / full_lora: DDP when multi-GPU (FSDP+PEFT is fragile).
    Full FT: FSDP when multi-GPU and memory is the bottleneck; otherwise DDP.
    Single GPU: auto (in-process).
    """
    gpus = 1 if n_gpus is None else max(1, int(n_gpus))
    if gpus <= 1:
        return "auto"
    if train_mode in {"lora", "qlora", "full_lora"}:
        return "ddp"
    if train_mode == "full":
        return "fsdp"
    return "ddp"
