"""Supervised fine-tuning is the only objective. There is no method flag.

IndicTrans2 is an encoder–decoder translation model. With parallel text the
objective is teacher-forced cross-entropy on the target tokens. That is what
AI4Bharat’s official LoRA script uses.

Why PPO, GRPO, DPO, and KTO are not offered is written for newcomers in
docs/why-sft.md. Parameter-efficiency modes (`lora`, `qlora`, `full_lora`,
`full`) only change which weights that update touches.
"""

from __future__ import annotations


def recommend_distribution(train_mode: str, n_gpus: int | None) -> str:
    """Default distribution from mode and GPU count. Memory is not inspected.

    One GPU stays ``auto`` (single process). Several GPUs use DDP, including
    full fine-tuning. ZeRO-2, ZeRO-3, and CPU offload are manual switches for
    when a full 1B run does not fit. See docs/distribution.md.
    """
    gpus = 1 if n_gpus is None else max(1, int(n_gpus))
    if gpus <= 1:
        return "auto"
    if train_mode in {"lora", "qlora", "full_lora", "full"}:
        return "ddp"
    return "ddp"
