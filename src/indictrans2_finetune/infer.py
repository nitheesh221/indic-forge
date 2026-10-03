"""Translate with a base IndicTrans2 checkpoint and an optional LoRA adapter."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from indictrans2_finetune.langs import default_model, normalize_lang
from indictrans2_finetune.toolkit import get_processor


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=2)
def load_pipeline(model_name: str, adapter: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.to(_device())
    model.eval()
    processor = get_processor(inference=True)
    return tokenizer, model, processor


def translate_batch(
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    model_name: str | None = None,
    adapter: str | None = None,
    max_length: int = 256,
    num_beams: int = 5,
) -> list[str]:
    src_lang = normalize_lang(src_lang)
    tgt_lang = normalize_lang(tgt_lang)
    model_name = model_name or default_model(src_lang, tgt_lang)
    tokenizer, model, processor = load_pipeline(model_name, adapter)
    device = next(model.parameters()).device
    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not cleaned:
        return []
    batch = processor.preprocess_batch(cleaned, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            use_cache=False,
            min_length=0,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=1,
        )
    decoded = tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return processor.postprocess_batch(decoded, lang=tgt_lang)


def maybe_adapter_base(adapter: Path | None) -> str | None:
    if adapter is None:
        return None
    meta = adapter / "adapter_config.json"
    if not meta.exists():
        return None
    import json

    payload = json.loads(meta.read_text(encoding="utf-8"))
    return payload.get("base_model_name_or_path")
