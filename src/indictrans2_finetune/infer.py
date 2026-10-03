"""Translate with a base IndicTrans2 checkpoint and an optional LoRA adapter."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from indictrans2_finetune.langs import default_model, normalize_lang
from indictrans2_finetune.toolkit import get_processor

# save_pretrained stores these in tokenizer_config.json. IndicTransTokenizer
# also passes them explicitly, so a reloaded merged model raises
# "multiple values for keyword argument 'src_vocab_file'".
_SAVED_TOKENIZER_KEYS = ("src_vocab_file", "tgt_vocab_file", "src_spm", "tgt_spm")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenizer_source(model_name: str) -> str:
    """Use the original Hub tokenizer for a merged full or full_lora directory."""
    path = Path(model_name)
    for candidate in (path, path.parent):
        meta_path = candidate / "finetune_meta.json"
        if not meta_path.is_file():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        base = payload.get("model_hf_id")
        if isinstance(base, str) and base.strip():
            return base
    return model_name


def sanitize_saved_tokenizer(directory: Path) -> None:
    config_path = Path(directory) / "tokenizer_config.json"
    if not config_path.is_file():
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not any(key in payload for key in _SAVED_TOKENIZER_KEYS):
        return
    for key in _SAVED_TOKENIZER_KEYS:
        payload.pop(key, None)
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_seq2seq(model_name: str, dtype: torch.dtype):
    kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


@lru_cache(maxsize=2)
def load_pipeline(model_name: str, adapter: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source(model_name), trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = _load_seq2seq(model_name, dtype)
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
