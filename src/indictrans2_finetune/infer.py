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


def _has_meta(model) -> bool:
    tensors = list(model.parameters()) + list(model.buffers())
    return any(getattr(tensor, "is_meta", False) for tensor in tensors)


def _load_state_dict(directory: Path) -> dict[str, torch.Tensor]:
    safetensors = directory / "model.safetensors"
    if safetensors.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors))
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        from safetensors.torch import load_file

        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shards = {name: load_file(str(directory / name)) for name in set(weight_map.values())}
        return {key: shards[name][key] for key, name in weight_map.items()}
    binary = directory / "pytorch_model.bin"
    if binary.is_file():
        try:
            return torch.load(binary, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(binary, map_location="cpu")
    raise FileNotFoundError(f"No model weights found in {directory}")


def overlay_checkpoint(model, directory: Path) -> dict[str, list[str]]:
    """Copy a merged checkpoint's tensors into an initialized base model.

    Saved IndicTrans2 checkpoints drop tied output weights and the sinusoidal
    position tables. Those must come from a properly built base model; loading
    them from an empty or meta-initialized model gives garbage translations.
    """
    state = _load_state_dict(Path(directory))
    own = model.state_dict()
    missing = [key for key in own if key not in state]
    unexpected = [key for key in state if key not in own]
    mismatched = [
        key for key in state if key in own and tuple(state[key].shape) != tuple(own[key].shape)
    ]
    if mismatched:
        raise RuntimeError(f"Checkpoint shapes do not match the base model: {mismatched[:5]}")
    model.load_state_dict({key: state[key] for key in state if key in own}, strict=False)
    _copy_output_projection(model, state)
    return {"missing": missing, "unexpected": unexpected}


def _copy_output_projection(model, state: dict[str, torch.Tensor]) -> None:
    head = getattr(model, "lm_head", None)
    if head is None or "lm_head.weight" in state:
        return
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else None
    embed = getattr(decoder, "embed_tokens", None)
    if embed is None or head.weight is embed.weight:
        return
    if head.weight.shape != embed.weight.shape:
        return
    with torch.no_grad():
        head.weight.copy_(embed.weight)


def _load_seq2seq(model_name: str, dtype: torch.dtype):
    kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def _load_model(model_name: str, dtype: torch.dtype):
    base = tokenizer_source(model_name)
    if base == model_name:
        model = _load_seq2seq(model_name, dtype)
        if _has_meta(model):
            raise RuntimeError(
                f"{model_name} loaded with empty weights. Train it with this repo so "
                "finetune_meta.json records the base checkpoint."
            )
        return model
    model = _load_seq2seq(base, dtype)
    overlay_checkpoint(model, Path(model_name))
    return model


@lru_cache(maxsize=2)
def load_pipeline(model_name: str, adapter: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source(model_name), trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = _load_model(model_name, dtype)
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
