"""Translate with a base IndicTrans2 checkpoint and an optional LoRA adapter."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from indictrans2_finetune.callbacks import assemble_translations, translation_slices
from indictrans2_finetune.langs import default_model, normalize_lang
from indictrans2_finetune.toolkit import get_processor

# save_pretrained stores these in tokenizer_config.json. IndicTransTokenizer
# also passes them explicitly, so a reloaded merged model raises
# "multiple values for keyword argument 'src_vocab_file'".
_SAVED_TOKENIZER_KEYS = ("src_vocab_file", "tgt_vocab_file", "src_spm", "tgt_spm")


def _device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local = os.environ.get("LOCAL_RANK")
    if local is not None and local.isdigit():
        index = int(local)
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return torch.device("cuda")


def inference_rank_and_world() -> tuple[int, int]:
    """Join the process group torchrun started. One GPU stays a single process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = int(os.environ.get("RANK", "0") or "0")
    if world <= 1:
        return 0, 1
    if not torch.cuda.is_available():
        raise SystemExit("Multi-GPU inference needs CUDA.")
    local = int(os.environ.get("LOCAL_RANK", "0") or "0")
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
    return dist.get_rank(), dist.get_world_size()


def shutdown_inference_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def clear_distributed_env() -> None:
    """Drop torchrun's world size before COMET starts.

    Lightning reads ``WORLD_SIZE`` and waits for every rank. The other GPU has
    already finished translating and exited, so scoring must look like one process.
    """
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_USE_AGENT_STORE",
    ):
        os.environ.pop(key, None)


def release_inference_model() -> None:
    """Drop the translation model so COMET or BERTScore can use the GPU."""
    load_pipeline.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_model_reference(model_name: str) -> str:
    """Return a Hub id, or the absolute path of a local model folder.

    A folder such as ``OUT2/indic-en-1b/final_model`` is not a Hub repo id.
    If that folder is missing, Hugging Face rejects the name before it says
    the path was not found.
    """
    raw = model_name.strip()
    path = Path(raw).expanduser()
    if path.is_dir():
        return str(path.resolve())
    looks_like_path = raw.startswith(("/", ".", "~")) or "\\" in raw or raw.count("/") > 1
    if looks_like_path:
        raise SystemExit(
            f"{raw} is not a model folder. Looked for {path.resolve()}. "
            "Pass the final_model directory written by training, or a Hub id "
            "such as ai4bharat/indictrans2-indic-en-1B."
        )
    return raw


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
    model_name = resolve_model_reference(model_name)
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
    batch_size: int = 8,
) -> list[str]:
    src_lang = normalize_lang(src_lang)
    tgt_lang = normalize_lang(tgt_lang)
    model_name = model_name or default_model(src_lang, tgt_lang)
    if batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    kept = [(offset, sentence.strip()) for offset, sentence in enumerate(sentences) if sentence and sentence.strip()]
    hypotheses = [""] * len(sentences)
    if not kept:
        return hypotheses
    tokenizer, model, processor = load_pipeline(model_name, adapter)
    device = next(model.parameters()).device
    for start in range(0, len(kept), batch_size):
        chunk = kept[start : start + batch_size]
        batch = processor.preprocess_batch(
            [sentence for _offset, sentence in chunk],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
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
        for (offset, _sentence), hypothesis in zip(chunk, processor.postprocess_batch(decoded, lang=tgt_lang)):
            hypotheses[offset] = hypothesis
    return hypotheses


def translate_across_ranks(
    sentences: list[str],
    src_lang: str,
    tgt_lang: str,
    model_name: str | None = None,
    adapter: str | None = None,
    max_length: int = 256,
    num_beams: int = 5,
    batch_size: int = 8,
) -> list[str]:
    """Translate on this rank's slice and return the sentences in source order.

    One process translates the whole list. Several processes each take every
    ``world``-th sentence, then the pieces are zipped back together.
    """
    rank, world = inference_rank_and_world()
    indices = translation_slices(len(sentences), rank, world) if world > 1 else list(range(len(sentences)))
    local_error = None
    rows: list[tuple[int, str]] = []
    try:
        if indices:
            local = translate_batch(
                [sentences[index] for index in indices],
                src_lang,
                tgt_lang,
                model_name=model_name,
                adapter=adapter,
                max_length=max_length,
                num_beams=num_beams,
                batch_size=batch_size,
            )
            if len(local) != len(indices):
                raise RuntimeError(
                    f"rank {rank} produced {len(local)} translations for {len(indices)} sentences"
                )
            rows = list(zip(indices, local))
    except Exception as exc:  # noqa: BLE001 — every rank must still enter the gather
        local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        rows = []
    if world == 1:
        if local_error:
            raise SystemExit(local_error)
        return [hypothesis for _index, hypothesis in rows]
    payload = {"rows": rows, "error": local_error}
    gathered: list[dict | None] = [None] * world
    dist.all_gather_object(gathered, payload)
    hypotheses, failure = assemble_translations(gathered, len(sentences))
    if failure or hypotheses is None:
        raise SystemExit(failure or "generation failed")
    return hypotheses


def maybe_adapter_base(adapter: Path | None) -> str | None:
    if adapter is None:
        return None
    meta = adapter / "adapter_config.json"
    if not meta.exists():
        return None
    import json

    payload = json.loads(meta.read_text(encoding="utf-8"))
    return payload.get("base_model_name_or_path")
