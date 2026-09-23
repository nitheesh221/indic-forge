"""Import IndicTransToolkit pieces without depending on a broken package __init__."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _module_from_pkg_file(mod_name: str, filename: str):
    import IndicTransToolkit as pkg  # package dir only; may still run __init__

    path = Path(pkg.__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_processor(*, inference: bool) -> Any:
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except Exception:
        module = _module_from_pkg_file("_it2_processor", "processor.py")
        IndicProcessor = module.IndicProcessor
    return IndicProcessor(inference=inference)


def get_collator(tokenizer, model, *, pad_to_multiple_of: int | None, label_pad_token_id: int = -100):
    try:
        from IndicTransToolkit.collator import IndicDataCollator

        return IndicDataCollator(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
        )
    except Exception:
        from transformers import DataCollatorForSeq2Seq

        return DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="longest",
            pad_to_multiple_of=pad_to_multiple_of,
            label_pad_token_id=label_pad_token_id,
        )
