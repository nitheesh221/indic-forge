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


class IndicBatchCollator:
    """Left-pad source tokens and right-pad labels.

    IndicTransToolkit's IndicDataCollator imports its padding helper inside the
    class body, so calling it raises NameError on Python 3.13.
    """

    def __init__(self, tokenizer, model=None, padding="longest", pad_to_multiple_of=None, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        if not features:
            return {}
        labels = None
        if "labels" in features[0]:
            labels = [list(feature.pop("labels")) for feature in features]
        padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = padding_side
        if labels is not None:
            batch["labels"] = self._pad_labels(labels)
            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                    labels=batch["labels"]
                )
        return batch

    def _pad_labels(self, labels: list[list[int]]):
        import torch

        width = max(len(row) for row in labels)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            width = ((width + multiple - 1) // multiple) * multiple
        return torch.tensor(
            [row + [self.label_pad_token_id] * (width - len(row)) for row in labels],
            dtype=torch.long,
        )


def get_collator(tokenizer, model, *, pad_to_multiple_of: int | None, label_pad_token_id: int = -100):
    return IndicBatchCollator(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        pad_to_multiple_of=pad_to_multiple_of,
        label_pad_token_id=label_pad_token_id,
    )
