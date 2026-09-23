"""Score a fine-tuned adapter on a prepared split."""

from __future__ import annotations

from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from indictrans2_finetune.data import validate_official_dir
from indictrans2_finetune.infer import maybe_adapter_base, translate_batch
from indictrans2_finetune.langs import default_model, normalize_lang


def evaluate_split(
    data_dir: Path,
    split: str,
    src_lang: str,
    tgt_lang: str,
    model_name: str | None = None,
    adapter: str | None = None,
    max_examples: int | None = None,
) -> dict:
    src_lang = normalize_lang(src_lang)
    tgt_lang = normalize_lang(tgt_lang)
    counts = validate_official_dir(data_dir, split, [src_lang], [tgt_lang])
    src_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{src_lang}"
    tgt_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{tgt_lang}"
    sources = src_path.read_text(encoding="utf-8").splitlines()
    references = tgt_path.read_text(encoding="utf-8").splitlines()
    if max_examples:
        sources = sources[:max_examples]
        references = references[:max_examples]
    model_name = model_name or maybe_adapter_base(Path(adapter) if adapter else None)
    model_name = model_name or default_model(src_lang, tgt_lang)
    predictions = translate_batch(
        sources, src_lang, tgt_lang, model_name=model_name, adapter=adapter
    )
    bleu = BLEU().corpus_score(predictions, [references]).score
    chrf = CHRF().corpus_score(predictions, [references]).score
    samples = [
        {"src": src, "ref": ref, "hyp": hyp}
        for src, ref, hyp in zip(sources[:5], references[:5], predictions[:5])
    ]
    return {
        "pair": f"{src_lang}-{tgt_lang}",
        "split": split,
        "n": counts[f"{src_lang}-{tgt_lang}"] if not max_examples else len(sources),
        "bleu": round(bleu, 2),
        "chrf": round(chrf, 2),
        "model": model_name,
        "adapter": adapter,
        "samples": samples,
    }
