"""Translation metrics. These are reported in CSV and never select checkpoints."""

from __future__ import annotations

import unicodedata

from sacrebleu.metrics import BLEU, CHRF, TER

METRIC_ORDER = ("chrf2pp", "comet", "bleu", "bertscore", "meteor", "ter", "exact_match")


def normalize_exact(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    return " ".join(normalized.split())


def exact_match(hypotheses: list[str], references: list[str]) -> float:
    if not references:
        return 0.0
    hits = sum(
        normalize_exact(hyp) == normalize_exact(ref) for hyp, ref in zip(hypotheses, references)
    )
    return 100.0 * hits / len(references)


def _chrf2pp(hypotheses: list[str], references: list[str]) -> float:
    return CHRF(word_order=2).corpus_score(hypotheses, [references]).score


def _bleu(hypotheses: list[str], references: list[str]) -> float:
    return BLEU(tokenize="flores200").corpus_score(hypotheses, [references]).score


def _ter(hypotheses: list[str], references: list[str]) -> float:
    return TER().corpus_score(hypotheses, [references]).score


def _comet(sources: list[str], hypotheses: list[str], references: list[str]) -> float:
    from comet import download_model, load_from_checkpoint

    checkpoint = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(checkpoint)
    samples = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(sources, hypotheses, references)
    ]
    prediction = model.predict(samples, batch_size=8, gpus=0, progress_bar=False)
    score = float(prediction.system_score)
    return score * 100.0 if score <= 1.5 else score


def _bertscore(hypotheses: list[str], references: list[str]) -> float:
    from bert_score import score

    _precision, _recall, f1 = score(
        hypotheses,
        references,
        model_type="xlm-roberta-large",
        verbose=False,
    )
    return float(f1.mean()) * 100.0


def _meteor(hypotheses: list[str], references: list[str]) -> float:
    import nltk
    from nltk.translate.meteor_score import meteor_score

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    scores = [
        meteor_score([ref.split()], hyp.split()) for hyp, ref in zip(hypotheses, references)
    ]
    return 100.0 * (sum(scores) / len(scores) if scores else 0.0)


_SCORERS = {
    "chrf2pp": lambda sources, hyps, refs: _chrf2pp(hyps, refs),
    "comet": _comet,
    "bleu": lambda sources, hyps, refs: _bleu(hyps, refs),
    "bertscore": lambda sources, hyps, refs: _bertscore(hyps, refs),
    "meteor": lambda sources, hyps, refs: _meteor(hyps, refs),
    "ter": lambda sources, hyps, refs: _ter(hyps, refs),
    "exact_match": lambda sources, hyps, refs: exact_match(hyps, refs),
}


def score_translations(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    metrics: list[str] | None = None,
) -> tuple[dict[str, float | None], dict[str, str]]:
    if not (len(sources) == len(hypotheses) == len(references)):
        raise ValueError("sources, hypotheses, and references must have the same length")
    selected = list(metrics or METRIC_ORDER)
    scores: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    for name in selected:
        try:
            scores[name] = round(float(_SCORERS[name](sources, hypotheses, references)), 4)
        except Exception as exc:  # noqa: BLE001 — one metric must not drop the others
            scores[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"
    return scores, errors
