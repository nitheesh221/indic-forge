"""Offline RCA report from a training run directory.

Reads training_logs.csv, eval_metrics.csv, eval_samples.csv, and
checkpoint_manifest.json, then writes one HTML file. Charts are inline SVG,
so the page works with no network. Each metric has its own chart. Findings
name the best step and the last step, not only the first and the last.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# Metrics already stored on a 0–100-ish scale. Each one is charted alone
# because a shared axis hides the small ones, and TER moves the other way.
_HIGHER_IS_BETTER = {
    "chrf2pp": True,
    "comet": True,
    "bleu": True,
    "bertscore": True,
    "meteor": True,
    "ter": False,
    "exact_match": True,
    "eval_loss": False,
    "eval_entropy": False,
    "eval_mean_token_accuracy": True,
    "loss": False,
    "learning_rate": True,
}
_METRIC_LABELS = {
    "loss": "Train loss",
    "eval_loss": "Validation loss",
    "learning_rate": "Learning rate",
    "eval_entropy": "Validation entropy",
    "eval_mean_token_accuracy": "Validation token accuracy",
    "chrf2pp": "chrF++",
    "comet": "COMET",
    "bleu": "BLEU",
    "bertscore": "BERTScore",
    "meteor": "METEOR",
    "ter": "TER",
    "exact_match": "Exact match",
}


def analyze_run(
    run_dir: Path,
    output_html: Path | None = None,
    worst_n: int = 25,
    latest_step_only: bool = True,
) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    logs = _read_csv(run_dir / "training_logs.csv")
    metrics = _read_csv(run_dir / "eval_metrics.csv")
    samples = _read_csv(run_dir / "eval_samples.csv")
    manifest = _read_json(run_dir / "checkpoint_manifest.json")
    meta = _find_meta(run_dir)

    findings = _build_findings(logs, metrics, samples, manifest)
    recommendation = _recommendation(logs, metrics, manifest)
    per_lang = _per_language_stats(samples, latest_step_only=latest_step_only)
    worst = _worst_examples(samples, worst_n=worst_n)
    charts = _chart_blocks(logs, metrics)
    output_html = output_html or (run_dir / "rca_report.html")
    output_html.write_text(
        _render_html(
            run_dir=run_dir,
            meta=meta,
            recommendation=recommendation,
            findings=findings,
            charts=charts,
            per_lang=per_lang,
            worst=worst,
            manifest=manifest,
            metric_tail=metrics[-8:] if metrics else [],
        ),
        encoding="utf-8",
    )
    return output_html


def _find_meta(run_dir: Path) -> dict[str, Any]:
    candidates = [
        run_dir / "final_model" / "finetune_meta.json",
        run_dir / "finetune_meta.json",
    ]
    for path in candidates:
        if path.is_file():
            return _read_json(path) or {}
    return {}


def _best_point(points: list[tuple[float, float]], higher_is_better: bool) -> tuple[float, float] | None:
    if not points:
        return None
    return max(points, key=lambda item: item[1]) if higher_is_better else min(points, key=lambda item: item[1])


def _recommendation(logs: list[dict], metrics: list[dict], manifest: dict | None) -> str:
    eval_loss = _numeric_col(logs, "eval_loss")
    best = _best_point(eval_loss, higher_is_better=False)
    if best is None:
        return (
            "No validation loss was logged. final_model is the weights from the "
            "last training step. Turn evaluation on to pick the best checkpoint."
        )
    best_step, best_value = best
    last_step, last_value = eval_loss[-1]
    manifest_note = _manifest_best_note(manifest)
    if abs(best_step - last_step) < 1e-9:
        text = (
            f"The best validation loss is {best_value:.4f} at step { _step(best_step) }, "
            "which is also the last eval. final_model matches the last checkpoint "
            "when eval and save use the same schedule."
        )
    else:
        text = (
            f"The best validation loss is {best_value:.4f} at step { _step(best_step) }. "
            f"The last eval is {last_value:.4f} at step { _step(last_step) }. "
            "On a normal run, final_model is the best step, not the last one. "
            "The last weights stay in the newest checkpoint folder."
        )
    if manifest_note:
        text = f"{text} {manifest_note}"
    chrf = _numeric_col(metrics, "chrf2pp")
    best_chrf = _best_point(chrf, higher_is_better=True)
    if best_chrf is not None and abs(best_chrf[0] - chrf[-1][0]) >= 1e-9:
        text += (
            f" chrF++ is highest at step { _step(best_chrf[0]) } "
            f"({best_chrf[1]:.2f}), not at the last step ({chrf[-1][1]:.2f}). "
            "chrF++ does not choose the saved checkpoint."
        )
    return text


def _manifest_best_note(manifest: dict | None) -> str:
    if not manifest or not manifest.get("checkpoints"):
        return ""
    scored = [row for row in manifest["checkpoints"] if row.get("score") is not None]
    if not scored:
        return ""
    higher = bool(manifest.get("greater_is_better"))
    best = max(scored, key=lambda row: float(row["score"])) if higher else min(
        scored, key=lambda row: float(row["score"])
    )
    name = Path(str(best.get("path", ""))).name
    return f"Manifest best checkpoint: {name} (score {best.get('score')})."


def _build_findings(
    logs: list[dict],
    metrics: list[dict],
    samples: list[dict],
    manifest: dict | None,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    train_loss = _numeric_col(logs, "loss")
    eval_loss = _numeric_col(logs, "eval_loss")
    best_eval = _best_point(eval_loss, higher_is_better=False)
    if best_eval is not None and len(eval_loss) >= 2 and train_loss:
        best_step, best_value = best_eval
        last_step, last_value = eval_loss[-1]
        train_fell = train_loss[-1][1] < train_loss[0][1] - 1e-4
        last_is_worse = last_value > best_value + 1e-4 and abs(last_step - best_step) >= 1e-9
        if train_fell and last_is_worse:
            findings.append(
                {
                    "level": "warn",
                    "title": "Possible overfitting",
                    "detail": (
                        f"Train loss fell from {train_loss[0][1]:.4f} to {train_loss[-1][1]:.4f}, "
                        f"while validation loss peaked at {best_value:.4f} on step { _step(best_step) } "
                        f"and the last eval is {last_value:.4f} on step { _step(last_step) }. "
                        "Use the best Top-N checkpoint, not the last one."
                    ),
                }
            )
        elif abs(train_loss[-1][1] - train_loss[0][1]) <= 1e-4 and abs(last_value - eval_loss[0][1]) <= 1e-4:
            findings.append(
                {
                    "level": "info",
                    "title": "Little loss movement",
                    "detail": (
                        "Loss barely moved from the first logged point to the last. "
                        "Check the learning rate, the amount of data, the language tags, "
                        "and that the checkpoint direction matches the pairs."
                    ),
                }
            )
    chrf = _numeric_col(metrics, "chrf2pp")
    best_chrf = _best_point(chrf, higher_is_better=True)
    if best_chrf is not None and len(chrf) >= 2:
        last_step, last_value = chrf[-1]
        if last_value + 0.5 < best_chrf[1] and abs(last_step - best_chrf[0]) >= 1e-9:
            findings.append(
                {
                    "level": "warn",
                    "title": "chrF++ regressed",
                    "detail": (
                        f"chrF++ peaked at {best_chrf[1]:.2f} on step { _step(best_chrf[0]) } "
                        f"and the last step is {last_value:.2f}. "
                        "This score is for reading the run. It does not choose final_model."
                    ),
                }
            )
    if samples:
        latest = max(_to_float(row.get("step")) or 0 for row in samples)
        latest_rows = [row for row in samples if (_to_float(row.get("step")) or 0) == latest]
        empty_hyp = sum(1 for row in latest_rows if not (row.get("hyp") or "").strip())
        if empty_hyp:
            findings.append(
                {
                    "level": "error",
                    "title": "Empty hypotheses",
                    "detail": f"{empty_hyp}/{len(latest_rows)} latest eval rows have an empty hyp.",
                }
            )
        exact = sum(
            1
            for row in latest_rows
            if _normalize(row.get("hyp")) == _normalize(row.get("ref")) and (row.get("ref") or "").strip()
        )
        if latest_rows:
            findings.append(
                {
                    "level": "info",
                    "title": "Exact-match rate (latest eval)",
                    "detail": (
                        f"{exact}/{len(latest_rows)} = {100.0 * exact / len(latest_rows):.1f}%. "
                        "Exact match stays near zero on real sentences. Use chrF++ by language below."
                    ),
                }
            )
    if manifest and manifest.get("checkpoints"):
        tops = [row for row in manifest["checkpoints"] if row.get("in_top_k")]
        lasts = [row for row in manifest["checkpoints"] if row.get("in_last_k")]
        last_names = ", ".join(
            Path(str(row.get("path", ""))).name
            for row in sorted(lasts, key=lambda item: _to_float(item.get("step")) or 0)
        ) or "none"
        findings.append(
            {
                "level": "info",
                "title": "Checkpoint retention",
                "detail": (
                    f"Kept {len(manifest['checkpoints'])} checkpoints "
                    f"(Top-N={manifest.get('top_k')}, Last-N={manifest.get('last_k')}; "
                    f"{len(tops)} in Top-N, {len(lasts)} in Last-N). "
                    f"Last-N folders: {last_names}."
                ),
            }
        )
    if not findings:
        findings.append(
            {
                "level": "info",
                "title": "No strong anomalies",
                "detail": "Review the charts, the language table, and the worst sentences below.",
            }
        )
    return findings


def _per_language_stats(samples: list[dict], latest_step_only: bool = True) -> list[dict]:
    if not samples:
        return []
    rows = samples
    if latest_step_only:
        latest = max(_to_float(row.get("step")) or 0 for row in samples)
        rows = [row for row in samples if (_to_float(row.get("step")) or 0) == latest]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('src_lang', '?')}→{row.get('tgt_lang', '?')}"
        buckets[key].append(row)
    stats = []
    for pair, items in buckets.items():
        exact = sum(
            1 for row in items if _normalize(row.get("hyp")) == _normalize(row.get("ref"))
        )
        refs = [row.get("ref") or "" for row in items]
        hyps = [row.get("hyp") or "" for row in items]
        chrf = _corpus_chrf(hyps, refs)
        avg_edit = sum(_token_edit_rate(ref, hyp) for ref, hyp in zip(refs, hyps)) / max(len(items), 1)
        stats.append(
            {
                "pair": pair,
                "n": len(items),
                "chrf2pp": None if chrf is None else round(chrf, 2),
                "exact_match_pct": round(100.0 * exact / len(items), 2),
                "mean_token_edit_rate": round(avg_edit, 4),
            }
        )
    stats.sort(key=lambda item: (item["chrf2pp"] is None, item["chrf2pp"] if item["chrf2pp"] is not None else 0))
    return stats


def _worst_examples(samples: list[dict], worst_n: int) -> list[dict]:
    """Worst latest sentences, with the best earlier chrF++ for the same source."""
    if not samples:
        return []
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in samples:
        key = (
            str(row.get("src_lang") or ""),
            str(row.get("tgt_lang") or ""),
            str(row.get("src") or ""),
        )
        grouped[key].append(row)
    scored = []
    for (src_lang, tgt_lang, src), rows in grouped.items():
        history = []
        for row in rows:
            step = _to_float(row.get("step")) or 0
            hyp = row.get("hyp") or ""
            ref = row.get("ref") or ""
            history.append(
                {
                    "step": step,
                    "hyp": hyp,
                    "ref": ref,
                    "chrf2pp": _sentence_chrf(hyp, ref),
                }
            )
        history.sort(key=lambda item: item["step"])
        latest = history[-1]
        ranked = [item for item in history if item["chrf2pp"] is not None]
        best = max(ranked, key=lambda item: item["chrf2pp"]) if ranked else latest
        latest_score = latest["chrf2pp"]
        best_score = best.get("chrf2pp")
        change = None
        if latest_score is not None and best_score is not None:
            change = round(latest_score - best_score, 2)
        scored.append(
            {
                "step": latest["step"],
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "src": src,
                "ref": latest["ref"],
                "hyp": latest["hyp"],
                "chrf2pp": None if latest_score is None else round(latest_score, 2),
                "best_chrf2pp": None if best_score is None else round(best_score, 2),
                "best_step": best["step"],
                "best_hyp": best["hyp"],
                "change": change,
                "token_edit_rate": round(_token_edit_rate(latest["ref"], latest["hyp"]), 4),
            }
        )
    scored.sort(
        key=lambda item: (
            item["chrf2pp"] is None,
            item["chrf2pp"] if item["chrf2pp"] is not None else 999,
        )
    )
    return scored[:worst_n]


def _corpus_chrf(hypotheses: list[str], references: list[str]) -> float | None:
    metric = _chrf_metric()
    if metric is None or not references:
        return None
    try:
        return float(metric.corpus_score(hypotheses, [references]).score)
    except Exception:
        return None


def _sentence_chrf(hypothesis: str, reference: str) -> float | None:
    metric = _chrf_metric()
    if metric is None:
        return None
    try:
        return float(metric.sentence_score(hypothesis or "", [reference or ""]).score)
    except Exception:
        return None


def _chrf_metric():
    if getattr(_chrf_metric, "cached", None) is not None or getattr(_chrf_metric, "failed", False):
        return getattr(_chrf_metric, "cached", None)
    try:
        from sacrebleu.metrics import CHRF

        _chrf_metric.cached = CHRF(word_order=2)
    except Exception:
        _chrf_metric.failed = True
        _chrf_metric.cached = None
    return _chrf_metric.cached


def _chart_blocks(logs: list[dict], metrics: list[dict]) -> list[str]:
    blocks = [
        _svg_overlay(
            "Loss",
            [
                ("Train loss", _numeric_col(logs, "loss"), "#0b6e4f"),
                ("Validation loss", _numeric_col(logs, "eval_loss"), "#9a6700"),
            ],
            lower_is_better=True,
        )
    ]
    for key in ("eval_entropy", "eval_mean_token_accuracy", "learning_rate"):
        blocks.append(_svg_one(logs, key))
    for key in ("chrf2pp", "comet", "bleu", "bertscore", "meteor", "ter", "exact_match"):
        points = _numeric_col(metrics, key)
        if points:
            blocks.append(_svg_series(_METRIC_LABELS[key], points, _HIGHER_IS_BETTER[key], "#1d4f91"))
    return [block for block in blocks if block]


def _svg_one(rows: list[dict], key: str) -> str:
    points = _numeric_col(rows, key)
    if not points:
        return ""
    return _svg_series(_METRIC_LABELS[key], points, _HIGHER_IS_BETTER[key], "#1d4f91")


def _svg_overlay(title: str, series: list[tuple[str, list[tuple[float, float]], str]], lower_is_better: bool) -> str:
    present = [(name, points, color) for name, points, color in series if points]
    if not present:
        return ""
    return _svg(title, present, lower_is_better)


def _svg_series(title: str, points: list[tuple[float, float]], higher_is_better: bool, color: str) -> str:
    return _svg(title, [(title, points, color)], lower_is_better=not higher_is_better)


def _svg(title: str, series: list[tuple[str, list[tuple[float, float]], str]], lower_is_better: bool) -> str:
    width, height = 640, 210
    left, right, top, bottom = 52, 16, 16, 32
    xs = [step for _name, points, _color in series for step, _value in points]
    ys = [value for _name, points, _color in series for _step, value in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    span = ymax - ymin
    pad = span * 0.15 if span else max(abs(ymin) * 0.05, 0.05)
    ymin -= pad
    ymax += pad
    if ymax == ymin:
        ymax = ymin + 1

    def x_of(step: float) -> float:
        return left + (step - xmin) / (xmax - xmin) * (width - left - right)

    def y_of(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * (height - top - bottom)

    grid = []
    for tick in range(4):
        value = ymin + (ymax - ymin) * tick / 3
        y = y_of(value)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#d5e0da" />'
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#4a5c55">{value:.3g}</text>'
        )
    paths = []
    marks = []
    legend = []
    focus = series[1][1] if len(series) > 1 else series[0][1]
    best = _best_point(focus, higher_is_better=not lower_is_better)
    last = focus[-1]
    for index, (name, points, color) in enumerate(series):
        coords = " ".join(f"{x_of(step):.1f},{y_of(value):.1f}" for step, value in points)
        paths.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{coords}" />'
        )
        legend.append(
            f'<rect x="{left + index * 170}" y="{height - 16}" width="10" height="10" fill="{color}" />'
            f'<text x="{left + 14 + index * 170}" y="{height - 7}" font-size="12" fill="#14201c">{html.escape(name)}</text>'
        )
    if best is not None:
        marks.append(
            f'<circle cx="{x_of(best[0]):.1f}" cy="{y_of(best[1]):.1f}" r="4.5" fill="#0b6e4f" />'
        )
    marks.append(
        f'<circle cx="{x_of(last[0]):.1f}" cy="{y_of(last[1]):.1f}" r="3.5" fill="#a11d2c" />'
    )
    direction = "lower is better" if lower_is_better else "higher is better"
    caption = f"{direction}. Green dot is the best point on this chart. Red dot is the last point."
    if best is not None:
        caption += f" Best {best[1]:.4g} at step {_step(best[0])}. Last {last[1]:.4g} at step {_step(last[0])}."
    return (
        '<figure class="chart">'
        f"<figcaption><strong>{html.escape(title)}</strong> {html.escape(caption)}</figcaption>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        + "".join(grid)
        + "".join(paths)
        + "".join(marks)
        + "".join(legend)
        + "</svg></figure>"
    )


def _token_edit_rate(ref: str | None, hyp: str | None) -> float:
    ref_tokens = _normalize(ref).split()
    hyp_tokens = _normalize(hyp).split()
    if not ref_tokens and not hyp_tokens:
        return 0.0
    distance = _levenshtein(ref_tokens, hyp_tokens)
    return distance / max(len(ref_tokens), 1)


def _levenshtein(left: list[str], right: list[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, token in enumerate(left, start=1):
        curr = [i]
        for j, other in enumerate(right, start=1):
            cost = 0 if token == other else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _numeric_col(rows: list[dict], key: str) -> list[tuple[float, float]]:
    out = []
    for row in rows:
        step = _to_float(row.get("step"))
        value = _to_float(row.get(key))
        if step is not None and value is not None:
            out.append((step, value))
    return out


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _step(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _render_html(
    run_dir: Path,
    meta: dict,
    recommendation: str,
    findings: list[dict],
    charts: list[str],
    per_lang: list[dict],
    worst: list[dict],
    manifest: dict | None,
    metric_tail: list[dict],
) -> str:
    meta_block = html.escape(json.dumps(meta, indent=2, ensure_ascii=False)) if meta else "No finetune_meta.json"
    findings_html = "".join(
        f'<div class="finding {html.escape(item["level"])}">'
        f"<strong>{html.escape(item['title'])}</strong>"
        f"<p>{html.escape(item['detail'])}</p></div>"
        for item in findings
    )
    lang_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['pair'])}</td>"
        f"<td>{row['n']}</td>"
        f"<td>{'' if row['chrf2pp'] is None else row['chrf2pp']}</td>"
        f"<td>{row['exact_match_pct']}</td>"
        f"<td>{row['mean_token_edit_rate']}</td>"
        "</tr>"
        for row in per_lang
    ) or "<tr><td colspan='5'>No eval_samples.csv</td></tr>"
    worst_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('src_lang', '')))}→{html.escape(str(row.get('tgt_lang', '')))}</td>"
        f"<td>{'' if row.get('chrf2pp') is None else row.get('chrf2pp')}</td>"
        f"<td>{'' if row.get('best_chrf2pp') is None else row.get('best_chrf2pp')} "
        f"(step { _step(float(row.get('best_step') or 0)) })</td>"
        f"<td>{'' if row.get('change') is None else row.get('change')}</td>"
        f"<td>{html.escape(str(row.get('src', '')))}</td>"
        f"<td>{html.escape(str(row.get('ref', '')))}</td>"
        f"<td>{html.escape(str(row.get('hyp', '')))}</td>"
        f"<td>{html.escape(str(row.get('best_hyp', '')))}</td>"
        "</tr>"
        for row in worst
    ) or "<tr><td colspan='8'>No eval samples</td></tr>"
    ckpt_rows = ""
    if manifest and manifest.get("checkpoints"):
        for row in manifest["checkpoints"]:
            flags = []
            if row.get("in_top_k"):
                flags.append("top-n")
            if row.get("in_last_k"):
                flags.append("last-n")
            ckpt_rows += (
                "<tr>"
                f"<td>{html.escape(Path(str(row.get('path', ''))).name)}</td>"
                f"<td>{row.get('step')}</td>"
                f"<td>{'' if row.get('score') is None else row.get('score')}</td>"
                f"<td>{html.escape(', '.join(flags))}</td>"
                "</tr>"
            )
    else:
        ckpt_rows = "<tr><td colspan='4'>No checkpoint_manifest.json</td></tr>"
    columns = _metric_columns(metric_tail)
    metric_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>"
        for row in metric_tail
    )
    metric_head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    charts_html = "".join(charts) or "<p>No numeric logs to chart.</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>IndicTrans2 RCA — {html.escape(run_dir.name)}</title>
<style>
  :root {{
    --bg: #eef3f1;
    --ink: #14201c;
    --muted: #4a5c55;
    --card: #ffffff;
    --line: #c5d4cd;
    --warn: #9a6700;
    --error: #a11d2c;
    --info: #1d4f91;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Source Sans 3", "IBM Plex Sans", "Segoe UI", sans-serif;
    color: var(--ink);
    background:
      linear-gradient(165deg, #d9ebe3 0%, transparent 42%),
      linear-gradient(340deg, #d6e0f0 0%, transparent 38%),
      var(--bg);
  }}
  header, main {{ max-width: 1100px; margin: 0 auto; }}
  header {{ padding: 2.2rem 1.5rem 0.5rem; }}
  header h1 {{
    margin: 0 0 0.35rem;
    font-size: clamp(1.8rem, 4vw, 2.5rem);
    letter-spacing: -0.02em;
  }}
  header p, .recommend p {{ margin: 0.4rem 0 0; color: var(--muted); }}
  main {{ padding: 0 1.5rem 3rem; display: grid; gap: 1.1rem; }}
  section {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.05rem 1.15rem 1.2rem;
  }}
  section h2 {{ margin: 0 0 0.7rem; font-size: 1.12rem; }}
  .recommend {{
    border-left: 4px solid #0b6e4f;
    background: #f3faf7;
  }}
  .charts {{
    display: grid;
    gap: 0.8rem;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  }}
  figure.chart {{ margin: 0; }}
  figure.chart figcaption {{ color: var(--muted); font-size: 0.86rem; margin-bottom: 0.25rem; }}
  figure.chart strong {{ color: var(--ink); }}
  svg {{ width: 100%; height: auto; background: #f8fbf9; border-radius: 10px; }}
  .finding {{
    border-left: 4px solid var(--info);
    background: #f3f7fb;
    padding: 0.7rem 0.85rem;
    margin-bottom: 0.55rem;
    border-radius: 0 10px 10px 0;
  }}
  .finding.warn {{ border-left-color: var(--warn); background: #fff9eb; }}
  .finding.error {{ border-left-color: var(--error); background: #fff0f1; }}
  .finding p {{ margin: 0.3rem 0 0; color: var(--muted); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{
    border-bottom: 1px solid var(--line);
    text-align: left;
    padding: 0.42rem 0.35rem;
    vertical-align: top;
  }}
  th {{ color: var(--muted); font-weight: 600; }}
  pre {{
    white-space: pre-wrap;
    background: #14201c;
    color: #eef3f1;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    overflow: auto;
    font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.82rem;
  }}
  .scroll {{ overflow: auto; max-height: 460px; }}
</style>
</head>
<body>
<header>
  <h1>IndicTrans2 training RCA</h1>
  <p>One file for <code>{html.escape(str(run_dir))}</code>. Charts are drawn in the page, so it opens without a network.</p>
</header>
<main>
  <section class="recommend">
    <h2>What to use</h2>
    <p>{html.escape(recommendation)}</p>
  </section>
  <section>
    <h2>Findings</h2>
    {findings_html}
  </section>
  <section>
    <h2>Charts</h2>
    <div class="charts">{charts_html}</div>
  </section>
  <section>
    <h2>Per-language chrF++ (latest eval, worst first)</h2>
    <div class="scroll">
      <table>
        <thead><tr><th>Pair</th><th>n</th><th>chrF++</th><th>Exact match %</th><th>Mean token edit rate</th></tr></thead>
        <tbody>{lang_rows}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Worst sentences</h2>
    <p>Sorted by the latest chrF++. The best earlier score for the same source shows whether the sentence used to be better.</p>
    <div class="scroll">
      <table>
        <thead><tr><th>Pair</th><th>Latest chrF++</th><th>Best chrF++</th><th>Change</th><th>Src</th><th>Ref</th><th>Latest hyp</th><th>Hyp at best step</th></tr></thead>
        <tbody>{worst_rows}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Checkpoints (Top-N and Last-N)</h2>
    <div class="scroll">
      <table>
        <thead><tr><th>Checkpoint</th><th>Step</th><th>Score</th><th>Kept as</th></tr></thead>
        <tbody>{ckpt_rows}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Recent eval metric rows</h2>
    <div class="scroll">
      <table>
        <thead><tr>{metric_head or "<th>none</th>"}</tr></thead>
        <tbody>{metric_rows or "<tr><td>No eval_metrics.csv</td></tr>"}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Run metadata</h2>
    <pre>{meta_block}</pre>
  </section>
</main>
</body>
</html>
"""


def _metric_columns(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    preferred = [
        "step",
        "eval_loss",
        "eval_entropy",
        "eval_mean_token_accuracy",
        "chrf2pp",
        "comet",
        "bleu",
        "bertscore",
        "meteor",
        "ter",
        "exact_match",
        "metric_errors",
    ]
    keys = list(dict.fromkeys([*preferred, *rows[0].keys()]))
    return [key for key in keys if any(row.get(key) not in (None, "") for row in rows)]
