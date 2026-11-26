"""Offline chart page for one inference run.

The page is written only when the caller asks for it. Charts are inline SVG,
so the file opens with no network. Sentence chrF++ is computed here for the
histogram and the worst-sentence table. It is not added to the CSV files.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

_HIGHER_IS_BETTER = {
    "chrf2pp": True,
    "comet": True,
    "bleu": True,
    "bertscore": True,
    "meteor": True,
    "ter": False,
    "exact_match": True,
}
_LABELS = {
    "chrf2pp": "chrF++",
    "comet": "COMET",
    "bleu": "BLEU",
    "bertscore": "BERTScore",
    "meteor": "METEOR",
    "ter": "TER",
    "exact_match": "Exact match",
}
_METRIC_ORDER = (
    "chrf2pp",
    "comet",
    "bleu",
    "bertscore",
    "meteor",
    "ter",
    "exact_match",
)


def default_report_path(metrics_csv: Path) -> Path:
    return Path(metrics_csv).parent / "inference_report.html"


def write_inference_report(
    prediction_rows: list[dict],
    metric_rows: list[dict],
    output_html: Path,
    *,
    source_name: str = "",
    worst_n: int = 25,
) -> Path:
    """Write the inference chart page. Returns the path that was written."""
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    scored = _score_sentences(prediction_rows)
    summary = _summary(metric_rows, len(prediction_rows))
    charts = _charts(metric_rows, scored)
    output_html.write_text(
        _render(
            source_name=source_name,
            summary=summary,
            charts=charts,
            metric_rows=metric_rows,
            worst=scored[:worst_n],
        ),
        encoding="utf-8",
    )
    return output_html


def _score_sentences(rows: list[dict]) -> list[dict]:
    metric = _chrf()
    scored = []
    for row in rows:
        value = None
        if metric is not None:
            try:
                value = float(
                    metric.sentence_score(
                        row.get("prediction") or "",
                        [row.get("reference") or ""],
                    ).score
                )
            except Exception:
                value = None
        scored.append({**row, "chrf2pp": value})
    scored.sort(key=lambda item: (item["chrf2pp"] is None, item["chrf2pp"] if item["chrf2pp"] is not None else 999))
    return scored


def _chrf():
    cached = getattr(_chrf, "cached", None)
    if cached is not None or getattr(_chrf, "failed", False):
        return cached
    try:
        from sacrebleu.metrics import CHRF

        _chrf.cached = CHRF(word_order=2)
    except Exception:
        _chrf.failed = True
        _chrf.cached = None
    return _chrf.cached


def _summary(metric_rows: list[dict], n_sentences: int) -> str:
    pairs = len(metric_rows)
    lead = f"{n_sentences} sentences across {pairs} language pair{'s' if pairs != 1 else ''}."
    ranked = [
        (f"{row.get('src_lang', '')}→{row.get('tgt_lang', '')}", _number(row.get("chrf2pp")))
        for row in metric_rows
    ]
    ranked = [(name, score) for name, score in ranked if score is not None]
    if not ranked:
        return lead + " No chrF++ score was written, so the charts use the other metrics."
    worst_name, worst_score = min(ranked, key=lambda item: item[1])
    best_name, best_score = max(ranked, key=lambda item: item[1])
    if len(ranked) == 1:
        return f"{lead} chrF++ is {worst_score:.2f} for {worst_name}."
    return (
        f"{lead} Lowest chrF++ is {worst_name} at {worst_score:.2f}. "
        f"Highest is {best_name} at {best_score:.2f}."
    )


def _charts(metric_rows: list[dict], sentences: list[dict]) -> list[str]:
    blocks = []
    present = [name for name in _METRIC_ORDER if any(_number(row.get(name)) is not None for row in metric_rows)]
    for name in present:
        points = []
        for row in metric_rows:
            value = _number(row.get(name))
            if value is None:
                continue
            label = f"{row.get('src_lang', '')}→{row.get('tgt_lang', '')}"
            points.append((label, value))
        if points:
            blocks.append(_hbar(_LABELS[name], points, _HIGHER_IS_BETTER[name]))
    histogram = _histogram(sentences)
    if histogram:
        blocks.append(histogram)
    return blocks


def _hbar(title: str, points: list[tuple[str, float]], higher_is_better: bool) -> str:
    ordered = sorted(points, key=lambda item: item[1], reverse=not higher_is_better)
    row_h = 28
    left = 168
    width = 720
    height = 36 + row_h * len(ordered)
    right = 64
    top = 16
    values = [value for _label, value in ordered]
    vmax = max(values)
    if vmax <= 0:
        vmax = 1
    bars = []
    for index, (label, value) in enumerate(ordered):
        y = top + index * row_h
        length = (value / vmax) * (width - left - right)
        bars.append(
            f'<text x="{left - 8}" y="{y + 16}" text-anchor="end" font-size="12" fill="#14201c">{html.escape(label)}</text>'
            f'<rect x="{left}" y="{y + 4}" width="{length:.1f}" height="16" rx="3" fill="#1d4f91" />'
            f'<text x="{left + length + 6:.1f}" y="{y + 16}" font-size="12" fill="#4a5c55">{value:.2f}</text>'
        )
    direction = "higher is better" if higher_is_better else "lower is better"
    caption = f"{direction}. Sorted with the weaker pairs first."
    return (
        '<figure class="chart">'
        f"<figcaption><strong>{html.escape(title)}</strong> {html.escape(caption)}</figcaption>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        + "".join(bars)
        + "</svg></figure>"
    )


def _histogram(sentences: list[dict]) -> str:
    values = [row["chrf2pp"] for row in sentences if row.get("chrf2pp") is not None]
    if not values:
        return ""
    bins = [0] * 10
    for value in values:
        index = min(9, max(0, int(value // 10)))
        bins[index] += 1
    width, height = 640, 220
    left, right, top, bottom = 44, 12, 16, 36
    peak = max(bins) or 1
    gap = 8
    inner = width - left - right
    bar_w = (inner - gap * 9) / 10
    parts = []
    for index, count in enumerate(bins):
        bar_h = 0 if count == 0 else (count / peak) * (height - top - bottom)
        x = left + index * (bar_w + gap)
        y = height - bottom - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="#0b6e4f" />'
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="11" fill="#4a5c55">{index * 10}</text>'
        )
        if count:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="11" fill="#14201c">{count}</text>'
            )
    caption = (
        f"{len(values)} sentences. Bars are sentence chrF++, from 0 up to 100. "
        f"Mean {sum(values) / len(values):.2f}."
    )
    return (
        '<figure class="chart">'
        f"<figcaption><strong>Sentence chrF++</strong> {html.escape(caption)}</figcaption>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Sentence chrF++">'
        + "".join(parts)
        + "</svg></figure>"
    )


def _render(
    source_name: str,
    summary: str,
    charts: list[str],
    metric_rows: list[dict],
    worst: list[dict],
) -> str:
    columns = _columns(metric_rows)
    head = "".join(f"<th>{html.escape(_LABELS.get(col, col))}</th>" for col in columns)
    body = []
    for row in metric_rows:
        cells = "".join(f"<td>{html.escape(_cell(row.get(col)))}</td>" for col in columns)
        body.append(f"<tr>{cells}</tr>")
    worst_rows = []
    for row in worst:
        score = row.get("chrf2pp")
        score_text = "" if score is None else f"{score:.2f}"
        worst_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('src_lang', '')))}→{html.escape(str(row.get('tgt_lang', '')))}</td>"
            f"<td>{html.escape(score_text)}</td>"
            f"<td>{html.escape(str(row.get('src', '')))}</td>"
            f"<td>{html.escape(str(row.get('reference', '')))}</td>"
            f"<td>{html.escape(str(row.get('prediction', '')))}</td>"
            "</tr>"
        )
    charts_html = "".join(charts) or "<p>No numeric scores to chart.</p>"
    title_src = source_name or "inference"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>IndicTrans2 inference — {html.escape(title_src)}</title>
<style>
  :root {{
    --bg: #eef3f1;
    --ink: #14201c;
    --muted: #4a5c55;
    --card: #ffffff;
    --line: #c5d4cd;
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
  header h1 {{ margin: 0 0 0.35rem; font-size: clamp(1.8rem, 4vw, 2.5rem); letter-spacing: -0.02em; }}
  header p, .recommend p {{ margin: 0.4rem 0 0; color: var(--muted); }}
  main {{ padding: 0 1.5rem 3rem; display: grid; gap: 1.1rem; }}
  section {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.05rem 1.15rem 1.2rem;
  }}
  section h2 {{ margin: 0 0 0.7rem; font-size: 1.12rem; }}
  .recommend {{ border-left: 4px solid #0b6e4f; background: #f3faf7; }}
  .charts {{ display: grid; gap: 0.8rem; }}
  figure.chart {{ margin: 0; }}
  figure.chart figcaption {{ color: var(--muted); font-size: 0.86rem; margin-bottom: 0.25rem; }}
  figure.chart strong {{ color: var(--ink); }}
  svg {{ width: 100%; height: auto; background: #f8fbf9; border-radius: 10px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{
    border-bottom: 1px solid var(--line);
    text-align: left;
    padding: 0.42rem 0.35rem;
    vertical-align: top;
  }}
  th {{ color: var(--muted); font-weight: 600; }}
  .scroll {{ overflow: auto; max-height: 460px; }}
</style>
</head>
<body>
<header>
  <h1>IndicTrans2 inference</h1>
  <p>One file for <code>{html.escape(title_src)}</code>. Charts are drawn in the page, so it opens without a network.</p>
</header>
<main>
  <section class="recommend">
    <h2>What the file shows</h2>
    <p>{html.escape(summary)}</p>
  </section>
  <section>
    <h2>Charts</h2>
    <div class="charts">{charts_html}</div>
  </section>
  <section>
    <h2>Scores by language</h2>
    <div class="scroll">
      <table>
        <thead><tr>{head or "<th>none</th>"}</tr></thead>
        <tbody>{''.join(body) or '<tr><td>No metric rows</td></tr>'}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Weakest sentences</h2>
    <p>Sorted by sentence chrF++, lowest first. This score is for the chart. It is not a second corpus metric.</p>
    <div class="scroll">
      <table>
        <thead><tr><th>Pair</th><th>chrF++</th><th>Source</th><th>Reference</th><th>Prediction</th></tr></thead>
        <tbody>{''.join(worst_rows) or '<tr><td>No sentences</td></tr>'}</tbody>
      </table>
    </div>
  </section>
</main>
</body>
</html>
"""


def _columns(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    preferred = ["src_lang", "tgt_lang", "n", "model", "adapter", *_METRIC_ORDER, "metric_errors"]
    keys = list(dict.fromkeys([*preferred, *rows[0].keys()]))
    return [key for key in keys if any(row.get(key) not in (None, "") for row in rows)]


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value)


def _number(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number
