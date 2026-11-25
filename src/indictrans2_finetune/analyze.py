"""Offline RCA report from a training run directory.

Reads training_logs.csv, eval_metrics.csv, eval_samples.csv, and
checkpoint_manifest.json, then writes a self-contained HTML dashboard for
root-cause analysis: loss curves, MT metric trends, per-language breakdown,
worst hypotheses, and checkpoint retention.
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
    per_lang = _per_language_stats(samples, latest_step_only=latest_step_only)
    worst = _worst_examples(samples, worst_n=worst_n, latest_step_only=latest_step_only)
    charts = {
        "loss": _series(logs, "step", ["loss", "eval_loss"]),
        "token": _series(logs, "step", ["eval_entropy", "eval_mean_token_accuracy"]),
        "mt": _series(
            metrics,
            "step",
            ["chrf2pp", "comet", "bleu", "bertscore", "meteor", "ter", "exact_match"],
        ),
        "lr": _series(logs, "step", ["learning_rate"]),
    }
    output_html = output_html or (run_dir / "rca_report.html")
    output_html.write_text(
        _render_html(
            run_dir=run_dir,
            meta=meta,
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


def _build_findings(
    logs: list[dict],
    metrics: list[dict],
    samples: list[dict],
    manifest: dict | None,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    train_loss = _numeric_col(logs, "loss")
    eval_loss = _numeric_col(logs, "eval_loss")
    if len(train_loss) >= 2 and len(eval_loss) >= 2:
        train_delta = train_loss[-1][1] - train_loss[0][1]
        eval_delta = eval_loss[-1][1] - eval_loss[0][1]
        if train_delta < -1e-4 and eval_delta > 1e-4:
            findings.append(
                {
                    "level": "warn",
                    "title": "Possible overfitting",
                    "detail": (
                        f"Train loss fell {train_delta:.4f} while eval_loss rose {eval_delta:.4f}. "
                        "Prefer an earlier Top-K checkpoint or stop earlier."
                    ),
                }
            )
        elif train_delta > -1e-4 and eval_delta > -1e-4:
            findings.append(
                {
                    "level": "info",
                    "title": "Little loss movement",
                    "detail": (
                        "Loss barely moved. Check learning rate, data size, language tags, "
                        "and that the checkpoint direction matches the pairs."
                    ),
                }
            )
    chrf = _numeric_col(metrics, "chrf2pp")
    if len(chrf) >= 2 and chrf[-1][1] + 0.5 < chrf[0][1]:
        findings.append(
            {
                "level": "warn",
                "title": "chrF++ regressed",
                "detail": (
                    f"chrF++ went from {chrf[0][1]:.2f} to {chrf[-1][1]:.2f}. "
                    "Inspect worst examples and consider the best Top-K step, not the last."
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
                    "detail": f"{exact}/{len(latest_rows)} = {100.0 * exact / len(latest_rows):.1f}%.",
                }
            )
    if manifest and manifest.get("checkpoints"):
        tops = [c for c in manifest["checkpoints"] if c.get("in_top_k")]
        lasts = [c for c in manifest["checkpoints"] if c.get("in_last_k")]
        findings.append(
            {
                "level": "info",
                "title": "Checkpoint retention",
                "detail": (
                    f"Kept {len(manifest['checkpoints'])} checkpoints "
                    f"(Top-K={manifest.get('top_k')}, Last-K={manifest.get('last_k')}; "
                    f"{len(tops)} in Top-K, {len(lasts)} in Last-K)."
                ),
            }
        )
    if not findings:
        findings.append(
            {
                "level": "info",
                "title": "No strong anomalies",
                "detail": "Review the charts and worst examples below for subtler issues.",
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
    for pair, items in sorted(buckets.items()):
        exact = sum(
            1 for row in items if _normalize(row.get("hyp")) == _normalize(row.get("ref"))
        )
        avg_edit = sum(_token_edit_rate(row.get("ref"), row.get("hyp")) for row in items) / max(
            len(items), 1
        )
        stats.append(
            {
                "pair": pair,
                "n": len(items),
                "exact_match_pct": round(100.0 * exact / len(items), 2),
                "mean_token_edit_rate": round(avg_edit, 4),
            }
        )
    return stats


def _worst_examples(samples: list[dict], worst_n: int, latest_step_only: bool) -> list[dict]:
    if not samples:
        return []
    rows = samples
    if latest_step_only:
        latest = max(_to_float(row.get("step")) or 0 for row in samples)
        rows = [row for row in samples if (_to_float(row.get("step")) or 0) == latest]
    scored = []
    for row in rows:
        ref = row.get("ref") or ""
        hyp = row.get("hyp") or ""
        rate = _token_edit_rate(ref, hyp)
        scored.append({**row, "token_edit_rate": round(rate, 4)})
    scored.sort(key=lambda item: item["token_edit_rate"], reverse=True)
    return scored[:worst_n]


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
    for i, a in enumerate(left, start=1):
        curr = [i]
        for j, b in enumerate(right, start=1):
            cost = 0 if a == b else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _series(rows: list[dict], x_key: str, y_keys: list[str]) -> dict[str, list]:
    xs: list[float] = []
    series = {key: [] for key in y_keys}
    for row in rows:
        x = _to_float(row.get(x_key))
        if x is None:
            continue
        present = False
        values = {}
        for key in y_keys:
            value = _to_float(row.get(key))
            values[key] = value
            if value is not None:
                present = True
        if not present:
            continue
        xs.append(x)
        for key in y_keys:
            series[key].append(values[key])
    return {"steps": xs, "series": series}


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
    findings: list[dict],
    charts: dict,
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
        f"<td>{row['exact_match_pct']}</td>"
        f"<td>{row['mean_token_edit_rate']}</td>"
        "</tr>"
        for row in per_lang
    ) or "<tr><td colspan='4'>No eval_samples.csv</td></tr>"
    worst_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('step', '')))}</td>"
        f"<td>{html.escape(str(row.get('src_lang', '')))}→{html.escape(str(row.get('tgt_lang', '')))}</td>"
        f"<td>{row.get('token_edit_rate', '')}</td>"
        f"<td>{html.escape(str(row.get('src', '')))}</td>"
        f"<td>{html.escape(str(row.get('ref', '')))}</td>"
        f"<td>{html.escape(str(row.get('hyp', '')))}</td>"
        "</tr>"
        for row in worst
    ) or "<tr><td colspan='6'>No eval samples</td></tr>"
    ckpt_rows = ""
    if manifest and manifest.get("checkpoints"):
        for row in manifest["checkpoints"]:
            flags = []
            if row.get("in_top_k"):
                flags.append("top-k")
            if row.get("in_last_k"):
                flags.append("last-k")
            ckpt_rows += (
                "<tr>"
                f"<td>{html.escape(Path(row['path']).name)}</td>"
                f"<td>{row.get('step')}</td>"
                f"<td>{'' if row.get('score') is None else row.get('score')}</td>"
                f"<td>{html.escape(', '.join(flags))}</td>"
                "</tr>"
            )
    else:
        ckpt_rows = "<tr><td colspan='4'>No checkpoint_manifest.json</td></tr>"
    metric_rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in _metric_columns(metric_tail))
        + "</tr>"
        for row in metric_tail
    )
    metric_head = "".join(f"<th>{html.escape(col)}</th>" for col in _metric_columns(metric_tail))
    payload = json.dumps(charts, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>IndicTrans2 RCA — {html.escape(run_dir.name)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #eef3f1;
    --ink: #14201c;
    --muted: #4a5c55;
    --card: #ffffff;
    --line: #c5d4cd;
    --accent: #0b6e4f;
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
  header {{
    padding: 2.5rem 1.5rem 1rem;
    max-width: 1100px;
    margin: 0 auto;
  }}
  header h1 {{
    margin: 0 0 0.35rem;
    font-size: clamp(1.8rem, 4vw, 2.6rem);
    letter-spacing: -0.02em;
    font-family: "IBM Plex Sans", "Source Sans 3", sans-serif;
    font-weight: 650;
  }}
  header p {{ margin: 0; color: var(--muted); max-width: 42rem; }}
  main {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 0 1.5rem 3rem;
    display: grid;
    gap: 1.25rem;
  }}
  section {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.1rem 1.2rem 1.25rem;
  }}
  section h2 {{
    margin: 0 0 0.75rem;
    font-size: 1.15rem;
  }}
  .grid {{
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }}
  .finding {{
    border-left: 4px solid var(--info);
    background: #f3f7fb;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
    border-radius: 0 10px 10px 0;
  }}
  .finding.warn {{ border-left-color: var(--warn); background: #fff9eb; }}
  .finding.error {{ border-left-color: var(--error); background: #fff0f1; }}
  .finding p {{ margin: 0.35rem 0 0; color: var(--muted); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.92rem;
  }}
  th, td {{
    border-bottom: 1px solid var(--line);
    text-align: left;
    padding: 0.45rem 0.4rem;
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
  canvas {{ width: 100% !important; max-height: 280px; }}
  .scroll {{ overflow: auto; max-height: 420px; }}
</style>
</head>
<body>
<header>
  <h1>IndicTrans2 training RCA</h1>
  <p>Offline analysis for <code>{html.escape(str(run_dir))}</code>. Charts use CSV logs from the run; translation metrics never choose checkpoints.</p>
</header>
<main>
  <section>
    <h2>Findings</h2>
    {findings_html}
  </section>
  <section>
    <h2>Run metadata</h2>
    <pre>{meta_block}</pre>
  </section>
  <section class="grid">
    <div>
      <h2>Loss</h2>
      <canvas id="chart-loss"></canvas>
    </div>
    <div>
      <h2>Token diagnostics</h2>
      <canvas id="chart-token"></canvas>
    </div>
    <div>
      <h2>MT metrics</h2>
      <canvas id="chart-mt"></canvas>
    </div>
    <div>
      <h2>Learning rate</h2>
      <canvas id="chart-lr"></canvas>
    </div>
  </section>
  <section>
    <h2>Per-language breakdown (latest eval)</h2>
    <div class="scroll">
      <table>
        <thead><tr><th>Pair</th><th>n</th><th>Exact match %</th><th>Mean token edit rate</th></tr></thead>
        <tbody>{lang_rows}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Worst hypotheses (latest eval)</h2>
    <div class="scroll">
      <table>
        <thead><tr><th>Step</th><th>Pair</th><th>Edit rate</th><th>Src</th><th>Ref</th><th>Hyp</th></tr></thead>
        <tbody>{worst_rows}</tbody>
      </table>
    </div>
  </section>
  <section>
    <h2>Checkpoints (Top-K ∪ Last-K)</h2>
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
</main>
<script>
const DATA = {payload};
function lineChart(id, title, keys, colors) {{
  const ctx = document.getElementById(id);
  if (!ctx) return;
  const pack = DATA[title];
  const datasets = keys.map((key, index) => ({{
    label: key,
    data: (pack.series[key] || []).map((y, i) => ({{ x: pack.steps[i], y }})),
    borderColor: colors[index % colors.length],
    backgroundColor: "transparent",
    spanGaps: true,
    tension: 0.2,
    pointRadius: 2,
  }})).filter(ds => ds.data.some(p => p.y !== null && p.y !== undefined));
  new Chart(ctx, {{
    type: "line",
    data: {{ datasets }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ type: "linear", title: {{ display: true, text: "step" }} }},
        y: {{ title: {{ display: true, text: "value" }} }},
      }},
      plugins: {{ legend: {{ position: "bottom" }} }},
    }},
  }});
}}
lineChart("chart-loss", "loss", ["loss", "eval_loss"], ["#0f6b5c", "#8a4b08"]);
lineChart("chart-token", "token", ["eval_entropy", "eval_mean_token_accuracy"], ["#1f4b7a", "#0f6b5c"]);
lineChart("chart-mt", "mt", ["chrf2pp", "comet", "bleu", "bertscore", "meteor", "ter", "exact_match"],
  ["#0f6b5c", "#1f4b7a", "#8a4b08", "#5c5346", "#8b1e1e", "#4a3f35", "#2f6f4e"]);
lineChart("chart-lr", "lr", ["learning_rate"], ["#1f4b7a"]);
</script>
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
