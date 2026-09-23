"""Convert JSONL/CSV/TSV into the official IndicTrans2 LoRA layout."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from indictrans2_finetune.langs import normalize_lang

Pair = tuple[str, str, str, str]


@dataclass
class PrepareReport:
    output_dir: Path
    pairs: dict[str, dict[str, int]]
    skipped: int

    def as_dict(self) -> dict:
        return {
            "output_dir": str(self.output_dir),
            "pairs": self.pairs,
            "skipped": self.skipped,
        }


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            yield row


def _read_table(path: Path) -> Iterator[dict]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        for row in reader:
            yield {key.strip(): (value or "").strip() for key, value in row.items() if key}


def iter_rows(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        yield from _read_jsonl(path)
    elif suffix in {".csv", ".tsv", ".tab"}:
        yield from _read_table(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (use .jsonl, .csv, or .tsv)")


def _pick(row: dict, *names: str) -> str | None:
    lower = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def rows_to_pairs(
    rows: Iterable[dict],
    default_src_lang: str | None = None,
    default_tgt_lang: str | None = None,
) -> tuple[list[Pair], int]:
    pairs: list[Pair] = []
    skipped = 0
    src_lang_default = normalize_lang(default_src_lang) if default_src_lang else None
    tgt_lang_default = normalize_lang(default_tgt_lang) if default_tgt_lang else None

    for row in rows:
        src_lang = _pick(row, "src_lang", "source_lang", "src_code") or src_lang_default
        tgt_lang = _pick(row, "tgt_lang", "target_lang", "tgt_code") or tgt_lang_default
        src = _pick(row, "src", "source", "en", "english", "sentence_src")
        tgt = _pick(row, "tgt", "target", "hi", "hindi", "sentence_tgt", "translation")

        if src_lang and src_lang in row and src is None:
            src = str(row[src_lang]).strip() or None
        if tgt_lang and tgt_lang in row and tgt is None:
            tgt = str(row[tgt_lang]).strip() or None

        if not src_lang or not tgt_lang:
            skipped += 1
            continue
        src_lang = normalize_lang(src_lang)
        tgt_lang = normalize_lang(tgt_lang)
        if src is None:
            src = _pick(row, src_lang)
        if tgt is None:
            tgt = _pick(row, tgt_lang)
        if not src or not tgt or src_lang == tgt_lang:
            skipped += 1
            continue
        pairs.append((src_lang, tgt_lang, _one_line(src), _one_line(tgt)))
    return pairs, skipped


def _one_line(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def write_official_layout(
    pairs: Iterable[Pair],
    output_dir: Path,
    split: str,
) -> dict[str, int]:
    buckets: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for src_lang, tgt_lang, src, tgt in pairs:
        buckets[(src_lang, tgt_lang)].append((src, tgt))

    counts: dict[str, int] = {}
    for (src_lang, tgt_lang), lines in buckets.items():
        pair_dir = output_dir / split / f"{src_lang}-{tgt_lang}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        src_path = pair_dir / f"{split}.{src_lang}"
        tgt_path = pair_dir / f"{split}.{tgt_lang}"
        with src_path.open("w", encoding="utf-8") as src_file, tgt_path.open(
            "w", encoding="utf-8"
        ) as tgt_file:
            for src, tgt in lines:
                src_file.write(src + "\n")
                tgt_file.write(tgt + "\n")
        counts[f"{src_lang}-{tgt_lang}"] = len(lines)
    return counts


def validate_official_dir(
    data_dir: Path,
    split: str,
    src_langs: list[str],
    tgt_langs: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src_lang in src_langs:
        for tgt_lang in tgt_langs:
            if src_lang == tgt_lang:
                continue
            src_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{src_lang}"
            tgt_path = data_dir / split / f"{src_lang}-{tgt_lang}" / f"{split}.{tgt_lang}"
            if not src_path.exists() or not tgt_path.exists():
                raise FileNotFoundError(
                    f"Missing {split} files for {src_lang}-{tgt_lang} under {data_dir}. "
                    "Expected:\n"
                    f"  {src_path}\n"
                    f"  {tgt_path}"
                )
            src_lines = src_path.read_text(encoding="utf-8").splitlines()
            tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines()
            if len(src_lines) != len(tgt_lines):
                raise ValueError(
                    f"{src_path.name} has {len(src_lines)} lines but "
                    f"{tgt_path.name} has {len(tgt_lines)}"
                )
            if not src_lines:
                raise ValueError(f"{src_path} is empty")
            counts[f"{src_lang}-{tgt_lang}"] = len(src_lines)
    return counts


def prepare_file(
    input_path: Path,
    output_dir: Path,
    split: str,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
) -> PrepareReport:
    pairs, skipped = rows_to_pairs(
        iter_rows(input_path),
        default_src_lang=src_lang,
        default_tgt_lang=tgt_lang,
    )
    if not pairs:
        raise ValueError(
            f"No parallel rows found in {input_path}. "
            "Need src/tgt text plus language codes (columns or --src-lang/--tgt-lang)."
        )
    counts = write_official_layout(pairs, output_dir, split)
    return PrepareReport(output_dir=output_dir, pairs={split: counts}, skipped=skipped)
