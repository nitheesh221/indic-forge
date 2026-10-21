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


def _read_json_document(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    if not text:
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        yield from _read_jsonl(path)
        return
    yield from _records_from_payload(payload, path)


def _records_from_payload(payload, path: Path) -> Iterator[dict]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = _unwrap_object(payload)
    else:
        raise ValueError(f"{path} must be a JSON object, a JSON array, or JSONL")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{path} item {index} must be a JSON object")
        yield row


def _unwrap_object(payload: dict) -> list:
    dict_lists = [
        value
        for value in payload.values()
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value[:20])
    ]
    if dict_lists:
        return max(dict_lists, key=len)
    list_fields = {key: value for key, value in payload.items() if isinstance(value, list)}
    if len(list_fields) >= 2:
        length = len(next(iter(list_fields.values())))
        if length and all(len(value) == length for value in list_fields.values()):
            scalars = {key: value for key, value in payload.items() if not isinstance(value, list)}
            return [{**scalars, **{key: value[index] for key, value in list_fields.items()}} for index in range(length)]
    if payload and all(isinstance(value, dict) for value in payload.values()):
        return list(payload.values())
    return [payload]


def iter_rows(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        yield from _read_json_document(path)
    elif suffix in {".csv", ".tsv", ".tab"}:
        yield from _read_table(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (use .jsonl, .csv, or .tsv)")


_SRC_KEYS = (
    "src",
    "source",
    "src_text",
    "source_text",
    "src_sentence",
    "source_sentence",
    "sentence_src",
    "original",
)
_TGT_KEYS = (
    "tgt",
    "target",
    "tgt_text",
    "target_text",
    "tgt_sentence",
    "target_sentence",
    "sentence_tgt",
    "translation",
    "translated",
)


def _as_text(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("text", "sentence", "value", "content", "src_text", "tgt_text"):
            found = _as_text(value.get(key))
            if found:
                return found
    return None


def _pick(row: dict, *names: str) -> str | None:
    lower = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        found = _as_text(lower.get(name.lower()))
        if found:
            return found
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
        src = _pick(row, *_SRC_KEYS, "en", "english")
        tgt = _pick(row, *_TGT_KEYS, "hi", "hindi")
        src, tgt = _pair_from_row(row, src, tgt, src_lang, tgt_lang)

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


def _pair_from_row(row: dict, src: str | None, tgt: str | None, src_lang: str | None, tgt_lang: str | None):
    containers = [row, *[value for value in row.values() if isinstance(value, dict)]]
    for container in containers:
        if src is None and src_lang and src_lang in container:
            src = _as_text(container.get(src_lang))
        if tgt is None and tgt_lang and tgt_lang in container:
            tgt = _as_text(container.get(tgt_lang))
        if src is None:
            src = _pick(container, *_SRC_KEYS)
        if tgt is None:
            tgt = _pick(container, *_TGT_KEYS)
        if src and tgt:
            return src, tgt
    return src, tgt


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


def list_pair_dirs(data_dir: Path, split: str) -> list[tuple[str, str, int]]:
    """Return (src, tgt, line count) for each prepared pair folder."""
    root = data_dir / split
    if not root.is_dir():
        return []
    found: list[tuple[str, str, int]] = []
    for pair_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if "-" not in pair_dir.name:
            continue
        src_name, tgt_name = pair_dir.name.split("-", 1)
        try:
            src_lang = normalize_lang(src_name)
            tgt_lang = normalize_lang(tgt_name)
        except ValueError:
            continue
        src_path = pair_dir / f"{split}.{src_lang}"
        tgt_path = pair_dir / f"{split}.{tgt_lang}"
        if not src_path.exists() or not tgt_path.exists():
            continue
        count = len(src_path.read_text(encoding="utf-8").splitlines())
        if count:
            found.append((src_lang, tgt_lang, count))
    return found


def select_training_pairs(
    data_dir: Path,
    model_hf_id: str,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
    split: str = "train",
) -> tuple[list[tuple[str, str, int]], list[dict[str, str | int]]]:
    """Keep pairs that match the checkpoint. Flags, when set, narrow that set."""
    from indictrans2_finetune.langs import direction_for, direction_of_model

    wanted_direction = direction_of_model(model_hf_id)
    src_filter = normalize_lang(src_lang) if src_lang else None
    tgt_filter = normalize_lang(tgt_lang) if tgt_lang else None
    discovered = list_pair_dirs(data_dir, split)
    kept: list[tuple[str, str, int]] = []
    skipped: list[dict[str, str | int]] = []
    for pair_src, pair_tgt, count in discovered:
        pair_name = f"{pair_src}-{pair_tgt}"
        if src_filter and pair_src != src_filter:
            skipped.append({"pair": pair_name, "n": count, "reason": "source filter"})
            continue
        if tgt_filter and pair_tgt != tgt_filter:
            skipped.append({"pair": pair_name, "n": count, "reason": "target filter"})
            continue
        direction = direction_for(pair_src, pair_tgt)
        if direction != wanted_direction:
            skipped.append({"pair": pair_name, "n": count, "reason": f"direction {direction}"})
            continue
        kept.append((pair_src, pair_tgt, count))
    if not kept:
        found = ", ".join(f"{src}-{tgt} ({count})" for src, tgt, count in discovered) or "none"
        if src_filter or tgt_filter:
            wanted = f"{src_filter or '*'} -> {tgt_filter or '*'}"
            raise SystemExit(f"No {split} rows for {wanted} under {data_dir}. Found: {found}.")
        skipped_text = ", ".join(f"{item['pair']} ({item['reason']})" for item in skipped) or "none"
        raise SystemExit(
            f"No {split} pairs match the {wanted_direction} checkpoint {model_hf_id}. "
            f"Skipped: {skipped_text}. Found: {found}."
        )
    return kept, skipped


def prepare_file(
    input_path: Path,
    output_dir: Path,
    split: str,
    src_lang: str | None = None,
    tgt_lang: str | None = None,
) -> PrepareReport:
    rows = list(iter_rows(input_path))
    pairs, skipped = rows_to_pairs(
        rows,
        default_src_lang=src_lang,
        default_tgt_lang=tgt_lang,
    )
    if not pairs:
        preview = ""
        if rows:
            preview = f" First record keys: {list(rows[0])}."
        raise ValueError(
            f"No parallel rows found in {input_path}.{preview} "
            "Need src/tgt text (src_text and tgt_text are accepted) plus language codes."
        )
    counts = write_official_layout(pairs, output_dir, split)
    return PrepareReport(output_dir=output_dir, pairs={split: counts}, skipped=skipped)
