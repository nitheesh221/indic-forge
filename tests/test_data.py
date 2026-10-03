"""Unit tests for data layout and language routing."""

import json
from pathlib import Path

import pytest

from indictrans2_finetune.data import prepare_file, rows_to_pairs, validate_official_dir
from indictrans2_finetune.langs import default_model, direction_for, normalize_lang

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_aliases():
    assert normalize_lang("hi") == "hin_Deva"
    assert normalize_lang("eng_Latn") == "eng_Latn"
    with pytest.raises(ValueError):
        normalize_lang("french")


def test_model_routing():
    assert direction_for("en", "hi") == "en-indic"
    assert direction_for("hin_Deva", "eng_Latn") == "indic-en"
    assert direction_for("hi", "ta") == "indic-indic"
    assert "en-indic-dist-200M" in default_model("eng_Latn", "hin_Deva")
    assert "indic-en" in default_model("hi", "en")
    assert "indic-indic" in default_model("hi", "ta")


def test_jsonl_prepare(tmp_path: Path):
    report = prepare_file(
        ROOT / "data/sample/train.jsonl",
        tmp_path,
        "train",
        src_lang="eng_Latn",
        tgt_lang="hin_Deva",
    )
    assert report.skipped == 0
    assert report.pairs["train"]["eng_Latn-hin_Deva"] == 40
    src = tmp_path / "train/eng_Latn-hin_Deva/train.eng_Latn"
    tgt = tmp_path / "train/eng_Latn-hin_Deva/train.hin_Deva"
    assert len(src.read_text(encoding="utf-8").splitlines()) == 40
    assert len(tgt.read_text(encoding="utf-8").splitlines()) == 40
    validate_official_dir(tmp_path, "train", ["eng_Latn"], ["hin_Deva"])


def test_csv_prepare_uses_defaults(tmp_path: Path):
    report = prepare_file(
        ROOT / "data/sample/train.csv",
        tmp_path,
        "train",
        src_lang="en",
        tgt_lang="hi",
    )
    assert report.pairs["train"]["eng_Latn-hin_Deva"] == 2


def test_mismatched_files_raise(tmp_path: Path):
    pair = tmp_path / "dev/eng_Latn-hin_Deva"
    pair.mkdir(parents=True)
    (pair / "dev.eng_Latn").write_text("a\nb\n", encoding="utf-8")
    (pair / "dev.hin_Deva").write_text("एक\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has 2 lines"):
        validate_official_dir(tmp_path, "dev", ["eng_Latn"], ["hin_Deva"])


def test_src_text_json_array(tmp_path: Path):
    path = tmp_path / "train.json"
    path.write_text(
        """[
  {
    "src_lang": "asm_Beng",
    "tgt_lang": "eng_Latn",
    "src_text": "নৱাগত ৰিয়া চক্ৰৱৰ্তীক নিয়োগ কৰা হৈছে।",
    "tgt_text": "Newcomer Rhea Chakraborty has been hired."
  }
]
""",
        encoding="utf-8",
    )
    report = prepare_file(path, tmp_path / "out", "train")
    assert report.skipped == 0
    assert report.pairs["train"]["asm_Beng-eng_Latn"] == 1


def test_wrapped_and_columnar_json(tmp_path: Path):
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "src_lang": "asm_Beng",
                        "tgt_lang": "eng_Latn",
                        "src_text": "এটা বাক্য।",
                        "tgt_text": "This is a sentence.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report = prepare_file(wrapped, tmp_path / "wrapped-out", "train")
    assert report.pairs["train"]["asm_Beng-eng_Latn"] == 1

    columnar = tmp_path / "columns.json"
    columnar.write_text(
        json.dumps(
            {
                "src_lang": "asm_Beng",
                "tgt_lang": "eng_Latn",
                "src_text": ["এটা বাক্য।"],
                "tgt_text": ["This is a sentence."],
            }
        ),
        encoding="utf-8",
    )
    report = prepare_file(columnar, tmp_path / "columns-out", "train")
    assert report.pairs["train"]["asm_Beng-eng_Latn"] == 1

    nested = tmp_path / "nested.json"
    nested.write_text(
        json.dumps(
            [
                {
                    "translation": {
                        "asm_Beng": "এটা বাক্য।",
                        "eng_Latn": "This is a sentence.",
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    report = prepare_file(nested, tmp_path / "nested-out", "train", "asm_Beng", "eng_Latn")
    assert report.pairs["train"]["asm_Beng-eng_Latn"] == 1


def test_rows_to_pairs_flores_keys():
    pairs, skipped = rows_to_pairs(
        [{"eng_Latn": "Hello", "hin_Deva": "नमस्ते", "src_lang": "eng_Latn", "tgt_lang": "hin_Deva"}]
    )
    assert skipped == 0
    assert pairs[0][2] == "Hello"
    assert pairs[0][3] == "नमस्ते"


def test_recipe_tab_languages_cover_sample():
    train = (ROOT / "data/sample/train.jsonl").read_text(encoding="utf-8")
    assert "PM-Kisan" in train
    assert "आधार" in train
