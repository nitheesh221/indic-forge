from pathlib import Path

from indictrans2_finetune.config import build_parser

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = (
    "train_lora.sh",
    "train_qlora.sh",
    "train_full_lora.sh",
    "train_full.sh",
)


def test_every_training_argument_is_documented_in_each_mode_script():
    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }
    for name in TRAIN_SCRIPTS:
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        missing = sorted(option for option in options if option not in text)
        assert missing == [], f"{name} is missing {missing}"


def test_every_translate_argument_is_documented_in_inference_script():
    expected = {
        "--text",
        "--file",
        "--json-file",
        "--src-lang",
        "--tgt-lang",
        "--model",
        "--adapter",
        "--eval-metrics",
        "--predictions-csv",
        "--metrics-csv",
        "--max-length",
        "--num-beams",
    }
    text = (ROOT / "scripts/inference.sh").read_text(encoding="utf-8")
    missing = sorted(option for option in expected if option not in text)
    assert missing == []
