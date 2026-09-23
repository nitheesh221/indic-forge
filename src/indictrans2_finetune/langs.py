"""FLORES-200 language tags used by IndicTrans2."""

from __future__ import annotations

LANGUAGES: dict[str, dict[str, str]] = {
    "asm_Beng": {"name": "Assamese", "script": "Bengali", "iso": "as"},
    "awa_Deva": {"name": "Awadhi", "script": "Devanagari", "iso": "awa"},
    "ben_Beng": {"name": "Bengali", "script": "Bengali", "iso": "bn"},
    "bho_Deva": {"name": "Bhojpuri", "script": "Devanagari", "iso": "bho"},
    "brx_Deva": {"name": "Bodo", "script": "Devanagari", "iso": "brx"},
    "doi_Deva": {"name": "Dogri", "script": "Devanagari", "iso": "doi"},
    "eng_Latn": {"name": "English", "script": "Latin", "iso": "en"},
    "gom_Deva": {"name": "Konkani", "script": "Devanagari", "iso": "gom"},
    "guj_Gujr": {"name": "Gujarati", "script": "Gujarati", "iso": "gu"},
    "hin_Deva": {"name": "Hindi", "script": "Devanagari", "iso": "hi"},
    "hne_Deva": {"name": "Chhattisgarhi", "script": "Devanagari", "iso": "hne"},
    "kan_Knda": {"name": "Kannada", "script": "Kannada", "iso": "kn"},
    "kas_Arab": {"name": "Kashmiri", "script": "Arabic", "iso": "ks"},
    "kas_Deva": {"name": "Kashmiri", "script": "Devanagari", "iso": "ks"},
    "kha_Latn": {"name": "Khasi", "script": "Latin", "iso": "kha"},
    "lus_Latn": {"name": "Mizo", "script": "Latin", "iso": "lus"},
    "mag_Deva": {"name": "Magahi", "script": "Devanagari", "iso": "mag"},
    "mai_Deva": {"name": "Maithili", "script": "Devanagari", "iso": "mai"},
    "mal_Mlym": {"name": "Malayalam", "script": "Malayalam", "iso": "ml"},
    "mar_Deva": {"name": "Marathi", "script": "Devanagari", "iso": "mr"},
    "mni_Beng": {"name": "Manipuri", "script": "Bengali", "iso": "mni"},
    "mni_Mtei": {"name": "Manipuri", "script": "Meitei", "iso": "mni"},
    "npi_Deva": {"name": "Nepali", "script": "Devanagari", "iso": "ne"},
    "ory_Orya": {"name": "Odia", "script": "Odia", "iso": "or"},
    "pan_Guru": {"name": "Punjabi", "script": "Gurmukhi", "iso": "pa"},
    "san_Deva": {"name": "Sanskrit", "script": "Devanagari", "iso": "sa"},
    "sat_Olck": {"name": "Santali", "script": "Ol Chiki", "iso": "sat"},
    "snd_Arab": {"name": "Sindhi", "script": "Arabic", "iso": "sd"},
    "snd_Deva": {"name": "Sindhi", "script": "Devanagari", "iso": "sd"},
    "tam_Taml": {"name": "Tamil", "script": "Tamil", "iso": "ta"},
    "tel_Telu": {"name": "Telugu", "script": "Telugu", "iso": "te"},
    "urd_Arab": {"name": "Urdu", "script": "Arabic", "iso": "ur"},
}

ALIASES = {
    "en": "eng_Latn",
    "english": "eng_Latn",
    "hi": "hin_Deva",
    "hindi": "hin_Deva",
    "bn": "ben_Beng",
    "bengali": "ben_Beng",
    "ta": "tam_Taml",
    "tamil": "tam_Taml",
    "te": "tel_Telu",
    "telugu": "tel_Telu",
    "ml": "mal_Mlym",
    "malayalam": "mal_Mlym",
    "kn": "kan_Knda",
    "kannada": "kan_Knda",
    "mr": "mar_Deva",
    "marathi": "mar_Deva",
    "gu": "guj_Gujr",
    "gujarati": "guj_Gujr",
    "pa": "pan_Guru",
    "punjabi": "pan_Guru",
    "or": "ory_Orya",
    "odia": "ory_Orya",
    "as": "asm_Beng",
    "assamese": "asm_Beng",
    "ur": "urd_Arab",
    "urdu": "urd_Arab",
    "ne": "npi_Deva",
    "nepali": "npi_Deva",
    "sa": "san_Deva",
    "sanskrit": "san_Deva",
}

MODELS = {
    "en-indic": {
        "distilled": "ai4bharat/indictrans2-en-indic-dist-200M",
        "base": "ai4bharat/indictrans2-en-indic-1B",
    },
    "indic-en": {
        "distilled": "ai4bharat/indictrans2-indic-en-dist-200M",
        "base": "ai4bharat/indictrans2-indic-en-1B",
    },
    "indic-indic": {
        "distilled": "ai4bharat/indictrans2-indic-indic-dist-320M",
        "base": "ai4bharat/indictrans2-indic-indic-1B",
    },
}

ENGLISH = "eng_Latn"


def normalize_lang(code: str) -> str:
    raw = code.strip()
    if raw in LANGUAGES:
        return raw
    key = raw.lower().replace("-", "_")
    if key in ALIASES:
        return ALIASES[key]
    raise ValueError(
        f"Unknown language '{code}'. Use a FLORES tag such as hin_Deva, or an alias like hi. "
        "Run `indictrans2-ft langs` for the full list."
    )


def direction_for(src: str, tgt: str) -> str:
    src = normalize_lang(src)
    tgt = normalize_lang(tgt)
    if src == tgt:
        raise ValueError("Source and target language must differ.")
    if src == ENGLISH and tgt != ENGLISH:
        return "en-indic"
    if src != ENGLISH and tgt == ENGLISH:
        return "indic-en"
    if src != ENGLISH and tgt != ENGLISH:
        return "indic-indic"
    raise ValueError(f"Unsupported direction {src} -> {tgt}.")


def default_model(src: str, tgt: str, size: str = "distilled") -> str:
    direction = direction_for(src, tgt)
    if size not in MODELS[direction]:
        raise ValueError("size must be 'distilled' or 'base'")
    return MODELS[direction][size]
