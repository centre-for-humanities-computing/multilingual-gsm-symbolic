"""Regression checks for the translation target registry."""

import ast
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_TRANSLATE_SCRIPT = _PROJECT_ROOT / "src/scripts/translate_templates.py"

_EU_OFFICIAL_LANGUAGE_CODES = {
    "bul",
    "hrv",
    "ces",
    "dan",
    "nld",
    "eng",
    "est",
    "fin",
    "fra",
    "deu",
    "ell",
    "hun",
    "gle",
    "ita",
    "lav",
    "lit",
    "mlt",
    "pol",
    "por",
    "ron",
    "slk",
    "slv",
    "spa",
    "swe",
}

_HUMAN_VALIDATED_LANGUAGE_CODES = {
    "ara",
    "dan",
    "fra",
    "hin",
    "isl",
    "jpn",
    "mar",
    "nld",
    "rus",
    "ukr",
    "zho",
}

_ETHNOLOGUE_2026_RANKED_LANGUAGE_CODES = {
    "afr",
    "amh",
    "asm",
    "azb",
    "bam",
    "bar",
    "ben",
    "bho",
    "ceb",
    "ctg",
    "dyu",
    "fuv",
    "gaz",
    "guj",
    "hat",
    "hau",
    "hne",
    "ibo",
    "ind",
    "jav",
    "kan",
    "kaz",
    "khm",
    "kin",
    "kmr",
    "kor",
    "ktu",
    "lin",
    "lug",
    "mag",
    "mai",
    "mal",
    "mos",
    "mya",
    "npi",
    "nso",
    "nya",
    "ory",
    "pbu",
    "pcm",
    "pes",
    "pnb",
    "run",
    "sck",
    "sin",
    "skr",
    "sna",
    "snd",
    "som",
    "sun",
    "swh",
    "tam",
    "tel",
    "tgl",
    "tha",
    "tsn",
    "tur",
    "uig",
    "urd",
    "uzn",
    "vie",
    "vjk",
    "wol",
    "xho",
    "yor",
    "yue",
    "zlm",
    "zul",
}


def _language_registry() -> dict[str, str]:
    module = ast.parse(_TRANSLATE_SCRIPT.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_LANGUAGE_NAMES" for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def test_translation_language_registry_is_limited_and_complete():
    languages = _language_registry()

    assert len(languages) == 100
    assert _EU_OFFICIAL_LANGUAGE_CODES <= languages.keys()
    assert _HUMAN_VALIDATED_LANGUAGE_CODES <= languages.keys()
    assert languages.keys() == (
        _EU_OFFICIAL_LANGUAGE_CODES | _HUMAN_VALIDATED_LANGUAGE_CODES | _ETHNOLOGUE_2026_RANKED_LANGUAGE_CODES
    )
