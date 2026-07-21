# /// script
# dependencies = ["openai", "tomli-w", "multilingual-gsm-symbolic"]
# [tool.uv.sources]
# multilingual-gsm-symbolic = { path = "../..", editable = true }
# ///
"""Translate symbolic templates and replacements using a local vLLM server.

Translates all natural-language fields in a single call (ensuring consistent
terminology across fields) and validates each template using the same checks
as the test suite. Templates that fail validation are retried up to 3 times.

Usage:
    uv run src/scripts/translate_templates.py --to nob
    uv run src/scripts/translate_templates.py --to fra --model Qwen/Qwen3-8B
    uv run src/scripts/translate_templates.py --to nob --base-url http://127.0.0.1:8000/v1
    uv run src/scripts/translate_templates.py --from dan --subfolder symbolic --to nob

The model is auto-detected when the vLLM server exposes exactly one model.
Set VLLM_BASE_URL, VLLM_API_KEY, and VLLM_MODEL to avoid repeating flags.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import time
import tomllib
from pathlib import Path

import tomli_w
from openai import OpenAI

from multilingual_gsm_symbolic.templates import AnnotatedQuestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DATA_ROOT = Path("src/multilingual_gsm_symbolic/data/templates")

_LANGUAGE_NAMES = {
    # Exactly 100 translation targets: all 24 official EU languages, all
    # human-validated template languages, then the highest-ranked languages
    # from the 2026 Ethnologue 200 table until the list reaches 100.
    # Closely related ranked varieties are deduplicated, while Hindi and Urdu
    # remain separate targets and Cantonese is retained alongside Chinese.
    # The validated `ara` and `zho` targets intentionally replace ranked `arb`
    # and `cmn`.
    "afr": "Afrikaans",  # Ethnologue 2026 rank 79
    "amh": "Amharic",  # Ethnologue 2026 rank 29
    "ara": "Arabic",  # human-validated override for `arb` (Ethnologue 2026 rank 5)
    "asm": "Assamese",  # Ethnologue 2026 rank 67
    "azb": "South Azerbaijani",  # Ethnologue 2026 rank 98
    "bam": "Bamanankan",  # Ethnologue 2026 rank 88
    "bar": "Bavarian",  # Ethnologue 2026 rank 94
    "ben": "Bengali",  # Ethnologue 2026 rank 7
    "bho": "Bhojpuri",  # Ethnologue 2026 rank 38
    "bul": "Bulgarian",  # EU official; Ethnologue 2026 rank 149
    "ceb": "Cebuano",  # Ethnologue 2026 rank 71
    "ces": "Czech",  # EU official; Ethnologue 2026 rank 110
    "ctg": "Chittagonian",  # Ethnologue 2026 rank 100
    "dan": "Danish",  # EU official; human-validated; Ethnologue 2026 rank 171
    "deu": "German",  # EU official; Ethnologue 2026 rank 12
    "dyu": "Jula",  # Ethnologue 2026 rank 102
    "ell": "Greek",  # EU official; Ethnologue 2026 rank 99
    "eng": "English",  # EU official; Ethnologue 2026 rank 1
    "est": "Estonian",  # EU official
    "fin": "Finnish",  # EU official; Ethnologue 2026 rank 176
    "fra": "French",  # EU official; human-validated; Ethnologue 2026 rank 6
    "fuv": "Nigerian Fulfulde",  # Ethnologue 2026 rank 80
    "gaz": "West Central Oromo",  # Ethnologue 2026 rank 64
    "gle": "Irish",  # EU official
    "guj": "Gujarati",  # Ethnologue 2026 rank 33
    "hat": "Haitian Creole",  # Ethnologue 2026 rank 92
    "hau": "Hausa",  # Ethnologue 2026 rank 20
    "hin": "Hindi",  # human-validated; Ethnologue 2026 rank 3
    "hrv": "Croatian",  # EU official; Ethnologue 2026 rank 169
    "hun": "Hungarian",  # EU official; Ethnologue 2026 rank 103
    "ibo": "Igbo",  # Ethnologue 2026 rank 53
    "ind": "Indonesian",  # Ethnologue 2026 rank 9
    "isl": "Icelandic",  # human-validated
    "ita": "Italian",  # EU official; Ethnologue 2026 rank 32
    "jav": "Javanese",  # Ethnologue 2026 rank 31
    "jpn": "Japanese",  # human-validated; Ethnologue 2026 rank 13
    "kan": "Kannada",  # Ethnologue 2026 rank 34
    "kaz": "Kazakh",  # Ethnologue 2026 rank 76
    "khm": "Khmer",  # Ethnologue 2026 rank 75
    "kmr": "Northern Kurdish",  # Ethnologue 2026 rank 82
    "kor": "Korean",  # Ethnologue 2026 rank 28
    "lav": "Latvian",  # EU official
    "lin": "Lingala",  # Ethnologue 2026 rank 45
    "lit": "Lithuanian",  # EU official
    "lug": "Ganda",  # Ethnologue 2026 rank 106
    "mal": "Malayalam",  # Ethnologue 2026 rank 48
    "mar": "Marathi",  # human-validated; Ethnologue 2026 rank 16
    "mai": "Maithili",  # Ethnologue 2026 rank 81
    "mag": "Magahi",  # Ethnologue 2026 rank 73
    "mlt": "Maltese",  # EU official
    "mos": "Moore",  # Ethnologue 2026 rank 112
    "mya": "Burmese",  # Ethnologue 2026 rank 42
    "nld": "Dutch",  # EU official; human-validated; Ethnologue 2026 rank 65
    "npi": "Nepali",  # Ethnologue 2026 rank 55
    "nso": "Northern Sotho",  # Ethnologue 2026 rank 93
    "nya": "Chichewa",  # Ethnologue 2026 rank 87
    "ory": "Odia",  # Ethnologue 2026 rank 47
    "pbu": "Northern Pashto",  # Ethnologue 2026 rank 63
    "pcm": "Nigerian Pidgin",  # Ethnologue 2026 rank 14
    "pes": "Western Persian",  # Ethnologue 2026 rank 27
    "pnb": "Western Punjabi",  # Ethnologue 2026 rank 22
    "pol": "Polish",  # EU official; Ethnologue 2026 rank 41
    "por": "Portuguese",  # EU official; Ethnologue 2026 rank 8
    "ron": "Romanian",  # EU official; Ethnologue 2026 rank 68
    "run": "Rundi",  # Ethnologue 2026 rank 101
    "rus": "Russian",  # human-validated; Ethnologue 2026 rank 11
    "sck": "Sadri",  # Ethnologue 2026 rank 111
    "sin": "Sinhala",  # Ethnologue 2026 rank 74
    "skr": "Saraiki",  # Ethnologue 2026 rank 60
    "slk": "Slovak",  # EU official; Ethnologue 2026 rank 157
    "slv": "Slovenian",  # EU official
    "snd": "Sindhi",  # Ethnologue 2026 rank 50
    "sna": "Shona",  # Ethnologue 2026 rank 90
    "som": "Somali",  # Ethnologue 2026 rank 66
    "spa": "Spanish",  # EU official; Ethnologue 2026 rank 4
    "sun": "Sundanese",  # Ethnologue 2026 rank 52
    "swh": "Swahili",  # Ethnologue 2026 rank 19
    "swe": "Swedish",  # EU official; Ethnologue 2026 rank 117
    "tam": "Tamil",  # Ethnologue 2026 rank 24
    "tel": "Telugu",  # Ethnologue 2026 rank 18
    "tha": "Thai",  # Ethnologue 2026 rank 30
    "tgl": "Tagalog",  # Ethnologue 2026 rank 23
    "tsn": "Setswana",  # Ethnologue 2026 rank 95
    "tur": "Turkish",  # Ethnologue 2026 rank 21
    "ukr": "Ukrainian",  # human-validated; Ethnologue 2026 rank 58
    "urd": "Urdu",  # Ethnologue 2026 rank 10
    "uig": "Uyghur",  # Ethnologue 2026 rank 96
    "uzn": "Northern Uzbek",  # Ethnologue 2026 rank 57
    "vie": "Vietnamese",  # Ethnologue 2026 rank 17
    "vjk": "Bajjika",  # Ethnologue 2026 rank 105
    "wol": "Wolof",  # Ethnologue 2026 rank 83
    "xho": "Xhosa",  # Ethnologue 2026 rank 78
    "yor": "Yoruba",  # Ethnologue 2026 rank 37
    "yue": "Cantonese",  # Ethnologue 2026 rank 25
    "zho": "Chinese",  # human-validated override for `cmn` (Ethnologue 2026 rank 2)
    "zlm": "Central Malay",  # Ethnologue 2026 rank 59
    "zul": "Zulu",  # Ethnologue 2026 rank 61
    "hne": "Chhattisgarhi",  # Ethnologue 2026 rank 84
    "kin": "Kinyarwanda",  # Ethnologue 2026 rank 85
    "ktu": "Kituba",  # Ethnologue 2026 rank 108
}

_TRANSLATE_FIELDS = ("question", "answer", "question_annotated", "answer_annotated")

_SYSTEM_PROMPT = """\
You are a precise translator of mathematical word problems from {src_name} to {tgt_name}.

You will receive a JSON object representing a template. Your task is to translate ONLY the `question_annotated` and `answer_annotated` fields. 
Return a JSON object containing ONLY those two translated keys. Do not include `question` or `answer` in your JSON output.

Rules:
1. Translate all natural-language prose, including default values inside placeholders.
   - Variable placeholders have the form {{varname,default}} — keep the {{varname,...}} syntax but translate the default value. E.g. {{animal,haj}} → {{animal,hai}} when translating to Norwegian. Default values are mid-sentence fragments: use lowercase and the uninflected/indefinite form (e.g. {{unit,måned}} not {{unit,Måned}} or {{unit,måneden}}).
   - Bare variable references {{varname}} (no default) — leave completely unchanged.
   - Init / conditions / answer blocks: lines starting with #init:, #conditions:, #answer: — do NOT alter these lines at all.
   - Init expressions: range(...), sample(...), arange(...), etc. — do NOT alter, except for when it is a list of words e.g. ["chair", "table"] — translate the words but keep the list syntax. E.g. ["stol", "bord"] in Danish.
   - Condition expressions: is_int(...), divides(...), True, etc. — do NOT alter.
   - Inline calc tags in answers: <<expr=result>> — copy them EXACTLY as they appear in the source, character for character.
   - The #### answer marker and its number — do NOT alter.
2. Use natural, idiomatic {tgt_name} phrasing.
3. Return ONLY valid JSON — no markdown fences, no explanation.
"""

_REPLACEMENTS_SYSTEM_PROMPT = """\
You are a precise translator from {src_name} to {tgt_name}.

Translate the VALUES in the JSON object to {tgt_name}. Keep all keys exactly as-is.
For list values, translate each element. For nested lists (e.g. [[singular, plural], ...]),
translate both forms. For lists of [word, number] pairs, translate only the word.

Do NOT translate:
- Units (kg, g, m, km, etc.)
- Numbers
- Names — replace with common {tgt_name} first names instead

Return ONLY valid JSON — no markdown fences, no explanation.
"""


def lang_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


def resolve_model(client: OpenAI, requested_model: str | None) -> str:
    """Return the requested model or auto-detect the single model served by vLLM."""
    if requested_model:
        return requested_model

    try:
        model_ids = [model.id for model in client.models.list().data]
    except Exception as exc:
        raise SystemExit(
            "Could not query the vLLM /v1/models endpoint. "
            "Pass --model explicitly or check --base-url. "
            f"Original error: {exc}"
        ) from exc

    if len(model_ids) == 1:
        logger.info("Auto-detected vLLM model: %s", model_ids[0])
        return model_ids[0]
    if not model_ids:
        raise SystemExit("The vLLM server returned no models. Pass --model after checking the server.")

    available = ", ".join(model_ids)
    raise SystemExit(
        "The vLLM server exposes multiple models; choose one with --model. "
        f"Available models: {available}"
    )


_RENDERED_NOTE = (
    "Note: `question` and `answer` are the rendered forms of `question_annotated` and "
    "`answer_annotated` respectively — all {var,default} placeholders are replaced by their "
    "default values and all <<expr=result>> tags by their result values. Use them as a "
    "reference for what the annotated fields should produce when rendered.\n\n"
)


def _build_initial_messages(src_data: dict, src: str, tgt: str) -> list[dict]:
    system = _SYSTEM_PROMPT.format(src_name=lang_name(src), tgt_name=lang_name(tgt))
    payload = {f: src_data[f] for f in _TRANSLATE_FIELDS if f in src_data}
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _RENDERED_NOTE + json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def translate_template_fields(
    client: OpenAI, src_data: dict, src: str, tgt: str, model: str
) -> tuple[dict, list[dict]]:
    """Translate all four prose fields in a single call. Returns translated data and conversation history."""
    messages = _build_initial_messages(src_data, src, tgt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,

        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "high"
            }
        },
    )
    raw = response.choices[0].message.content.strip()
    messages = messages + [{"role": "assistant", "content": raw}]
    return json.loads(raw), messages


def fix_template_fields(client: OpenAI, model: str, feedback: str, messages: list[dict]) -> tuple[dict, list[dict]]:
    """Continue the translation conversation with error feedback."""
    messages = messages + [
        {"role": "user", "content": f"The translation has the following issues — fix ONLY what is needed:\n{feedback}"},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,

        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "high"
            }
        },
    )
    raw = response.choices[0].message.content.strip()
    messages = messages + [{"role": "assistant", "content": raw}]
    return json.loads(raw), messages


def _reconstruct_messages(src_data: dict, tgt_data: dict, src: str, tgt: str) -> list[dict]:
    """Build a synthetic conversation for an existing translation so retries use the same path."""
    messages = _build_initial_messages(src_data, src, tgt)
    tgt_payload = {f: tgt_data[f] for f in ("question_annotated", "answer_annotated") if f in tgt_data}
    return messages + [{"role": "assistant", "content": json.dumps(tgt_payload, ensure_ascii=False, indent=2)}]


def _strip_answer_annotated_defaults(tgt_data: dict) -> dict:
    """Remove any var defaults the model added to answer_annotated placeholders.

    answer_annotated uses bare {var} syntax; defaults only appear in question_annotated.
    The model sometimes copies the {var,default} pattern from question_annotated — this undoes that.
    """
    _RE_VAR = re.compile(r"\{([^}]+)\}")

    def _strip(m: re.Match) -> str:
        inner = m.group(1)
        if "," in inner:
            return "{" + inner.split(",")[0].strip() + "}"
        return m.group(0)

    if "answer_annotated" in tgt_data:
        tgt_data = dict(tgt_data)
        tgt_data["answer_annotated"] = _RE_VAR.sub(_strip, tgt_data["answer_annotated"])
    return tgt_data


def _render_deterministic_fields(tgt_data: dict, replacements: dict) -> dict:
    try:
        template = AnnotatedQuestion(
            **{
                k: tgt_data[k]
                for k in (
                    "question",
                    "answer",
                    "question_annotated",
                    "answer_annotated",
                    "id_orig",
                    "id_shuffled",
                    "language",
                )
                if k in tgt_data
            }
        )
        defaults = template._get_full_default_assignments(replacements)
        tgt_data["question"] = template.format_question(defaults)
        tgt_data["answer"] = template.format_answer(defaults)
    except Exception:
        pass
    return tgt_data


def translate_template(client: OpenAI, src_data: dict, src: str, tgt: str, model: str, tgt_replacements: dict) -> tuple[dict, list[dict]]:
    tgt_data = dict(src_data)
    translated_fields, messages = translate_template_fields(client, src_data, src, tgt, model)
    tgt_data.update(translated_fields)
    tgt_data = _strip_answer_annotated_defaults(tgt_data)
    tgt_data = _render_deterministic_fields(tgt_data, tgt_replacements)
    tgt_data["language"] = tgt
    tgt_data["creation"] = (
        f"machine-translated from {lang_name(src)} using {model}, "
        f"based on {lang_name(src)} templates; computationally validated"
    )
    return tgt_data, messages


def translate_replacements(client: OpenAI, src_data: dict, src: str, tgt: str, model: str) -> dict:
    system = _REPLACEMENTS_SYSTEM_PROMPT.format(src_name=lang_name(src), tgt_name=lang_name(tgt))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(src_data, ensure_ascii=False, indent=2)},
        ],

        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "high"
            }
        },
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def _var_name(placeholder: str) -> str:
    """Extract variable name from a placeholder like '{var,default}' → 'var'."""
    return placeholder.strip("{}").split(",")[0].strip()


def verify_syntax(src: dict, tgt: dict) -> list[str]:
    """Check that template syntax is preserved."""
    issues = []
    _RE_VAR = re.compile(r"\{[^}]+\}")
    for field in ("question_annotated", "answer_annotated"):
        src_names = {_var_name(p) for p in _RE_VAR.findall(src.get(field, ""))}
        tgt_names = {_var_name(p) for p in _RE_VAR.findall(tgt.get(field, ""))}
        missing = src_names - tgt_names
        if missing:
            issues.append(f"{field}: missing placeholders {missing}")

    for marker in ("#init:", "#conditions:", "#answer:"):
        if marker in src.get("question_annotated", "") and marker not in tgt.get("question_annotated", ""):
            issues.append(f"question_annotated: missing block marker '{marker}'")

    _RE_CALC = re.compile(r"<<[^>]+>>")
    src_calcs = _RE_CALC.findall(src.get("answer_annotated", ""))
    tgt_calcs = _RE_CALC.findall(tgt.get("answer_annotated", ""))
    if src_calcs != tgt_calcs:
        issues.append(f"answer_annotated: calc tags differ — src: {src_calcs}, tgt: {tgt_calcs}")

    if "####" in src.get("answer", "") and "####" not in tgt.get("answer", ""):
        issues.append("answer: missing #### marker")

    return issues


def verify_renders(tgt_data: dict, replacements: dict) -> list[str]:
    """Check that rendering with default assignments reproduces question and answer.

    Mirrors test_template_formatting_matches_original and test_default_assignments_are_valid.
    """
    issues = []
    try:
        template = AnnotatedQuestion(
            **{
                k: tgt_data[k]
                for k in (
                    "question",
                    "answer",
                    "question_annotated",
                    "answer_annotated",
                    "id_orig",
                    "id_shuffled",
                    "language",
                )
                if k in tgt_data
            }
        )
    except Exception as e:
        return [f"failed to construct AnnotatedQuestion: {e}"]

    try:
        defaults = template._get_full_default_assignments(replacements)
        formatted_q = template.format_question(defaults)
        formatted_a = template.format_answer(defaults)
        if formatted_q != template.question:
            issues.append(f"question mismatch:\n  rendered: {formatted_q!r}\n  expected: {template.question!r}")
        if formatted_a != template.answer:
            issues.append(f"answer mismatch:\n  rendered: {formatted_a!r}\n  expected: {template.answer!r}")
    except Exception as e:
        issues.append(f"render error: {e}")

    return issues


def process_template(
    i: int,
    src_file: Path,
    client: OpenAI,
    args: argparse.Namespace,
    tgt: str,
    tgt_symbolic: Path,
    tgt_replacements: dict,
    total_files: int
) -> tuple[str, list[str]] | None:
    tgt_file = tgt_symbolic / src_file.name

    with src_file.open("rb") as f:
        src_data = tomllib.load(f)

    # If translation already exists, validate it first; only redo if broken.
    if tgt_file.exists() and not args.overwrite:
        with tgt_file.open("rb") as f:
            tgt_data = tomllib.load(f)
        issues = verify_syntax(src_data, tgt_data) + verify_renders(tgt_data, tgt_replacements)
        if not issues:
            logger.info("[%d/%d] %s OK (skipping)", i + 1, total_files, src_file.name)
            return None
        logger.warning(
            "[%d/%d] %s has issues, fixing: %s", i + 1, total_files, src_file.name, "; ".join(issues)
        )
        messages = _reconstruct_messages(src_data, tgt_data, args.src, tgt)
        feedback = "\n".join(issues)
    else:
        logger.info("[%d/%d] Translating %s", i + 1, total_files, src_file.name)
        tgt_data, messages = translate_template(client, src_data, args.src, tgt, args.model, tgt_replacements)
        issues = verify_syntax(src_data, tgt_data) + verify_renders(tgt_data, tgt_replacements)
        feedback = "\n".join(issues)

    for attempt in range(1, args.retries + 1):
        if not issues:
            break
        logger.warning("[%d/%d] %s Attempt %d/%d failed, retrying with feedback", i + 1, total_files, src_file.name, attempt, args.retries)
        time.sleep(1)
        try:
            translated_fields, messages = fix_template_fields(client, args.model, feedback, messages)
            tgt_data.update(translated_fields)
            tgt_data = _strip_answer_annotated_defaults(tgt_data)
            tgt_data = _render_deterministic_fields(tgt_data, tgt_replacements)
            issues = verify_syntax(src_data, tgt_data) + verify_renders(tgt_data, tgt_replacements)
            feedback = "\n".join(issues)
        except Exception as e:
            issues = [f"retry error: {e}"]
            break

    if issues:
        logger.warning("[%d/%d] Unresolved issues in %s: %s", i + 1, total_files, src_file.name, "; ".join(issues))
        tgt_data["ignore"] = True
        error_result = (src_file.name, issues)
    else:
        logger.info("[%d/%d] %s OK", i + 1, total_files, src_file.name)
        error_result = None

    with tgt_file.open("wb") as f:
        f.write(tomli_w.dumps(tgt_data).encode("utf-8"))
    logger.info("[%d/%d] Written %s", i + 1, total_files, tgt_file.name)
    return error_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate symbolic math templates between languages.")
    parser.add_argument("--from", dest="src", default="eng", help="Source language code (default: eng)")
    parser.add_argument("--to", dest="tgt", required=True, help="Target language code (e.g. nob, or 'all')")
    parser.add_argument(
        "--subfolder",
        default="test_metric",
        help="Template subfolder within the language directory (default: test_metric)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="vLLM OpenAI-compatible base URL (default: %(default)s; env: VLLM_BASE_URL)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", "EMPTY"),
        help="API key configured on the vLLM server (default: EMPTY; env: VLLM_API_KEY)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL"),
        help="Served vLLM model name. Auto-detected when exactly one model is served (env: VLLM_MODEL)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("VLLM_TIMEOUT", "300")),
        help="Per-request timeout in seconds (default: %(default)s; env: VLLM_TIMEOUT)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-translate already existing files")
    parser.add_argument("--retries", type=int, default=2, help="Max retries for templates failing validation")
    args = parser.parse_args()

    src_dir = _DATA_ROOT / args.src

    if not src_dir.exists():
        raise SystemExit(f"Source language directory not found: {src_dir}")

    client = OpenAI(
        base_url=args.base_url.rstrip("/"),
        api_key=args.api_key,
        timeout=args.timeout,
    )
    args.model = resolve_model(client, args.model)
    logger.info("Using vLLM endpoint %s with model %s", args.base_url, args.model)

    targets = [lang for lang in _LANGUAGE_NAMES if lang != args.src] if args.tgt == "all" else [args.tgt]
    overall_errors = []

    for tgt in targets:
        logger.info("--- Processing target language: %s ---", tgt)
        tgt_dir = _DATA_ROOT / tgt
        tgt_symbolic = tgt_dir / args.subfolder

        tgt_symbolic.mkdir(parents=True, exist_ok=True)

        # Load target replacements (needed for render validation)
        rep_src = src_dir / "replacements.json"
        rep_tgt = tgt_dir / "replacements.json"
        if rep_src.exists() and (args.overwrite or not rep_tgt.exists()):
            logger.info("Translating replacements.json (%s → %s)", args.src, tgt)
            with rep_src.open(encoding="utf-8") as f:
                src_replacements = json.load(f)
            tgt_replacements = translate_replacements(client, src_replacements, args.src, tgt, args.model)
            with rep_tgt.open("w", encoding="utf-8") as f:
                json.dump(tgt_replacements, f, ensure_ascii=False, indent=4)
            logger.info("Written %s", rep_tgt)
        else:
            logger.info("Skipping replacements.json (already exists or no source)")

        tgt_replacements = json.loads(rep_tgt.read_text(encoding="utf-8")) if rep_tgt.exists() else {}

        # Translate / fix templates
        template_files = sorted((src_dir / args.subfolder).glob("*.toml"))
        errors: list[tuple[str, list[str]]] = []
        total_files = len(template_files)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(
                    process_template, i, src_file, client, args, tgt, tgt_symbolic, tgt_replacements, total_files
                )
                for i, src_file in enumerate(template_files)
            ]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    errors.append(res)

        if errors:
            logger.warning("\n%d templates had unresolved issues in %s:", len(errors), tgt)
            for name, issues in errors:
                logger.warning("  %s: %s", name, "; ".join(issues))
            overall_errors.extend([(tgt, name, issues) for name, issues in errors])
        else:
            logger.info("All %d templates translated and verified successfully for %s.", len(template_files), tgt)

    if overall_errors:
        logger.warning("\n--- OVERALL ERRORS ---")
        for tgt, name, issues in overall_errors:
            logger.warning("[%s] %s: %s", tgt, name, "; ".join(issues))


if __name__ == "__main__":
    main()
