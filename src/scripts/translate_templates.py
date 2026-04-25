# /// script
# dependencies = ["openai"]
# ///
"""Translate symbolic templates and replacements between languages using GPT.

Translates all natural-language fields in a single call (ensuring consistent
terminology across fields) and validates each template using the same checks
as the test suite. Templates that fail validation are retried up to 3 times.

Usage:
    uv run src/scripts/translate_templates.py --from dan --to nob
    uv run src/scripts/translate_templates.py --from eng --to fra --model gpt-5.4
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from openai import OpenAI

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DATA_ROOT = Path("src/multilingual_gsm_symbolic/data/templates")

_LANGUAGE_NAMES = {
    "eng": "English",
    "dan": "Danish",
    "nob": "Norwegian Bokmål",
    "nno": "Norwegian Nynorsk",
    "swe": "Swedish",
    "deu": "German",
    "fra": "French",
    "nld": "Dutch",
    "fin": "Finnish",
    "isl": "Icelandic",
    "spa": "Spanish",
    "ita": "Italian",
    "pol": "Polish",
    "por": "Portuguese",
}

_TRANSLATE_FIELDS = ("question", "answer", "question_annotated", "answer_annotated")

_SYSTEM_PROMPT = """\
You are a precise translator of mathematical word problems from {src_name} to {tgt_name}.

You will receive a JSON object with fields to translate. Return a JSON object with the same keys
and translated values. All four fields share the same problem — use consistent terminology
throughout (e.g. the same word for every object, same names in every field).

Rules:
1. Translate all natural-language prose, including default values inside placeholders.
   - Variable placeholders have the form {{varname,default}} — keep the {{varname,...}} syntax but translate the default value. E.g. {{animal,haj}} → {{animal,hai}} when translating to Norwegian. Default values are mid-sentence fragments: use lowercase and the uninflected/indefinite form (e.g. {{unit,måned}} not {{unit,Måned}} or {{unit,måneden}}).
   - Bare variable references {{varname}} (no default) — leave completely unchanged.
   - Init / conditions / answer blocks: lines starting with #init:, #conditions:, #answer: — do NOT alter these lines at all.
   - Init expressions: range(...), sample(...), arange(...), etc. — do NOT alter.
   - Condition expressions: is_int(...), divides(...), True, etc. — do NOT alter.
   - Inline calc tags in answers: <<expr=result>> — copy them EXACTLY as they appear in the source, character for character.
   - The #### answer marker and its number — do NOT alter.
   - Numbers, units, currency symbols, fractions — do NOT alter.
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
        temperature=0.2,
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
        temperature=0.4,
    )
    raw = response.choices[0].message.content.strip()
    messages = messages + [{"role": "assistant", "content": raw}]
    return json.loads(raw), messages


def _reconstruct_messages(src_data: dict, tgt_data: dict, src: str, tgt: str) -> list[dict]:
    """Build a synthetic conversation for an existing translation so retries use the same path."""
    messages = _build_initial_messages(src_data, src, tgt)
    tgt_payload = {f: tgt_data[f] for f in _TRANSLATE_FIELDS if f in tgt_data}
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


def translate_template(client: OpenAI, src_data: dict, src: str, tgt: str, model: str) -> tuple[dict, list[dict]]:
    tgt_data = dict(src_data)
    translated_fields, messages = translate_template_fields(client, src_data, src, tgt, model)
    tgt_data.update(translated_fields)
    tgt_data = _strip_answer_annotated_defaults(tgt_data)
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
        temperature=0.1,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate symbolic math templates between languages.")
    parser.add_argument("--from", dest="src", required=True, help="Source language code (e.g. dan)")
    parser.add_argument("--to", dest="tgt", required=True, help="Target language code (e.g. nob)")
    parser.add_argument("--model", default="gpt-5.4-nano", help="OpenAI model to use")
    parser.add_argument("--overwrite", action="store_true", help="Re-translate already existing files")
    parser.add_argument("--retries", type=int, default=2, help="Max retries for templates failing validation")
    args = parser.parse_args()

    src_dir = _DATA_ROOT / args.src
    tgt_dir = _DATA_ROOT / args.tgt
    tgt_symbolic = tgt_dir / "symbolic"

    if not src_dir.exists():
        raise SystemExit(f"Source language directory not found: {src_dir}")

    tgt_symbolic.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    # Load target replacements (needed for render validation)
    rep_src = src_dir / "replacements.json"
    rep_tgt = tgt_dir / "replacements.json"
    if rep_src.exists() and (args.overwrite or not rep_tgt.exists()):
        logger.info("Translating replacements.json (%s → %s)", args.src, args.tgt)
        with rep_src.open(encoding="utf-8") as f:
            src_replacements = json.load(f)
        tgt_replacements = translate_replacements(client, src_replacements, args.src, args.tgt, args.model)
        with rep_tgt.open("w", encoding="utf-8") as f:
            json.dump(tgt_replacements, f, ensure_ascii=False, indent=4)
        logger.info("Written %s", rep_tgt)
    else:
        logger.info("Skipping replacements.json (already exists)")

    tgt_replacements = json.loads(rep_tgt.read_text(encoding="utf-8")) if rep_tgt.exists() else {}

    # Translate / fix templates
    template_files = sorted((src_dir / "symbolic").glob("*.json"))
    errors: list[tuple[str, list[str]]] = []

    for i, src_file in enumerate(template_files):
        tgt_file = tgt_symbolic / src_file.name

        with src_file.open(encoding="utf-8") as f:
            src_data = json.load(f)

        # If translation already exists, validate it first; only redo if broken.
        if tgt_file.exists() and not args.overwrite:
            with tgt_file.open(encoding="utf-8") as f:
                tgt_data = json.load(f)
            issues = verify_syntax(src_data, tgt_data) + verify_renders(tgt_data, tgt_replacements)
            if not issues:
                logger.info("[%d/%d] %s OK (skipping)", i + 1, len(template_files), src_file.name)
                continue
            logger.warning(
                "[%d/%d] %s has issues, fixing: %s", i + 1, len(template_files), src_file.name, "; ".join(issues)
            )
            messages = _reconstruct_messages(src_data, tgt_data, args.src, args.tgt)
            feedback = "\n".join(issues)
        else:
            logger.info("[%d/%d] Translating %s", i + 1, len(template_files), src_file.name)
            tgt_data, messages = translate_template(client, src_data, args.src, args.tgt, args.model)
            issues = verify_syntax(src_data, tgt_data) + verify_renders(tgt_data, tgt_replacements)
            feedback = "\n".join(issues)

        for attempt in range(1, args.retries + 1):
            if not issues:
                break
            logger.warning("  Attempt %d/%d failed, retrying with feedback", attempt, args.retries)
            time.sleep(1)
            try:
                translated_fields, messages = fix_template_fields(client, args.model, feedback, messages)
                tgt_data.update(translated_fields)
                tgt_data = _strip_answer_annotated_defaults(tgt_data)
                issues = verify_syntax(src_data, tgt_data) + verify_renders(tgt_data, tgt_replacements)
                feedback = "\n".join(issues)
            except Exception as e:
                issues = [f"retry error: {e}"]
                break

        if issues:
            logger.warning("  Unresolved issues in %s: %s", src_file.name, "; ".join(issues))
            errors.append((src_file.name, issues))
        else:
            logger.info("  OK")

        with tgt_file.open("w", encoding="utf-8") as f:
            json.dump(tgt_data, f, ensure_ascii=False, indent=2)

    if errors:
        logger.warning("\n%d templates had unresolved issues:", len(errors))
        for name, issues in errors:
            logger.warning("  %s: %s", name, "; ".join(issues))
    else:
        logger.info("All %d templates translated and verified successfully.", len(template_files))


if __name__ == "__main__":
    main()
