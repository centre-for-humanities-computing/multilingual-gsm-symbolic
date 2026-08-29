"""Generate synthetic English train templates using separate opencode instances.

Every model call spawns a fresh `opencode run` process (a separate opencode
instance) so the free x-preview-f-free model can be used without limits.
"""

import json
import os
import re
import subprocess
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

from multilingual_gsm_symbolic._helpers import format_numbers_by_language
from multilingual_gsm_symbolic.load_data import load_replacements
from multilingual_gsm_symbolic.templates import AnnotatedQuestion
from multilingual_gsm_symbolic.validation import validate_template_against_pytest_checks

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "train"
MODEL = os.getenv("SYNTHETIC_MODEL", "opencode/x-preview-f-free")
CREATION = f"machine-generated from GSM8K using {MODEL} via opencode"
OPENCODE_BIN = os.getenv(
    "OPENCODE_BIN",
    r"C:\Users\riley\AppData\Local\Programs\nodejs\node_modules\opencode-ai\bin\opencode.exe",
)
MAX_TEMPLATE_ATTEMPTS = int(os.getenv("SYNTHETIC_MAX_ATTEMPTS", "10"))
SURFACE_ATTEMPTS = int(os.getenv("SYNTHETIC_SURFACE_ATTEMPTS", "6"))
MAX_JOBS = int(os.getenv("SYNTHETIC_MAX_JOBS", "100"))
WORKERS = int(os.getenv("SYNTHETIC_TEMPLATE_WORKERS", "6"))
CALL_TIMEOUT_SECONDS = float(os.getenv("SYNTHETIC_CALL_TIMEOUT", "420"))
FIDELITY = os.getenv("SYNTHETIC_FIDELITY", "surface").lower()
if FIDELITY not in {"surface", "answer"}:
    raise RuntimeError("SYNTHETIC_FIDELITY must be 'surface' or 'answer'.")

templates_dir = Path(__file__).parent / "src" / "multilingual_gsm_symbolic" / "data" / "templates" / "eng" / "test"
output_dir = templates_dir.parent / "train"
log_path = output_dir / "generation_log.jsonl"
replacements_path = templates_dir.parent / "replacements.json"

# ---------------------------------------------------------------------------
# Gather style examples and already-generated source ids.
# ---------------------------------------------------------------------------

example_texts: list[str] = []
generated_source_ids: set[int] = set()
template_numbers: list[int] = []

for directory in (templates_dir, output_dir):
    if not directory.exists():
        continue
    for p in sorted(directory.glob("*.toml")):
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if len(example_texts) < 3 and p.parent == templates_dir:
            example_texts.append(txt)
        if p.parent != output_dir:
            continue
        try:
            data = tomllib.loads(txt)
            id_orig = int(data["id_orig"])
        except (KeyError, TypeError, ValueError):
            print(f"Warning: {p} missing id_orig, skipping")
            continue
        if data.get("ignore"):
            # Failed templates are repairable: treat them as not yet generated.
            print(f"Note: {p.name} is marked ignore=true; will regenerate (repair pass)")
            continue
        if str(data.get("creation", "")).startswith("machine-generated from GSM8K"):
            generated_source_ids.add(id_orig)
        template_numbers.append(int(p.stem))

replacements = load_replacements("eng")

replacement_summary_lines = []
for name, values in replacements.items():
    preview = ", ".join(json.dumps(v, ensure_ascii=False) for v in list(values)[:4])
    replacement_summary_lines.append(f"- {name}: [{preview}, ...] ({len(values)} entries)")
REPLACEMENT_SUMMARY = "\n".join(replacement_summary_lines)

STYLE_EXAMPLES = "\n\n".join(example_texts)

# A fully worked input/output pair showing the exact expected response format.
FEW_SHOT = """\
Example input:
{"source_id": 0,
 "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
 "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\\n#### 72"}

Example correct response (body only - exactly this TOML, nothing else):
question_annotated = \"\"\"
{name,Natalia} sold clips to {a,48} of her friends in April, and then she sold half as many clips in May. How many clips did {name,Natalia} sell altogether in April and May?

#init:
- name = sample(names)
- $a = range(10, 210, 2)

#conditions:
- a > 0

#answer: a + a//2
\"\"\"

answer_annotated = \"\"\"
{name} sold {a}/2 = <<{a}/2={a//2}>>{a//2} clips in May.
{name} sold {a}+{a//2} = <<{a}+{a//2}={a+a//2}>>{a+a//2} clips altogether in April and May.
#### {a+a//2}
\"\"\""""

SYSTEM_PROMPT = f"""\
You are creating one new English symbolic TOML template for the multilingual-gsm-symbolic project.

RESPONSE CONTRACT (violating this fails instantly):
- Respond with plain TOML only: exactly two keys, `question_annotated` first, then `answer_annotated`, both TOML multiline basic strings (\"\"\"...\"\"\").
- Do NOT return `question`, `answer`, `id_orig`, `id_shuffled`, `creation`, `language`, markdown fences, explanations, or any text before/after the TOML.

TEMPLATE FORMAT (see style examples below):
- `question_annotated`: line 1 is the question with placeholders; then an `#init:` block; then optional `#conditions:` block; then `#answer: <expression>`.
- `answer_annotated`: ONLY the worked solution template (no #init/#conditions/#answer).

FIDELITY RULE (most important):
- The question line must equal the source question verbatim except that exact numeric/name spans may be replaced by placeholders `{{var,default}}`. Keep every other word, punctuation mark, and their order identical. The default value must be the original value from the source.
- The answer template must equal the source answer verbatim except numbers replaced by `{{expr}}` placeholders that evaluate to the original number at the defaults. Copy the source wording line by line; keep all `<<calculation=result>>` markers exactly where they are, e.g. source `<<48/2=24>>24` becomes `<<{{a}}/2={{a//2}}>>{{a//2}}`.
- Every `<<lhs=rhs>>` must satisfy eval(lhs) == rhs numerically after substitution.
- End the answer template with the final line `#### {{<#answer expression>}}`.

PLACEHOLDER RULES:
- Question placeholders are `{{variable,default}}` (default = the source value). Repeated uses of one variable repeat the same default.
- Answer placeholders are `{{expression}}` only - never `{{variable,default}}` in the answer. A person's name inside the answer uses the bare form `{{name}}` (it evaluates to the sampled name).
- Every placeholder variable must be defined in #init.

INIT RULES:
- First variable on each init line is prefixed with `$`, e.g. `- $a = range(10, 210, 2)`.
- String variables use `- name = sample(names)` or `- name = sample(["Alice", "Bob"])` (no $ prefix needed for sample of strings? ALWAYS include $ on numeric ranges; for strings write without $).
- Each init line defines exactly ONE variable; right side must be iterable (`range(...)`, `arange(...)`, or one-argument `sample([...])`). For a fixed value use `range(12, 13)` or `sample([12])`, never a bare literal.
- Init lines CANNOT reference variables defined in other init lines.
- `range(start, end[, step])` has Python-style EXCLUSIVE end. Never `range(x, x)`.
- Keep ranges modest but ensure AT LEAST 100 distinct valid numeric combinations and AT MOST 100000 (count = product of option counts after conditions filter).

CONDITIONS RULES:
- Conditions are boolean expressions over init variables only, one per `- ` line, e.g. `- divides(a, 2)` or `- a * b < 10000`.
- Allowed helpers in conditions/answers: divides(a, b), is_int(x), int(x), float(x), str(x), round(x), len(x), Fraction(x), plus arithmetic/comparisons/and/or/not. No subscripts, no dict lookups, no format specs like :.2f, no ^, no assignments.
- Add conditions to guarantee integer results and positivity (e.g. divisibility).

ANSWER EXPRESSION (#answer:) must be a single arithmetic expression over init variables equal to the original #### value at the defaults.

VARIABLE NAMING: short lowercase names matching what they hold (a, b, price, rate, mins, name). If the question mentions a person's name, create `name = sample(names_female)` / `sample(names_male)` / `sample(names)` as appropriate; use `{{name,default}}` in the question and bare `{{name}}` in the answer.

Available English replacement lists (use these names AS-IS inside sample(); do not invent others):
{REPLACEMENT_SUMMARY}

Parser helpers available: range(start,end[,step]) exclusive end; arange(start,end[,step]) decimal strings; sample(items) picks one item.

SELF-CHECK BEFORE RESPONDING:
1. Question line == source question with ONLY value spans swapped for {{var,source_value}} placeholders.
2. All defaults reproduce the source question AND source answer character-for-character (after substitution), including <<...>> blocks and #### value.
3. Every <<lhs=rhs>> evaluates correctly at the defaults and at ALL sampled combinations.
4. >= 100 and <= 100000 valid numeric combinations.
5. One variable per init line, $ prefix, no cross-references, exclusive range ends, conditions boolean-only.

Style examples of complete existing templates (match this style):
{STYLE_EXAMPLES}

{FEW_SHOT}
"""

_QUESTION_PLACEHOLDER_RE = re.compile(r"\{([^},]+)(?:,[^}]+)?\}")
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_RE_FENCE = re.compile(r"^\s*```[^\n`]*\n|\n```\s*$")


def call_model(prompt: str) -> str:
    """Spawn a separate headless opencode instance and return its response."""
    # Redirect through files instead of pipes: killed children can leave
    # grandchildren holding pipe handles open, which would block forever.
    import tempfile

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as out, tempfile.TemporaryFile(
        mode="w+", encoding="utf-8", errors="replace"
    ) as err:
        proc = subprocess.Popen(
            [OPENCODE_BIN, "run", "-m", MODEL, prompt],
            stdout=out,
            stderr=err,
            creationflags=0x00000200,  # CREATE_NEW_PROCESS_GROUP
        )
        try:
            returncode = proc.wait(timeout=CALL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
            )
            raise RuntimeError(f"opencode call timed out after {CALL_TIMEOUT_SECONDS}s")
        out.seek(0)
        err.seek(0)
        stderr_text = err.read()
        if returncode != 0:
            raise RuntimeError(f"opencode exited with code {returncode}: {stderr_text[-500:]}")
        return out.read()


def _is_formatting_mismatch_error(error: Exception) -> bool:
    return "doesn't match original" in str(error)


def _retry_instruction(error: Exception) -> tuple[str, str]:
    message = str(error)
    if _is_formatting_mismatch_error(error):
        which = "question" if "Formatted question" in message else "answer"
        return (
            "fidelity",
            f"The rendered {which} did not match the source. Copy the source {which} verbatim; replace ONLY exact "
            "value spans with placeholders. Keep <<lhs=rhs>> markers and every word identical.",
        )
    if any(
        token in message
        for token in (
            "bare question placeholder",
            "Example assignment",
            "Could not derive value",
            "not found in assignments",
            "is not defined",
            "Multiple default candidates",
        )
    ):
        return (
            "defaults",
            "Every placeholder needs `{var,default}` with default reachable from its own single-variable #init line. "
            "Make sure each default is produced by its range/sample and satisfies every condition.",
        )
    if "combinations" in message or "Too many combinations" in message:
        return (
            "combinations",
            "Use independent modest ranges yielding between 100 and 100000 numeric combinations after conditions.",
        )
    if any(
        token in message
        for token in (
            "Unsupported AST",
            "invalid syntax",
            "invalid decimal",
            "Unterminated string",
            "index out of range",
            "can only concatenate",
            "missing '#init:'",
            "missing '#answer:'",
            "derived variable",
        )
    ):
        return (
            "syntax",
            "Stay in the safe subset: one variable per #init line; only range, arange, or one-argument sample; no "
            "subscripts/format specs/bitwise operators/tuples; put #init, #conditions and #answer in question_annotated.",
        )
    if "computed" in message and "expected" in message:
        return (
            "chevron",
            "Some <<lhs=rhs>> block computes the wrong number. Ensure the right-hand expression equals the left-hand "
            "evaluation at the defaults, e.g. <<{a}/2={a//2}>>{a//2}.",
        )
    return (
        "validation",
        "Return a fresh template in the safe subset. Define every used variable in a single-variable #init line and "
        "make its defaults exactly reproduce the source.",
    )


def validate_question_annotated_placeholders(template: AnnotatedQuestion) -> None:
    defined_variables = {variable.lstrip("$") for variable in template.variables}
    for match in _QUESTION_PLACEHOLDER_RE.finditer(template.question_template):
        placeholder = match.group(0)
        variable = match.group(1).strip()
        if "," not in placeholder:
            raise ValueError(
                f"question_annotated has bare question placeholder missing a default value: {placeholder}; "
                f"include a default value like {{{variable},default}} or remove the placeholder."
            )
        if variable not in defined_variables:
            raise ValueError(f"question_annotated uses {placeholder}, but {variable} is not defined in #init:.")


def annotated_question_from_toml_text(text: str) -> AnnotatedQuestion:
    data = tomllib.loads(text)
    for key in ("question", "answer", "question_annotated", "answer_annotated"):
        if key in data and isinstance(data[key], str):
            data[key] = data[key].strip("\n")
    data.pop("ignore", None)
    template = AnnotatedQuestion(**data)
    validate_question_annotated_placeholders(template)
    return template


def _toml_multiline_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return '"""\n' + escaped + '\n"""'


def build_template_toml(example: dict, source_id: int, id_shuffled: int, body: str) -> str:
    return (
        f"question = {_toml_multiline_string(example['question'])}\n\n"
        f"answer = {_toml_multiline_string(example['answer'])}\n\n"
        f"id_orig = {source_id}\n"
        f"id_shuffled = {id_shuffled}\n\n"
        f"{body.strip()}\n\n"
        f'creation = "{CREATION}"\n\n'
        'language = "eng"'
    )


def serialize_template(template: AnnotatedQuestion) -> str:
    return (
        f"question = {_toml_multiline_string(template.question)}\n\n"
        f"answer = {_toml_multiline_string(template.answer)}\n\n"
        f"id_orig = {template.id_orig}\n"
        f"id_shuffled = {template.id_shuffled}\n\n"
        f"question_annotated = {_toml_multiline_string(template.question_annotated)}\n\n"
        f"answer_annotated = {_toml_multiline_string(template.answer_annotated)}\n\n"
        f'creation = "{template.creation}"\n\n'
        f'language = "{template.language}"'
    )


def clean_model_output(output_text: str) -> str:
    text = _RE_THINK.sub("", output_text)
    text = re.sub(r"^\s*```(?:toml)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    start = text.find("question_annotated")
    if start > 0:
        text = text[start:]
    return text.strip()


def create_template(source_id: int, example: dict, id_shuffled: int) -> dict:
    user_prompt = {
        "source_id": source_id,
        "question": example["question"],
        "answer": example["answer"],
    }
    base_prompt = SYSTEM_PROMPT + "\n\nNow generate the template body for this input (respond with TOML only):\n" + json.dumps(
        user_prompt, ensure_ascii=False
    )

    last_error = None
    last_failure_kind = None
    last_template_text = None
    attempts = []
    answer_rebased = False
    for attempt in range(1, MAX_TEMPLATE_ATTEMPTS + 1):
        print(f"[id {source_id}] attempt {attempt}/{MAX_TEMPLATE_ATTEMPTS}", flush=True)
        prompt = base_prompt
        if last_error is not None:
            kind, instruction = _retry_instruction(last_error)
            prompt = (
                base_prompt
                + "\n\nA previous candidate failed validation. Return a completely new template that fixes the "
                f"difference shown below; do not quote the failed one.\nFailure category: {kind}\nRepair "
                f"instruction: {instruction}\n\nValidation error details (expected vs rendered):\n"
                + str(last_error)[:1500]
            )
        fidelity_this_attempt = FIDELITY
        if FIDELITY == "surface" and attempt > SURFACE_ATTEMPTS:
            # Last resort per project decision: rebase the stored answer onto the
            # template rendering so formatting checks pass.
            fidelity_this_attempt = "answer"
        try:
            raw = call_model(prompt)
        except subprocess.TimeoutExpired:
            last_error = RuntimeError(f"opencode call timed out after {CALL_TIMEOUT_SECONDS}s")
            last_failure_kind = "api"
            attempts.append({"attempt": attempt, "error": str(last_error), "failure_kind": "api"})
            time.sleep(min(20 * attempt, 120))
            continue
        except Exception as e:
            last_error = e
            last_failure_kind = "api"
            attempts.append({"attempt": attempt, "error": str(e)[:300], "failure_kind": "api"})
            # Back off on API errors so a throttled provider can recover.
            time.sleep(min(20 * attempt, 120))
            continue

        body = clean_model_output(raw)
        if "question_annotated" not in body or "answer_annotated" not in body:
            last_error = RuntimeError("Response did not contain question_annotated/answer_annotated TOML.")
            last_failure_kind = "format"
            attempts.append({"attempt": attempt, "error": str(last_error), "failure_kind": "format"})
            continue

        template_text = build_template_toml(example, source_id, id_shuffled, body)
        last_template_text = template_text
        attempt_log: dict = {"attempt": attempt}

        try:
            template = annotated_question_from_toml_text(template_text)
            validate_template_against_pytest_checks(
                template, replacements, source=f"{id_shuffled:04d}.toml", fidelity=fidelity_this_attempt
            )
            template.generate_questions(n=10, replacements=replacements, verbose=False)
        except Exception as e:
            last_error = e
            last_failure_kind, _ = _retry_instruction(e)
            attempt_log["error"] = str(e)[:2000]
            attempt_log["failure_kind"] = last_failure_kind
            attempts.append(attempt_log)
            print(f"[id {source_id}] validation failure ({last_failure_kind}): {str(e)[:200]}", flush=True)
            continue

        attempts.append(attempt_log)
        answer_rebased = template.answer.strip() != example["answer"].strip()
        log_entry = {"source_id": source_id, "id_shuffled": id_shuffled, "success": True, "attempts": attempts}
        if answer_rebased:
            log_entry["answer_rebased"] = True
        return {
            "template_text": serialize_template(template),
            "log_entry": log_entry,
        }

    print(f"[id {source_id}] FAILED after {MAX_TEMPLATE_ATTEMPTS} attempts: {str(last_error)[:200]}", flush=True)
    if last_template_text:
        fallback_text = (
            last_template_text
            + f"\n\nignore = true  # failed after {MAX_TEMPLATE_ATTEMPTS} attempts: {last_error}"
        )
    else:
        fallback_text = f'ignore = true  # failed: "{str(last_error)[:200]}"'

    return {
        "template_text": fallback_text,
        "log_entry": {
            "source_id": source_id,
            "id_shuffled": id_shuffled,
            "success": False,
            "error": str(last_error),
            "failure_kind": last_failure_kind,
            "attempts": attempts,
        },
    }


def main() -> None:
    status_file = Path(__file__).parent / "logs" / "driver_status.json"
    gsm8k_train = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split=GSM8K_SPLIT)
    next_template_number = max(template_numbers, default=-1) + 1

    jobs = []
    for source_id, example in enumerate(gsm8k_train):
        if source_id in generated_source_ids:
            continue
        # Normalize stored text through the eng number formatter so the rendered
        # output (which inserts thousands separators for >=10000) matches verbatim.
        example = {
            "question": format_numbers_by_language(example["question"], "eng"),
            "answer": format_numbers_by_language(example["answer"], "eng"),
        }
        jobs.append((source_id, example, next_template_number + len(jobs)))
        if len(jobs) >= MAX_JOBS:
            break

    remaining_total = sum(1 for sid in range(len(gsm8k_train)) if sid not in generated_source_ids)
    status_file.parent.mkdir(exist_ok=True)
    try:
        _run_jobs(jobs, next_template_number)
        if not jobs:
            print("[repair] All English questions templatized.", flush=True)
    finally:
        status_file.write_text(
            json.dumps({"jobs": len(jobs), "remaining_total": remaining_total}),
            encoding="utf-8",
        )
    print(f"STATUS: batch={len(jobs)} remaining_total={remaining_total} done_on_disk={len(template_numbers)}")


def _run_jobs(jobs: list, next_template_number: int) -> None:
    print(f"Generating {len(jobs)} templates (ids {next_template_number}..{next_template_number + len(jobs) - 1})")
    output_dir.mkdir(exist_ok=True)
    successes = failures = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(create_template, *job): job for job in jobs}
        for future in as_completed(futures):
            source_id, _, id_shuffled = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {
                    "template_text": None,
                    "log_entry": {"source_id": source_id, "id_shuffled": id_shuffled, "success": False, "error": str(e)},
                }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result["log_entry"], ensure_ascii=False) + "\n")
            toml = result["template_text"]
            if toml is None:
                failures += 1
                print(f"Skipping id_shuffled={id_shuffled} due to generation error", flush=True)
                continue
            if result["log_entry"].get("success"):
                successes += 1
            else:
                failures += 1
            (output_dir / f"{id_shuffled:04d}.toml").write_text(toml.strip() + "\n", encoding="utf-8")
            status = "OK " if result["log_entry"].get("success") else "IGN"
            print(f"{status} wrote {output_dir / f'{id_shuffled:04d}.toml'} (source {source_id})", flush=True)

    total = successes + failures
    rate = failures / total * 100 if total else 0.0
    print(f"DONE batch: {successes}/{total} succeeded, {failures} failed ({rate:.1f}% error rate)")


if __name__ == "__main__":
    main()
