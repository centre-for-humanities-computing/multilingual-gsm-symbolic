import json
import os
import re
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from subprocess import Popen, TimeoutExpired
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

from datasets import load_dataset
from openai import OpenAI

from multilingual_gsm_symbolic.load_data import load_replacements
from multilingual_gsm_symbolic.templates import AnnotatedQuestion
from multilingual_gsm_symbolic.validation import (
    validate_template_against_pytest_checks,
)

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "train"
MODEL = "Qwen/Qwen3-235B-A22B-Thinking-2507"
CREATION = f"machine-generated from GSM8K using {MODEL}"
MAX_TEMPLATE_ATTEMPTS = 10
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
VLLM_HEALTH_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
VLLM_STARTUP_TIMEOUT_SECONDS = 600


# gather generated template ids and aggregate toml texts
generated_source_ids = set()
template_numbers = []

toml_text = ""
templates_dir = Path(__file__).parent / "src" / "multilingual_gsm_symbolic" / "data" / "templates" / "eng" / "test"
output_dir = templates_dir.parent / "train"
log_path = output_dir / "generation_log.jsonl"
replacements_path = templates_dir.parent / "replacements.json"
replacement_keys = list(json.loads(replacements_path.read_text(encoding="utf-8")))
template_text_count = 0
if templates_dir.exists():
    for p in sorted(templates_dir.glob("*.toml"))[:5]:
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        toml_text += "\n\n" + txt
        template_text_count += 1

if output_dir.exists():
    for p in sorted(output_dir.glob("*.toml")):
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if template_text_count < 5:
            toml_text += "\n\n" + txt
            template_text_count += 1
        try:
            data = tomllib.loads(txt)
            id_orig = data["id_orig"]
        except (KeyError, TypeError):
            print(f"Warning: {p} missing id_orig, skipping")
            continue
        id_orig = int(id_orig)

        if str(data.get("creation", "")).startswith("machine-generated from GSM8K"):
            generated_source_ids.add(id_orig)
        template_numbers.append(int(p.stem))


SYSTEM_PROMPT = f"""\
You are creating one new English symbolic TOML template for the multilingual-gsm-symbolic project.

Return plain text for only the variable middle of a single TOML template.
Do not return `question`, `answer`, `id_orig`, `id_shuffled`, `creation`, or `language` because those are added separately.
Return only `question_annotated` and `answer_annotated` in valid TOML format.
Following the existing project style, `#init`, `#conditions`, and `#answer` live inside `question_annotated`. `answer_annotated` should only contain the worked solution template.

Here are existing templates to match the project style:
{toml_text}

Here are the available English replacement list names from replacements.json. Use these list names as-is; do not invent replacement list names:
{replacement_keys}

Template rules:
- Use placeholders like {{variable,default_value}} in question_annotated. Defaults must render the exact concrete question. Use the same default in every occurrence of that variable's placeholder even if the placeholder appears multiple times. 
- Do not use placeholders in answer_annotated. Render the exact concrete answer in answer_annotated with no placeholders.
- Use bare {{variable}} or Python expressions in answer_annotated. These must render the exact concrete answer.
- In init, the first variable on each init line must be prefixed with $, for example: "$price = range(1, 20)".
- In init, every right-hand side must be iterable; for fixed values use `range(12, 13)` or `sample([12])`, not `12`.
- Every placeholder must have a default value. For example, {{price,20}}. This default must be reachable from that variable's init expression. 
- range(start, end[, step]) uses a Python-style exclusive end. Never emit range(x, x), because it has no possible values.
- String variables can use sample([...]) or use the project replacement lists
- Keep ranges modest and add conditions for divisibility, positivity, integer percentages, or other constraints.
- conditions must contain boolean expressions only. Do not put assignments or dictionary lookups there.
- Prefer clean integer arithmetic. Avoid final answers that rely on floating point rounding.
- Following the style of existing templates, sample enough values to ensure that the template can produce at least 100 question variations, but not more than 100 thousand question variations. 
- When sampling things that there are project replacement lists for always use those lists. 

Parser helpers available in synthetic templates:
- Init sampling helpers: range(start,end[,step]) uses Python-style exclusive end and must have at least one possible value; arange(start,end[,step]) samples a decimal string; sample(items[,n]) samples one item or n unique items; sample_sequential(items,n) samples n consecutive cyclic items; range_str(start,end,step,words) samples (word, number) from a 1-indexed word list.
- Init enumeration helpers: range_list(start,end[,step]) enumerates Python-style integers with exclusive end; list(x) converts iterables to lists.
- Never use range(x, x), because it has no possible values.
- Conditions and answer helpers: is_int(x) tests integer-valued numbers; divides(a,b) tests a % b == 0 and rejects b == 0; int(x), float(x), str(x), round(x), len(x), and Fraction(x) convert or format values.
- Expressions may use constants, variables, lists/tuples, arithmetic, comparisons, and/or/not. Do not invent helpers beyond this list.


- Do not invent unsupported helper functions or undefined replacement list names.
- Init lines cannot reference other variables defined in init. For example you CANNOT: define {{var1}} in init and then use {{var1}} in the init expression for {{var2}}. All init expressions must be independently executable without relying on other variables.
- If the question involves a currency you must sample from the currency replacement list and define a variable for the currency
- All variables must be defined in init. Do not use undefined variables or variables that are only defined in the question_annotated. Before finalizing the TOML, check that all variables used in question_annotated and answer_annotated are defined in init.
"""


_QUESTION_PLACEHOLDER_RE = re.compile(r"\{([^},]+)(?:,[^}]+)?\}")


@contextmanager
def local_vllm_server():
    process = Popen(
        [
            "vllm",
            "serve",
            MODEL,
            "--host",
            VLLM_HOST,
            "--port",
            str(VLLM_PORT),
            "--quantization",
            "nvfp4",
            "--max-model-len",
            "32768",
        ],
    )
    deadline = monotonic() + VLLM_STARTUP_TIMEOUT_SECONDS
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}.")
        try:
            with urlopen(VLLM_HEALTH_URL, timeout=1):
                break
        except URLError:
            if monotonic() >= deadline:
                process.terminate()
                process.wait()
                raise TimeoutError(f"vLLM did not become healthy within {VLLM_STARTUP_TIMEOUT_SECONDS} seconds.")
            sleep(1)
    try:
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except TimeoutExpired:
            process.kill()
            process.wait()


def _is_formatting_mismatch_error(error: Exception) -> bool:
    message = str(error)
    return message.startswith("Formatted question doesn't match original") or message.startswith(
        "Formatted answer doesn't match original"
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
    return '"""\n' + value.replace('"""', '\\"\\"\\"') + '\n"""'


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


def create_template(source_id: int, example: dict, id_shuffled: int) -> dict:
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    user_prompt = {
        "source_id": source_id,
        "question": example["question"],
        "answer": example["answer"],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_prompt)},
    ]
    replacements = load_replacements("eng")

    last_error = None
    last_template_text = None
    attempts = []
    for attempt in range(1, MAX_TEMPLATE_ATTEMPTS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        # strip markdown toml tags that the model may include, and also strip leading/trailing whitespace
        output_text = response.choices[0].message.content or ""
        body = re.sub(r"^\s*```[^\n`]*\n|\n```\s*$", "", output_text.strip())
        template_text = build_template_toml(example, source_id, id_shuffled, body)
        last_template_text = template_text
        attempt_log = {"attempt": attempt, "template_text": template_text}

        try:
            template = annotated_question_from_toml_text(template_text)
            validate_template_against_pytest_checks(template, replacements, source=f"{id_shuffled:04d}.toml")
            template.generate_questions(n=10, replacements=replacements, verbose=False)
        except Exception as e:
            last_error = e
            attempt_log["error"] = str(e)
            attempts.append(attempt_log)
            if _is_formatting_mismatch_error(e):
                print(
                    "Formatting mismatch for "
                    f"id {source_id} on attempt {attempt}/{MAX_TEMPLATE_ATTEMPTS}; "
                    "details sent back to the model for retry."
                )
            else:
                print(f"Error generating questions for id {source_id} on attempt {attempt}/{MAX_TEMPLATE_ATTEMPTS}: {e}")
            messages.append(
                {
                    "role": "assistant",
                    "content": output_text,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous TOML template failed validation.\n\n"
                        f"Validation error:\n{e}\n\n"
                        "Return corrected TOML for only `question_annotated` and `answer_annotated`."
                    ),
                }
            )
            continue

        attempts.append(attempt_log)
        return {
            "template_text": template_text,
            "log_entry": {"source_id": source_id, "id_shuffled": id_shuffled, "success": True, "attempts": attempts},
        }

    last_error = "formatting mismatch" if _is_formatting_mismatch_error(last_error) else last_error
    print(f"Error generating questions after {MAX_TEMPLATE_ATTEMPTS} attempts: {last_error}")
    
    return {
        "template_text": (
            last_template_text
            + "\n\n"
            + "ignore = true  # failed after "
            + str(MAX_TEMPLATE_ATTEMPTS)
            + " attempts: "
            + str(last_error)
        ),
        "log_entry": {
            "source_id": source_id,
            "id_shuffled": id_shuffled,
            "success": False,
            "error": str(last_error),
            "attempts": attempts,
        },
    }


gsm8k_train = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split=GSM8K_SPLIT)
next_template_number = max(template_numbers, default=-1) + 1

jobs = []
for source_id, example in enumerate(gsm8k_train):
    if source_id in generated_source_ids:
        continue
    if len(jobs) >= 90:
        break
    jobs.append((source_id, example, next_template_number + len(jobs)))

output_dir.mkdir(exist_ok=True)
with local_vllm_server(), ThreadPoolExecutor(max_workers=int(os.getenv("SYNTHETIC_TEMPLATE_WORKERS", "3"))) as executor:
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
            print(f"Skipping template with id_shuffled={id_shuffled} due to generation error")
            continue
        (output_dir / f"{id_shuffled:04d}.toml").write_text(toml.strip() + "\n", encoding="utf-8")
        print(f"Wrote {output_dir / f'{id_shuffled:04d}.toml'} for source id {source_id} and id_shuffled {id_shuffled}")
