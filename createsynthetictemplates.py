import json
import os
import re
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion
from multilingual_gsm_symbolic.load_data import load_replacements

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "train"
MODEL = "gpt-5.4-nano-2026-03-17"
CREATION = f"machine-generated from GSM8K using {MODEL}"
MAX_TEMPLATE_ATTEMPTS = 3


# gather generated template ids and aggregate toml texts
generated_source_ids = set()
template_numbers = []

toml_text = ""
templates_dir = Path(__file__).parent / "src" / "multilingual_gsm_symbolic" / "data" / "templates" / "eng" / "test"
output_dir = templates_dir.parent / "train"
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

Return plain text for a single TOML template that meets the following requirements:

Here are existing templates to match the project style:
{toml_text}

Here are the available English replacement list names from replacements.json. Use these list names as-is; do not invent replacement list names:
{replacement_keys}

Template rules:
- The TOML must contain the following top-level keys in this exact order: question, answer, id_orig, id_shuffled, question_annotated, answer_annotated, creation, language.
- The concrete question and answer must match the supplied GSM8K item exactly, including the final #### answer line. These shoul include the original values from the GSM8K, with no placeholder syntax.
- Use placeholders like {{variable,default_value}} in question_annotated. Defaults must render the exact concrete question.
- Use bare {{variable}} or Python expressions in answer_annotated. These must render the exact concrete answer.
- In init, prefix numeric sampled variables with $, for example: "$price = range(1, 20)".
- Every placeholder default value must be reachable from that variable's init expression. For example, {{price,20}} must have an init expression that can produce 20.
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
- If the question involves a currency you must samples from the currecy replacement list and define a variable for the currency
- All variables must be defined in init. Do not use undefined variables or variables that are only defined in the question_annotated. Before finalizing the TOML, check that all variables used in question_annotated and answer_annotated are defined in init.
"""


_QUESTION_PLACEHOLDER_RE = re.compile(r"\{([^},]+)(?:,[^}]+)?\}")


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
    if "init" in data:
        data["init_section"] = data.pop("init")
    if "conditions" in data:
        data["conditions_section"] = data.pop("conditions")
    data.pop("ignore", None)
    template = AnnotatedQuestion(**data)
    validate_question_annotated_placeholders(template)
    return template


def create_template(source_id: int, example: dict, id_shuffled: int) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        api_key = api_key.encode("ascii", "ignore").decode()

    client = OpenAI(api_key=api_key)
    user_prompt = {
        "source_id": source_id,
        "id_orig": source_id,
        "id_shuffled": id_shuffled,
        "creation": CREATION,
        "language": "eng",
        "question": example["question"],
        "answer": example["answer"],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_prompt)},
    ]
    replacements = load_replacements("eng")

    last_error = None
    for attempt in range(1, MAX_TEMPLATE_ATTEMPTS + 1):
        response = client.responses.create(
            model=MODEL,
            reasoning={"effort": "medium"},
            input=messages,
        )

        try:
            annotated_question_from_toml_text(response.output_text).generate_questions(
                n=10, replacements=replacements, verbose=False
            )
        except Exception as e:
            last_error = e
            print(f"Error generating questions for id {source_id} on attempt {attempt}/{MAX_TEMPLATE_ATTEMPTS}: {e}")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous TOML template failed validation with this error:\n"
                        f"{e}\n\n"
                        "Return a corrected single TOML template only."
                    ),
                }
            )
            continue

        return response.output_text

    print(f"Error generating questions after {MAX_TEMPLATE_ATTEMPTS} attempts: {last_error}")
    return (
        response.output_text
        + "\n\n"
        + "ignore = true  # failed after "
        + str(MAX_TEMPLATE_ATTEMPTS)
        + " attempts: "
        + str(last_error)
    )


gsm8k_train = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split=GSM8K_SPLIT)
next_template_number = max(template_numbers, default=-1) + 1

jobs = []
for source_id, example in enumerate(gsm8k_train):
    if source_id in generated_source_ids:
        continue
    if len(jobs) >= 100:
        break
    jobs.append((source_id, example, next_template_number + len(jobs)))

output_dir.mkdir(exist_ok=True)
with ThreadPoolExecutor(max_workers=int(os.getenv("SYNTHETIC_TEMPLATE_WORKERS", "5"))) as executor:
    futures = {executor.submit(create_template, *job): job for job in jobs}
    for future in as_completed(futures):
        _, _, id_shuffled = futures[future]
        toml = future.result()
        if toml is None:
            print(f"Skipping template with id_shuffled={id_shuffled} due to generation error")
            continue
        (output_dir / f"{id_shuffled:04d}.toml").write_text(toml.strip() + "\n", encoding="utf-8")
        print(
            f"Wrote {output_dir / f'{id_shuffled:04d}.toml'} for source id {futures[future][0]} and id_shuffled {id_shuffled}"
        )
