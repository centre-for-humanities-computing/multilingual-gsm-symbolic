# multilingual-gsm-symbolic

A Python package for generating diverse multilingual math word problems from symbolic templates.
It ports the symbolic GSM (Grade School Math) engine originally developed as part of the
[m-gsm-symbolic](https://github.com/centre-for-humanities-computing/m-gsm-symbolic) project.

Given a symbolic template like:

```
A fog bank rolls in over a city at {speed,3} miles/hour.
The city is {width,14} miles wide.
#init:
- $speed = range(1, 20)
- $width = range(2, 100)
#conditions:
- is_int(width / speed)
#answer: width // speed
```

the package generates many concrete, valid problem instances by systematically sampling
variable values that satisfy the declared constraints.

## Installation

```bash
pip install multilingual-gsm-symbolic
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add multilingual-gsm-symbolic
```

## Quickstart

```python
from multilingual_gsm_symbolic import load_data, load_replacements

# Load English templates (default)
templates = load_data()          # 100 English templates
templates_dan = load_data("dan") # 100 Danish templates

# Load language-specific replacement values (used in some templates)
replacements = load_replacements()       # English
replacements_dan = load_replacements("dan")

# Generate concrete questions from a template
template = templates[0]
questions = template.generate_questions(n=10, language="eng", replacements=replacements)

for q in questions:
    print(q.question)
    print(q.answer)
    print()
```

You can also load the bundled concrete GSM problems directly:

```python
from multilingual_gsm_symbolic import load_gsm_eng, load_gsm_dan

problems = load_gsm_eng()  # list of GSMProblem
for p in problems[:3]:
    print(p.id_orig, p.question[:60])
```

## Template format

Templates are JSON files with four fields:

| Field | Description |
|---|---|
| `question` | Concrete question (the original example) |
| `answer` | Concrete answer with calculation steps |
| `question_annotated` | Template with variable placeholders and `#init` / `#conditions` / `#answer` sections |
| `answer_annotated` | Answer template with inline expressions |

### Annotated question syntax

```
{variable, default_value}   — placeholder in the question text
#init:
- $var = range(low, high)   — variable sampled from a range
- $var = sample([a, b, c])  — variable sampled from a list
#conditions:
- is_int(x / y)             — constraint that must hold for a combination to be valid
#answer: x * y + z          — Python expression evaluated to produce the numeric answer
```

### Available helper functions

| Function | Description |
|---|---|
| `range(start, end[, step])` | All integers in `[start, end)` |
| `sample([a, b, c])` | One value from the list |
| `range_sample(start, end, step)` | Uniform sample from a range |
| `sample_sequential(items, n)` | `n` consecutive items from a list |
| `arange_sample(start, end, step)` | Sample from `np.arange(start, end, step)` |
| `is_int(x)` | True if `x` is an integer |
| `divides(a, b)` | True if `a` divides `b` |
| `frac_format(x)` | Format `x` as a fraction string |

## API reference

### `load_data(language="eng", directory=None) → list[AnnotatedQuestion]`

Load symbolic templates.

- `language` — `"eng"` (default) or `"dan"`, or any language code for which a template folder exists
- `directory` — override the bundled data; load templates from this path instead

### `load_replacements(language="eng") → dict`

Load language-specific named values (e.g. lists of names, places) used inside templates.

### `load_gsm_eng(directory_path=...) → list[GSMProblem]`
### `load_gsm_dan(directory_path=...) → list[GSMProblem]`

Load the 100 bundled concrete problems for English / Danish.

### `AnnotatedQuestion`

Core class. Constructed from a JSON template file via `AnnotatedQuestion.from_json(path)`.

Key methods:

| Method | Description |
|---|---|
| `generate_questions(n, language, replacements)` | Generate `n` concrete `Question` instances |
| `get_default_assignments(replacements)` | Extract the example variable values from the template |
| `format_question(assignments, language)` | Render the question text for a given assignment |
| `format_answer(assignments, language)` | Render the answer text for a given assignment |

### `Question`

Dataclass holding a single generated problem: `question`, `answer`, `id_orig`, `id_shuffled`.

### `GSMProblem`

Pydantic model for a concrete problem loaded from disk: `question`, `answer`, `id_orig`, `filepath`.

## Data

The package ships with **100 English** and **100 Danish** symbolic templates derived from
[GSM8k](https://huggingface.co/datasets/openai/gsm8k) and localized into Danish as part of
the [m-gsm-symbolic](https://github.com/centre-for-humanities-computing/m-gsm-symbolic) project.

## Development

```bash
# Install with dev dependencies
make install

# Run tests (405 tests across all bundled templates)
make test

# Lint
make lint
```
