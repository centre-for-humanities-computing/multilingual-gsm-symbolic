# multilingual-gsm-symbolic

[![tests](https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic/actions/workflows/tests.yml/badge.svg)](https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/multilingual-gsm-symbolic.svg?style=flat&logo=pypi&logoColor=white)](https://pypi.org/project/multilingual-gsm-symbolic/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/badge/type--checked-ty-blue?style=flat)](https://github.com/astral-sh/ty)


A Python package for generating synthetic multilingual math problems from symbolic templates. Allows you to create more than a thousand examples from just one problem and allows you to test if the LLMs actually understand the problem or whether it was just lucky pattern-matching.


![Example of a symbolic template and generated questions](https://raw.githubusercontent.com/centre-for-humanities-computing/multilingual-gsm-symbolic/main/images/example.png)

## ⏳ Installation

```bash
pip install multilingual-gsm-symbolic
```

## 👩‍💻 Get started

```python
from multilingual_gsm_symbolic import load_data, load_replacements, available_languages

# see possible languages
languages = available_languages()

lang = "eng"
print(languages[lang])
# {"number of samples": 100}

# Load English templates (default)
templates = load_data(lang)

# Load language-specific replacement values (used in some templates)
replacements = load_replacements(lang)

# Generate concrete questions from a template
template = templates[0]
questions = template.generate_questions(n=10, language="eng", replacements=replacements)

for q in questions:
    print(q.question)
    print(q.answer)
    print()
```

## 📋 Template format

Templates are JSON files with four fields:

| Field                | Description                                                                          |
| -------------------- | ------------------------------------------------------------------------------------ |
| `question`           | Concrete question (the original example)                                             |
| `answer`             | Concrete answer with calculation steps                                               |
| `question_annotated` | Template with variable placeholders and `#init` / `#conditions` / `#answer` sections |
| `answer_annotated`   | Answer template with inline expressions                                              |

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

<details>
<summary>Example: fog bank problem</summary>

```json
{
  "question": "A fog bank rolls in over a city at 3 miles/hour. The city is 42 miles wide. How many hours will it take for the fog bank to cover the city?",
  "question_annotated": "A fog bank rolls in over a city at {speed,3} miles/hour. The city is {width,42} miles wide. How many hours will it take for the fog bank to cover the city?\n#init:\n- $speed = range(1, 20)\n- $width = range(2, 100)\n#conditions:\n- is_int(width / speed)\n#answer: width // speed",
  "answer": "At 3 miles/hour, it will take 42/3=14 hours for the fog to cover the city.",
  "answer_annotated": "At {speed} miles/hour, it will take {width}/{speed}={width//speed} hours for the fog to cover the city."
}
```

</details>

<details>
<summary>Example: shopping problem</summary>

```json
{
  "question": "A store sells apples for $2 each and oranges for $3 each. If you buy 4 apples and 5 oranges, how much do you spend?",
  "question_annotated": "A store sells apples for ${apple_price,2} each and oranges for ${orange_price,3} each. If you buy {n_apples,4} apples and {n_oranges,5} oranges, how much do you spend?\n#init:\n- $apple_price = range(1, 10)\n- $orange_price = range(1, 10)\n- $n_apples = range(1, 20)\n- $n_oranges = range(1, 20)\n#conditions:\n- True\n#answer: apple_price * n_apples + orange_price * n_oranges",
  "answer": "You spend 4*2 + 5*3 = 8 + 15 = $23.",
  "answer_annotated": "You spend {n_apples}*{apple_price} + {n_oranges}*{orange_price} = {n_apples*apple_price} + {n_oranges*orange_price} = ${apple_price*n_apples + orange_price*n_oranges}."
}
```

</details>


## 🗃️ Data

The English templates are derived from Apple's [GSM-Symbolic](https://machinelearning.apple.com/research/gsm-symbolic) paper.
The Danish templates are manual translations and localizations of the English set, validated both computationally and manually.
The original concrete problems are from [GSM8k](https://huggingface.co/datasets/openai/gsm8k).

| Language | Code  | Templates |
| -------- | ----- | --------- |
| English  | `eng` | 100       |
| Danish   | `dan` | 100       |


### Writing a custom template

Here is a complete example — a "speed × time = distance" problem with randomised values and a divisibility constraint:

```json
{
  "question": "A car travels at 60 mph for 3 hours. How far does it travel?",
  "answer": "Distance = speed × time = 60 × 3 = 180 miles.\n#### 180",
  "id_orig": 0,
  "id_shuffled": 0,
  "question_annotated": "A car travels at {speed,60} mph for {hours,3} hours. How far does it travel?\n#init:\n- $speed = range(20, 100, 10)\n- $hours = range(1, 9)\n#conditions:\n- is_int(speed * hours / 10)\n#answer: speed * hours",
  "answer_annotated": "Distance = speed × time = {speed} × {hours} = {speed * hours} miles.\n#### {speed * hours}"
}
```

Save it as a `.json` file and load it directly:

```python
from multilingual_gsm_symbolic.gsm_parser import AnnotatedQuestion

template = AnnotatedQuestion.from_json("my_template.json")
questions = template.generate_questions(n=5, language="eng", replacements={})
for q in questions:
    print(q.question)
    print(q.answer)
```

**Init functions** available in `#init` lines:

| Function | Returns |
| -------- | ------- |
| `range(start, end[, step])` | integers in `[start, end)` |
| `arange(start, end[, step])` | evenly-spaced floats |
| `sample(items[, n])` | one item (or `n` items) from a list |
| `sample_sequential(items, n)` | `n` consecutive items from a list |
| `range_str(start, end, step, word_list)` | `(word, int)` pairs, e.g. `("three", 3)` |

**Condition functions** available in `#conditions` lines:

| Function | Returns |
| -------- | ------- |
| `is_int(x)` | `True` if `x` is a whole number |
| `divides(a, b)` | `True` if `a % b == 0` |
| `Fraction(x)` | fraction string, e.g. `"3/4"` |

## 📖 API reference

### <kbd>function</kbd> `load_data`

```python
load_data(language="eng", directory=None) → list[AnnotatedQuestion]
```

Load symbolic templates.

| Argument    | Type                      | Description                                                      |
| ----------- | ------------------------- | ---------------------------------------------------------------- |
| `language`  | `str`                     | Language code, e.g. `"eng"` (default) or `"dan"`                 |
| `directory` | `Path \| None`            | Override the bundled data; load templates from this path instead |
| **RETURNS** | `list[AnnotatedQuestion]` | The loaded templates                                             |

### <kbd>function</kbd> `load_replacements`

```python
load_replacements(language="eng") → dict
```

Load language-specific named values (e.g. lists of names, places) used inside templates.

| Argument    | Type   | Description                              |
| ----------- | ------ | ---------------------------------------- |
| `language`  | `str`  | Language code, e.g. `"eng"` (default)    |
| **RETURNS** | `dict` | Mapping of replacement name → value list |

### <kbd>function</kbd> `load_gsm`

```python
load_gsm(language="eng", directory=None) → list[GSMProblem]
```

Load the bundled concrete problems for a given language.

| Argument    | Type               | Description                           |
| ----------- | ------------------ | ------------------------------------- |
| `language`  | `str`              | Language code, e.g. `"eng"` (default) |
| `directory` | `Path \| None`     | Override the bundled data directory   |
| **RETURNS** | `list[GSMProblem]` | The loaded concrete problems          |

### <kbd>class</kbd> `AnnotatedQuestion`

Core class representing a symbolic template. Constructed from a JSON template file via `AnnotatedQuestion.from_json(path)`.

#### <sup><kbd>method</kbd> `AnnotatedQuestion.generate_questions`</sup>

Generate concrete `Question` instances from the template.

| Argument       | Type             | Description                                 |
| -------------- | ---------------- | ------------------------------------------- |
| `n`            | `int`            | Number of questions to generate             |
| `language`     | `str`            | Language code for rendered text             |
| `replacements` | `dict`           | Replacement values from `load_replacements` |
| **RETURNS**    | `list[Question]` | The generated questions                     |

#### <sup><kbd>method</kbd> `AnnotatedQuestion.get_default_assignments`</sup>

Extract the example variable values from the template.

| Argument       | Type   | Description                                 |
| -------------- | ------ | ------------------------------------------- |
| `replacements` | `dict` | Replacement values from `load_replacements` |
| **RETURNS**    | `dict` | Mapping of variable name → default value    |

#### <sup><kbd>method</kbd> `AnnotatedQuestion.format_question`</sup>

Render the question text for a given variable assignment.

| Argument      | Type   | Description                     |
| ------------- | ------ | ------------------------------- |
| `assignments` | `dict` | Variable name → value mapping   |
| `language`    | `str`  | Language code for rendered text |
| **RETURNS**   | `str`  | The rendered question string    |

#### <sup><kbd>method</kbd> `AnnotatedQuestion.format_answer`</sup>

Render the answer text for a given variable assignment.

| Argument      | Type   | Description                     |
| ------------- | ------ | ------------------------------- |
| `assignments` | `dict` | Variable name → value mapping   |
| `language`    | `str`  | Language code for rendered text |
| **RETURNS**   | `str`  | The rendered answer string      |

### <kbd>class</kbd> `Question`

Dataclass holding a single generated problem.

| Attribute     | Type  | Description                      |
| ------------- | ----- | -------------------------------- |
| `question`    | `str` | The rendered question text       |
| `answer`      | `str` | The rendered answer text         |
| `id_orig`     | `int` | Index of the original template   |
| `id_shuffled` | `int` | Index within the shuffled sample |

### <kbd>class</kbd> `GSMProblem`

Pydantic model for a concrete problem loaded from disk.

| Attribute  | Type   | Description                     |
| ---------- | ------ | ------------------------------- |
| `question` | `str`  | The question text               |
| `answer`   | `str`  | The answer text                 |
| `id_orig`  | `int`  | Original problem index          |
| `filepath` | `Path` | Path to the source file on disk |

## Acknowledgement

The symbolic template engine and the danish subset were originally developed as part of the [m-gsm-symbolic](https://github.com/centre-for-humanities-computing/m-gsm-symbolic) project at the [Centre for Humanities Computing](https://chc.au.dk/) by:

- [Kenneth Enevoldsen](https://github.com/KennethEnevoldsen)
- [Simon Mosegaard](https://github.com/SMosegaard)
- [Enniw](https://github.com/Enniwhere)

The initial template format was derived from Apple's [GSM-Symbolic](https://machinelearning.apple.com/research/gsm-symbolic) paper and the original concrete problems are from [GSM8k](https://huggingface.co/datasets/openai/gsm8k).
