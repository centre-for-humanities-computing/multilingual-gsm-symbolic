# multilingual-gsm-symbolic

A Python package for generating synthetic multilingual math problems from symbolic templates.
See the [Data](#data) section for available languages.

![Example of a symbolic template and generated questions](https://raw.githubusercontent.com/centre-for-humanities-computing/multilingual-gsm-symbolic/main/images/example.png)

## Installation

```bash
pip install multilingual-gsm-symbolic
```

## Quickstart

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

## Template format

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

### Available helper functions

| Function                          | Description                               |
| --------------------------------- | ----------------------------------------- |
| `range(start, end[, step])`       | All integers in `[start, end)`            |
| `sample([a, b, c])`               | One value from the list                   |
| `range_sample(start, end, step)`  | Uniform sample from a range               |
| `sample_sequential(items, n)`     | `n` consecutive items from a list         |
| `arange_sample(start, end, step)` | Sample from `np.arange(start, end, step)` |
| `is_int(x)`                       | True if `x` is an integer                 |
| `divides(a, b)`                   | True if `a` divides `b`                   |
| `frac_format(x)`                  | Format `x` as a fraction string           |

## 📖 API reference

### `load_data(language="eng", directory=None) → list[AnnotatedQuestion]`

Load symbolic templates.

- `language` — `"eng"` (default) or `"dan"`, or any language code for which a template folder exists
- `directory` — override the bundled data; load templates from this path instead

### `load_replacements(language="eng") → dict`

Load language-specific named values (e.g. lists of names, places) used inside templates.

### `load_gsm(language="eng", directory=None) → list[GSMProblem]`

Load the bundled concrete problems for a given language.

### `AnnotatedQuestion`

Core class. Constructed from a JSON template file via `AnnotatedQuestion.from_json(path)`.

Key methods:

| Method                                          | Description                                           |
| ----------------------------------------------- | ----------------------------------------------------- |
| `generate_questions(n, language, replacements)` | Generate `n` concrete `Question` instances            |
| `get_default_assignments(replacements)`         | Extract the example variable values from the template |
| `format_question(assignments, language)`        | Render the question text for a given assignment       |
| `format_answer(assignments, language)`          | Render the answer text for a given assignment         |

### `Question`

Dataclass holding a single generated problem: `question`, `answer`, `id_orig`, `id_shuffled`.

### `GSMProblem`

Pydantic model for a concrete problem loaded from disk: `question`, `answer`, `id_orig`, `filepath`.

## Data

The English templates are derived from Apple's [GSM-Symbolic](https://machinelearning.apple.com/research/gsm-symbolic) paper.
The Danish templates are manual translations and localizations of the English set, validated both computationally and manually.
The original concrete problems are from [GSM8k](https://huggingface.co/datasets/openai/gsm8k).

| Language | Code  | Templates |
| -------- | ----- | --------- |
| English  | `eng` | 100       |
| Danish   | `dan` | 100       |

## Acknowledgement

The symbolic template engine and the danish subset were originally developed as part of the [m-gsm-symbolic](https://github.com/centre-for-humanities-computing/m-gsm-symbolic) project at the [Centre for Humanities Computing](https://chc.au.dk/) by:

- [Kenneth Enevoldsen](https://github.com/KennethEnevoldsen)
- [Simon Mosegaard](https://github.com/SMosegaard)
- [Enniw](https://github.com/Enniwhere)

The initial template format was derived from Apple's [GSM-Symbolic](https://machinelearning.apple.com/research/gsm-symbolic) paper and the original concrete problems are from [GSM8k](https://huggingface.co/datasets/openai/gsm8k).
