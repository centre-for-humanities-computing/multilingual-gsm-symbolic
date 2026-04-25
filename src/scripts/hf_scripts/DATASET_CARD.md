---
language:
- en
- da
- de
- is
license: mit
task_ids:
  - text-generation
pretty_name: Multilingual GSM-Symbolic
size_categories:
- 1K<n<10K
tags:
- math
- reasoning
- symbolic
- multilingual
---

# Multilingual GSM-Symbolic

**Multilingual GSM-Symbolic** is a benchmark for evaluating arithmetic reasoning in large language models across multiple languages. It extends Apple's [GSM-Symbolic](https://machinelearning.apple.com/research/gsm-symbolic) approach by providing symbolic templates that generate thousands of structurally equivalent but numerically distinct math problems. Templates and generation are handled by the [`multilingual-gsm-symbolic`](https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic) package.

The dataset lets you test whether a model genuinely understands a problem or merely pattern-matches on the specific numbers it saw during training. Comparing performance on `original` vs. `synthetic` splits directly measures this gap.

<img src="figures/example.png" alt="Template illustration showing a symbolic template rendered into English and German examples">

Want to add your own language to the dataset? It only requires validating 100 templates. Read more about how to contribute [here](https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic#want-to-add-a-new-language).

## Dataset Structure

| Split | Description |
|-------|-------------|
| `original` | The concrete GSM problems (one per template, all languages) |
| `synthetic` | 20 generated variants per template per language |
| `original_{lang}` | Original problems for a specific language |
| `synthetic_{lang}` | Synthetic problems for a specific language |

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `question` | string | The math problem |
| `answer` | string | Step-by-step solution ending with `#### <number>` |
| `target` | string | The final numeric answer (extracted from `answer`) |
| `language` | string | Three-letter language code (e.g. `eng`, `dan`) |
| `source_id` | int | Problem index in the original GSM8K dataset |

### Answer format

Answers follow the GSM8K convention — reasoning steps followed by a final numeric answer:

```
At 3 miles/hour, it will take 42/3=14 hours for the fog to cover the city.
#### 14
```

## Loading the Dataset

### With 🤗 datasets

```python
from datasets import load_dataset

# All languages
ds = load_dataset("danish-foundation-models/multilingual-gsm-symbolic", split="synthetic")

# Language-specific split
eng = load_dataset("danish-foundation-models/multilingual-gsm-symbolic", split="synthetic_eng")
deu = load_dataset("danish-foundation-models/multilingual-gsm-symbolic", split="synthetic_deu")
```

### With inspect-ai

You can evaluate with inspect-ai simply using:

```bash
inspect eval hf/danish-foundation-models/multilingual-gsm-symbolic/synthetic_eng --model openai/gpt-5.4-nano --reasoning-effort high
```

you can of course change to the original split or a specific languages if you wish.

## Evaluation Results

As a sanity check for the results we evaluated the model with [inspect-ai](https://inspect.ai) using `openai/gpt-5.4-nano` (reasoning effort: high).
Each split was run with 4 epochs (to estimate degree of certainty in the answers); original splits contain 100 problems × 4 epochs = 400 samples,
synthetic splits contain 2 000 problems × 4 epochs = 8 000 samples. German and Icelandic synthetic splits have fewer problems as some templates are pending human validation.

| Split | Language | Accuracy |
|-------|----------|----------|
| `original_eng` | English | 90.0% |
| `synthetic_eng` | English | 75.2% |
| `original_dan` | Danish | 83.2% |
| `synthetic_dan` | Danish | 70.2% |
| `original_deu` | German | — |
| `synthetic_deu` | German | — |
| `original_isl` | Icelandic | — |
| `synthetic_isl` | Icelandic | — |

The gap between `original` and `synthetic` accuracy reflects performance degradation on novel
number combinations — a proxy for how much a model relies on memorisation vs. genuine reasoning.

### Performance degradation (original → synthetic)

<img src="figures/distribution.png" width="70%" alt="Distribution of synthetic set accuracy with memorisation gap arrow">

### Language gap (English vs. Danish, synthetic)

<img src="figures/language_gap.png" width="70%" alt="Overlapping accuracy distributions for English and Danish">

### Accuracy by problem difficulty (reasoning steps)

<img src="figures/by_steps.png" width="70%" alt="Accuracy as a function of the number of reasoning steps">


### Reproducing the evaluation

```bash
inspect eval hf/danish-foundation-models/multilingual-gsm-symbolic/{subset} \
  --model openai/gpt-5.4-nano --reasoning-effort high --epochs 4
```

Eval logs are available in the [`logs/`](logs/) folder of the
[GitHub repository](https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic).

## Citation

If you use this dataset, please cite the project in which it was constructed (Danish Foundation Models) as well as the original GSM-Symbolic paper:

```bibtex
@misc{enevoldsen2023danishfoundationmodels,
      title={Danish Foundation Models}, 
      author={Kenneth Enevoldsen and Lasse Hansen and Dan S. Nielsen and Rasmus A. F. Egebæk and Søren V. Holm and Martin C. Nielsen and Martin Bernstorff and Rasmus Larsen and Peter B. Jørgensen and Malte Højmark-Bertelsen and Peter B. Vahlstrup and Per Møldrup-Dalum and Kristoffer Nielbo},
      year={2023},
      eprint={2311.07264},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2311.07264}, 
}

@article{mirzadeh2024gsmsymbolic,
  title={GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models},
  author={Mirzadeh, Iman and others},
  year={2024}
}
```

## Acknowledgements

This dataset was developed at the [Centre for Humanities Computing](https://chc.au.dk/) by [Kenneth Enevoldsen](https://github.com/KennethEnevoldsen) using the [`multilingual-gsm-symbolic`](https://github.com/centre-for-humanities-computing/multilingual-gsm-symbolic) package.
