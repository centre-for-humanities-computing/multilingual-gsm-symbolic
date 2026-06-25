# Italian Inspect AI Evaluation

This guide explains how to run the Italian GSM-Symbolic Inspect AI tasks with
an OpenAI-compatible SGLang or vLLM server.

The Italian tasks live in `eval_ita.py`:

- `eval_ita.py@synthetic_ita`: generated variants from the Italian symbolic templates.
- `eval_ita.py@original_ita`: the localized default examples from the same templates.

Both tasks load only:

```text
src/multilingual_gsm_symbolic/data/templates/ita/symbolic
```

Templates in `ita/exclude` and the language-level `ita/ignore` marker are not
used by these evals.

## Setup

From the repository root:

```bash
uv sync
```

Verify that Inspect can see the Italian tasks:

```bash
./.venv/bin/inspect list tasks eval_ita.py
```

Expected output:

```text
eval_ita.py@original_ita
eval_ita.py@synthetic_ita
```

## Run With SGLang

Start an SGLang OpenAI-compatible server in one terminal. This example uses
port `8000`; SGLang often defaults to `30000`, so keep the port consistent with
the Inspect command.

```bash
python -m sglang.launch_server \
  --model-path google/gemma-3-12b-it \
  --host 0.0.0.0 \
  --port 8000
```

Check that the server is reachable:

```bash
curl http://localhost:8000/v1/models
```

Run the synthetic Italian eval:

```bash
export SGLANG_API_KEY=dummy

./.venv/bin/inspect eval eval_ita.py@synthetic_ita \
  --model sglang/google/gemma-3-12b-it \
  --model-base-url http://localhost:8000/v1 \
  --max-connections 4 \
  --max-tokens 2048 \
  --temperature 0 \
  --log-dir logs/ita-gemma-3-12b-it-sglang
```

Run the original/default Italian eval:

```bash
export SGLANG_API_KEY=dummy

./.venv/bin/inspect eval eval_ita.py@original_ita \
  --model sglang/google/gemma-3-12b-it \
  --model-base-url http://localhost:8000/v1 \
  --max-connections 4 \
  --max-tokens 2048 \
  --temperature 0 \
  --log-dir logs/ita-gemma-3-12b-it-sglang
```

## Run With vLLM

Start a vLLM OpenAI-compatible server in one terminal:

```bash
export HF_TOKEN=your_huggingface_token

vllm serve google/gemma-3-12b-it \
  --host 0.0.0.0 \
  --port 8000
```

Check that the server is reachable:

```bash
curl http://localhost:8000/v1/models
```

Run the synthetic Italian eval:

```bash
export VLLM_API_KEY=dummy

./.venv/bin/inspect eval eval_ita.py@synthetic_ita \
  --model vllm/google/gemma-3-12b-it \
  --model-base-url http://localhost:8000/v1 \
  --max-connections 4 \
  --max-tokens 2048 \
  --temperature 0 \
  --log-dir logs/ita-gemma-3-12b-it-vllm
```

Run the original/default Italian eval:

```bash
export VLLM_API_KEY=dummy

./.venv/bin/inspect eval eval_ita.py@original_ita \
  --model vllm/google/gemma-3-12b-it \
  --model-base-url http://localhost:8000/v1 \
  --max-connections 4 \
  --max-tokens 2048 \
  --temperature 0 \
  --log-dir logs/ita-gemma-3-12b-it-vllm
```

## More Synthetic Variants

By default, `synthetic_ita` generates one variant per template. To run more:

```bash
./.venv/bin/inspect eval eval_ita.py@synthetic_ita \
  -T variants_per_template=3 \
  --model sglang/google/gemma-3-12b-it \
  --model-base-url http://localhost:8000/v1 \
  --max-connections 4 \
  --max-tokens 2048 \
  --temperature 0 \
  --log-dir logs/ita-gemma-3-12b-it-sglang-v3
```

Use the same `-T variants_per_template=3` argument with `vllm/...` if running against vLLM.

## Inspect Results

Results are saved in the `logs/` directory and are easily inspectable using the inspect AI vscode extension.
