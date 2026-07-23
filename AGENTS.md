# Repository agent notes

## Bulk translation runtime

Use the official `Qwen/Qwen3.5-122B-A10B-FP8` checkpoint for bulk template translation. 
The vLLM server should be started in a tmux session named `vllm_qwen` on port `8000`.

### Server command

Start the Qwen vLLM replica using the following command inside a tmux session:

```bash
tmux new-session -d -s vllm_qwen \
  'CUDA_HOME=/work/.venv/lib/python3.12/site-packages/nvidia/cu13 \
   PATH=/work/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH \
   VLLM_USE_DEEP_GEMM=0 \
   VLLM_MOE_USE_DEEP_GEMM=0 \
   VLLM_USE_FLASHINFER_SAMPLER=0 \
   uv run vllm serve Qwen/Qwen3.5-122B-A10B-FP8 \
     --port 8000 \
     --max-model-len 32768 \
     --max-num-seqs 512 \
     --reasoning-parser qwen3 \
     --moe-backend triton \
     --attention-backend TRITON_ATTN > /work/vllm_qwen.log 2>&1'
```

Check health and logs with:

```bash
curl -f http://127.0.0.1:8000/health
tail -f /work/vllm_qwen.log
```

## Translation request policy

Reasoning is enabled by default by passing `chat_template_kwargs` via `extra_body` directly in the `translate_templates.py` Python script. 
This triggers the Qwen model to emit a large number of internal thinking tokens for each request, drastically improving translation and formatting consistency. 

With reasoning enabled, translation is significantly more compute-heavy:
- Individual `.toml` templates and keys within `replacements.json` take ~50-60 seconds.

## Concurrency and Throughput

To overcome the latency of reasoning models, `translate_templates.py` was rewritten to support high concurrency natively in Python:
- Uses a `ThreadPoolExecutor` with **32 workers** (`max_workers=32`).
- Processes 32 templates simultaneously per language.
- `replacements.json` is also translated concurrently on a key-by-key (object-by-object) basis using the same 32 workers.
- The script checks existing `.toml` files and natively skips templates that have `ignore = true`, rather than infinitely trying to resolve broken edge cases.
- The vLLM server is configured with `--max-num-seqs 512` to handle this heavy batching.

The script also supports a `--to all` argument, which iterates sequentially over all target languages while maintaining the 32x concurrency for the templates and replacements within each language.

## Running bulk translations

To start a full run across all languages (e.g., from `eng_metric` to all languages), you can spawn a single tmux session. You no longer need `xargs` parallelization for the languages since the Python script inherently saturates the GPU with 32 workers per language.

```bash
tmux new-session -d -s translate_all 'uv run src/scripts/translate_templates.py --from eng_metric --to all --subfolder symbolic > translate_all.log 2>&1'
```

The script writes each verified template immediately and logs its success or failure line by line.

## Failure history worth retaining

- The python script requires explicit concurrency. Sequential loops took up to ~160 hours for a full bulk translation. With 32x workers, throughput is maximized against vLLM's continuous batching.
- The `extra_body` payload with `enable_thinking: True` and `reasoning_effort: "high"` is strictly necessary for preserving JSON integrity and translation nuance.
- Qwen-122B uses `--reasoning-parser qwen3` along with `--moe-backend triton` and `--attention-backend TRITON_ATTN` for proper functionality in this environment.
