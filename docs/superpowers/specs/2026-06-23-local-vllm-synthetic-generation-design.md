# Local vLLM Synthetic Generation Design

## Goal

Generate synthetic templates with a locally launched Qwen3-235B-A22B-Thinking-2507 vLLM server, using NVFP4 quantization and up to ten validation attempts per template.

## Design

`createsynthetictemplates.py` starts `vllm serve Qwen/Qwen3-235B-A22B-Thinking-2507 --quantization nvfp4` on localhost, waits for its `/health` endpoint, and terminates only the process it launched when generation ends. The existing OpenAI SDK connects to vLLM's OpenAI-compatible `/v1` endpoint and uses chat completions; prompt and validation-feedback messages stay unchanged.

`MAX_TEMPLATE_ATTEMPTS` becomes 10. The existing three generation workers and validation/logging behavior remain unchanged.

## Error Handling and Verification

The launcher reports server exit or startup timeout before generating. A unit test will assert the vLLM command includes the model and NVFP4 quantization, and that the request path targets the local compatible endpoint without a real server.
