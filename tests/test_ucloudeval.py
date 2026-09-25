import importlib.machinery
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "paper" / "scripts" / "ucloudeval"
loader = importlib.machinery.SourceFileLoader("ucloudeval", str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
ucloudeval = importlib.util.module_from_spec(spec)
sys.modules[loader.name] = ucloudeval
loader.exec_module(ucloudeval)


def test_task_fingerprint_uses_data_prompt_model_and_reasoning(monkeypatch):
    monkeypatch.setattr(ucloudeval, "parquet_digest", lambda *_: "data-a")
    task = ucloudeval.Task("original_eng", "eng", "test_original", "prompt-a")
    run = ucloudeval.ModelRun("Qwen/Qwen3-4B", reasoning="on")
    base = ucloudeval.task_fingerprint(task, run, "revision-a")

    assert base == ucloudeval.task_fingerprint(task, run, "revision-b")
    assert base != ucloudeval.task_fingerprint(
        ucloudeval.Task("original_eng", "eng", "test_original", "prompt-b"),
        run,
        "revision-a",
    )
    assert base != ucloudeval.task_fingerprint(
        task, ucloudeval.ModelRun("Qwen/Qwen3-4B", reasoning="off"), "revision-a"
    )
    assert base != ucloudeval.task_fingerprint(
        task, ucloudeval.ModelRun("Qwen/Qwen3-8B", reasoning="on"), "revision-a"
    )


def test_task_fingerprint_changes_with_parquet(monkeypatch):
    task = ucloudeval.Task("original_eng", "eng", "test_original", "prompt")
    run = ucloudeval.ModelRun("model")
    monkeypatch.setattr(ucloudeval, "parquet_digest", lambda *_: "data-a")
    first = ucloudeval.task_fingerprint(task, run, "main")
    monkeypatch.setattr(ucloudeval, "parquet_digest", lambda *_: "data-b")
    assert first != ucloudeval.task_fingerprint(task, run, "main")


def test_task_fingerprint_ignores_model_provider_prefix(monkeypatch):
    monkeypatch.setattr(ucloudeval, "parquet_digest", lambda *_: "data")
    task = ucloudeval.Task("original_eng", "eng", "test_original", "prompt")
    assert ucloudeval.task_fingerprint(
        task, ucloudeval.ModelRun("vllm/Qwen/Qwen3-4B", reasoning="on"), "main"
    ) == ucloudeval.task_fingerprint(
        task, ucloudeval.ModelRun("openai/Qwen/Qwen3-4B", reasoning="on"), "main"
    )


def test_inspect_command_uses_stock_eval_set_identity():
    task = ucloudeval.Task("original_eng", "eng", "test_original", "prompt")
    command = ucloudeval.inspect_command(ucloudeval.ModelRun("model"), [task])
    assert "--id" not in command


def test_attention_backend_is_model_specific():
    assert ucloudeval.attention_backend("swiss-ai/Apertus-70B-Instruct-2509") == "FLASH_ATTN"
    assert ucloudeval.attention_backend("google/gemma-3-27b-it") == "FLASH_ATTN"
    assert ucloudeval.attention_backend("google/gemma-4-12B-it") == "TRITON_ATTN"


def test_vllm_environment_sets_selected_attention_backend(monkeypatch):
    env = ucloudeval.vllm_environment({}, "swiss-ai/Apertus-70B-Instruct-2509")
    server_args = ucloudeval.json.loads(env["VLLM_DEFAULT_SERVER_ARGS"])
    assert server_args["attention_backend"] == "FLASH_ATTN"


def test_explicit_attention_backend_override_wins():
    env = ucloudeval.vllm_environment(
        {"attention_backend": "FLEX_ATTENTION"},
        "swiss-ai/Apertus-70B-Instruct-2509",
    )
    server_args = ucloudeval.json.loads(env["VLLM_DEFAULT_SERVER_ARGS"])
    assert server_args["attention_backend"] == "FLEX_ATTENTION"


def test_run_inspect_skips_completed_and_records_success(tmp_path, monkeypatch):
    done = ucloudeval.Task("done", "eng", "test_original", "prompt")
    pending = ucloudeval.Task("pending", "eng", "test_synthetic", "prompt")
    run = ucloudeval.ModelRun("model")
    monkeypatch.setattr(
        ucloudeval,
        "task_fingerprint",
        lambda task, _run, _revision: task.task_id,
    )
    monkeypatch.setattr(
        ucloudeval,
        "inspect_targets",
        lambda *_args: [(tmp_path, "main", [done, pending])],
    )
    monkeypatch.setattr(ucloudeval, "vllm_environment", lambda *_: {})
    monkeypatch.setattr(ucloudeval, "cleanup_vllm_processes", lambda: None)
    commands = []
    monkeypatch.setattr(
        ucloudeval.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )
    ucloudeval.save_completed_tasks(tmp_path, {"done"})

    ucloudeval.run_inspect(run, [done, pending], False, tmp_path, "main", None, {})

    assert any("pending@main" in part for part in commands[0])
    assert not any("done@main" in part for part in commands[0])
    assert ucloudeval.load_completed_tasks(tmp_path) == {"done", "pending"}
